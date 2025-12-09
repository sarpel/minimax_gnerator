"""Batch Processor

Async batch processing engine for parallel audio sample generation.
Handles task distribution, rate limiting, and error handling.

Features:
- Async task processing with configurable concurrency
- Rate limiting per provider
- Automatic retry with exponential backoff
- Progress tracking integration
- Error handling and logging
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import cast

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from wakegen.core.exceptions import GenerationError, ProviderError
from wakegen.core.protocols import TTSProvider
from wakegen.core.types import ProviderType
from wakegen.generation.progress import ProgressTracker
from wakegen.generation.rate_limiter import RateLimiter
from wakegen.models.audio import AudioSample
from wakegen.models.generation import GenerationParameters, GenerationResult

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for batch audio generation processing.

    Attributes:
        max_concurrent_tasks: Maximum number of concurrent generation tasks
        retry_attempts: Number of retry attempts for failed tasks
        timeout_seconds: Timeout for individual generation tasks
        rate_limits: Rate limits per provider type
    """

    max_concurrent_tasks: int = 5
    retry_attempts: int = 3
    timeout_seconds: int = 300
    rate_limits: dict[str, tuple[int, int]] = None  # type: ignore

    def __post_init__(self) -> None:
        """Initialize default rate limits if not provided."""
        if self.rate_limits is None:
            self.rate_limits = {
                "commercial": (
                    10,
                    60,
                ),  # 10 requests per minute for commercial providers
                "free": (5, 60),  # 5 requests per minute for free providers
            }


class BatchProcessor:
    """Async batch processor for parallel audio sample generation.

    This class handles:
    - Distributing generation tasks across multiple providers
    - Rate limiting to respect API constraints
    - Retry logic for transient failures
    - Progress tracking and reporting
    - Error handling and logging
    """

    def __init__(self, config: BatchConfig):
        """Initialize the batch processor.

        Args:
            config: Batch processing configuration
        """
        self.config = config
        self.rate_limiters: dict[str, RateLimiter] = {}
        self.progress_tracker: ProgressTracker | None = None

        # Initialize rate limiters for each provider type
        for provider_type, (max_requests, period_seconds) in config.rate_limits.items():
            self.rate_limiters[provider_type] = RateLimiter(
                max_requests=max_requests, period_seconds=period_seconds
            )

    def set_progress_tracker(self, progress_tracker: ProgressTracker) -> None:
        """Set the progress tracker for this batch processor.

        Args:
            progress_tracker: Progress tracker instance
        """
        self.progress_tracker = progress_tracker

    async def _process_single_task(
        self, provider: TTSProvider, params: GenerationParameters, task_id: str
    ) -> GenerationResult:
        """Process a single generation task with retry logic.

        Args:
            provider: Audio provider instance
            params: Generation parameters
            task_id: Unique task identifier

        Returns:
            GenerationResult with the generated audio

        Raises:
            GenerationError: If task fails after all retry attempts
        """
        provider_type = (
            "commercial"
            if hasattr(provider, "_is_commercial") and provider._is_commercial
            else "free"
        )

        @retry(
            stop=stop_after_attempt(self.config.retry_attempts),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            retry=retry_if_exception_type((ProviderError, asyncio.TimeoutError)),
            reraise=True,
        )
        async def _generate_with_retry() -> GenerationResult:
            # Apply rate limiting
            await self.rate_limiters[provider_type].wait_for_token()

            # Update progress
            if self.progress_tracker:
                await self.progress_tracker.update_task_status(task_id, "processing")

            try:
                # Issue C-002 Fix: Use correct method signature from TTSProvider protocol
                # provider.generate(text, voice_id, output_path) instead of generate_audio(params)
                output_path = self._generate_output_path(params)

                # Generate audio with timeout
                await asyncio.wait_for(
                    provider.generate(params.text, params.voice_id, output_path),
                    timeout=self.config.timeout_seconds,
                )

                # Create GenerationResult from the generated audio
                result = self._create_generation_result(params, output_path)

                # Update progress on success
                if self.progress_tracker:
                    await self.progress_tracker.update_task_status(task_id, "completed")

                return result

            except asyncio.TimeoutError:
                logger.warning(
                    f"Task {task_id} timed out after {self.config.timeout_seconds} seconds"
                )
                if self.progress_tracker:
                    await self.progress_tracker.update_task_status(task_id, "timeout")
                raise

            except ProviderError as e:
                logger.warning(f"Provider error for task {task_id}: {e!s}")
                if self.progress_tracker:
                    await self.progress_tracker.update_task_status(
                        task_id, "provider_error"
                    )
                raise

            except Exception as e:
                logger.error(f"Unexpected error in task {task_id}: {e!s}")
                if self.progress_tracker:
                    await self.progress_tracker.update_task_status(task_id, "error")
                raise GenerationError(
                    f"Generation failed for task {task_id}: {e!s}"
                ) from e

        return await _generate_with_retry()

    def _generate_output_path(self, params: GenerationParameters) -> str:
        """Generate a unique output path for the audio file.

        Args:
            params: Generation parameters

        Returns:
            Unique file path for the generated audio
        """
        # Create temp directory for generated audio
        temp_dir = tempfile.gettempdir()
        audio_dir = os.path.join(temp_dir, "wakegen_audio")
        os.makedirs(audio_dir, exist_ok=True)

        # Create unique filename based on text, voice, and timestamp
        timestamp = int(time.time() * 1000)
        safe_text = params.text[:20].replace(" ", "_").replace("/", "_")
        filename = f"{safe_text}_{params.voice_id}_{timestamp}.wav"
        return os.path.join(audio_dir, filename)

    def _create_generation_result(
        self, params: GenerationParameters, output_path: str
    ) -> GenerationResult:
        """Create a GenerationResult from generated audio.

        Args:
            params: Original generation parameters
            output_path: Path to the generated audio file

        Returns:
            GenerationResult with audio metadata
        """
        # Get audio duration if file exists
        duration = None
        if os.path.exists(output_path):
            try:
                import soundfile as sf

                info = sf.info(output_path)
                duration = info.duration
            except Exception:
                pass

        audio_sample = AudioSample(
            file_path=output_path,
            text=params.text,
            voice_id=params.voice_id,
            provider=cast(
                ProviderType, getattr(self, "_current_provider_type", "edge_tts")
            ),
            duration_seconds=duration,
        )

        return GenerationResult(
            parameters=params,
            audio_data=audio_sample,
            generation_time=0.0,  # Could be tracked if needed
            provider_used=audio_sample.provider,
            success=True,
            error_message=None,
        )

    async def _worker(
        self,
        provider: TTSProvider,
        task_queue: asyncio.Queue[tuple[str, GenerationParameters]],
        results_queue: asyncio.Queue[
            tuple[str, GenerationResult | None, Exception | None]
        ],
    ) -> None:
        """Worker coroutine that processes tasks from the queue.

        Args:
            provider: TTS provider instance
            task_queue: Queue for incoming tasks (task_id, params)
            results_queue: Queue for results (task_id, result, error)
        """
        while True:
            try:
                task_id, params = await task_queue.get()

                try:
                    # Process the task
                    result = await self._process_single_task(provider, params, task_id)

                    # Put result in results queue
                    await results_queue.put((task_id, result, None))

                except Exception as e:
                    # Put error in results queue
                    await results_queue.put((task_id, None, e))

                finally:
                    task_queue.task_done()

            except asyncio.CancelledError:
                # Worker was cancelled, exit gracefully
                break
            except Exception as e:
                logger.error(f"Worker error: {e!s}")
                break

    async def process_batch(
        self, provider: TTSProvider, tasks: list[tuple[str, GenerationParameters]]
    ) -> AsyncIterator[tuple[str, GenerationResult | None, Exception | None]]:
        """Process a batch of generation tasks asynchronously.

        Args:
            provider: Audio provider instance
            tasks: List of (task_id, GenerationParameters) tuples

        Yields:
            Tuple of (task_id, result, error) for each completed task
        """
        if not tasks:
            return

        # Initialize progress if tracker is set
        if self.progress_tracker:
            await self.progress_tracker.initialize_batch(len(tasks))

        # Create queues
        task_queue: asyncio.Queue[tuple[str, GenerationParameters]] = asyncio.Queue()
        results_queue: asyncio.Queue[
            tuple[str, GenerationResult | None, Exception | None]
        ] = asyncio.Queue()

        # Put all tasks in the queue
        for task_id, params in tasks:
            await task_queue.put((task_id, params))

        # Create worker tasks
        worker_tasks = []
        for _ in range(min(self.config.max_concurrent_tasks, len(tasks))):
            worker_task = asyncio.create_task(
                self._worker(provider, task_queue, results_queue)
            )
            worker_tasks.append(worker_task)

        # Process results as they come in
        results_processed = 0
        while results_processed < len(tasks):
            task_id, result, error = await results_queue.get()
            yield task_id, result, error
            results_processed += 1

            # Update overall progress
            if self.progress_tracker:
                await self.progress_tracker.update_overall_progress(
                    results_processed, len(tasks)
                )

        # Wait for all tasks to be processed
        await task_queue.join()

        # Cancel worker tasks
        for worker_task in worker_tasks:
            worker_task.cancel()

        # Wait for workers to finish
        await asyncio.gather(*worker_tasks, return_exceptions=True)

        # Finalize progress
        if self.progress_tracker:
            await self.progress_tracker.finalize_batch()

    async def process_with_fallback(
        self,
        primary_provider: TTSProvider,
        fallback_providers: list[TTSProvider],
        tasks: list[tuple[str, GenerationParameters]],
    ) -> AsyncIterator[tuple[str, GenerationResult | None, Exception | None]]:
        """Process tasks with fallback to other providers if primary fails.

        Args:
            primary_provider: Primary audio provider
            fallback_providers: List of fallback providers
            tasks: List of (task_id, GenerationParameters) tuples

        Yields:
            Tuple of (task_id, result, error) for each completed task
        """
        # First try with primary provider
        # Issue M-003 Fix: Track task params properly for fallback
        task_params_map = {task_id: params for task_id, params in tasks}

        async for task_id, result, error in self.process_batch(primary_provider, tasks):
            if error is None:
                # Success with primary provider
                yield task_id, result, None
            else:
                # Try with fallback providers - use the actual failed task's params
                actual_params = task_params_map.get(task_id)
                if actual_params is None:
                    yield task_id, None, error
                    continue

                fallback_error = None
                fallback_success = False
                for fallback_provider in fallback_providers:
                    try:
                        fallback_result = await self._process_single_task(
                            fallback_provider, actual_params, task_id
                        )
                        yield task_id, fallback_result, None
                        fallback_success = True
                        break
                    except Exception as e:
                        fallback_error = e
                        continue

                if not fallback_success and fallback_error:
                    yield task_id, None, fallback_error

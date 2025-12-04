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
from typing import List, Dict, Any, Optional, AsyncIterator, Tuple
from collections import defaultdict
from dataclasses import dataclass

import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from wakegen.generation.rate_limiter import RateLimiter
from wakegen.generation.progress import ProgressTracker
from wakegen.models.generation import GenerationParameters, GenerationResult
from wakegen.core.exceptions import GenerationError, ProviderError
from wakegen.core.protocols import TTSProvider

logger = logging.getLogger(__name__)

@dataclass
class BatchConfig:
    """Configuration for batch processing.

    Attributes:
        max_concurrent_tasks: Maximum number of concurrent generation tasks
        retry_attempts: Number of retry attempts for failed tasks
        timeout_seconds: Timeout for individual generation tasks
        rate_limits: Rate limits per provider type
    """
    max_concurrent_tasks: int = 5
    retry_attempts: int = 3
    timeout_seconds: int = 300
    rate_limits: Dict[str, Tuple[int, int]] = None  # provider_type: (max_requests, period_seconds)

    def __post_init__(self):
        """Initialize default rate limits if not provided."""
        if self.rate_limits is None:
            self.rate_limits = {
                "commercial": (10, 60),  # 10 requests per minute for commercial providers
                "free": (5, 60),         # 5 requests per minute for free providers
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
        self.rate_limiters: Dict[str, RateLimiter] = {}
        self.progress_tracker: Optional[ProgressTracker] = None

        # Initialize rate limiters for each provider type
        for provider_type, (max_requests, period_seconds) in config.rate_limits.items():
            self.rate_limiters[provider_type] = RateLimiter(
                max_requests=max_requests,
                period_seconds=period_seconds
            )

    def set_progress_tracker(self, progress_tracker: ProgressTracker) -> None:
        """Set the progress tracker for this batch processor.

        Args:
            progress_tracker: Progress tracker instance
        """
        self.progress_tracker = progress_tracker

    async def _process_single_task(
        self,
        provider: TTSProvider,
        params: GenerationParameters,
        task_id: str
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
        provider_type = "commercial" if hasattr(provider, '_is_commercial') and provider._is_commercial else "free"

        @retry(
            stop=stop_after_attempt(self.config.retry_attempts),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            retry=retry_if_exception_type((ProviderError, asyncio.TimeoutError)),
            reraise=True
        )
        async def _generate_with_retry():
            # Apply rate limiting
            await self.rate_limiters[provider_type].wait_for_token()

            # Update progress
            if self.progress_tracker:
                await self.progress_tracker.update_task_status(task_id, "processing")

            try:
                # Generate audio with timeout
                result = await asyncio.wait_for(
                    provider.generate_audio(params),
                    timeout=self.config.timeout_seconds
                )

                # Update progress on success
                if self.progress_tracker:
                    await self.progress_tracker.update_task_status(task_id, "completed")

                return result

            except asyncio.TimeoutError:
                logger.warning(f"Task {task_id} timed out after {self.config.timeout_seconds} seconds")
                if self.progress_tracker:
                    await self.progress_tracker.update_task_status(task_id, "timeout")
                raise

            except ProviderError as e:
                logger.warning(f"Provider error for task {task_id}: {str(e)}")
                if self.progress_tracker:
                    await self.progress_tracker.update_task_status(task_id, "provider_error")
                raise

            except Exception as e:
                logger.error(f"Unexpected error in task {task_id}: {str(e)}")
                if self.progress_tracker:
                    await self.progress_tracker.update_task_status(task_id, "error")
                raise GenerationError(f"Generation failed for task {task_id}: {str(e)}") from e

        return await _generate_with_retry()

    async def _worker(
        self,
        provider: TTSProvider,
        task_queue: asyncio.Queue,
        results_queue: asyncio.Queue
    ) -> None:
        """Worker coroutine that processes tasks from the queue.

        Args:
            provider: Audio provider instance
            task_queue: Queue of tasks to process
            results_queue: Queue to put results
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
                logger.error(f"Worker error: {str(e)}")
                break

    async def process_batch(
        self,
        provider: TTSProvider,
        tasks: List[Tuple[str, GenerationParameters]]
    ) -> AsyncIterator[Tuple[str, Optional[GenerationResult], Optional[Exception]]]:
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
        task_queue = asyncio.Queue()
        results_queue = asyncio.Queue()

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
                await self.progress_tracker.update_overall_progress(results_processed, len(tasks))

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
        fallback_providers: List[TTSProvider],
        tasks: List[Tuple[str, GenerationParameters]]
    ) -> AsyncIterator[Tuple[str, Optional[GenerationResult], Optional[Exception]]]:
        """Process tasks with fallback to other providers if primary fails.

        Args:
            primary_provider: Primary audio provider
            fallback_providers: List of fallback providers
            tasks: List of (task_id, GenerationParameters) tuples

        Yields:
            Tuple of (task_id, result, error) for each completed task
        """
        # First try with primary provider
        async for task_id, result, error in self.process_batch(primary_provider, tasks):
            if error is None:
                # Success with primary provider
                yield task_id, result, None
            else:
                # Try with fallback providers
                fallback_error = None
                for fallback_provider in fallback_providers:
                    try:
                        fallback_result = await self._process_single_task(
                            fallback_provider, tasks[0][1], task_id  # Use first task's params as example
                        )
                        yield task_id, fallback_result, None
                        break
                    except Exception as e:
                        fallback_error = e
                        continue

                if fallback_error:
                    yield task_id, None, fallback_error
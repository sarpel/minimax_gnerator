"""Generation Orchestrator

Main coordinator for the wake word generation engine.
Orchestrates all components to generate diverse audio samples.

Responsibilities:
- Coordinate variation engine, batch processor, checkpoint system
- Manage provider selection and fallback
- Handle generation lifecycle (start, pause, resume, cancel)
- Provide high-level API for generation sessions
- Manage resource cleanup
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

from wakegen.core.exceptions import ConfigError, GenerationError
from wakegen.core.protocols import TTSProvider
from wakegen.core.types import ProviderType
from wakegen.generation.batch_processor import BatchConfig, BatchProcessor
from wakegen.generation.checkpoint import CheckpointConfig, CheckpointManager
from wakegen.generation.progress import ProgressConfig, ProgressTracker
from wakegen.generation.variation_engine import VariationEngine, VariationParameters
from wakegen.models.config import GenerationConfig
from wakegen.models.generation import GenerationParameters, GenerationResult
from wakegen.providers.registry import get_provider

logger = logging.getLogger(__name__)


class GenerationOrchestrator:
    """Main orchestrator for wake word audio sample generation.

    This class coordinates all generation components to create diverse audio samples.
    It handles the complete lifecycle of generation sessions including:
    - Variation parameter generation
    - Batch processing with rate limiting
    - Checkpoint and resume capability
    - Progress tracking and reporting
    - Error handling and recovery
    """

    def __init__(self, config: GenerationConfig):
        """Initialize the generation orchestrator.

        Args:
            config: Generation configuration
        """
        self.config = config
        self._session_id = str(uuid.uuid4())

        # Initialize components
        self.variation_engine: VariationEngine | None = None
        self.batch_processor: BatchProcessor | None = None
        self.checkpoint_manager: CheckpointManager | None = None
        self.progress_tracker: ProgressTracker | None = None

        # Task tracking
        self._current_checkpoint_id: str | None = None
        self._task_count = 0

        # Initialize components
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize all generation components."""
        # Initialize checkpoint manager
        checkpoint_config = CheckpointConfig(
            db_path=getattr(self.config, "checkpoint_db_path", "checkpoints.db"),
            cleanup_interval=getattr(self.config, "checkpoint_cleanup_interval", 3600),
            max_checkpoints=getattr(self.config, "max_checkpoints", 10),
        )
        self.checkpoint_manager = CheckpointManager(checkpoint_config)

        # Initialize progress tracker
        progress_config = ProgressConfig(
            refresh_rate=getattr(self.config, "progress_refresh_rate", 0.1),
            show_task_details=getattr(self.config, "show_task_details", True),
            console_width=getattr(self.config, "console_width", 80),
        )
        self.progress_tracker = ProgressTracker(progress_config)

        # Initialize batch processor
        batch_config = BatchConfig(
            max_concurrent_tasks=getattr(self.config, "max_concurrent_tasks", 5),
            retry_attempts=getattr(self.config, "retry_attempts", 3),
            timeout_seconds=getattr(self.config, "task_timeout_seconds", 300),
            rate_limits=getattr(
                self.config, "rate_limits", {"commercial": (10, 60), "free": (5, 60)}
            ),
        )
        self.batch_processor = BatchProcessor(batch_config)
        self.batch_processor.set_progress_tracker(self.progress_tracker)

    async def generate(
        self,
        wake_words: list[str],
        count: int,
        output_dir: str,
        voice_ids: list[str] | None = None,
        resume_from_checkpoint: str | None = None,
    ) -> list[GenerationResult]:
        """Generate audio samples for the given wake words.

        Args:
            wake_words: List of wake words to generate samples for
            count: Number of samples to generate
            output_dir: Directory to save generated audio files
            voice_ids: List of voice IDs to use (optional)
            resume_from_checkpoint: Checkpoint ID to resume from (optional)

        Returns:
            List of successfully generated results

        Raises:
            GenerationError: If generation fails
        """
        try:
            # Validate inputs
            if not wake_words:
                raise GenerationError("At least one wake word must be provided")

            if count <= 0:
                raise GenerationError("Count must be positive")

            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Check if resuming from checkpoint
            if resume_from_checkpoint:
                return await self._resume_generation(resume_from_checkpoint, output_dir)

            # Create new generation session
            return await self._start_new_generation(
                wake_words, count, output_dir, voice_ids
            )

        except Exception as e:
            logger.error(f"Generation failed: {e!s}")
            if self.progress_tracker:
                await self.progress_tracker.show_error(str(e))
            raise GenerationError(f"Generation failed: {e!s}") from e

    async def _start_new_generation(
        self,
        wake_words: list[str],
        count: int,
        output_dir: str,
        voice_ids: list[str] | None = None,
    ) -> list[GenerationResult]:
        """Start a new generation session.

        Args:
            wake_words: List of wake words
            count: Number of samples to generate
            output_dir: Output directory
            voice_ids: List of voice IDs

        Returns:
            List of generation results
        """
        # Create checkpoint ID
        self._current_checkpoint_id = str(uuid.uuid4())

        # Determine voice IDs to use
        if voice_ids is None:
            # Use default voices from config
            voice_ids = self.config.default_voice_ids or [
                "tr-TR-PinarNeural",
                "tr-TR-AhmetNeural",
            ]

        # Create variation parameters
        variation_params = VariationParameters(
            text_variations=wake_words,
            voice_ids=voice_ids,
            speed_range=self.config.speed_range or (0.8, 1.2),
            pitch_range=self.config.pitch_range or (0.9, 1.1),
        )

        # Initialize variation engine
        self.variation_engine = VariationEngine(variation_params)

        # Estimate total combinations needed
        total_combinations = self.variation_engine.estimate_total_combinations()
        combinations_needed = min(count, total_combinations)

        # Create checkpoint
        checkpoint_config = {
            "wake_words": wake_words,
            "count": count,
            "output_dir": output_dir,
            "voice_ids": voice_ids,
            # Issue M-007 Fix: Use asdict() for dataclass instead of .model_dump()
            "variation_params": asdict(variation_params),
        }

        if self.variation_engine is None:
            raise GenerationError("Variation engine not initialized")
        if self.checkpoint_manager is None:
            raise GenerationError("Checkpoint manager not initialized")

        await self.checkpoint_manager.create_checkpoint(
            session_id=self._session_id,
            checkpoint_id=self._current_checkpoint_id,
            total_tasks=combinations_needed,
            config=checkpoint_config,
        )

        # Generate parameter combinations
        params_list = []
        task_ids = []

        async for params in self._generate_parameters_with_checkpoint(
            combinations_needed
        ):
            task_id = f"task_{self._task_count}"
            params_list.append(params)
            task_ids.append(task_id)
            self._task_count += 1

            # Save initial task state
            await self.checkpoint_manager.save_task_state(
                checkpoint_id=self._current_checkpoint_id,
                task_id=task_id,
                status="pending",
                parameters=params,
            )

            if len(params_list) >= combinations_needed:
                break

        # Initialize progress tracking
        if self.progress_tracker is None:
            # Just warn, don't fail if progress tracker is missing
            logger.warning("Progress tracker not initialized")
        else:
            await self.progress_tracker.initialize_batch(len(params_list))

        # Process batch
        if self.batch_processor is None:
            raise GenerationError("Batch processor not initialized")

        results = []
        async for task_id, result, error in self.batch_processor.process_batch(
            provider=self._get_primary_provider(),
            tasks=list(zip(task_ids, params_list)),
        ):
            if result:
                # Save successful result
                await self._save_generated_result(result, output_dir)
                results.append(result)

                # Save checkpoint progress
                await self.checkpoint_manager.save_task_state(
                    checkpoint_id=self._current_checkpoint_id,
                    task_id=task_id,
                    status="completed",
                    result=result,
                )
            else:
                # Save error state
                await self.checkpoint_manager.save_task_state(
                    checkpoint_id=self._current_checkpoint_id,
                    task_id=task_id,
                    status="failed",
                    error=error,
                )

        # Mark checkpoint as completed
        await self.checkpoint_manager.mark_checkpoint_completed(
            self._current_checkpoint_id
        )

        return results

    async def _generate_parameters_with_checkpoint(
        self, max_combinations: int
    ) -> AsyncIterator[GenerationParameters]:
        """Generate parameters with checkpoint support.

        Args:
            max_combinations: Maximum number of combinations to generate

        Yields:
            GenerationParameters objects
        """
        # Generate parameters from variation engine
        if self.variation_engine is None:
            raise GenerationError("Variation engine not initialized")

        params_generator = self.variation_engine.generate_variations(max_combinations)

        # Yield parameters one by one
        for params in params_generator:
            yield params

    async def _resume_generation(
        self, checkpoint_id: str, output_dir: str
    ) -> list[GenerationResult]:
        """Resume generation from a checkpoint.

        Args:
            checkpoint_id: Checkpoint ID to resume from
            output_dir: Output directory for new results

        Returns:
            List of generation results
        """
        if self.checkpoint_manager is None:
            raise GenerationError("Checkpoint manager not initialized")

        # Restore from checkpoint
        config, pending_tasks = await self.checkpoint_manager.restore_from_checkpoint(
            checkpoint_id
        )

        self._current_checkpoint_id = checkpoint_id
        self._task_count = len(pending_tasks)

        # Initialize progress tracking
        if self.progress_tracker:
            await self.progress_tracker.initialize_batch(len(pending_tasks))

        if self.batch_processor is None:
            raise GenerationError("Batch processor not initialized")

        # Process pending tasks
        results = []
        async for task_id, result, error in self.batch_processor.process_batch(
            provider=self._get_primary_provider(), tasks=pending_tasks
        ):
            if result:
                # Save successful result
                await self._save_generated_result(result, output_dir)
                results.append(result)

                # Save checkpoint progress
                await self.checkpoint_manager.save_task_state(
                    checkpoint_id=self._current_checkpoint_id,
                    task_id=task_id,
                    status="completed",
                    result=result,
                )
            else:
                # Save error state
                await self.checkpoint_manager.save_task_state(
                    checkpoint_id=self._current_checkpoint_id,
                    task_id=task_id,
                    status="failed",
                    error=error,
                )

        # Mark checkpoint as completed
        await self.checkpoint_manager.mark_checkpoint_completed(
            self._current_checkpoint_id
        )

        return results

    def _get_primary_provider(self) -> TTSProvider:
        """Get the primary audio provider based on configuration.

        Returns:
            Primary audio provider instance

        Raises:
            GenerationError: If no suitable provider is found
        """
        from wakegen.models.config import ProviderConfig

        # Create provider config from the main config
        provider_config = ProviderConfig()

        # Try to get commercial provider first (MiniMax is the primary commercial provider)
        if self.config.use_commercial_providers:
            try:
                return get_provider(ProviderType.MINIMAX, provider_config)
            except Exception as e:
                logger.warning(f"Commercial provider (MiniMax) unavailable: {e!s}")

        # Fall back to free provider (Edge TTS is the primary free provider)
        try:
            return get_provider(ProviderType.EDGE_TTS, provider_config)
        except Exception as e:
            raise GenerationError(f"No available providers: {e!s}") from e

    async def _save_generated_result(
        self, result: GenerationResult, output_dir: str
    ) -> str:
        """Save a generated result to the output directory.

        Args:
            result: Generation result to save
            output_dir: Output directory

        Returns:
            Path to the saved file
        """
        output_path = Path(output_dir)

        # Create unique filename
        timestamp = int(time.time() * 1000)
        filename = (
            f"{result.parameters.text.replace(' ', '_')}_"
            f"{result.parameters.voice_id}_"
            f"{timestamp}.wav"
        )

        file_path = output_path / filename

        # Save audio data using soundfile
        # Note: AudioSample.file_path contains the source path of the generated audio
        # For now, we copy the file if it exists, otherwise log a warning
        import shutil

        source_path = result.audio_data.file_path
        if source_path and Path(source_path).exists():
            shutil.copy2(source_path, file_path)
        else:
            logger.warning(f"Source audio file not found: {source_path}")

        logger.info(f"Saved audio sample: {file_path}")
        return str(file_path)

    async def get_generation_status(self) -> dict[str, Any]:
        """Get the status of the current generation session.

        Returns:
            Dictionary with generation status information
        """
        if not self._current_checkpoint_id:
            return {"status": "not_started"}

        try:
            if self.checkpoint_manager is None:
                return {
                    "status": "error",
                    "error": "Checkpoint manager not initialized",
                }

            checkpoint_status = await self.checkpoint_manager.get_checkpoint_status(
                self._current_checkpoint_id
            )

            progress_status = (
                self.progress_tracker.get_current_status()
                if self.progress_tracker
                else {}
            )

            return {
                "checkpoint_status": checkpoint_status,
                "progress_status": progress_status,
                "session_id": self._session_id,
                "checkpoint_id": self._current_checkpoint_id,
            }
        except Exception as e:
            logger.error(f"Failed to get generation status: {e!s}")
            return {"status": "error", "error": str(e)}

    async def cancel_generation(self) -> None:
        """Cancel the current generation session."""
        if not self._current_checkpoint_id:
            return

        try:
            if self.checkpoint_manager is None:
                logger.error("Checkpoint manager not initialized")
                return

            # Mark checkpoint as failed
            await self.checkpoint_manager.mark_checkpoint_failed(
                self._current_checkpoint_id, "Generation cancelled by user"
            )

            # Clean up progress tracker
            if self.progress_tracker:
                await self.progress_tracker.show_error("Generation cancelled")

            logger.info("Generation cancelled successfully")
        except Exception as e:
            logger.error(f"Failed to cancel generation: {e!s}")
            raise GenerationError(f"Failed to cancel generation: {e!s}") from e

    async def cleanup(self) -> None:
        """Clean up resources and close connections."""
        try:
            if self.checkpoint_manager:
                await self.checkpoint_manager.close()

            if self.progress_tracker:
                await self.progress_tracker.__aexit__(None, None, None)

            logger.info("Generation orchestrator cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {e!s}")
            raise GenerationError(f"Cleanup failed: {e!s}") from e

    async def __aenter__(self) -> GenerationOrchestrator:
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Async context manager exit."""
        await self.cleanup()

    def create_turkish_generation_config(
        self, wake_words: list[str], voice_ids: list[str] | None = None
    ) -> VariationParameters:
        """Create Turkish-specific variation parameters.

        Args:
            wake_words: List of Turkish wake words
            voice_ids: List of Turkish voice IDs (optional)

        Returns:
            VariationParameters configured for Turkish
        """
        if voice_ids is None:
            voice_ids = ["tr-TR-PinarNeural", "tr-TR-AhmetNeural", "tr-TR-EmelNeural"]

        if self.variation_engine is None:
            raise GenerationError(
                "Variation engine not initialized. Please call generate() first or initialize variation engine."
            )

        return self.variation_engine.create_turkish_parameters(wake_words, voice_ids)

    async def generate_turkish_samples(
        self,
        wake_words: list[str],
        count: int,
        output_dir: str,
        voice_ids: list[str] | None = None,
    ) -> list[GenerationResult]:
        """Generate Turkish-specific audio samples.

        Args:
            wake_words: List of Turkish wake words
            count: Number of samples to generate
            output_dir: Output directory
            voice_ids: List of Turkish voice IDs (optional)

        Returns:
            List of generation results
        """
        # Create Turkish variation parameters
        turkish_params = self.create_turkish_generation_config(wake_words, voice_ids)

        # Update variation engine
        self.variation_engine = VariationEngine(turkish_params)

        # Generate samples
        return await self.generate(wake_words, count, output_dir, voice_ids)

    async def generate_with_fallback(
        self,
        wake_words: list[str],
        count: int,
        output_dir: str,
        voice_ids: list[str] | None = None,
    ) -> list[GenerationResult]:
        """Generate samples with fallback to other providers.

        Args:
            wake_words: List of wake words
            count: Number of samples to generate
            output_dir: Output directory
            voice_ids: List of voice IDs (optional)

        Returns:
            List of generation results
        """
        from wakegen.models.config import ProviderConfig

        provider_config = ProviderConfig()
        fallback_providers = []

        # Get all available providers
        # Issue C-001 & C-003 Fix: Use correct ProviderType enums and proper exception handling
        try:
            if self.config.use_commercial_providers:
                fallback_providers.append(
                    get_provider(ProviderType.MINIMAX, provider_config)
                )
        except (ConfigError, Exception) as e:
            logger.debug(f"MiniMax provider unavailable for fallback: {e}")

        try:
            fallback_providers.append(
                get_provider(ProviderType.EDGE_TTS, provider_config)
            )
        except (ConfigError, Exception) as e:
            logger.debug(f"Edge TTS provider unavailable for fallback: {e}")

        if not fallback_providers:
            raise GenerationError("No fallback providers available")

        # Create checkpoint ID
        self._current_checkpoint_id = str(uuid.uuid4())

        # Determine voice IDs to use
        if voice_ids is None:
            voice_ids = self.config.default_voice_ids or [
                "tr-TR-PinarNeural",
                "tr-TR-AhmetNeural",
            ]

        # Create variation parameters
        variation_params = VariationParameters(
            text_variations=wake_words,
            voice_ids=voice_ids,
            speed_range=self.config.speed_range or (0.8, 1.2),
            pitch_range=self.config.pitch_range or (0.9, 1.1),
        )

        # Initialize variation engine
        self.variation_engine = VariationEngine(variation_params)

        # Estimate total combinations needed
        total_combinations = self.variation_engine.estimate_total_combinations()
        combinations_needed = min(count, total_combinations)

        # Generate parameter combinations
        params_list = []
        task_ids = []

        # Issue M-006 Fix: generate_variations returns a sync Iterator, not AsyncIterator
        for params in self.variation_engine.generate_variations(combinations_needed):
            task_id = f"task_{self._task_count}"
            params_list.append(params)
            task_ids.append(task_id)
            self._task_count += 1

            if len(params_list) >= combinations_needed:
                break

        # Initialize progress tracking
        if self.progress_tracker is not None:
            await self.progress_tracker.initialize_batch(len(params_list))

        # Process with fallback
        if self.batch_processor is None:
            raise GenerationError("Batch processor not initialized")

        results = []
        async for task_id, result, error in self.batch_processor.process_with_fallback(
            primary_provider=self._get_primary_provider(),
            fallback_providers=fallback_providers,
            tasks=list(zip(task_ids, params_list)),
        ):
            if result:
                # Save successful result
                await self._save_generated_result(result, output_dir)
                results.append(result)
            # Errors are already handled by the batch processor

        return results

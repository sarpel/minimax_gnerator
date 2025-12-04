"""
Augmentation Pipeline

This module provides the main augmentation pipeline that coordinates
all augmentation components to create realistic environmental variations
of wake word samples. It's the central orchestrator for Phase 4.

Key Features:
- Async-first architecture for efficient processing
- Coordination of all augmentation types
- Environment profile integration
- Resource-efficient processing
- Error handling and graceful degradation
- Progress tracking and logging
"""

from __future__ import annotations
import asyncio
import os
import tempfile
import uuid
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from wakegen.core.types import EnvironmentProfile, AugmentationType
from wakegen.core.exceptions import AugmentationError
from wakegen.augmentation.profiles import AugmentationProfile, get_profile
from wakegen.augmentation.noise.mixer import NoiseMixer
from wakegen.augmentation.noise.events import NoiseEventGenerator
from wakegen.augmentation.noise.profiles import NoiseProfileManager
from wakegen.augmentation.room.simulator import RoomSimulator
from wakegen.augmentation.microphone.simulator import MicrophoneSimulator
from wakegen.augmentation.effects.time_domain import TimeDomainEffects
from wakegen.augmentation.effects.dynamics import DynamicsProcessor
from wakegen.augmentation.effects.degradation import AudioDegrader
from wakegen.utils.audio import load_audio
import soundfile as sf
import numpy as np
import librosa
import logging
import random

# Configure logging
logger = logging.getLogger(__name__)

class AugmentationPipeline:
    """
    Main augmentation pipeline that coordinates all augmentation components.

    This is the central orchestrator for Phase 4 that applies multiple
    augmentation types in sequence to create realistic environmental
    variations of wake word samples.
    """

    def __init__(
        self,
        profile: AugmentationProfile,
        sample_rate: int = 24000,
        temp_dir: Optional[str] = None
    ):
        """
        Initialize the augmentation pipeline.

        Args:
            profile: AugmentationProfile defining the environment characteristics.
            sample_rate: Target sample rate for all processing.
            temp_dir: Optional directory for temporary files (None for system temp).
        """
        self.profile = profile
        self.sample_rate = sample_rate
        self.temp_dir = temp_dir or tempfile.gettempdir()

        # Initialize all augmentation components
        self._init_components()

        # Validate the profile
        self._validate_profile()

        logger.info(f"Initialized augmentation pipeline with profile: {profile.name}")

    def _init_components(self) -> None:
        """Initialize all augmentation components."""
        self.noise_mixer = NoiseMixer(self.sample_rate)
        self.noise_event_gen = NoiseEventGenerator(self.sample_rate)
        self.noise_profile_manager = NoiseProfileManager()
        self.room_simulator = RoomSimulator(self.sample_rate)
        self.mic_simulator = MicrophoneSimulator(self.sample_rate)
        self.time_effects = TimeDomainEffects(self.sample_rate)
        self.dynamics_processor = DynamicsProcessor(self.sample_rate)
        self.audio_degrader = AudioDegrader(self.sample_rate)

    def _validate_profile(self) -> None:
        """Validate that the augmentation profile is valid."""
        if not self.profile.augmentation_types:
            raise AugmentationError("Augmentation profile must specify at least one augmentation type")

        # Validate that all specified augmentation types are supported
        supported_types = {
            AugmentationType.BACKGROUND_NOISE,
            AugmentationType.ROOM_SIMULATION,
            AugmentationType.MICROPHONE_SIMULATION,
            AugmentationType.TIME_STRETCH,
            AugmentationType.PITCH_SHIFT,
            AugmentationType.COMPRESSION,
            AugmentationType.DEGRADATION
        }

        for aug_type in self.profile.augmentation_types:
            if aug_type not in supported_types:
                raise AugmentationError(f"Unsupported augmentation type: {aug_type}")

    async def apply(
        self,
        input_path: str,
        output_path: str,
        intermediate_dir: Optional[str] = None
    ) -> str:
        """
        Apply the full augmentation pipeline to an audio file.

        Args:
            input_path: Path to input audio file.
            output_path: Path to save final augmented audio.
            intermediate_dir: Optional directory for intermediate files (for debugging).

        Returns:
            Path to the final augmented audio file.

        Raises:
            AugmentationError: If augmentation pipeline fails.
        """
        try:
            # Create intermediate directory if specified
            if intermediate_dir:
                os.makedirs(intermediate_dir, exist_ok=True)

            # Load the original audio
            original_audio, original_sr = load_audio(input_path)

            # Resample to target sample rate if needed
            if original_sr != self.sample_rate:
                original_audio = librosa.resample(
                    original_audio,
                    orig_sr=original_sr,
                    target_sr=self.sample_rate
                )
                logger.debug(f"Resampled audio from {original_sr}Hz to {self.sample_rate}Hz")

            # Apply augmentations in sequence
            processed_audio = original_audio

            # Track intermediate steps for debugging
            intermediate_files = {}

            # Apply each augmentation type in the specified order
            for aug_type in self.profile.augmentation_types:
                try:
                    if aug_type == AugmentationType.BACKGROUND_NOISE:
                        processed_audio = await self._apply_background_noise(processed_audio)
                        if intermediate_dir:
                            intermediate_files["noise"] = self._save_intermediate(
                                processed_audio, intermediate_dir, "after_noise"
                            )

                    elif aug_type == AugmentationType.ROOM_SIMULATION:
                        processed_audio = await self._apply_room_simulation(processed_audio)
                        if intermediate_dir:
                            intermediate_files["room"] = self._save_intermediate(
                                processed_audio, intermediate_dir, "after_room"
                            )

                    elif aug_type == AugmentationType.MICROPHONE_SIMULATION:
                        processed_audio = await self._apply_microphone_simulation(processed_audio)
                        if intermediate_dir:
                            intermediate_files["mic"] = self._save_intermediate(
                                processed_audio, intermediate_dir, "after_mic"
                            )

                    elif aug_type == AugmentationType.TIME_STRETCH:
                        processed_audio = await self._apply_time_stretch(processed_audio)
                        if intermediate_dir:
                            intermediate_files["time_stretch"] = self._save_intermediate(
                                processed_audio, intermediate_dir, "after_time_stretch"
                            )

                    elif aug_type == AugmentationType.PITCH_SHIFT:
                        processed_audio = await self._apply_pitch_shift(processed_audio)
                        if intermediate_dir:
                            intermediate_files["pitch_shift"] = self._save_intermediate(
                                processed_audio, intermediate_dir, "after_pitch_shift"
                            )

                    elif aug_type == AugmentationType.COMPRESSION:
                        processed_audio = await self._apply_compression(processed_audio)
                        if intermediate_dir:
                            intermediate_files["compression"] = self._save_intermediate(
                                processed_audio, intermediate_dir, "after_compression"
                            )

                    elif aug_type == AugmentationType.DEGRADATION:
                        processed_audio = await self._apply_degradation(processed_audio)
                        if intermediate_dir:
                            intermediate_files["degradation"] = self._save_intermediate(
                                processed_audio, intermediate_dir, "after_degradation"
                            )

                except Exception as e:
                    logger.error(f"Failed to apply {aug_type}: {str(e)}")
                    raise AugmentationError(f"Augmentation failed at {aug_type} stage: {str(e)}") from e

            # Save the final result
            self._save_audio(processed_audio, output_path)
            logger.info(f"Successfully saved augmented audio to {output_path}")

            return output_path

        except Exception as e:
            logger.error(f"Augmentation pipeline failed: {str(e)}")
            raise AugmentationError(f"Augmentation pipeline failed: {str(e)}") from e

    async def _apply_background_noise(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply background noise augmentation.

        Args:
            audio: Input audio signal.

        Returns:
            Audio with background noise added.
        """
        try:
            # Get SNR from profile's noise characteristics
            # The profile has a noise_profile object, not noise_profile_id
            snr_db = random.uniform(*self.profile.noise_profile.snr_range)

            # Generate base noise
            noise_duration = len(audio) / self.sample_rate
            base_noise = self.noise_mixer.generate_noise(
                noise_duration,
                self.profile.noise_profile.base_noise_type
            )

            # Add noise events if this profile has them
            if self.profile.noise_profile.typical_events:
                events = self.noise_event_gen.generate_random_events(
                    noise_duration,
                    self.profile.noise_profile.event_density,
                    self.profile.noise_profile.typical_events
                )
                base_noise = self.noise_event_gen.apply_events_to_noise(base_noise, events)

            # Mix with original audio
            return self.noise_mixer.mix_with_noise(audio, base_noise, snr_db)

        except Exception as e:
            raise AugmentationError(f"Background noise application failed: {str(e)}") from e

    async def _apply_room_simulation(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply room simulation augmentation.

        Args:
            audio: Input audio signal.

        Returns:
            Audio with room effects applied.
        """
        try:
            # Generate room impulse response
            rir = self.room_simulator.create_room_impulse_response(self.profile.room_params)

            # Apply room simulation with moderate wet/dry mix
            return self.room_simulator.apply_room_simulation(audio, rir, wet_dry_mix=0.6)

        except Exception as e:
            raise AugmentationError(f"Room simulation failed: {str(e)}") from e

    async def _apply_microphone_simulation(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply microphone simulation augmentation.

        Args:
            audio: Input audio signal.

        Returns:
            Audio with microphone effects applied.
        """
        try:
            return self.mic_simulator.apply_microphone_effect(audio, self.profile.microphone_profile)

        except Exception as e:
            raise AugmentationError(f"Microphone simulation failed: {str(e)}") from e

    async def _apply_time_stretch(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply time stretching augmentation.

        Args:
            audio: Input audio signal.

        Returns:
            Time-stretched audio.
        """
        try:
            stretch_factor = self.profile.time_effects.get("time_stretch_factor", 1.0)
            if stretch_factor != 1.0:
                return self.time_effects.time_stretch(audio, stretch_factor)
            return audio

        except Exception as e:
            raise AugmentationError(f"Time stretching failed: {str(e)}") from e

    async def _apply_pitch_shift(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply pitch shifting augmentation.

        Args:
            audio: Input audio signal.

        Returns:
            Pitch-shifted audio.
        """
        try:
            pitch_steps = self.profile.time_effects.get("pitch_steps", 0.0)
            if pitch_steps != 0.0:
                return self.time_effects.pitch_shift(audio, pitch_steps)
            return audio

        except Exception as e:
            raise AugmentationError(f"Pitch shifting failed: {str(e)}") from e

    async def _apply_compression(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply dynamics compression augmentation.

        Args:
            audio: Input audio signal.

        Returns:
            Compressed audio.
        """
        try:
            if self.profile.dynamics_effects.get("effect_type") == "compression":
                return self.dynamics_processor.apply_compression(
                    audio,
                    **{k: v for k, v in self.profile.dynamics_effects.items()
                       if k != "effect_type"}
                )
            elif self.profile.dynamics_effects.get("effect_type") == "limiting":
                return self.dynamics_processor.apply_limiting(
                    audio,
                    **{k: v for k, v in self.profile.dynamics_effects.items()
                       if k != "effect_type"}
                )
            return audio

        except Exception as e:
            raise AugmentationError(f"Dynamics processing failed: {str(e)}") from e

    async def _apply_degradation(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply quality degradation augmentation.

        Args:
            audio: Input audio signal.

        Returns:
            Degraded audio.
        """
        try:
            if self.profile.degradation_effects.get("degradation_type") == "random":
                severity = self.profile.degradation_effects.get("severity", 0.5)
                return self.audio_degrader.apply_random_degradation(audio, severity)
            else:
                return self.audio_degrader.apply_degradation(
                    audio,
                    **self.profile.degradation_effects
                )

        except Exception as e:
            raise AugmentationError(f"Audio degradation failed: {str(e)}") from e

    def _save_intermediate(
        self,
        audio: np.ndarray,
        directory: str,
        filename_prefix: str
    ) -> str:
        """
        Save intermediate audio file for debugging.

        Args:
            audio: Audio signal to save.
            directory: Directory to save in.
            filename_prefix: Prefix for filename.

        Returns:
            Path to saved file.
        """
        try:
            filename = f"{filename_prefix}_{uuid.uuid4().hex[:8]}.wav"
            filepath = os.path.join(directory, filename)
            self._save_audio(audio, filepath)
            return filepath
        except Exception as e:
            logger.warning(f"Failed to save intermediate file: {str(e)}")
            return ""

    def _save_audio(self, audio: np.ndarray, filepath: str) -> None:
        """
        Save audio to file with proper directory creation.

        Args:
            audio: Audio signal to save.
            filepath: Full path to save file.

        Raises:
            AugmentationError: If saving fails.
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # Save with soundfile
            sf.write(filepath, audio, self.sample_rate)

        except Exception as e:
            raise AugmentationError(f"Failed to save audio to {filepath}: {str(e)}") from e

    async def batch_augment(
        self,
        input_paths: List[str],
        output_dir: str,
        prefix: str = "augmented_",
        suffix: str = "",
        intermediate_dir: Optional[str] = None
    ) -> List[str]:
        """
        Apply augmentation to multiple files in batch.

        Args:
            input_paths: List of input file paths.
            output_dir: Directory to save augmented files.
            prefix: Prefix for output filenames.
            suffix: Suffix for output filenames.
            intermediate_dir: Optional directory for intermediate files.

        Returns:
            List of paths to augmented files.

        Raises:
            AugmentationError: If batch processing fails.
        """
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            results = []
            for i, input_path in enumerate(input_paths):
                try:
                    # Generate output filename
                    input_name = os.path.basename(input_path)
                    output_name = f"{prefix}{input_name}{suffix}"
                    output_path = os.path.join(output_dir, output_name)

                    # Apply augmentation
                    result_path = await self.apply(input_path, output_path, intermediate_dir)
                    results.append(result_path)

                    logger.info(f"Processed {i+1}/{len(input_paths)}: {input_path} -> {output_path}")

                except Exception as e:
                    logger.error(f"Failed to process {input_path}: {str(e)}")
                    continue

            return results

        except Exception as e:
            raise AugmentationError(f"Batch augmentation failed: {str(e)}") from e

    def get_augmentation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the augmentation pipeline configuration.

        Returns:
            Dictionary with pipeline configuration details.
        """
        return {
            "profile_name": self.profile.name,
            "profile_description": self.profile.description,
            "sample_rate": self.sample_rate,
            "augmentation_types": [str(at) for at in self.profile.augmentation_types],
            "noise_profile": self.profile.noise_profile.name,
            "room_parameters": {
                "size": f"{self.profile.room_params.length}x{self.profile.room_params.width}x{self.profile.room_params.height}m",
                "rt60": f"{self.profile.room_params.rt60}s",
                "absorption": self.profile.room_params.absorption
            },
            "microphone_profile": self.profile.microphone_profile.name,
            "time_effects": self.profile.time_effects,
            "dynamics_effects": self.profile.dynamics_effects,
            "degradation_effects": self.profile.degradation_effects
        }

# Convenience function for easy pipeline creation
def create_pipeline(
    profile_id: EnvironmentProfile,
    sample_rate: int = 24000
) -> AugmentationPipeline:
    """
    Create an augmentation pipeline with the specified environment profile.

    Args:
        profile_id: EnvironmentProfile enum value.
        sample_rate: Target sample rate.

    Returns:
        Configured AugmentationPipeline instance.
    """
    profile = get_profile(profile_id)
    return AugmentationPipeline(profile, sample_rate)
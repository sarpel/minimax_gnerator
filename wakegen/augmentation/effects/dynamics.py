"""
Dynamics Effects

This module provides dynamic range processing effects including
compression, limiting, and expansion. These effects help simulate
different recording conditions and microphone characteristics.

Key Features:
- Audio compression with configurable parameters
- Limiting to prevent clipping
- Noise gating and expansion
- Efficient processing suitable for real-time
- Parameter validation and error handling
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple
from wakegen.core.exceptions import AugmentationError
from wakegen.utils.audio import load_audio
import soundfile as sf

class DynamicsProcessor:
    """
    Provides dynamic range processing effects for audio.

    Implements compression, limiting, and expansion algorithms.
    """

    def __init__(self, sample_rate: int = 24000):
        """
        Initialize the dynamics processor.

        Args:
            sample_rate: Target sample rate for processing.
        """
        self.sample_rate = sample_rate
        self._validate_sample_rate(sample_rate)

    def _validate_sample_rate(self, sample_rate: int) -> None:
        """Validate sample rate is suitable for dynamics processing."""
        if not (8000 <= sample_rate <= 48000):
            raise AugmentationError(f"Sample rate {sample_rate}Hz is out of valid range (8000-48000Hz)")

    def apply_compression(
        self,
        audio: np.ndarray,
        threshold_db: float = -20.0,
        ratio: float = 4.0,
        attack_ms: float = 10.0,
        release_ms: float = 100.0,
        knee_db: float = 5.0,
        makeup_gain_db: float = 0.0
    ) -> np.ndarray:
        """
        Apply compression to audio signal.

        Args:
            audio: Input audio signal.
            threshold_db: Threshold in dB (signals above this are compressed).
            ratio: Compression ratio (4:1 means 4dB in -> 1dB out).
            attack_ms: Attack time in milliseconds.
            release_ms: Release time in milliseconds.
            knee_db: Knee width in dB (softens the compression transition).
            makeup_gain_db: Gain to apply after compression to compensate for level loss.

        Returns:
            Compressed audio signal.

        Raises:
            AugmentationError: If compression fails or parameters invalid.
        """
        self._validate_compression_parameters(
            threshold_db, ratio, attack_ms, release_ms, knee_db, makeup_gain_db
        )

        try:
            # Convert parameters to samples
            attack_samples = int(attack_ms * self.sample_rate / 1000.0)
            release_samples = int(release_ms * self.sample_rate / 1000.0)

            # Initialize envelope and gain
            envelope = np.zeros_like(audio)
            gain = np.ones_like(audio)
            compressed = np.zeros_like(audio)

            # Simple compressor implementation
            for i in range(len(audio)):
                # Calculate current level in dB
                current_level_db = 20.0 * np.log10(np.abs(audio[i]) + 1e-10)

                # Apply knee smoothing
                if knee_db > 0:
                    # Soft knee implementation
                    knee_start = threshold_db - knee_db/2.0
                    knee_end = threshold_db + knee_db/2.0

                    if current_level_db > knee_start and current_level_db < knee_end:
                        # Linear transition through knee
                        knee_position = (current_level_db - knee_start) / knee_db
                        effective_threshold = threshold_db - (knee_db/2.0) + (knee_position * knee_db)
                    else:
                        effective_threshold = threshold_db
                else:
                    effective_threshold = threshold_db

                # Calculate gain reduction
                if current_level_db > effective_threshold:
                    # Amount over threshold
                    over_db = current_level_db - effective_threshold
                    # Apply compression ratio
                    gain_reduction_db = over_db * (1.0 - 1.0/ratio)
                    gain_reduction = 10.0 ** (-gain_reduction_db / 20.0)
                else:
                    gain_reduction = 1.0

                # Apply envelope following (simplified attack/release)
                if i == 0:
                    envelope[i] = current_level_db
                else:
                    # Attack: when signal increases
                    if current_level_db > envelope[i-1]:
                        envelope[i] = envelope[i-1] + (current_level_db - envelope[i-1]) / attack_samples
                    # Release: when signal decreases
                    else:
                        envelope[i] = envelope[i-1] + (current_level_db - envelope[i-1]) / release_samples

                # Apply gain reduction
                gain[i] = gain_reduction
                compressed[i] = audio[i] * gain_reduction

            # Apply makeup gain
            makeup_gain = 10.0 ** (makeup_gain_db / 20.0)
            compressed = compressed * makeup_gain

            # Ensure we don't exceed original peak level
            original_peak = np.max(np.abs(audio))
            compressed_peak = np.max(np.abs(compressed))

            if compressed_peak > original_peak and original_peak > 0:
                compressed = compressed * (original_peak / compressed_peak)

            return compressed

        except Exception as e:
            raise AugmentationError(f"Failed to apply compression: {str(e)}") from e

    def _validate_compression_parameters(
        self,
        threshold_db: float,
        ratio: float,
        attack_ms: float,
        release_ms: float,
        knee_db: float,
        makeup_gain_db: float
    ) -> None:
        """Validate compression parameters are reasonable."""
        if threshold_db > 0:
            raise AugmentationError("Compression threshold must be negative (below 0dB)")
        if threshold_db < -60:
            raise AugmentationError("Compression threshold too low (minimum -60dB)")
        if ratio < 1.0:
            raise AugmentationError("Compression ratio must be >= 1.0")
        if ratio > 20.0:
            raise AugmentationError("Compression ratio too high (maximum 20:1)")
        if attack_ms <= 0 or release_ms <= 0:
            raise AugmentationError("Attack and release times must be positive")
        if knee_db < 0:
            raise AugmentationError("Knee width cannot be negative")
        if abs(makeup_gain_db) > 24:
            raise AugmentationError("Makeup gain must be between -24dB and +24dB")

    def apply_limiting(
        self,
        audio: np.ndarray,
        threshold_db: float = -3.0,
        release_ms: float = 50.0
    ) -> np.ndarray:
        """
        Apply limiting to prevent audio from exceeding threshold.

        Args:
            audio: Input audio signal.
            threshold_db: Maximum allowed level in dB.
            release_ms: Release time in milliseconds.

        Returns:
            Limited audio signal.

        Raises:
            AugmentationError: If limiting fails.
        """
        if threshold_db >= 0:
            raise AugmentationError("Limiting threshold must be negative (below 0dB)")
        if release_ms <= 0:
            raise AugmentationError("Release time must be positive")

        try:
            # Convert threshold to linear
            threshold_linear = 10.0 ** (threshold_db / 20.0)

            # Simple limiter implementation
            limited = np.zeros_like(audio)
            envelope = 0.0

            release_samples = int(release_ms * self.sample_rate / 1000.0)

            for i in range(len(audio)):
                # Calculate current level
                current_level = np.abs(audio[i])

                # Update envelope (simplified)
                if current_level > envelope:
                    envelope = current_level
                else:
                    envelope = envelope + (current_level - envelope) / release_samples

                # Apply limiting
                if envelope > threshold_linear:
                    gain_reduction = threshold_linear / envelope
                    limited[i] = audio[i] * gain_reduction
                else:
                    limited[i] = audio[i]

            return limited

        except Exception as e:
            raise AugmentationError(f"Failed to apply limiting: {str(e)}") from e

    def apply_expansion(
        self,
        audio: np.ndarray,
        threshold_db: float = -30.0,
        ratio: float = 2.0,
        attack_ms: float = 5.0,
        release_ms: float = 50.0
    ) -> np.ndarray:
        """
        Apply expansion to increase dynamic range (opposite of compression).

        Args:
            audio: Input audio signal.
            threshold_db: Threshold in dB (signals below this are expanded).
            ratio: Expansion ratio (2:1 means signals below threshold are reduced more).
            attack_ms: Attack time in milliseconds.
            release_ms: Release time in milliseconds.

        Returns:
            Expanded audio signal.

        Raises:
            AugmentationError: If expansion fails.
        """
        if ratio < 1.0:
            raise AugmentationError("Expansion ratio must be >= 1.0")
        if attack_ms <= 0 or release_ms <= 0:
            raise AugmentationError("Attack and release times must be positive")

        try:
            # Convert parameters to samples
            attack_samples = int(attack_ms * self.sample_rate / 1000.0)
            release_samples = int(release_ms * self.sample_rate / 1000.0)

            # Initialize variables
            envelope = np.zeros_like(audio)
            gain = np.ones_like(audio)
            expanded = np.zeros_like(audio)

            for i in range(len(audio)):
                # Calculate current level in dB
                current_level_db = 20.0 * np.log10(np.abs(audio[i]) + 1e-10)

                # Calculate gain change for expansion
                if current_level_db < threshold_db:
                    # Amount below threshold
                    below_db = threshold_db - current_level_db
                    # Apply expansion ratio (inverse of compression)
                    gain_change_db = below_db * (ratio - 1.0)
                    gain_change = 10.0 ** (-gain_change_db / 20.0)
                else:
                    gain_change = 1.0

                # Apply envelope following
                if i == 0:
                    envelope[i] = current_level_db
                else:
                    if current_level_db < envelope[i-1]:
                        envelope[i] = envelope[i-1] + (current_level_db - envelope[i-1]) / attack_samples
                    else:
                        envelope[i] = envelope[i-1] + (current_level_db - envelope[i-1]) / release_samples

                gain[i] = gain_change
                expanded[i] = audio[i] * gain_change

            return expanded

        except Exception as e:
            raise AugmentationError(f"Failed to apply expansion: {str(e)}") from e

    async def apply_dynamics(
        self,
        input_path: str,
        output_path: str,
        effect_type: str = "compression",
        **effect_params
    ) -> None:
        """
        Apply dynamics processing to an audio file and save the result.

        Args:
            input_path: Path to input audio file.
            output_path: Path to save processed audio.
            effect_type: Type of dynamics effect ('compression', 'limiting', 'expansion').
            **effect_params: Parameters specific to the effect type.

        Raises:
            AugmentationError: If dynamics processing fails.
        """
        try:
            # Load input audio
            audio, sr = load_audio(input_path)

            # Resample if needed
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)

            # Apply selected effect
            if effect_type == "compression":
                processed = self.apply_compression(audio, **effect_params)
            elif effect_type == "limiting":
                processed = self.apply_limiting(audio, **effect_params)
            elif effect_type == "expansion":
                processed = self.apply_expansion(audio, **effect_params)
            else:
                raise AugmentationError(f"Unknown dynamics effect type: {effect_type}")

            # Save result
            sf.write(output_path, processed, self.sample_rate)

        except Exception as e:
            raise AugmentationError(f"Failed to apply dynamics effects: {str(e)}") from e

    def get_dynamics_preset(self, preset_name: str) -> dict:
        """
        Get pre-configured dynamics parameters for common scenarios.

        Args:
            preset_name: Name of preset ('voice_compression', 'aggressive_limiting',
                        'noise_reduction', 'dynamic_boost').

        Returns:
            Dictionary with dynamics parameters.

        Raises:
            AugmentationError: If preset name is unknown.
        """
        presets = {
            "voice_compression": {
                "effect_type": "compression",
                "threshold_db": -24.0,
                "ratio": 3.0,
                "attack_ms": 5.0,
                "release_ms": 100.0,
                "knee_db": 3.0,
                "makeup_gain_db": 6.0,
                "description": "Gentle compression for voice recordings"
            },
            "aggressive_limiting": {
                "effect_type": "limiting",
                "threshold_db": -1.0,
                "release_ms": 20.0,
                "description": "Aggressive limiting to prevent clipping"
            },
            "noise_reduction": {
                "effect_type": "expansion",
                "threshold_db": -40.0,
                "ratio": 3.0,
                "attack_ms": 2.0,
                "release_ms": 50.0,
                "description": "Expand dynamic range to reduce background noise"
            },
            "dynamic_boost": {
                "effect_type": "compression",
                "threshold_db": -30.0,
                "ratio": 2.0,
                "attack_ms": 1.0,
                "release_ms": 200.0,
                "knee_db": 2.0,
                "makeup_gain_db": 3.0,
                "description": "Subtle compression to boost quiet passages"
            }
        }

        if preset_name not in presets:
            available = list(presets.keys())
            raise AugmentationError(f"Unknown dynamics preset '{preset_name}'. Available: {available}")

        return presets[preset_name].copy()
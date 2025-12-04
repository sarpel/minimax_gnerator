"""
Degradation Effects

This module provides audio quality degradation effects that simulate
different recording conditions, transmission artifacts, and low-quality
microphones. These effects help create more diverse training data.

Key Features:
- Bit depth reduction
- Sample rate reduction
- MP3 compression artifacts
- Telephone bandwidth limiting
- Random dropout simulation
- Efficient processing for batch operations
"""

from __future__ import annotations
import numpy as np
import librosa
from typing import Optional, Tuple
from wakegen.core.exceptions import AugmentationError
from wakegen.utils.audio import load_audio
import soundfile as sf
import random

class AudioDegrader:
    """
    Provides audio quality degradation effects to simulate
    different recording and transmission conditions.
    """

    def __init__(self, sample_rate: int = 24000):
        """
        Initialize the audio degrader.

        Args:
            sample_rate: Target sample rate for processing.
        """
        self.sample_rate = sample_rate
        self._validate_sample_rate(sample_rate)

    def _validate_sample_rate(self, sample_rate: int) -> None:
        """Validate sample rate is suitable for degradation processing."""
        if not (8000 <= sample_rate <= 48000):
            raise AugmentationError(f"Sample rate {sample_rate}Hz is out of valid range (8000-48000Hz)")

    def reduce_bit_depth(
        self,
        audio: np.ndarray,
        target_bits: int = 16
    ) -> np.ndarray:
        """
        Reduce the bit depth of audio to simulate low-quality recordings.

        Args:
            audio: Input audio signal (normalized to -1.0 to 1.0).
            target_bits: Target bit depth (8, 16, 24, etc.).

        Returns:
            Audio with reduced bit depth.

        Raises:
            AugmentationError: If bit depth is invalid.
        """
        if target_bits not in [8, 12, 16, 20, 24, 32]:
            raise AugmentationError(f"Unsupported bit depth: {target_bits}")

        try:
            # Scale to target bit range
            max_val = 2.0 ** (target_bits - 1) - 1.0
            scaled = audio * max_val

            # Quantize (round to nearest integer)
            quantized = np.round(scaled)

            # Clip to valid range
            quantized = np.clip(quantized, -max_val, max_val)

            # Scale back to -1.0 to 1.0 range
            return quantized / max_val

        except Exception as e:
            raise AugmentationError(f"Failed to reduce bit depth: {str(e)}") from e

    def apply_bandwidth_limiting(
        self,
        audio: np.ndarray,
        low_cut: float = 300.0,
        high_cut: float = 3400.0
    ) -> np.ndarray:
        """
        Apply bandwidth limiting to simulate telephone or low-quality mic.

        Args:
            audio: Input audio signal.
            low_cut: Low frequency cutoff in Hz.
            high_cut: High frequency cutoff in Hz.

        Returns:
            Bandwidth-limited audio.

        Raises:
            AugmentationError: If frequency limits are invalid.
        """
        if low_cut <= 0 or high_cut <= 0:
            raise AugmentationError("Frequency cutoffs must be positive")
        if low_cut >= high_cut:
            raise AugmentationError("Low cutoff must be less than high cutoff")
        if high_cut > self.sample_rate / 2:
            raise AugmentationError("High cutoff exceeds Nyquist frequency")

        try:
            # Apply bandpass filter using FFT
            fft_audio = np.fft.rfft(audio)
            freqs = np.fft.rfftfreq(len(audio), 1.0/self.sample_rate)

            # Create bandpass filter
            filter_mask = (freqs >= low_cut) & (freqs <= high_cut)

            # Apply filter
            fft_filtered = fft_audio * filter_mask

            # Inverse FFT
            return np.fft.irfft(fft_filtered)

        except Exception as e:
            raise AugmentationError(f"Failed to apply bandwidth limiting: {str(e)}") from e

    def add_transmission_artifacts(
        self,
        audio: np.ndarray,
        dropout_probability: float = 0.01,
        dropout_duration_ms: float = 5.0
    ) -> np.ndarray:
        """
        Add transmission artifacts like packet loss and dropouts.

        Args:
            audio: Input audio signal.
            dropout_probability: Probability of dropout per sample.
            dropout_duration_ms: Typical dropout duration in milliseconds.

        Returns:
            Audio with transmission artifacts.

        Raises:
            AugmentationError: If parameters are invalid.
        """
        if not (0.0 <= dropout_probability <= 0.1):
            raise AugmentationError("Dropout probability must be between 0.0 and 0.1")
        if dropout_duration_ms <= 0:
            raise AugmentationError("Dropout duration must be positive")

        try:
            result = audio.copy()
            dropout_samples = int(dropout_duration_ms * self.sample_rate / 1000.0)

            i = 0
            while i < len(result):
                if random.random() < dropout_probability:
                    # Apply dropout
                    dropout_end = min(i + dropout_samples, len(result))
                    result[i:dropout_end] = 0.0
                    i = dropout_end
                else:
                    i += 1

            return result

        except Exception as e:
            raise AugmentationError(f"Failed to add transmission artifacts: {str(e)}") from e

    def apply_mp3_artifacts(
        self,
        audio: np.ndarray,
        bitrate_kbps: int = 64
    ) -> np.ndarray:
        """
        Simulate MP3 compression artifacts.

        Args:
            audio: Input audio signal.
            bitrate_kbps: Simulated MP3 bitrate (32, 64, 96, 128, etc.).

        Returns:
            Audio with MP3-like artifacts.

        Raises:
            AugmentationError: If bitrate is invalid.
        """
        if bitrate_kbps not in [32, 48, 64, 96, 128, 192, 256, 320]:
            raise AugmentationError(f"Unsupported MP3 bitrate: {bitrate_kbps}")

        try:
            # Simulate MP3 artifacts by adding quantization noise
            # and applying mild bandwidth limiting

            # Add quantization noise (more aggressive for lower bitrates)
            noise_level = 10.0 ** (-40.0 + (bitrate_kbps / 8.0))  # -40dB to -10dB range
            noise = np.random.normal(0, noise_level * 0.01, len(audio))
            noisy_audio = audio + noise

            # Apply bandwidth limiting (more aggressive for lower bitrates)
            if bitrate_kbps <= 64:
                high_cut = 12000.0
            elif bitrate_kbps <= 128:
                high_cut = 16000.0
            else:
                high_cut = 18000.0

            return self.apply_bandwidth_limiting(noisy_audio, low_cut=50.0, high_cut=high_cut)

        except Exception as e:
            raise AugmentationError(f"Failed to apply MP3 artifacts: {str(e)}") from e

    def apply_random_degradation(
        self,
        audio: np.ndarray,
        severity: float = 0.5
    ) -> np.ndarray:
        """
        Apply random degradation effects based on severity level.

        Args:
            audio: Input audio signal.
            severity: Degradation severity (0.1 = mild, 1.0 = severe).

        Returns:
            Degraded audio.

        Raises:
            AugmentationError: If severity is invalid.
        """
        if not (0.1 <= severity <= 1.0):
            raise AugmentationError("Severity must be between 0.1 and 1.0")

        try:
            degraded = audio.copy()

            # Apply multiple degradation effects based on severity
            if random.random() < severity * 0.8:
                # Bit depth reduction
                target_bits = max(8, 16 - int(severity * 8))
                degraded = self.reduce_bit_depth(degraded, target_bits)

            if random.random() < severity * 0.7:
                # Bandwidth limiting
                low_cut = 100.0 + (severity * 400.0)
                high_cut = 8000.0 - (severity * 3000.0)
                degraded = self.apply_bandwidth_limiting(degraded, low_cut, high_cut)

            if random.random() < severity * 0.5:
                # Transmission artifacts
                dropout_prob = severity * 0.01
                degraded = self.add_transmission_artifacts(degraded, dropout_prob)

            if random.random() < severity * 0.6:
                # MP3 artifacts
                bitrate = max(32, 320 - int(severity * 288))
                degraded = self.apply_mp3_artifacts(degraded, bitrate)

            return degraded

        except Exception as e:
            raise AugmentationError(f"Failed to apply random degradation: {str(e)}") from e

    async def apply_degradation(
        self,
        input_path: str,
        output_path: str,
        degradation_type: str = "random",
        **effect_params
    ) -> None:
        """
        Apply degradation effects to an audio file and save the result.

        Args:
            input_path: Path to input audio file.
            output_path: Path to save processed audio.
            degradation_type: Type of degradation ('bit_depth', 'bandwidth',
                           'transmission', 'mp3', 'random').
            **effect_params: Parameters specific to the degradation type.

        Raises:
            AugmentationError: If degradation fails.
        """
        try:
            # Load input audio
            audio, sr = load_audio(input_path)

            # Resample if needed
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)

            # Apply selected degradation
            if degradation_type == "bit_depth":
                processed = self.reduce_bit_depth(audio, **effect_params)
            elif degradation_type == "bandwidth":
                processed = self.apply_bandwidth_limiting(audio, **effect_params)
            elif degradation_type == "transmission":
                processed = self.add_transmission_artifacts(audio, **effect_params)
            elif degradation_type == "mp3":
                processed = self.apply_mp3_artifacts(audio, **effect_params)
            elif degradation_type == "random":
                processed = self.apply_random_degradation(audio, **effect_params)
            else:
                raise AugmentationError(f"Unknown degradation type: {degradation_type}")

            # Save result
            sf.write(output_path, processed, self.sample_rate)

        except Exception as e:
            raise AugmentationError(f"Failed to apply degradation effects: {str(e)}") from e

    def get_degradation_preset(self, preset_name: str) -> dict:
        """
        Get pre-configured degradation parameters for common scenarios.

        Args:
            preset_name: Name of preset ('telephone', 'old_recording',
                        'voip_call', 'low_quality_mic', 'broken_transmission').

        Returns:
            Dictionary with degradation parameters.

        Raises:
            AugmentationError: If preset name is unknown.
        """
        presets = {
            "telephone": {
                "degradation_type": "bandwidth",
                "low_cut": 300.0,
                "high_cut": 3400.0,
                "description": "Simulate telephone bandwidth (300-3400Hz)"
            },
            "old_recording": {
                "degradation_type": "random",
                "severity": 0.7,
                "description": "Simulate old, degraded recording"
            },
            "voip_call": {
                "degradation_type": "mp3",
                "bitrate_kbps": 48,
                "description": "Simulate VoIP call quality (48kbps MP3)"
            },
            "low_quality_mic": {
                "degradation_type": "bit_depth",
                "target_bits": 12,
                "description": "Simulate low-quality microphone (12-bit)"
            },
            "broken_transmission": {
                "degradation_type": "transmission",
                "dropout_probability": 0.05,
                "dropout_duration_ms": 20.0,
                "description": "Simulate broken transmission with dropouts"
            }
        }

        if preset_name not in presets:
            available = list(presets.keys())
            raise AugmentationError(f"Unknown degradation preset '{preset_name}'. Available: {available}")

        return presets[preset_name].copy()
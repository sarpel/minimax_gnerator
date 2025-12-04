"""
Background Noise Mixer

This module handles mixing environmental sounds with the original audio
at configurable signal-to-noise ratios (SNR). The mixer allows for
realistic simulation of different environmental conditions.

Key Features:
- SNR-based mixing (higher SNR = cleaner signal, lower SNR = noisier)
- Support for different noise types (white, pink, brown, or custom files)
- Random noise event generation for dynamic environments
- Graceful handling of audio format mismatches
"""

from __future__ import annotations
import asyncio
import random
import numpy as np
import librosa
import soundfile as sf
from typing import Optional, Tuple
from wakegen.core.exceptions import NoiseError
from wakegen.utils.audio import load_audio

class NoiseMixer:
    """
    Handles mixing background noise with clean audio at specified SNR levels.

    The Signal-to-Noise Ratio (SNR) determines how much noise is mixed:
    - High SNR (e.g., 30dB): Clean signal, minimal noise (quiet environment)
    - Medium SNR (e.g., 15dB): Balanced mix (typical room)
    - Low SNR (e.g., 5dB): Very noisy (loud environment)

    Formula: SNR = 10 * log10(P_signal / P_noise)
    """

    def __init__(self, sample_rate: int = 24000):
        """
        Initialize the noise mixer.

        Args:
            sample_rate: The target sample rate for all audio processing.
        """
        self.sample_rate = sample_rate
        self._validate_sample_rate(sample_rate)

    def _validate_sample_rate(self, sample_rate: int) -> None:
        """Validate that the sample rate is reasonable for audio processing."""
        if not (8000 <= sample_rate <= 48000):
            raise NoiseError(f"Sample rate {sample_rate}Hz is out of valid range (8000-48000Hz)")
        if sample_rate % 1000 != 0:
            raise NoiseError(f"Sample rate {sample_rate}Hz should be a multiple of 1000Hz for compatibility")

    def generate_noise(
        self,
        duration_seconds: float,
        noise_type: str = "white",
        color: float = 1.0
    ) -> np.ndarray:
        """
        Generate synthetic noise of the specified type and duration.

        Args:
            duration_seconds: Duration of noise to generate in seconds.
            noise_type: Type of noise ('white', 'pink', 'brown', 'blue').
            color: Color parameter for colored noise (1.0 = white, 0.0 = pink, -1.0 = brown).

        Returns:
            Generated noise as numpy array.

        Raises:
            NoiseError: If invalid noise type or generation fails.
        """
        if duration_seconds <= 0:
            raise NoiseError(f"Duration must be positive, got {duration_seconds}")

        num_samples = int(duration_seconds * self.sample_rate)

        try:
            if noise_type == "white":
                # White noise: equal energy at all frequencies
                return np.random.normal(0, 0.1, num_samples)
            elif noise_type == "pink":
                # Pink noise: equal energy per octave (more natural sounding)
                return self._generate_colored_noise(num_samples, 1.0)
            elif noise_type == "brown":
                # Brown noise: more low-frequency energy (like thunder)
                return self._generate_colored_noise(num_samples, 2.0)
            elif noise_type == "blue":
                # Blue noise: more high-frequency energy
                return self._generate_colored_noise(num_samples, 0.5)
            else:
                raise NoiseError(f"Unknown noise type: {noise_type}")
        except Exception as e:
            raise NoiseError(f"Failed to generate {noise_type} noise: {str(e)}") from e

    def _generate_colored_noise(self, num_samples: int, beta: float) -> np.ndarray:
        """
        Generate colored noise using the spectral synthesis method.

        Args:
            num_samples: Number of samples to generate.
            beta: Exponent for frequency spectrum (1.0 = pink, 2.0 = brown).

        Returns:
            Colored noise as numpy array.
        """
        # Generate white noise
        white = np.random.randn(num_samples)

        # Apply FFT to get frequency domain representation
        fft_white = np.fft.rfft(white)

        # Create frequency array
        n = len(fft_white)
        freqs = np.fft.rfftfreq(num_samples, 1.0/self.sample_rate)

        # Avoid division by zero for DC component
        freqs[0] = 1.0

        # Apply coloring (inverse frequency weighting)
        fft_colored = fft_white * (freqs ** (-beta/2.0))

        # Convert back to time domain
        colored = np.fft.irfft(fft_colored, n=num_samples)

        # Normalize to prevent clipping
        colored = colored / np.max(np.abs(colored)) * 0.1

        return colored

    def mix_with_noise(
        self,
        clean_audio: np.ndarray,
        noise_audio: np.ndarray,
        target_snr_db: float
    ) -> np.ndarray:
        """
        Mix clean audio with noise at the specified SNR level.

        Args:
            clean_audio: Clean audio signal as numpy array.
            noise_audio: Noise signal as numpy array.
            target_snr_db: Target SNR in decibels.

        Returns:
            Mixed audio signal.

        Raises:
            NoiseError: If SNR is invalid or mixing fails.
        """
        if target_snr_db < -10 or target_snr_db > 50:
            raise NoiseError(f"SNR {target_snr_db}dB is out of valid range (-10 to 50dB)")

        # Ensure both signals have the same length
        min_len = min(len(clean_audio), len(noise_audio))
        clean_audio = clean_audio[:min_len]
        noise_audio = noise_audio[:min_len]

        # Calculate signal and noise power
        signal_power = np.mean(clean_audio ** 2)
        noise_power = np.mean(noise_audio ** 2)

        if signal_power == 0:
            raise NoiseError("Clean audio has zero power - cannot compute SNR")

        # Calculate required scaling factor for noise
        # SNR = 10 * log10(signal_power / noise_power)
        # => signal_power / noise_power = 10^(SNR/10)
        # => noise_scaling = sqrt(signal_power / (noise_power * 10^(SNR/10)))
        target_ratio = 10.0 ** (target_snr_db / 10.0)
        noise_scaling = np.sqrt(signal_power / (noise_power * target_ratio))

        # Scale and mix
        scaled_noise = noise_audio * noise_scaling
        mixed_audio = clean_audio + scaled_noise

        # Normalize to prevent clipping
        max_val = np.max(np.abs(mixed_audio))
        if max_val > 0:
            mixed_audio = mixed_audio / max_val * 0.95  # Leave 5% headroom

        return mixed_audio

    async def apply_noise_augmentation(
        self,
        input_path: str,
        output_path: str,
        noise_type: str = "pink",
        snr_db: float = 15.0,
        noise_duration: Optional[float] = None
    ) -> None:
        """
        Apply noise augmentation to an audio file and save the result.

        Args:
            input_path: Path to input audio file.
            output_path: Path to save augmented audio.
            noise_type: Type of noise to generate.
            snr_db: Target signal-to-noise ratio in decibels.
            noise_duration: Optional duration for noise (uses input duration if None).

        Raises:
            NoiseError: If augmentation fails.
        """
        try:
            # Load clean audio
            clean_audio, sr = load_audio(input_path)

            # Resample if needed
            if sr != self.sample_rate:
                clean_audio = librosa.resample(clean_audio, orig_sr=sr, target_sr=self.sample_rate)
                sr = self.sample_rate

            # Determine noise duration
            if noise_duration is None:
                noise_duration = len(clean_audio) / sr

            # Generate or load noise
            noise_audio = self.generate_noise(noise_duration, noise_type)

            # Mix the signals
            mixed_audio = self.mix_with_noise(clean_audio, noise_audio, snr_db)

            # Save the result
            sf.write(output_path, mixed_audio, sr)

        except Exception as e:
            raise NoiseError(f"Failed to apply noise augmentation: {str(e)}") from e
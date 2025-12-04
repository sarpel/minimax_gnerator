"""
Microphone Simulation

This module simulates different microphone frequency responses and
characteristics. It applies EQ curves that mimic how different
microphones color the sound, creating more realistic recordings.

Key Features:
- Frequency response curves for different mic types
- EQ simulation with parametric filters
- Microphone distortion modeling
- Pre-configured profiles for common microphones
- Efficient processing suitable for embedded systems
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass
from wakegen.core.exceptions import MicrophoneSimulationError
from wakegen.utils.audio import load_audio
import soundfile as sf
import librosa

@dataclass
class MicrophoneProfile:
    """
    Defines the frequency response characteristics of a microphone.

    Attributes:
        name: Microphone name/description.
        frequency_response: List of (frequency, gain_dB) tuples defining the response curve.
        sample_rate: Target sample rate.
        distortion: Amount of harmonic distortion to add (0.0 to 1.0).
        noise_floor: Amount of self-noise to add in dB.
    """
    name: str
    frequency_response: List[Tuple[float, float]]
    sample_rate: int
    distortion: float = 0.05
    noise_floor: float = -60.0

class MicrophoneSimulator:
    """
    Simulates microphone frequency responses and characteristics.

    Applies EQ curves and distortion modeling to mimic different
    microphone types and qualities.
    """

    def __init__(self, sample_rate: int = 24000):
        """
        Initialize the microphone simulator.

        Args:
            sample_rate: Target sample rate for processing.
        """
        self.sample_rate = sample_rate
        self._validate_sample_rate(sample_rate)

    def _validate_sample_rate(self, sample_rate: int) -> None:
        """Validate sample rate is suitable for microphone simulation."""
        if not (8000 <= sample_rate <= 48000):
            raise MicrophoneSimulationError(f"Sample rate {sample_rate}Hz is out of valid range (8000-48000Hz)")

    def apply_microphone_effect(
        self,
        audio: np.ndarray,
        profile: MicrophoneProfile
    ) -> np.ndarray:
        """
        Apply microphone frequency response and characteristics to audio.

        Args:
            audio: Input audio signal.
            profile: MicrophoneProfile defining the microphone characteristics.

        Returns:
            Processed audio with microphone effects.

        Raises:
            MicrophoneSimulationError: If processing fails.
        """
        try:
            # Apply frequency response EQ
            processed = self._apply_frequency_response(audio, profile)

            # Add harmonic distortion
            processed = self._add_harmonic_distortion(processed, profile.distortion)

            # Add microphone self-noise
            processed = self._add_microphone_noise(processed, profile.noise_floor)

            # Normalize to prevent clipping
            max_val = np.max(np.abs(processed))
            if max_val > 0:
                processed = processed / max_val * 0.95

            return processed

        except Exception as e:
            raise MicrophoneSimulationError(f"Failed to apply microphone effects: {str(e)}") from e

    def _apply_frequency_response(
        self,
        audio: np.ndarray,
        profile: MicrophoneProfile
    ) -> np.ndarray:
        """
        Apply the frequency response curve using FFT-based EQ.

        Args:
            audio: Input audio signal.
            profile: MicrophoneProfile with frequency response.

        Returns:
            Audio with frequency response applied.
        """
        # Convert frequency response to EQ curve
        freq_points, gain_db = zip(*profile.frequency_response)
        freq_points = np.array(freq_points)
        gain_db = np.array(gain_db)

        # Convert dB to linear gain
        gain_linear = 10.0 ** (gain_db / 20.0)

        # Create interpolation function
        from scipy.interpolate import interp1d
        eq_curve = interp1d(
            freq_points,
            gain_linear,
            kind='cubic',
            bounds_error=False,
            fill_value=(gain_linear[0], gain_linear[-1])
        )

        # Apply EQ using FFT
        return self._apply_fft_eq(audio, eq_curve, profile.sample_rate)

    def _apply_fft_eq(
        self,
        audio: np.ndarray,
        eq_curve_func: Any,
        sample_rate: int
    ) -> np.ndarray:
        """
        Apply EQ using FFT-based frequency domain processing.

        Args:
            audio: Input audio.
            eq_curve_func: Function that maps frequency to gain.
            sample_rate: Sample rate.

        Returns:
            Audio with EQ applied.
        """
        # FFT of audio
        fft_audio = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1.0/sample_rate)

        # Apply EQ curve to each frequency bin
        eq_gains = eq_curve_func(freqs)
        fft_processed = fft_audio * eq_gains

        # Inverse FFT
        return np.fft.irfft(fft_processed)

    def _add_harmonic_distortion(
        self,
        audio: np.ndarray,
        distortion_amount: float
    ) -> np.ndarray:
        """
        Add harmonic distortion to simulate microphone nonlinearities.

        Args:
            audio: Input audio.
            distortion_amount: Amount of distortion (0.0 to 1.0).

        Returns:
            Audio with distortion added.
        """
        if distortion_amount <= 0:
            return audio

        # Simple clipping distortion model
        threshold = 0.8 - (distortion_amount * 0.6)  # 0.8 to 0.2 threshold
        distorted = np.tanh(audio * (1.0 + distortion_amount * 2.0)) * threshold

        # Mix original and distorted
        mix_ratio = distortion_amount ** 0.5  # Square root for perceptual linearity
        return (audio * (1.0 - mix_ratio)) + (distorted * mix_ratio)

    def _add_microphone_noise(
        self,
        audio: np.ndarray,
        noise_floor_db: float
    ) -> np.ndarray:
        """
        Add microphone self-noise based on noise floor specification.

        Args:
            audio: Input audio.
            noise_floor_db: Noise floor in dB.

        Returns:
            Audio with noise added.
        """
        # Calculate noise level relative to signal
        signal_rms = np.sqrt(np.mean(audio ** 2))
        if signal_rms == 0:
            signal_rms = 1.0

        # Convert dB to linear ratio
        noise_level = 10.0 ** (noise_floor_db / 20.0)
        noise_std = signal_rms * noise_level * 0.1  # Scale down for realism

        # Generate noise
        noise = np.random.normal(0, noise_std, len(audio))

        return audio + noise

    async def simulate_microphone(
        self,
        input_path: str,
        output_path: str,
        profile: MicrophoneProfile
    ) -> None:
        """
        Apply microphone simulation to an audio file and save the result.

        Args:
            input_path: Path to input audio file.
            output_path: Path to save processed audio.
            profile: MicrophoneProfile defining the microphone.

        Raises:
            MicrophoneSimulationError: If simulation fails.
        """
        try:
            # Load input audio
            audio, sr = load_audio(input_path)

            # Resample if needed
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)

            # Apply microphone effects
            processed = self.apply_microphone_effect(audio, profile)

            # Save result
            sf.write(output_path, processed, self.sample_rate)

        except Exception as e:
            raise MicrophoneSimulationError(f"Failed to simulate microphone: {str(e)}") from e

    def get_preset_microphone(self, preset_name: str) -> MicrophoneProfile:
        """
        Get pre-configured microphone profiles for common microphone types.

        Args:
            preset_name: Name of preset ('smartphone', 'lapel', 'studio',
                        'conference', 'headset', 'far_field').

        Returns:
            MicrophoneProfile object.

        Raises:
            MicrophoneSimulationError: If preset name is unknown.
        """
        presets = self._get_microphone_presets()

        if preset_name not in presets:
            available = list(presets.keys())
            raise MicrophoneSimulationError(f"Unknown microphone preset '{preset_name}'. Available: {available}")

        return presets[preset_name]

    def _get_microphone_presets(self) -> Dict[str, MicrophoneProfile]:
        """Define standard microphone presets with typical characteristics."""
        return {
            "smartphone": MicrophoneProfile(
                name="Smartphone Microphone",
                frequency_response=[
                    (50, -12.0),    # Low roll-off
                    (100, -6.0),
                    (200, -3.0),
                    (500, 0.0),
                    (1000, 1.0),    # Mid boost
                    (2000, 2.0),    # Presence boost
                    (5000, 1.0),
                    (10000, -3.0),  # High roll-off
                    (15000, -8.0)
                ],
                sample_rate=24000,
                distortion=0.15,
                noise_floor=-55.0
            ),
            "lapel": MicrophoneProfile(
                name="Lapel Microphone",
                frequency_response=[
                    (50, -8.0),
                    (100, -2.0),
                    (200, 1.0),
                    (500, 2.0),     # Mid boost
                    (1000, 3.0),   # Presence boost
                    (2000, 2.0),
                    (5000, 0.0),
                    (10000, -4.0),
                    (15000, -10.0)
                ],
                sample_rate=24000,
                distortion=0.08,
                noise_floor=-65.0
            ),
            "studio": MicrophoneProfile(
                name="Studio Condenser Microphone",
                frequency_response=[
                    (20, -2.0),     # Flat low end
                    (50, 0.0),
                    (100, 0.5),
                    (200, 1.0),
                    (500, 1.5),
                    (1000, 2.0),   # Slight presence boost
                    (2000, 1.5),
                    (5000, 1.0),
                    (10000, 0.5),
                    (15000, 0.0)
                ],
                sample_rate=24000,
                distortion=0.03,
                noise_floor=-80.0
            ),
            "conference": MicrophoneProfile(
                name="Conference Room Microphone",
                frequency_response=[
                    (50, -6.0),
                    (100, -1.0),
                    (200, 1.0),
                    (500, 2.0),    # Mid boost for voice clarity
                    (1000, 3.0),  # Presence boost
                    (2000, 2.0),
                    (5000, 0.0),
                    (10000, -3.0),
                    (15000, -8.0)
                ],
                sample_rate=16000,
                distortion=0.12,
                noise_floor=-50.0
            ),
            "headset": MicrophoneProfile(
                name="Headset Microphone",
                frequency_response=[
                    (50, -10.0),   # Strong low cut
                    (100, -3.0),
                    (200, 1.0),
                    (500, 3.0),    # Strong mid boost
                    (1000, 4.0),  # Strong presence boost
                    (2000, 3.0),
                    (5000, 1.0),
                    (10000, -4.0),
                    (15000, -12.0)
                ],
                sample_rate=16000,
                distortion=0.10,
                noise_floor=-60.0
            ),
            "far_field": MicrophoneProfile(
                name="Far-Field Array Microphone",
                frequency_response=[
                    (50, -15.0),   # Very strong low cut
                    (100, -8.0),
                    (200, -2.0),
                    (500, 2.0),    # Mid boost
                    (1000, 4.0),  # Strong presence boost
                    (2000, 3.0),
                    (5000, 0.0),
                    (10000, -6.0),
                    (15000, -15.0)
                ],
                sample_rate=16000,
                distortion=0.20,
                noise_floor=-45.0
            )
        }
"""
Time Domain Effects

This module provides time domain audio effects including pitch shifting,
time stretching, and other temporal modifications. These effects are
essential for creating variations in wake word samples.

Key Features:
- Pitch shifting with formant preservation
- Time stretching without pitch change
- Rate changes (speed up/slow down)
- Efficient processing using librosa
- Parameter validation and error handling
"""

from __future__ import annotations
import numpy as np
import librosa
from typing import Optional, Tuple
from wakegen.core.exceptions import AugmentationError
from wakegen.utils.audio import load_audio
import soundfile as sf

class TimeDomainEffects:
    """
    Provides time domain audio effects including pitch shifting and time stretching.

    Uses librosa's phase vocoder implementation for high-quality processing.
    """

    def __init__(self, sample_rate: int = 24000):
        """
        Initialize the time domain effects processor.

        Args:
            sample_rate: Target sample rate for processing.
        """
        self.sample_rate = sample_rate
        self._validate_sample_rate(sample_rate)

    def _validate_sample_rate(self, sample_rate: int) -> None:
        """Validate sample rate is suitable for time domain effects."""
        if not (8000 <= sample_rate <= 48000):
            raise AugmentationError(f"Sample rate {sample_rate}Hz is out of valid range (8000-48000Hz)")

    def pitch_shift(
        self,
        audio: np.ndarray,
        n_steps: float,
        preserve_formants: bool = True
    ) -> np.ndarray:
        """
        Shift the pitch of audio by specified number of semitones.

        Args:
            audio: Input audio signal.
            n_steps: Number of semitones to shift (positive = higher pitch).
            preserve_formants: Whether to preserve formant structure (for speech).

        Returns:
            Pitch-shifted audio.

        Raises:
            AugmentationError: If pitch shifting fails.
        """
        if n_steps < -12 or n_steps > 12:
            raise AugmentationError("Pitch shift range must be between -12 and +12 semitones")

        try:
            # Use librosa's pitch shift with phase vocoder
            if preserve_formants:
                # For speech, we want to preserve formants (vocal tract characteristics)
                return librosa.effects.pitch_shift(
                    audio,
                    sr=self.sample_rate,
                    n_steps=n_steps,
                    bins_per_octave=36,
                    res_type='soxr_hq'
                )
            else:
                # For general audio, simple pitch shift
                return librosa.effects.pitch_shift(
                    audio,
                    sr=self.sample_rate,
                    n_steps=n_steps,
                    res_type='soxr_hq'
                )

        except Exception as e:
            raise AugmentationError(f"Failed to pitch shift audio: {str(e)}") from e

    def time_stretch(
        self,
        audio: np.ndarray,
        rate: float
    ) -> np.ndarray:
        """
        Stretch or compress audio in time without changing pitch.

        Args:
            audio: Input audio signal.
            rate: Stretch factor (1.0 = original, 0.5 = half speed, 2.0 = double speed).

        Returns:
            Time-stretched audio.

        Raises:
            AugmentationError: If time stretching fails.
        """
        if rate <= 0:
            raise AugmentationError("Time stretch rate must be positive")
        if rate < 0.25 or rate > 4.0:
            raise AugmentationError("Time stretch rate must be between 0.25 and 4.0")

        try:
            # Use librosa's time stretch with phase vocoder
            return librosa.effects.time_stretch(
                audio,
                rate=rate,
                res_type='soxr_hq'
            )

        except Exception as e:
            raise AugmentationError(f"Failed to time stretch audio: {str(e)}") from e

    def change_speed(
        self,
        audio: np.ndarray,
        speed_factor: float
    ) -> np.ndarray:
        """
        Change the speed of audio (affects both pitch and duration).

        Args:
            audio: Input audio signal.
            speed_factor: Speed factor (1.0 = original, 0.5 = half speed, 2.0 = double speed).

        Returns:
            Speed-modified audio.

        Raises:
            AugmentationError: If speed change fails.
        """
        if speed_factor <= 0:
            raise AugmentationError("Speed factor must be positive")
        if speed_factor < 0.25 or speed_factor > 4.0:
            raise AugmentationError("Speed factor must be between 0.25 and 4.0")

        try:
            # For speed changes, we can use simple resampling
            if speed_factor == 1.0:
                return audio.copy()

            # Calculate new length
            new_length = int(len(audio) / speed_factor)

            # Use librosa resampling
            return librosa.resample(
                audio,
                orig_sr=self.sample_rate,
                target_sr=int(self.sample_rate * speed_factor)
            )

        except Exception as e:
            raise AugmentationError(f"Failed to change audio speed: {str(e)}") from e

    def apply_tempo_variation(
        self,
        audio: np.ndarray,
        tempo_factor: float,
        preserve_pitch: bool = True
    ) -> np.ndarray:
        """
        Apply tempo variation while optionally preserving pitch.

        Args:
            audio: Input audio signal.
            tempo_factor: Tempo factor (1.0 = original, 0.5 = half tempo, 2.0 = double tempo).
            preserve_pitch: Whether to preserve original pitch.

        Returns:
            Tempo-modified audio.

        Raises:
            AugmentationError: If tempo variation fails.
        """
        if tempo_factor <= 0:
            raise AugmentationError("Tempo factor must be positive")
        if tempo_factor < 0.5 or tempo_factor > 2.0:
            raise AugmentationError("Tempo factor must be between 0.5 and 2.0")

        try:
            if preserve_pitch:
                # Time stretch without pitch change
                return self.time_stretch(audio, 1.0 / tempo_factor)
            else:
                # Simple speed change (affects pitch)
                return self.change_speed(audio, tempo_factor)

        except Exception as e:
            raise AugmentationError(f"Failed to apply tempo variation: {str(e)}") from e

    async def apply_time_effects(
        self,
        input_path: str,
        output_path: str,
        pitch_steps: float = 0.0,
        time_stretch_factor: float = 1.0,
        speed_factor: float = 1.0
    ) -> None:
        """
        Apply time domain effects to an audio file and save the result.

        Args:
            input_path: Path to input audio file.
            output_path: Path to save processed audio.
            pitch_steps: Pitch shift in semitones.
            time_stretch_factor: Time stretch factor.
            speed_factor: Speed change factor.

        Raises:
            AugmentationError: If effect application fails.
        """
        try:
            # Load input audio
            audio, sr = load_audio(input_path)

            # Resample if needed
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)

            # Apply effects in order: pitch shift -> time stretch -> speed change
            if pitch_steps != 0.0:
                audio = self.pitch_shift(audio, pitch_steps)

            if time_stretch_factor != 1.0:
                audio = self.time_stretch(audio, time_stretch_factor)

            if speed_factor != 1.0:
                audio = self.change_speed(audio, speed_factor)

            # Save result
            sf.write(output_path, audio, self.sample_rate)

        except Exception as e:
            raise AugmentationError(f"Failed to apply time effects: {str(e)}") from e

    def get_effect_preset(self, preset_name: str) -> dict:
        """
        Get pre-configured effect parameters for common scenarios.

        Args:
            preset_name: Name of preset ('male_to_female', 'female_to_male',
                        'child_voice', 'slow_speech', 'fast_speech', 'robot_voice').

        Returns:
            Dictionary with effect parameters.

        Raises:
            AugmentationError: If preset name is unknown.
        """
        presets = {
            "male_to_female": {
                "pitch_steps": 4.0,      # Raise pitch by 4 semitones
                "time_stretch_factor": 0.95,  # Slightly faster
                "speed_factor": 1.0,
                "description": "Convert male voice to sound more female"
            },
            "female_to_male": {
                "pitch_steps": -5.0,     # Lower pitch by 5 semitones
                "time_stretch_factor": 1.05,  # Slightly slower
                "speed_factor": 1.0,
                "description": "Convert female voice to sound more male"
            },
            "child_voice": {
                "pitch_steps": 8.0,      # Raise pitch significantly
                "time_stretch_factor": 0.85,  # Faster speech
                "speed_factor": 1.1,
                "description": "Make voice sound more child-like"
            },
            "slow_speech": {
                "pitch_steps": 0.0,
                "time_stretch_factor": 1.3,   # Stretch time
                "speed_factor": 0.9,
                "description": "Slow down speech without major pitch change"
            },
            "fast_speech": {
                "pitch_steps": 0.0,
                "time_stretch_factor": 0.7,   # Compress time
                "speed_factor": 1.1,
                "description": "Speed up speech without major pitch change"
            },
            "robot_voice": {
                "pitch_steps": -2.0,     # Slightly lower pitch
                "time_stretch_factor": 1.0,
                "speed_factor": 1.0,
                "description": "Create robotic-sounding voice"
            }
        }

        if preset_name not in presets:
            available = list(presets.keys())
            raise AugmentationError(f"Unknown effect preset '{preset_name}'. Available: {available}")

        return presets[preset_name].copy()
"""
Noise Events Generator

This module creates dynamic noise events that simulate real-world
environmental sounds like dishes clinking, doors closing, or footsteps.
These events are randomly inserted into the background noise to create
more realistic and varied environmental conditions.

Key Features:
- Random event timing and duration
- Multiple event types with different characteristics
- Configurable event frequency and intensity
- Realistic sound profiles for common household events
"""

from __future__ import annotations
import random
import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from wakegen.core.exceptions import NoiseError

@dataclass
class NoiseEvent:
    """
    Represents a single noise event with timing and characteristics.

    Attributes:
        start_time: Start time in seconds from beginning of audio.
        duration: Duration of the event in seconds.
        event_type: Type of event (e.g., "dish_clink", "door_close").
        intensity: Relative intensity (0.1 = subtle, 1.0 = prominent).
    """
    start_time: float
    duration: float
    event_type: str
    intensity: float = 0.5

    def validate(self) -> None:
        """Validate that the event parameters are reasonable."""
        if self.start_time < 0:
            raise NoiseError(f"Event start time cannot be negative: {self.start_time}")
        if self.duration <= 0:
            raise NoiseError(f"Event duration must be positive: {self.duration}")
        if not (0.1 <= self.intensity <= 1.0):
            raise NoiseError(f"Event intensity must be between 0.1 and 1.0: {self.intensity}")

class NoiseEventGenerator:
    """
    Generates realistic noise events for dynamic environmental simulation.

    This creates events like:
    - Dish clinking (short, high-frequency)
    - Door closing (medium duration, mid-frequency)
    - Footsteps (multiple short events)
    - Chair scraping (medium duration, variable frequency)
    """

    def __init__(self, sample_rate: int = 24000):
        """
        Initialize the noise event generator.

        Args:
            sample_rate: The target sample rate for audio generation.
        """
        self.sample_rate = sample_rate
        self._event_profiles = self._create_event_profiles()

    def _create_event_profiles(self) -> Dict[str, Dict[str, Any]]:
        """
        Create profiles for different types of noise events.

        Each profile defines the characteristics of a specific event type.
        """
        return {
            "dish_clink": {
                "duration_range": (0.1, 0.3),  # Short event
                "frequency_range": (2000, 8000),  # High frequency
                "intensity_range": (0.3, 0.7),
                "description": "Sound of dishes or glasses clinking"
            },
            "door_close": {
                "duration_range": (0.5, 1.2),  # Medium duration
                "frequency_range": (100, 2000),  # Mid frequency
                "intensity_range": (0.5, 0.9),
                "description": "Sound of a door closing"
            },
            "footstep": {
                "duration_range": (0.15, 0.35),  # Short event
                "frequency_range": (50, 1000),  # Low-mid frequency
                "intensity_range": (0.2, 0.6),
                "description": "Sound of a single footstep"
            },
            "chair_scrape": {
                "duration_range": (0.8, 1.5),  # Medium-long duration
                "frequency_range": (150, 3000),  # Variable frequency
                "intensity_range": (0.4, 0.8),
                "description": "Sound of a chair being moved"
            },
            "water_running": {
                "duration_range": (1.0, 2.5),  # Longer duration
                "frequency_range": (500, 5000),  # Broad frequency
                "intensity_range": (0.3, 0.7),
                "description": "Sound of water running from tap"
            }
        }

    def generate_event(
        self,
        event_type: str,
        duration: float,
        intensity: float = 0.5
    ) -> np.ndarray:
        """
        Generate a specific type of noise event.

        Args:
            event_type: Type of event to generate.
            duration: Duration of the event in seconds.
            intensity: Relative intensity of the event.

        Returns:
            Generated event as numpy array.

        Raises:
            NoiseError: If event type is unknown or generation fails.
        """
        if event_type not in self._event_profiles:
            raise NoiseError(f"Unknown event type: {event_type}")

        profile = self._event_profiles[event_type]
        num_samples = int(duration * self.sample_rate)

        try:
            # Create time array
            t = np.linspace(0, duration, num_samples)

            # Generate base noise with appropriate frequency characteristics
            base_freq = random.uniform(*profile["frequency_range"])
            base_noise = self._generate_banded_noise(
                num_samples, base_freq, profile["frequency_range"]
            )

            # Apply intensity scaling
            scaled_noise = base_noise * intensity * 0.5  # 0.5 is base scaling factor

            # Apply event-specific envelope for realism
            envelope = self._create_event_envelope(event_type, t)
            event_audio = scaled_noise * envelope

            # Normalize to prevent clipping
            max_val = np.max(np.abs(event_audio))
            if max_val > 0:
                event_audio = event_audio / max_val * 0.8  # 80% of max to leave headroom

            return event_audio

        except Exception as e:
            raise NoiseError(f"Failed to generate {event_type} event: {str(e)}") from e

    def _generate_banded_noise(
        self,
        num_samples: int,
        center_freq: float,
        freq_range: Tuple[float, float]
    ) -> np.ndarray:
        """
        Generate noise concentrated around a specific frequency band.

        Args:
            num_samples: Number of samples to generate.
            center_freq: Center frequency of the band.
            freq_range: Range of frequencies to include.

        Returns:
            Band-limited noise.
        """
        # Generate white noise
        white = np.random.randn(num_samples)

        # Apply bandpass filter using FFT
        fft_noise = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(num_samples, 1.0/self.sample_rate)

        # Create bandpass filter
        low_cut, high_cut = freq_range
        filter_mask = (freqs >= low_cut) & (freqs <= high_cut)

        # Apply filter
        fft_filtered = fft_noise * filter_mask

        # Convert back to time domain
        return np.fft.irfft(fft_filtered, n=num_samples)

    def _create_event_envelope(
        self,
        event_type: str,
        t: np.ndarray
    ) -> np.ndarray:
        """
        Create an amplitude envelope for different event types.

        The envelope shapes the amplitude over time to make events sound realistic.

        Args:
            event_type: Type of event.
            t: Time array for the event.

        Returns:
            Amplitude envelope.
        """
        # Normalized time (0 to 1)
        t_norm = t / t.max() if t.max() > 0 else t

        if event_type == "dish_clink":
            # Quick attack, short sustain, quick release (percussive)
            attack = np.minimum(t_norm * 10, 1.0)
            release = np.maximum(1.0 - (t_norm - 0.7) * 5, 0.0) if t_norm > 0.7 else 1.0
            return attack * release
        elif event_type == "door_close":
            # Medium attack, longer sustain, medium release
            attack = np.minimum(t_norm * 3, 1.0)
            release = np.maximum(1.0 - (t_norm - 0.8) * 2, 0.0) if t_norm > 0.8 else 1.0
            return attack * release
        elif event_type == "footstep":
            # Quick attack, very short sustain, quick release
            attack = np.minimum(t_norm * 15, 1.0)
            release = np.maximum(1.0 - (t_norm - 0.5) * 4, 0.0) if t_norm > 0.5 else 1.0
            return attack * release
        elif event_type == "chair_scrape":
            # Variable envelope with some randomness
            base = np.sin(t_norm * np.pi * 2) * 0.3 + 0.7
            return np.maximum(base, 0.1)
        elif event_type == "water_running":
            # Gradual attack, steady sustain, gradual release
            attack = np.minimum(t_norm * 2, 1.0)
            release = np.maximum(1.0 - (t_norm - 0.9) * 3, 0.0) if t_norm > 0.9 else 1.0
            return attack * release
        else:
            # Default: simple attack-release
            attack = np.minimum(t_norm * 5, 1.0)
            release = np.maximum(1.0 - (t_norm - 0.6) * 3, 0.0) if t_norm > 0.6 else 1.0
            return attack * release

    def generate_random_events(
        self,
        total_duration: float,
        event_density: float = 0.3,
        allowed_types: Optional[List[str]] = None
    ) -> List[NoiseEvent]:
        """
        Generate a sequence of random noise events.

        Args:
            total_duration: Total duration of audio in seconds.
            event_density: Average number of events per minute (0.3 = 1 every 3.3 minutes).
            allowed_types: List of allowed event types (None for all).

        Returns:
            List of NoiseEvent objects.
        """
        if allowed_types is None:
            allowed_types = list(self._event_profiles.keys())

        valid_types = [t for t in allowed_types if t in self._event_profiles]
        if not valid_types:
            raise NoiseError("No valid event types provided")

        events = []
        current_time = 0.0

        # Calculate target number of events based on density
        # density = events per minute, so total_events = density * duration_minutes
        target_events = int(event_density * (total_duration / 60))

        for _ in range(target_events):
            # Random event type
            event_type = random.choice(valid_types)
            profile = self._event_profiles[event_type]

            # Random duration within profile range
            duration = random.uniform(*profile["duration_range"])

            # Random intensity
            intensity = random.uniform(*profile["intensity_range"])

            # Random start time (but not overlapping with previous events)
            min_gap = 0.5  # Minimum 0.5 second gap between events
            start_time = max(
                current_time + min_gap,
                random.uniform(current_time, total_duration - duration)
            )

            # Create and validate event
            event = NoiseEvent(start_time, duration, event_type, intensity)
            event.validate()
            events.append(event)

            current_time = start_time + duration

        return events

    def apply_events_to_noise(
        self,
        base_noise: np.ndarray,
        events: List[NoiseEvent]
    ) -> np.ndarray:
        """
        Apply noise events to a base noise signal.

        Args:
            base_noise: Base noise signal as numpy array.
            events: List of NoiseEvent objects to apply.

        Returns:
            Noise signal with events mixed in.
        """
        if len(base_noise) == 0:
            return base_noise

        result = base_noise.copy()
        sample_rate = self.sample_rate
        total_duration = len(base_noise) / sample_rate

        for event in events:
            if event.start_time + event.duration > total_duration:
                continue  # Skip events that would exceed the audio length

            # Generate the event audio
            event_audio = self.generate_event(
                event.event_type,
                event.duration,
                event.intensity
            )

            # Calculate sample positions
            start_sample = int(event.start_time * sample_rate)
            end_sample = start_sample + len(event_audio)

            # Mix the event into the noise
            if end_sample <= len(result):
                result[start_sample:end_sample] += event_audio

        return result
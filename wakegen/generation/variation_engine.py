"""Variation Engine

This module generates parameter combinations for creating diverse wake word audio samples.
It handles the Cartesian product of text variations, voice parameters, and audio effects.

The variation engine is responsible for:
- Generating text variations (prosody, intonation, emphasis)
- Creating parameter combinations (voice × speed × pitch × prosody)
- Ensuring diverse sample generation
- Supporting Turkish wake word variations
"""

from __future__ import annotations

import itertools
import random
from typing import List, Dict, Any, Tuple, Optional, Iterator
from dataclasses import dataclass

from wakegen.models.generation import GenerationParameters
from wakegen.core.exceptions import GenerationError

@dataclass
class VariationParameters:
    """Parameters for generating variations of wake words.

    Attributes:
        text_variations: Different ways to say the wake word (e.g., "hey katya", "katya", "hey katya!")
        voice_ids: List of voice IDs to use for generation
        speed_range: Range of speech speeds (0.5 to 2.0 typically)
        pitch_range: Range of pitch values (0.5 to 2.0 typically)
        prosody_variations: Different intonation patterns
        emphasis_positions: Positions to emphasize in the text
    """
    text_variations: List[str]
    voice_ids: List[str]
    speed_range: Tuple[float, float] = (0.8, 1.2)
    pitch_range: Tuple[float, float] = (0.9, 1.1)
    prosody_variations: List[str] = None
    emphasis_positions: List[int] = None

    def __post_init__(self):
        """Initialize default variations if not provided."""
        if self.prosody_variations is None:
            # Default prosody variations for natural speech
            self.prosody_variations = ["normal", "happy", "questioning", "urgent"]
        if self.emphasis_positions is None:
            # Default emphasis positions (none by default)
            self.emphasis_positions = []

class VariationEngine:
    """Engine for generating diverse parameter combinations for wake word samples.

    This class creates the Cartesian product of all variation parameters to ensure
    diverse audio sample generation. It's designed to be memory-efficient and
    work well with async generation pipelines.
    """

    def __init__(self, parameters: VariationParameters):
        """Initialize the variation engine with generation parameters.

        Args:
            parameters: Variation parameters containing text, voices, and ranges
        """
        self.parameters = parameters
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate that parameters are within reasonable bounds."""
        if not self.parameters.text_variations:
            raise GenerationError("At least one text variation must be provided")

        if not self.parameters.voice_ids:
            raise GenerationError("At least one voice ID must be provided")

        if self.parameters.speed_range[0] <= 0 or self.parameters.speed_range[1] <= 0:
            raise GenerationError("Speed range values must be positive")

        if self.parameters.pitch_range[0] <= 0 or self.parameters.pitch_range[1] <= 0:
            raise GenerationError("Pitch range values must be positive")

        if len(self.parameters.text_variations) > 100:
            raise GenerationError("Too many text variations (max 100)")

        if len(self.parameters.voice_ids) > 50:
            raise GenerationError("Too many voice IDs (max 50)")

    def _generate_speed_values(self, count: int = 3) -> List[float]:
        """Generate speed values within the specified range.

        Args:
            count: Number of speed values to generate

        Returns:
            List of speed values evenly distributed in the range
        """
        start, end = self.parameters.speed_range
        return [start + (end - start) * (i / (count - 1)) for i in range(count)]

    def _generate_pitch_values(self, count: int = 3) -> List[float]:
        """Generate pitch values within the specified range.

        Args:
            count: Number of pitch values to generate

        Returns:
            List of pitch values evenly distributed in the range
        """
        start, end = self.parameters.pitch_range
        return [start + (end - start) * (i / (count - 1)) for i in range(count)]

    def _generate_text_with_emphasis(self, text: str) -> List[str]:
        """Generate text variations with different emphasis patterns.

        Args:
            text: Base text to vary

        Returns:
            List of text variations with different emphasis
        """
        if not self.parameters.emphasis_positions:
            return [text]

        variations = []
        words = text.split()

        # Generate variations with emphasis at different positions
        for pos in self.parameters.emphasis_positions:
            if pos < len(words):
                emphasized_words = words.copy()
                emphasized_words[pos] = f"<emphasis>{emphasized_words[pos]}</emphasis>"
                variations.append(" ".join(emphasized_words))

        return variations if variations else [text]

    def generate_variations(self, max_combinations: Optional[int] = None) -> Iterator[GenerationParameters]:
        """Generate all parameter combinations as GenerationParameters objects.

        This method creates the Cartesian product of:
        - Text variations (with emphasis)
        - Voice IDs
        - Speed values
        - Pitch values
        - Prosody variations

        Args:
            max_combinations: Maximum number of combinations to generate (None for all)

        Returns:
            Iterator of GenerationParameters objects

        Yields:
            GenerationParameters for each combination
        """
        # Generate all text variations including emphasis
        all_texts = []
        for base_text in self.parameters.text_variations:
            emphasis_variations = self._generate_text_with_emphasis(base_text)
            all_texts.extend(emphasis_variations)

        # Generate speed and pitch values
        speed_values = self._generate_speed_values()
        pitch_values = self._generate_pitch_values()

        # Create Cartesian product of all parameters
        combinations = itertools.product(
            all_texts,
            self.parameters.voice_ids,
            speed_values,
            pitch_values,
            self.parameters.prosody_variations
        )

        # Limit combinations if requested
        if max_combinations is not None:
            combinations = itertools.islice(combinations, max_combinations)

        # Convert to GenerationParameters objects
        for text, voice_id, speed, pitch, prosody in combinations:
            yield GenerationParameters(
                text=text,
                voice_id=voice_id,
                speed=speed,
                pitch=pitch,
                prosody=prosody,
                emphasis_positions=self.parameters.emphasis_positions
            )

    def estimate_total_combinations(self) -> int:
        """Estimate the total number of combinations that would be generated.

        Returns:
            Estimated total combinations count
        """
        # Calculate text variations with emphasis
        text_count = len(self.parameters.text_variations)
        if self.parameters.emphasis_positions:
            # Each text can have multiple emphasis variations
            text_count *= len(self.parameters.emphasis_positions) + 1

        speed_count = 3  # Default number of speed values
        pitch_count = 3  # Default number of pitch values
        prosody_count = len(self.parameters.prosody_variations)
        voice_count = len(self.parameters.voice_ids)

        return text_count * voice_count * speed_count * pitch_count * prosody_count

    def generate_turkish_variations(self, base_word: str) -> List[str]:
        """Generate Turkish-specific variations for wake words.

        Args:
            base_word: Base Turkish wake word

        Returns:
            List of Turkish variations
        """
        # Common Turkish wake word patterns
        variations = [
            base_word,
            f"hey {base_word}",
            f"{base_word} lütfen",
            f"hey {base_word} lütfen",
            f"{base_word}!",
            f"hey {base_word}!",
            f"{base_word}, lütfen",
            f"hey {base_word}, lütfen"
        ]

        # Add some informal variations
        informal_variations = [
            f"hey {base_word} canım",
            f"{base_word} canım",
            f"hey {base_word} abla",
            f"hey {base_word} abi"
        ]

        return variations + informal_variations

    def create_turkish_parameters(self, wake_words: List[str], voice_ids: List[str]) -> VariationParameters:
        """Create Turkish-specific variation parameters.

        Args:
            wake_words: List of Turkish wake words
            voice_ids: List of Turkish voice IDs

        Returns:
            VariationParameters configured for Turkish
        """
        # Generate all Turkish text variations
        all_texts = []
        for word in wake_words:
            all_texts.extend(self.generate_turkish_variations(word))

        return VariationParameters(
            text_variations=all_texts,
            voice_ids=voice_ids,
            speed_range=(0.7, 1.3),  # Wider range for Turkish intonation
            pitch_range=(0.8, 1.2),  # Wider range for Turkish pitch
            prosody_variations=["normal", "friendly", "polite", "urgent"],
            emphasis_positions=[0, 1]  # Emphasize first and second words
        )
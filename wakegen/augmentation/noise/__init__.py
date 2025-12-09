"""
WakeGen Augmentation Noise Module

This module provides background noise generation and mixing capabilities
for creating realistic environmental variations.
"""

from .events import NoiseEvent, NoiseEventGenerator
from .mixer import NoiseMixer
from .profiles import NoiseProfile, NoiseProfileManager

__all__ = [
    "NoiseEvent",
    "NoiseEventGenerator",
    "NoiseMixer",
    "NoiseProfile",
    "NoiseProfileManager",
]

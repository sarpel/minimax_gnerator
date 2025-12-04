"""
WakeGen Augmentation Noise Module

This module provides background noise generation and mixing capabilities
for creating realistic environmental variations.
"""

from .mixer import NoiseMixer
from .events import NoiseEventGenerator, NoiseEvent
from .profiles import NoiseProfileManager, NoiseProfile

__all__ = [
    "NoiseMixer",
    "NoiseEventGenerator",
    "NoiseEvent",
    "NoiseProfileManager",
    "NoiseProfile"
]
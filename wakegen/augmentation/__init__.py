"""
WakeGen Augmentation Module

This module provides audio augmentation capabilities for creating realistic
environmental variations of wake word samples. It includes:

- Background noise mixing with configurable SNR
- Room simulation with different acoustics
- Microphone frequency response simulation
- Time domain effects (pitch shift, time stretch)
- Quality degradation effects
- Pre-built environment profiles

The augmentation pipeline coordinates all these components to create
realistic variations that improve wake word model robustness.
"""

from .pipeline import AugmentationPipeline
from .profiles import get_profile, EnvironmentProfile

__all__ = [
    "AugmentationPipeline",
    "get_profile",
    "EnvironmentProfile"
]
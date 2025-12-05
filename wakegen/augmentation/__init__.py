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
- Device-specific augmentation presets
- Telephony and distance simulation

The augmentation pipeline coordinates all these components to create
realistic variations that improve wake word model robustness.
"""

from .pipeline import AugmentationPipeline
from .profiles import get_profile, EnvironmentProfile
from .device_presets import (
    TargetDevice,
    DevicePreset,
    DevicePresetManager,
    get_device_preset,
    list_device_presets,
    device_preset_manager,
)

__all__ = [
    # Core pipeline
    "AugmentationPipeline",
    # Environment profiles
    "get_profile",
    "EnvironmentProfile",
    # Device presets
    "TargetDevice",
    "DevicePreset",
    "DevicePresetManager",
    "get_device_preset",
    "list_device_presets",
    "device_preset_manager",
]
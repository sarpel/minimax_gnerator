# This file initializes the 'core' module.
# The core module contains the fundamental building blocks of our application,
# such as basic types, exceptions, and protocols (interfaces).

from wakegen.core.exceptions import (  # Additional exceptions that were missing from exports (Issue 8):
    AudioError,
    AugmentationError,
    ConfigError,
    GenerationError,
    MicrophoneSimulationError,
    NoiseError,
    ProviderError,
    QualityAssuranceError,
    RoomSimulationError,
    WakeGenError,
)
from wakegen.core.protocols import TTSProvider
from wakegen.core.types import (
    AudioFormat,
    AugmentationType,
    EnvironmentProfile,
    Gender,
    ProviderType,
    QualityLevel,
)

__all__ = [
    # Types
    "ProviderType",
    "AudioFormat",
    "QualityLevel",
    "Gender",
    "AugmentationType",
    "EnvironmentProfile",
    # Exceptions
    "WakeGenError",
    "ProviderError",
    "ConfigError",
    "AudioError",
    "GenerationError",
    "AugmentationError",
    # Additional exceptions (Issue 8):
    "NoiseError",
    "RoomSimulationError",
    "MicrophoneSimulationError",
    "QualityAssuranceError",
    # Protocols
    "TTSProvider",
]

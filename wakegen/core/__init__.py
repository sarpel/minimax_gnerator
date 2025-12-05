# This file initializes the 'core' module.
# The core module contains the fundamental building blocks of our application,
# such as basic types, exceptions, and protocols (interfaces).

from wakegen.core.types import (
    ProviderType,
    AudioFormat,
    QualityLevel,
    Gender,
    AugmentationType,
    EnvironmentProfile,
)
from wakegen.core.exceptions import (
    WakeGenError,
    ProviderError,
    ConfigError,
    AudioError,
    GenerationError,
    AugmentationError,
)
from wakegen.core.protocols import TTSProvider

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
    # Protocols
    "TTSProvider",
]
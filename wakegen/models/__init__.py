# This file initializes the 'models' module.
# This module contains the data structures (Pydantic models) used to represent
# our application's data, such as audio samples, voices, and configuration.

from wakegen.models.audio import AudioSample, ProviderCapabilities, Voice
from wakegen.models.config import GenerationConfig, ProviderConfig
from wakegen.models.generation import (
                                       GenerationParameters,
                                       GenerationRequest,
                                       GenerationResponse,
                                       GenerationResult,
)

__all__ = [
    # Audio models
    "Voice",
    "AudioSample",
    "ProviderCapabilities",
    # Config models
    "ProviderConfig",
    "GenerationConfig",
    # Generation models
    "GenerationRequest",
    "GenerationResponse",
    "GenerationParameters",
    "GenerationResult",
]

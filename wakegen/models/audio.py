from __future__ import annotations
from typing import Optional, List
from pydantic import BaseModel, Field
from wakegen.core.types import ProviderType, Gender

# We use Pydantic 'BaseModel' to define our data structures.
# Pydantic automatically validates the data types for us.

class Voice(BaseModel):
    """
    Represents a specific voice available from a TTS provider.
    """
    id: str = Field(..., description="The unique identifier for the voice (e.g., 'en-US-AriaNeural')")
    name: str = Field(..., description="The human-readable name of the voice")
    gender: Gender = Field(..., description="The gender of the voice")
    language: str = Field(..., description="The language code (e.g., 'en-US')")
    provider: ProviderType = Field(..., description="The provider this voice belongs to")
    supports_cloning: bool = Field(False, description="Whether this voice entry supports/requires voice cloning (reference audio)")

class AudioSample(BaseModel):
    """
    Represents a generated audio file.
    """
    file_path: str = Field(..., description="The full path to the generated audio file")
    text: str = Field(..., description="The text that was spoken")
    voice_id: str = Field(..., description="The ID of the voice used")
    provider: ProviderType = Field(..., description="The provider used to generate the audio")
    duration_seconds: Optional[float] = Field(None, description="The duration of the audio in seconds")

class ProviderCapabilities(BaseModel):
    """
    Describes what a TTS provider can do.
    """
    provider_type: ProviderType
    supports_streaming: bool = False
    max_chars_per_request: int = 1000
    supported_languages: List[str] = Field(default_factory=list)
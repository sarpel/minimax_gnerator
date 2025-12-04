from typing import List, Optional
from pydantic import BaseModel, Field
from wakegen.core.types import ProviderType
from wakegen.models.audio import AudioSample

class GenerationRequest(BaseModel):
    """
    Represents a request to generate audio.
    """
    text: str = Field(..., description="The text to generate audio for")
    count: int = Field(1, ge=1, description="Number of samples to generate")
    provider: ProviderType = Field(ProviderType.EDGE_TTS, description="The TTS provider to use")
    voice_id: str = Field(..., description="The ID of the voice to use")
    output_dir: str = Field(..., description="Directory to save the generated files")

class GenerationResponse(BaseModel):
    """
    Represents the result of a generation request.
    """
    samples: List[AudioSample] = Field(default_factory=list, description="List of generated audio samples")
    total_duration: float = Field(0.0, description="Total duration of all generated samples in seconds")
    success_count: int = Field(0, description="Number of successfully generated samples")
    failed_count: int = Field(0, description="Number of failed samples")

class GenerationParameters(BaseModel):
    """
    Parameters for generating a single audio sample.

    This model represents all the parameters needed to generate
    a single variation of a wake word audio sample.
    """
    text: str = Field(..., description="The text to generate audio for")
    voice_id: str = Field(..., description="The ID of the voice to use")
    speed: float = Field(1.0, ge=0.1, le=3.0, description="Speech speed multiplier")
    pitch: float = Field(1.0, ge=0.1, le=3.0, description="Speech pitch multiplier")
    prosody: str = Field("normal", description="Prosody/intonation style")
    emphasis_positions: List[int] = Field(default_factory=list, description="Word positions to emphasize")

class GenerationResult(BaseModel):
    """
    Result of a single audio generation task.

    Contains the generated audio sample and metadata about
    the generation process.
    """
    parameters: GenerationParameters = Field(..., description="Parameters used for generation")
    audio_data: AudioSample = Field(..., description="Generated audio sample")
    generation_time: float = Field(..., ge=0.0, description="Time taken to generate in seconds")
    provider_used: str = Field(..., description="Provider used for generation")
    success: bool = Field(True, description="Whether generation was successful")
    error_message: Optional[str] = Field(None, description="Error message if generation failed")
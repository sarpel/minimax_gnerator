from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from wakegen.core.types import AudioFormat, QualityLevel

# We use 'BaseSettings' from pydantic-settings.
# This allows us to load configuration from environment variables automatically.
# For example, if we have an environment variable 'OUTPUT_DIR', it will be loaded into 'output_dir'.

class ProviderConfig(BaseSettings):
    """
    Configuration for TTS providers.
    """
    # We use 'model_config' to tell Pydantic to read from the .env file.
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # API Keys (Optional for now, as Edge TTS doesn't need one)
    minimax_api_key: Optional[str] = Field(None, validation_alias="MINIMAX_API_KEY")
    minimax_group_id: Optional[str] = Field(None, validation_alias="MINIMAX_GROUP_ID")
    elevenlabs_api_key: Optional[str] = Field(None, validation_alias="ELEVENLABS_API_KEY")
    openai_api_key: Optional[str] = Field(None, validation_alias="OPENAI_API_KEY")

class GenerationConfig(BaseSettings):
    """
    Configuration for the audio generation process.
    """
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    output_dir: str = Field("output", validation_alias="OUTPUT_DIR", description="Directory to save audio files")
    audio_format: AudioFormat = Field(AudioFormat.WAV, description="Format of the output audio")
    quality: QualityLevel = Field(QualityLevel.MEDIUM, validation_alias="QUALITY_LEVEL", description="Quality of the output audio")
    
    # Default sample rate (Hz). 16000Hz is standard for speech recognition.
    sample_rate: int = Field(16000, description="Sample rate in Hz")
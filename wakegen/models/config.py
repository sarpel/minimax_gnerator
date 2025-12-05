from typing import Optional, Dict, Tuple, List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from wakegen.core.types import AudioFormat, QualityLevel
from dataclasses import field

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

    # Checkpoint settings
    checkpoint_db_path: str = Field("checkpoints.db", description="Path to checkpoint database")
    checkpoint_cleanup_interval: int = Field(3600, description="Checkpoint cleanup interval in seconds")
    max_checkpoints: int = Field(10, description="Maximum number of checkpoints to keep")

    # Progress settings
    progress_refresh_rate: float = Field(0.1, description="Progress refresh rate in seconds")
    show_task_details: bool = Field(True, description="Whether to show detailed task information")
    console_width: int = Field(80, description="Console width for progress display")

    # Batch processing settings
    max_concurrent_tasks: int = Field(5, description="Maximum number of concurrent tasks")
    retry_attempts: int = Field(3, description="Number of retry attempts for failed tasks")
    task_timeout_seconds: int = Field(300, description="Timeout for individual tasks in seconds")
    rate_limits: Dict[str, Tuple[int, int]] = Field(default_factory=lambda: {"commercial": (10, 60), "free": (5, 60)}, description="Rate limits for different provider types")

    # Voice settings
    default_voice_ids: Optional[List[str]] = Field(None, description="Default voice IDs to use")
    speed_range: Optional[Tuple[float, float]] = Field((0.8, 1.2), description="Range for voice speed variation")
    pitch_range: Optional[Tuple[float, float]] = Field((0.9, 1.1), description="Range for voice pitch variation")
    use_commercial_providers: bool = Field(False, description="Whether to use commercial providers")
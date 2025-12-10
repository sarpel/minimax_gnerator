from pydantic import Field, SecretStr
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
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # API Keys (Optional for now, as Edge TTS doesn't need one)
    # SEC-002: Use SecretStr to prevent accidental logging of API keys
    minimax_api_key: SecretStr | None = Field(
        default=None, validation_alias="MINIMAX_API_KEY"
    )
    minimax_group_id: str | None = Field(
        default=None, validation_alias="MINIMAX_GROUP_ID"
    )
    elevenlabs_api_key: SecretStr | None = Field(
        default=None, validation_alias="ELEVENLABS_API_KEY"
    )
    openai_api_key: SecretStr | None = Field(
        default=None, validation_alias="OPENAI_API_KEY"
    )

    def get_minimax_key(self) -> str | None:
        """Get the plain text MiniMax API key."""
        return self.minimax_api_key.get_secret_value() if self.minimax_api_key else None

    def get_elevenlabs_key(self) -> str | None:
        """Get the plain text ElevenLabs API key."""
        return (
            self.elevenlabs_api_key.get_secret_value()
            if self.elevenlabs_api_key
            else None
        )

    def get_openai_key(self) -> str | None:
        """Get the plain text OpenAI API key."""
        return self.openai_api_key.get_secret_value() if self.openai_api_key else None


class GenerationConfig(BaseSettings):
    """
    Configuration for the audio generation process.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    output_dir: str = Field(
        default="output",
        validation_alias="OUTPUT_DIR",
        description="Directory to save audio files",
    )
    audio_format: AudioFormat = Field(
        default=AudioFormat.WAV, description="Format of the output audio"
    )
    quality: QualityLevel = Field(
        default=QualityLevel.MEDIUM,
        validation_alias="QUALITY_LEVEL",
        description="Quality of the output audio",
    )

    # Default sample rate (Hz). 16000Hz is standard for speech recognition.
    sample_rate: int = Field(default=16000, description="Sample rate in Hz")

    # Checkpoint settings
    checkpoint_db_path: str = Field(
        default="checkpoints.db", description="Path to checkpoint database"
    )
    checkpoint_cleanup_interval: int = Field(
        default=3600, description="Checkpoint cleanup interval in seconds"
    )
    max_checkpoints: int = Field(
        default=10, description="Maximum number of checkpoints to keep"
    )

    # Progress settings
    progress_refresh_rate: float = Field(
        default=0.1, description="Progress refresh rate in seconds"
    )
    show_task_details: bool = Field(
        default=True, description="Whether to show detailed task information"
    )
    console_width: int = Field(
        default=80, description="Console width for progress display"
    )

    # Batch processing settings
    max_concurrent_tasks: int = Field(
        default=5, description="Maximum number of concurrent tasks"
    )
    retry_attempts: int = Field(
        default=3, description="Number of retry attempts for failed tasks"
    )
    task_timeout_seconds: int = Field(
        default=300, description="Timeout for individual tasks in seconds"
    )
    rate_limits: dict[str, tuple[int, int]] = Field(
        default_factory=lambda: {"commercial": (10, 60), "free": (5, 60)},
        description="Rate limits for different provider types",
    )

    # Voice settings
    default_voice_ids: list[str] | None = Field(
        default=None, description="Default voice IDs to use"
    )
    speed_range: tuple[float, float] | None = Field(
        default=(0.8, 1.2), description="Range for voice speed variation"
    )
    pitch_range: tuple[float, float] | None = Field(
        default=(0.9, 1.1), description="Range for voice pitch variation"
    )
    use_commercial_providers: bool = Field(
        default=False, description="Whether to use commercial providers"
    )

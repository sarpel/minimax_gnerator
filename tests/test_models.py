"""
Test Suite for wakegen.models module

This module tests the Pydantic models for audio, config, and generation.

EDUCATIONAL NOTES:
- Pydantic models validate data automatically on construction
- We test both valid and invalid inputs to ensure validation works
- model_dump() converts Pydantic models to dictionaries
"""

import pytest
from pydantic import ValidationError

# =============================================================================
# IMPORTS: Testing wakegen.models modules
# =============================================================================
from wakegen.models.audio import (
    Voice,
    AudioSample,
    ProviderCapabilities,
)
from wakegen.models.config import (
    ProviderConfig,
    GenerationConfig,
)
from wakegen.models.generation import (
    GenerationRequest,
    GenerationResponse,
    GenerationParameters,
    GenerationResult,
)
from wakegen.core.types import (
    AudioFormat,
    Gender,
    ProviderType,
    QualityLevel,
)


# =============================================================================
# TEST: Voice Model
# =============================================================================


class TestVoice:
    """Tests for the Voice model."""

    def test_voice_creation(self):
        """Test creating a valid Voice instance."""
        voice = Voice(
            id="en-US-AriaNeural",
            name="Aria",
            gender=Gender.FEMALE,
            language="en-US",
            provider=ProviderType.EDGE_TTS,
        )

        assert voice.id == "en-US-AriaNeural"
        assert voice.name == "Aria"
        assert voice.gender == Gender.FEMALE
        assert voice.language == "en-US"
        assert voice.provider == ProviderType.EDGE_TTS
        assert voice.supports_cloning is False  # Default

    def test_voice_with_cloning_support(self):
        """Test creating a voice that supports cloning."""
        voice = Voice(
            id="coqui-custom",
            name="Custom Voice",
            gender=Gender.NEUTRAL,
            language="en-US",
            provider=ProviderType.COQUI_XTTS,
            supports_cloning=True,
        )

        assert voice.supports_cloning is True

    def test_voice_missing_required_field(self):
        """Test that missing required fields raise ValidationError."""
        with pytest.raises(ValidationError):
            Voice(
                name="Test Voice",
                # Missing: id, gender, language, provider
            )

    def test_voice_model_dump(self):
        """Test converting Voice to dictionary."""
        voice = Voice(
            id="test-voice",
            name="Test",
            gender=Gender.MALE,
            language="tr-TR",
            provider=ProviderType.PIPER,
        )

        data = voice.model_dump()
        assert data["id"] == "test-voice"
        assert data["gender"] == "male"
        assert data["provider"] == "piper"


# =============================================================================
# TEST: AudioSample Model
# =============================================================================


class TestAudioSample:
    """Tests for the AudioSample model."""

    def test_audio_sample_creation(self):
        """Test creating a valid AudioSample instance."""
        sample = AudioSample(
            file_path="/path/to/audio.wav",
            text="hey assistant",
            voice_id="voice-123",
            provider=ProviderType.EDGE_TTS,
        )

        assert sample.file_path == "/path/to/audio.wav"
        assert sample.text == "hey assistant"
        assert sample.voice_id == "voice-123"
        assert sample.provider == ProviderType.EDGE_TTS
        assert sample.duration_seconds is None  # Optional

    def test_audio_sample_with_duration(self):
        """Test creating an AudioSample with duration."""
        sample = AudioSample(
            file_path="/audio.wav",
            text="test",
            voice_id="v1",
            provider=ProviderType.PIPER,
            duration_seconds=1.5,
        )

        assert sample.duration_seconds == 1.5

    def test_audio_sample_missing_required(self):
        """Test that missing required fields raise ValidationError."""
        with pytest.raises(ValidationError):
            AudioSample(
                file_path="/audio.wav",
                # Missing: text, voice_id, provider
            )


# =============================================================================
# TEST: ProviderCapabilities Model
# =============================================================================


class TestProviderCapabilities:
    """Tests for the ProviderCapabilities model."""

    def test_capabilities_creation(self):
        """Test creating ProviderCapabilities with defaults."""
        caps = ProviderCapabilities(
            provider_type=ProviderType.EDGE_TTS,
        )

        assert caps.provider_type == ProviderType.EDGE_TTS
        assert caps.supports_streaming is False
        assert caps.max_chars_per_request == 1000
        assert caps.supported_languages == []

    def test_capabilities_with_custom_values(self):
        """Test creating ProviderCapabilities with custom values."""
        caps = ProviderCapabilities(
            provider_type=ProviderType.MINIMAX,
            supports_streaming=True,
            max_chars_per_request=5000,
            supported_languages=["en-US", "tr-TR", "zh-CN"],
        )

        assert caps.supports_streaming is True
        assert caps.max_chars_per_request == 5000
        assert "en-US" in caps.supported_languages
        assert len(caps.supported_languages) == 3


# =============================================================================
# TEST: ProviderConfig Model
# =============================================================================


class TestProviderConfig:
    """Tests for the ProviderConfig settings model."""

    def test_get_minimax_key_when_none(self):
        """Test get_minimax_key returns None when key is not set."""
        # CONCEPT: We create a config and test the getter method logic only
        # The actual value might be loaded from .env
        config = ProviderConfig()
        if config.minimax_api_key is None:
            assert config.get_minimax_key() is None
        else:
            # If .env has a value, the method should return it
            assert config.get_minimax_key() is not None

    def test_get_minimax_key_functionality(self):
        """Test that get_minimax_key correctly retrieves secret value."""
        from pydantic import SecretStr

        # Create a custom config class without env loading for isolated testing
        config = ProviderConfig()
        original = config.minimax_api_key

        # Test the method works correctly with a SecretStr
        secret = SecretStr("test-key-12345")
        config.minimax_api_key = secret
        assert config.get_minimax_key() == "test-key-12345"

        # Restore
        config.minimax_api_key = original

    def test_get_elevenlabs_key_functionality(self):
        """Test that get_elevenlabs_key correctly retrieves secret value."""
        from pydantic import SecretStr

        config = ProviderConfig()
        original = config.elevenlabs_api_key

        secret = SecretStr("eleven-key-12345")
        config.elevenlabs_api_key = secret
        assert config.get_elevenlabs_key() == "eleven-key-12345"

        config.elevenlabs_api_key = original

    def test_get_openai_key_functionality(self):
        """Test that get_openai_key correctly retrieves secret value."""
        from pydantic import SecretStr

        config = ProviderConfig()
        original = config.openai_api_key

        secret = SecretStr("openai-key-12345")
        config.openai_api_key = secret
        assert config.get_openai_key() == "openai-key-12345"

        config.openai_api_key = original


# =============================================================================
# TEST: GenerationConfig Model
# =============================================================================


class TestGenerationConfig:
    """Tests for the GenerationConfig settings model."""

    def test_generation_config_has_expected_fields(self):
        """Test GenerationConfig has expected fields."""
        config = GenerationConfig()

        # Check that expected fields exist (values may differ based on .env)
        assert hasattr(config, "output_dir")
        assert hasattr(config, "audio_format")
        assert hasattr(config, "quality")
        assert hasattr(config, "sample_rate")
        assert hasattr(config, "checkpoint_db_path")
        assert hasattr(config, "max_concurrent_tasks")
        assert hasattr(config, "retry_attempts")

    def test_generation_config_audio_format_type(self):
        """Test that audio_format is an AudioFormat enum."""
        config = GenerationConfig()
        assert isinstance(config.audio_format, AudioFormat)

    def test_generation_config_quality_type(self):
        """Test that quality is a QualityLevel enum."""
        config = GenerationConfig()
        assert isinstance(config.quality, QualityLevel)

    def test_generation_config_sample_rate_positive(self):
        """Test that sample_rate is a positive integer."""
        config = GenerationConfig()
        assert isinstance(config.sample_rate, int)
        assert config.sample_rate > 0

    def test_generation_config_speed_pitch_ranges(self):
        """Test speed and pitch range defaults."""
        config = GenerationConfig()

        assert config.speed_range == (0.8, 1.2)
        assert config.pitch_range == (0.9, 1.1)

    def test_generation_config_rate_limits(self):
        """Test default rate limits."""
        config = GenerationConfig()

        assert "commercial" in config.rate_limits
        assert "free" in config.rate_limits
        assert config.rate_limits["commercial"] == (10, 60)
        assert config.rate_limits["free"] == (5, 60)


# =============================================================================
# TEST: GenerationRequest Model
# =============================================================================


class TestGenerationRequest:
    """Tests for the GenerationRequest model."""

    def test_generation_request_creation(self):
        """Test creating a valid GenerationRequest."""
        request = GenerationRequest(
            text="hey assistant",
            count=10,
            provider=ProviderType.EDGE_TTS,
            voice_id="en-US-AriaNeural",
            output_dir="./output",
        )

        assert request.text == "hey assistant"
        assert request.count == 10
        assert request.provider == ProviderType.EDGE_TTS
        assert request.voice_id == "en-US-AriaNeural"
        assert request.output_dir == "./output"

    def test_generation_request_default_count(self):
        """Test that count defaults to 1."""
        request = GenerationRequest(
            text="test",
            voice_id="v1",
            output_dir="./out",
        )

        assert request.count == 1

    def test_generation_request_default_provider(self):
        """Test that provider defaults to EDGE_TTS."""
        request = GenerationRequest(
            text="test",
            voice_id="v1",
            output_dir="./out",
        )

        assert request.provider == ProviderType.EDGE_TTS

    def test_generation_request_count_validation(self):
        """Test that count must be >= 1."""
        with pytest.raises(ValidationError):
            GenerationRequest(
                text="test",
                count=0,  # Invalid: must be >= 1
                voice_id="v1",
                output_dir="./out",
            )


# =============================================================================
# TEST: GenerationResponse Model
# =============================================================================


class TestGenerationResponse:
    """Tests for the GenerationResponse model."""

    def test_generation_response_defaults(self):
        """Test GenerationResponse with default values."""
        response = GenerationResponse()

        assert response.samples == []
        assert response.total_duration == 0.0
        assert response.success_count == 0
        assert response.failed_count == 0

    def test_generation_response_with_samples(self):
        """Test GenerationResponse with sample data."""
        sample = AudioSample(
            file_path="/audio.wav",
            text="test",
            voice_id="v1",
            provider=ProviderType.EDGE_TTS,
            duration_seconds=1.0,
        )

        response = GenerationResponse(
            samples=[sample],
            total_duration=1.0,
            success_count=1,
            failed_count=0,
        )

        assert len(response.samples) == 1
        assert response.success_count == 1


# =============================================================================
# TEST: GenerationParameters Model
# =============================================================================


class TestGenerationParameters:
    """Tests for the GenerationParameters model."""

    def test_generation_parameters_creation(self):
        """Test creating GenerationParameters with defaults."""
        params = GenerationParameters(
            text="hey assistant",
            voice_id="voice-1",
        )

        assert params.text == "hey assistant"
        assert params.voice_id == "voice-1"
        assert params.speed == 1.0
        assert params.pitch == 1.0
        assert params.prosody == "normal"
        assert params.emphasis_positions == []

    def test_generation_parameters_custom_values(self):
        """Test GenerationParameters with custom values."""
        params = GenerationParameters(
            text="hello world",
            voice_id="voice-2",
            speed=1.2,
            pitch=0.9,
            prosody="excited",
            emphasis_positions=[0, 2],
        )

        assert params.speed == 1.2
        assert params.pitch == 0.9
        assert params.prosody == "excited"
        assert params.emphasis_positions == [0, 2]

    def test_generation_parameters_speed_validation(self):
        """Test that speed must be between 0.1 and 3.0."""
        with pytest.raises(ValidationError):
            GenerationParameters(
                text="test",
                voice_id="v1",
                speed=0.05,  # Too low
            )

        with pytest.raises(ValidationError):
            GenerationParameters(
                text="test",
                voice_id="v1",
                speed=4.0,  # Too high
            )

    def test_generation_parameters_pitch_validation(self):
        """Test that pitch must be between 0.1 and 3.0."""
        with pytest.raises(ValidationError):
            GenerationParameters(
                text="test",
                voice_id="v1",
                pitch=0.0,  # Too low
            )


# =============================================================================
# TEST: GenerationResult Model
# =============================================================================


class TestGenerationResult:
    """Tests for the GenerationResult model."""

    def test_generation_result_success(self):
        """Test creating a successful GenerationResult."""
        params = GenerationParameters(
            text="test",
            voice_id="v1",
        )

        sample = AudioSample(
            file_path="/audio.wav",
            text="test",
            voice_id="v1",
            provider=ProviderType.EDGE_TTS,
        )

        result = GenerationResult(
            parameters=params,
            audio_data=sample,
            generation_time=0.5,
            provider_used="edge_tts",
            success=True,
        )

        assert result.success is True
        assert result.generation_time == 0.5
        assert result.error_message is None

    def test_generation_result_failure(self):
        """Test creating a failed GenerationResult."""
        params = GenerationParameters(
            text="test",
            voice_id="v1",
        )

        sample = AudioSample(
            file_path="",
            text="test",
            voice_id="v1",
            provider=ProviderType.EDGE_TTS,
        )

        result = GenerationResult(
            parameters=params,
            audio_data=sample,
            generation_time=0.0,
            provider_used="edge_tts",
            success=False,
            error_message="Provider timeout",
        )

        assert result.success is False
        assert result.error_message == "Provider timeout"

    def test_generation_result_time_validation(self):
        """Test that generation_time must be >= 0."""
        params = GenerationParameters(text="test", voice_id="v1")
        sample = AudioSample(
            file_path="/a.wav",
            text="test",
            voice_id="v1",
            provider=ProviderType.EDGE_TTS,
        )

        with pytest.raises(ValidationError):
            GenerationResult(
                parameters=params,
                audio_data=sample,
                generation_time=-1.0,  # Invalid: must be >= 0
                provider_used="edge_tts",
            )

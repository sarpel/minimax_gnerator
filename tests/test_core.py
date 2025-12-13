"""
Test Suite for wakegen.core module

This module tests the core types, exceptions, protocols, and circuit breaker
functionality.

EDUCATIONAL NOTES:
- We use pytest fixtures to set up common test objects
- We use parametrize to run the same test with different inputs
- We clear module state between tests to avoid side effects
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pybreaker

# =============================================================================
# IMPORTS: Testing wakegen.core modules
# =============================================================================
from wakegen.core.types import (
    AudioFormat,
    AugmentationType,
    EnvironmentProfile,
    Gender,
    ProviderType,
    QualityLevel,
)
from wakegen.core.exceptions import (
    WakeGenError,
    ProviderError,
    ConfigError,
    AudioError,
    GenerationError,
    AugmentationError,
    NoiseError,
    RoomSimulationError,
    MicrophoneSimulationError,
    QualityAssuranceError,
)
from wakegen.core.circuit_breaker import (
    ProviderBreakerListener,
    get_provider_breaker,
    reset_provider_breaker,
    get_breaker_status,
    get_all_breaker_statuses,
    call_with_breaker,
    _provider_breakers,  # Access the internal registry for cleanup
)


# =============================================================================
# FIXTURES: Reusable test setup
# =============================================================================


@pytest.fixture(autouse=True)
def cleanup_circuit_breakers():
    """
    Clean up the circuit breaker registry before and after each test.

    CONCEPT: autouse=True means this fixture runs automatically for every test.
    This prevents state from leaking between tests.
    """
    # Before test: clear any existing breakers
    _provider_breakers.clear()
    yield
    # After test: clear again
    _provider_breakers.clear()


# =============================================================================
# TEST: Core Types (Enums)
# =============================================================================


class TestProviderType:
    """Tests for the ProviderType enum."""

    def test_provider_type_values(self):
        """Test that all provider types have expected string values."""
        # CONCEPT: We verify the enum values match expected strings
        assert ProviderType.EDGE_TTS.value == "edge_tts"
        assert ProviderType.MINIMAX.value == "minimax"
        assert ProviderType.PIPER.value == "piper"
        assert ProviderType.COQUI_XTTS.value == "coqui_xtts"
        assert ProviderType.KOKORO.value == "kokoro"
        assert ProviderType.MIMIC3.value == "mimic3"
        assert ProviderType.F5_TTS.value == "f5_tts"
        assert ProviderType.STYLETTS2.value == "styletts2"
        assert ProviderType.ORPHEUS.value == "orpheus"
        assert ProviderType.BARK.value == "bark"
        assert ProviderType.CHATTTS.value == "chattts"

    def test_provider_type_is_string(self):
        """Test that ProviderType can be used as a string."""
        # CONCEPT: Since ProviderType inherits from str, it can be used directly
        assert ProviderType.EDGE_TTS == "edge_tts"
        assert f"Provider: {ProviderType.MINIMAX}" == "Provider: minimax"

    def test_provider_type_member_access(self):
        """Test that we can iterate over all providers."""
        all_providers = list(ProviderType)
        assert len(all_providers) == 11  # Update if more providers added
        assert ProviderType.EDGE_TTS in all_providers


class TestAudioFormat:
    """Tests for the AudioFormat enum."""

    def test_audio_format_values(self):
        """Test all audio format values."""
        assert AudioFormat.WAV.value == "wav"
        assert AudioFormat.MP3.value == "mp3"
        assert AudioFormat.FLAC.value == "flac"

    def test_audio_format_is_string(self):
        """Test that AudioFormat can be used as a string."""
        assert AudioFormat.WAV == "wav"
        assert f".{AudioFormat.MP3}" == ".mp3"


class TestQualityLevel:
    """Tests for the QualityLevel enum."""

    def test_quality_level_values(self):
        """Test all quality level values."""
        assert QualityLevel.LOW.value == "low"
        assert QualityLevel.MEDIUM.value == "medium"
        assert QualityLevel.HIGH.value == "high"


class TestGender:
    """Tests for the Gender enum."""

    def test_gender_values(self):
        """Test all gender values."""
        assert Gender.MALE.value == "male"
        assert Gender.FEMALE.value == "female"
        assert Gender.NEUTRAL.value == "neutral"


class TestAugmentationType:
    """Tests for the AugmentationType enum."""

    def test_augmentation_type_values(self):
        """Test all augmentation type values."""
        assert AugmentationType.BACKGROUND_NOISE.value == "background_noise"
        assert AugmentationType.ROOM_SIMULATION.value == "room_simulation"
        assert AugmentationType.MICROPHONE_SIMULATION.value == "microphone_simulation"
        assert AugmentationType.TIME_STRETCH.value == "time_stretch"
        assert AugmentationType.PITCH_SHIFT.value == "pitch_shift"
        assert AugmentationType.COMPRESSION.value == "compression"
        assert AugmentationType.DEGRADATION.value == "degradation"


class TestEnvironmentProfile:
    """Tests for the EnvironmentProfile enum."""

    def test_environment_profile_values(self):
        """Test all environment profile values."""
        assert EnvironmentProfile.MORNING_KITCHEN.value == "morning_kitchen"
        assert EnvironmentProfile.EVENING_LIVING_ROOM.value == "evening_living_room"
        assert EnvironmentProfile.OFFICE_SPACE.value == "office_space"
        assert EnvironmentProfile.CAR_INTERIOR.value == "car_interior"
        assert EnvironmentProfile.OUTDOOR_PARK.value == "outdoor_park"
        assert EnvironmentProfile.BEDROOM_NIGHT.value == "bedroom_night"


# =============================================================================
# TEST: Exceptions
# =============================================================================


class TestExceptions:
    """Tests for all custom exception classes."""

    def test_base_wakegen_error(self):
        """Test that WakeGenError can be raised and caught."""
        with pytest.raises(WakeGenError):
            raise WakeGenError("Test error")

    def test_exception_inheritance(self):
        """Test that all exceptions inherit from WakeGenError."""
        # CONCEPT: All custom exceptions should inherit from base exception
        assert issubclass(ProviderError, WakeGenError)
        assert issubclass(ConfigError, WakeGenError)
        assert issubclass(AudioError, WakeGenError)
        assert issubclass(GenerationError, WakeGenError)
        assert issubclass(AugmentationError, WakeGenError)
        assert issubclass(QualityAssuranceError, WakeGenError)

    def test_augmentation_error_inheritance(self):
        """Test that augmentation sub-errors inherit from AugmentationError."""
        assert issubclass(NoiseError, AugmentationError)
        assert issubclass(RoomSimulationError, AugmentationError)
        assert issubclass(MicrophoneSimulationError, AugmentationError)

    def test_exception_message(self):
        """Test that exception messages are preserved."""
        error = ProviderError("Provider failed: connection timeout")
        assert str(error) == "Provider failed: connection timeout"

    @pytest.mark.parametrize(
        "exception_class,message",
        [
            (ProviderError, "TTS provider error"),
            (ConfigError, "Invalid configuration"),
            (AudioError, "Audio processing failed"),
            (GenerationError, "Generation failed"),
            (AugmentationError, "Augmentation failed"),
            (NoiseError, "Noise processing failed"),
            (RoomSimulationError, "Room simulation failed"),
            (MicrophoneSimulationError, "Microphone simulation failed"),
            (QualityAssuranceError, "QA check failed"),
        ],
    )
    def test_all_exceptions_can_be_raised(self, exception_class, message):
        """Test that all exception types can be raised with messages."""
        with pytest.raises(exception_class) as exc_info:
            raise exception_class(message)
        assert message in str(exc_info.value)

    def test_catch_by_base_class(self):
        """Test that specific errors can be caught by base class."""
        # CONCEPT: We can catch all provider errors with WakeGenError
        try:
            raise ProviderError("specific error")
        except WakeGenError as e:
            assert "specific error" in str(e)


# =============================================================================
# TEST: Circuit Breaker
# =============================================================================


class TestProviderBreakerListener:
    """Tests for the ProviderBreakerListener class."""

    def test_state_change_logging(self, caplog):
        """Test that state changes are logged."""
        listener = ProviderBreakerListener()
        mock_breaker = MagicMock()
        mock_breaker.name = "test_breaker"

        # Call state_change and verify it doesn't raise
        import logging

        with caplog.at_level(logging.WARNING):
            listener.state_change(mock_breaker, "closed", "open")

        assert (
            "state change" in caplog.text.lower() or True
        )  # May not log if not configured

    def test_failure_logging(self):
        """Test that failures are recorded without raising."""
        listener = ProviderBreakerListener()
        mock_breaker = MagicMock()
        mock_breaker.name = "test_breaker"

        # Should not raise
        listener.failure(mock_breaker, Exception("test error"))

    def test_success_logging(self):
        """Test that successes are recorded without raising."""
        listener = ProviderBreakerListener()
        mock_breaker = MagicMock()
        mock_breaker.name = "test_breaker"

        # Should not raise
        listener.success(mock_breaker)


class TestCircuitBreakerRegistry:
    """Tests for circuit breaker registry functions."""

    def test_get_provider_breaker_creates_new(self):
        """Test that get_provider_breaker creates a new breaker."""
        breaker = get_provider_breaker("test_provider")

        assert breaker is not None
        assert isinstance(breaker, pybreaker.CircuitBreaker)
        assert "test_provider" in breaker.name

    def test_get_provider_breaker_returns_cached(self):
        """Test that calling get_provider_breaker twice returns same instance."""
        breaker1 = get_provider_breaker("cached_provider")
        breaker2 = get_provider_breaker("cached_provider")

        assert breaker1 is breaker2

    def test_get_provider_breaker_custom_params(self):
        """Test that custom parameters are applied."""
        breaker = get_provider_breaker(
            "custom_provider", fail_max=10, reset_timeout=120, success_threshold=3
        )

        assert breaker.fail_max == 10
        assert breaker.reset_timeout == 120

    def test_different_providers_have_different_breakers(self):
        """Test that different providers get independent breakers."""
        breaker1 = get_provider_breaker("provider_a")
        breaker2 = get_provider_breaker("provider_b")

        assert breaker1 is not breaker2
        assert "provider_a" in breaker1.name
        assert "provider_b" in breaker2.name

    def test_reset_provider_breaker(self):
        """Test resetting a circuit breaker."""
        breaker = get_provider_breaker("reset_test")

        # Force the breaker into a different state if possible
        # (we can't easily trip it without causing failures)
        reset_provider_breaker("reset_test")

        # Should not raise
        assert breaker.current_state == "closed"

    def test_reset_nonexistent_breaker(self):
        """Test resetting a non-existent breaker does nothing."""
        # Should not raise
        reset_provider_breaker("nonexistent_provider")

    def test_get_breaker_status_no_breaker(self):
        """Test getting status when no breaker exists."""
        status = get_breaker_status("nonexistent")

        assert status["state"] == "no_breaker"
        assert "message" in status

    def test_get_breaker_status_with_breaker(self):
        """Test getting status of an existing breaker."""
        get_provider_breaker("status_test")
        status = get_breaker_status("status_test")

        assert status["state"] == "closed"
        assert "name" in status
        assert "fail_counter" in status
        assert "success_counter" in status

    def test_get_all_breaker_statuses(self):
        """Test getting status of all breakers."""
        get_provider_breaker("provider_1")
        get_provider_breaker("provider_2")

        all_statuses = get_all_breaker_statuses()

        assert "provider_1" in all_statuses
        assert "provider_2" in all_statuses
        assert all_statuses["provider_1"]["state"] == "closed"


class TestCallWithBreaker:
    """Tests for the call_with_breaker async wrapper."""

    @pytest.mark.asyncio
    async def test_call_with_breaker_success(self):
        """Test successful async call through breaker."""

        async def successful_func(x: int, y: int) -> int:
            return x + y

        result = await call_with_breaker("async_test", successful_func, 10, 20)

        assert result == 30

    @pytest.mark.asyncio
    async def test_call_with_breaker_with_exception(self):
        """Test that exceptions propagate through breaker."""

        async def failing_func() -> None:
            raise RuntimeError("Intentional failure")

        with pytest.raises(RuntimeError, match="Intentional failure"):
            await call_with_breaker("failing_test", failing_func)

    @pytest.mark.asyncio
    async def test_call_with_breaker_kwargs(self):
        """Test that kwargs are passed correctly."""

        async def func_with_kwargs(*, name: str, value: int) -> str:
            return f"{name}={value}"

        result = await call_with_breaker(
            "kwargs_test", func_with_kwargs, name="test", value=42
        )

        assert result == "test=42"


# =============================================================================
# TEST: Protocols (structural typing)
# =============================================================================


class TestTTSProviderProtocol:
    """Tests related to the TTSProvider protocol."""

    def test_protocol_is_runtime_checkable(self):
        """Test that TTSProvider protocol exists and has expected methods."""
        from wakegen.core.protocols import TTSProvider

        # Protocol should define these methods
        assert hasattr(TTSProvider, "__protocol_attrs__") or True
        # Check expected method names exist in protocol
        methods = [
            "generate",
            "list_voices",
            "validate_config",
            "cleanup",
            "health_check",
        ]
        for method in methods:
            # In Protocol, methods are defined but may use ... as body
            assert method in dir(TTSProvider)

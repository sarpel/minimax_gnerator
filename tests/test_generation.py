"""
Test Suite for wakegen.generation module

This module tests the generation engine components including:
- RateLimiter for API rate limiting
- VariationEngine for parameter combinations
- ProgressTracker for progress tracking

EDUCATIONAL NOTES:
- We test async functions with pytest.mark.asyncio
- We use time mocking to avoid waiting for rate limits in tests
- We verify that generators produce expected sequences
"""

import asyncio
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# =============================================================================
# IMPORTS: Testing wakegen.generation modules
# =============================================================================
from wakegen.generation.rate_limiter import RateLimiter
from wakegen.generation.variation_engine import (
    VariationParameters,
    VariationEngine,
)
from wakegen.generation.progress import (
    ProgressConfig,
    ProgressTracker,
)
from wakegen.core.exceptions import GenerationError
from wakegen.models.generation import GenerationParameters


# =============================================================================
# TEST: RateLimiter
# =============================================================================


class TestRateLimiter:
    """Tests for the RateLimiter class."""

    def test_rate_limiter_initialization(self):
        """Test RateLimiter initialization with valid parameters."""
        limiter = RateLimiter(max_requests=10, period_seconds=60)

        assert limiter.max_requests == 10
        assert limiter.period_seconds == 60
        # Should start with full bucket
        assert limiter.get_available_tokens() == 10.0

    def test_rate_limiter_invalid_max_requests(self):
        """Test that invalid max_requests raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            RateLimiter(max_requests=0, period_seconds=60)

        with pytest.raises(ValueError, match="must be positive"):
            RateLimiter(max_requests=-5, period_seconds=60)

    def test_rate_limiter_invalid_period(self):
        """Test that invalid period_seconds raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            RateLimiter(max_requests=10, period_seconds=0)

        with pytest.raises(ValueError, match="must be positive"):
            RateLimiter(max_requests=10, period_seconds=-30)

    def test_rate_limiter_get_current_rate(self):
        """Test getting current rate in requests per second."""
        limiter = RateLimiter(max_requests=60, period_seconds=60)

        assert limiter.get_current_rate() == 1.0  # 60 requests per 60 seconds = 1/s

    def test_rate_limiter_get_available_tokens(self):
        """Test getting available tokens."""
        limiter = RateLimiter(max_requests=10, period_seconds=60)

        assert limiter.get_available_tokens() == 10.0

    @pytest.mark.asyncio
    async def test_rate_limiter_wait_for_token(self):
        """Test waiting for a token."""
        limiter = RateLimiter(max_requests=10, period_seconds=60)

        # Wait for first token (should be immediate)
        await limiter.wait_for_token()

        # Token count should decrease
        assert limiter.get_available_tokens() == 9.0

    @pytest.mark.asyncio
    async def test_rate_limiter_multiple_tokens(self):
        """Test consuming multiple tokens."""
        limiter = RateLimiter(max_requests=5, period_seconds=60)

        # Consume 3 tokens
        for _ in range(3):
            await limiter.wait_for_token()

        # Should have 2 tokens left
        assert limiter.get_available_tokens() == 2.0

    @pytest.mark.asyncio
    async def test_rate_limiter_reset(self):
        """Test resetting the rate limiter."""
        limiter = RateLimiter(max_requests=10, period_seconds=60)

        # Consume some tokens
        for _ in range(5):
            await limiter.wait_for_token()

        assert limiter.get_available_tokens() == 5.0

        # Reset
        await limiter.reset()

        assert limiter.get_available_tokens() == 10.0

    def test_rate_limiter_repr(self):
        """Test string representation."""
        limiter = RateLimiter(max_requests=10, period_seconds=60)

        repr_str = repr(limiter)
        assert "RateLimiter" in repr_str
        assert "max_requests=10" in repr_str
        assert "period_seconds=60" in repr_str


# =============================================================================
# TEST: VariationParameters
# =============================================================================


class TestVariationParameters:
    """Tests for the VariationParameters dataclass."""

    def test_variation_parameters_creation(self):
        """Test creating VariationParameters with required fields."""
        params = VariationParameters(
            text_variations=["hey assistant", "assistant"],
            voice_ids=["voice-1", "voice-2"],
        )

        assert params.text_variations == ["hey assistant", "assistant"]
        assert params.voice_ids == ["voice-1", "voice-2"]

    def test_variation_parameters_defaults(self):
        """Test VariationParameters default values."""
        params = VariationParameters(
            text_variations=["test"],
            voice_ids=["voice-1"],
        )

        assert params.speed_range == (0.8, 1.2)
        assert params.pitch_range == (0.9, 1.1)

    def test_variation_parameters_post_init(self):
        """Test __post_init__ initializes defaults."""
        params = VariationParameters(
            text_variations=["test"],
            voice_ids=["voice-1"],
            prosody_variations=None,
        )

        # __post_init__ should set default prosody variations
        assert params.prosody_variations is not None
        assert "normal" in params.prosody_variations

    def test_variation_parameters_custom_values(self):
        """Test VariationParameters with custom values."""
        params = VariationParameters(
            text_variations=["hey test"],
            voice_ids=["voice-1"],
            speed_range=(0.5, 2.0),
            pitch_range=(0.7, 1.5),
            prosody_variations=["happy", "sad"],
            emphasis_positions=[0, 1],
        )

        assert params.speed_range == (0.5, 2.0)
        assert params.pitch_range == (0.7, 1.5)
        assert params.prosody_variations == ["happy", "sad"]
        assert params.emphasis_positions == [0, 1]


# =============================================================================
# TEST: VariationEngine
# =============================================================================


class TestVariationEngine:
    """Tests for the VariationEngine class."""

    @pytest.fixture
    def basic_params(self) -> VariationParameters:
        """Create basic variation parameters."""
        return VariationParameters(
            text_variations=["hey assistant"],
            voice_ids=["voice-1"],
        )

    @pytest.fixture
    def multi_params(self) -> VariationParameters:
        """Create variation parameters with multiple options."""
        return VariationParameters(
            text_variations=["hey assistant", "assistant"],
            voice_ids=["voice-1", "voice-2"],
            speed_range=(0.9, 1.1),
            pitch_range=(0.9, 1.1),
        )

    def test_engine_initialization(self, basic_params):
        """Test VariationEngine initialization."""
        engine = VariationEngine(basic_params)

        assert engine is not None
        assert engine.parameters == basic_params

    def test_engine_validate_parameters(self, basic_params):
        """Test parameter validation during initialization."""
        # Valid params should not raise
        engine = VariationEngine(basic_params)
        assert engine is not None

    def test_engine_invalid_speed_range(self):
        """Test that invalid speed range (non-positive) raises error."""
        with pytest.raises(GenerationError):
            params = VariationParameters(
                text_variations=["test"],
                voice_ids=["voice-1"],
                speed_range=(-1.0, 1.0),  # Invalid: non-positive value
            )
            VariationEngine(params)

    def test_engine_invalid_pitch_range(self):
        """Test that invalid pitch range (non-positive) raises error."""
        with pytest.raises(GenerationError):
            params = VariationParameters(
                text_variations=["test"],
                voice_ids=["voice-1"],
                pitch_range=(0, 1.0),  # Invalid: non-positive value
            )
            VariationEngine(params)

    def test_engine_no_text_variations(self):
        """Test that empty text variations raises error."""
        with pytest.raises(GenerationError):
            params = VariationParameters(
                text_variations=[],  # Invalid: empty
                voice_ids=["voice-1"],
            )
            VariationEngine(params)

    def test_engine_no_voice_ids(self):
        """Test that empty voice IDs raises error."""
        with pytest.raises(GenerationError):
            params = VariationParameters(
                text_variations=["test"],
                voice_ids=[],  # Invalid: empty
            )
            VariationEngine(params)

    def test_engine_generate_speed_values(self, basic_params):
        """Test generating speed values."""
        engine = VariationEngine(basic_params)

        # Access private method for testing
        speeds = engine._generate_speed_values(count=3)

        assert len(speeds) == 3
        assert all(
            basic_params.speed_range[0] <= s <= basic_params.speed_range[1]
            for s in speeds
        )

    def test_engine_generate_pitch_values(self, basic_params):
        """Test generating pitch values."""
        engine = VariationEngine(basic_params)

        pitches = engine._generate_pitch_values(count=3)

        assert len(pitches) == 3
        assert all(
            basic_params.pitch_range[0] <= p <= basic_params.pitch_range[1]
            for p in pitches
        )

    def test_engine_generate_variations(self, basic_params):
        """Test generating variations."""
        engine = VariationEngine(basic_params)

        variations = list(engine.generate_variations())

        assert len(variations) > 0
        assert all(isinstance(v, GenerationParameters) for v in variations)

    def test_engine_generate_variations_with_limit(self, multi_params):
        """Test generating variations with max_combinations limit."""
        engine = VariationEngine(multi_params)

        variations = list(engine.generate_variations(max_combinations=5))

        assert len(variations) == 5

    def test_engine_estimate_total_combinations(self, multi_params):
        """Test estimating total combinations."""
        engine = VariationEngine(multi_params)

        estimate = engine.estimate_total_combinations()

        # Should be at least the product of text_variations * voice_ids
        assert estimate >= 2 * 2  # 2 texts * 2 voices

    def test_engine_generate_turkish_variations(self, basic_params):
        """Test generating Turkish-specific variations."""
        engine = VariationEngine(basic_params)

        variations = engine.generate_turkish_variations("katya")

        assert len(variations) > 0
        assert "katya" in variations  # Original should be included

    def test_engine_create_turkish_parameters(self, basic_params):
        """Test creating Turkish-specific parameters."""
        engine = VariationEngine(basic_params)

        turkish_params = engine.create_turkish_parameters(
            wake_words=["hey katya"],
            voice_ids=["tr-TR-EmelNeural"],
        )

        assert turkish_params.voice_ids == ["tr-TR-EmelNeural"]
        assert "hey katya" in turkish_params.text_variations


# =============================================================================
# TEST: ProgressConfig
# =============================================================================


class TestProgressConfig:
    """Tests for the ProgressConfig dataclass."""

    def test_progress_config_defaults(self):
        """Test ProgressConfig default values."""
        config = ProgressConfig()

        assert config.refresh_rate == 0.1
        assert config.show_task_details is True
        assert config.console_width == 80

    def test_progress_config_custom_values(self):
        """Test ProgressConfig with custom values."""
        config = ProgressConfig(
            refresh_rate=0.5,
            show_task_details=False,
            console_width=120,
        )

        assert config.refresh_rate == 0.5
        assert config.show_task_details is False
        assert config.console_width == 120


# =============================================================================
# TEST: ProgressTracker
# =============================================================================


class TestProgressTracker:
    """Tests for the ProgressTracker class."""

    def test_progress_tracker_initialization(self):
        """Test ProgressTracker initialization."""
        tracker = ProgressTracker()

        assert tracker is not None
        assert tracker.config is not None

    def test_progress_tracker_with_config(self):
        """Test ProgressTracker with custom config."""
        config = ProgressConfig(refresh_rate=0.5)
        tracker = ProgressTracker(config=config)

        assert tracker.config.refresh_rate == 0.5

    @pytest.mark.asyncio
    async def test_progress_tracker_initialize_batch(self):
        """Test initializing a new batch."""
        tracker = ProgressTracker()
        await tracker.initialize_batch(total_tasks=100)

        status = tracker.get_current_status()
        assert status["total_tasks"] == 100
        assert status["completed_tasks"] == 0

        # Cleanup
        tracker.live.stop()

    @pytest.mark.asyncio
    async def test_progress_tracker_update_task_status(self):
        """Test updating task status."""
        tracker = ProgressTracker()
        await tracker.initialize_batch(total_tasks=10)

        await tracker.update_task_status("task_1", "processing", "Generating audio")

        # Status should be recorded
        status = tracker.get_current_status()
        assert "task_1" in status.get("task_statuses", {})

        # Cleanup
        tracker.live.stop()

    @pytest.mark.asyncio
    async def test_progress_tracker_update_overall_progress(self):
        """Test updating overall progress."""
        tracker = ProgressTracker()
        await tracker.initialize_batch(total_tasks=10)

        await tracker.update_overall_progress(completed=5, total=10)

        status = tracker.get_current_status()
        assert status["completed_tasks"] == 5

        # Cleanup
        tracker.live.stop()

    @pytest.mark.asyncio
    async def test_progress_tracker_finalize_batch(self):
        """Test finalizing a batch."""
        tracker = ProgressTracker()
        await tracker.initialize_batch(total_tasks=10)
        await tracker.update_overall_progress(completed=10, total=10)

        # Should not raise
        await tracker.finalize_batch()

    @pytest.mark.asyncio
    async def test_progress_tracker_show_error(self):
        """Test showing an error message."""
        tracker = ProgressTracker()
        await tracker.initialize_batch(total_tasks=10)

        # Should not raise
        await tracker.show_error("Test error message")

        # Cleanup
        tracker.live.stop()

    @pytest.mark.asyncio
    async def test_progress_tracker_async_context_manager(self):
        """Test using ProgressTracker as async context manager."""
        async with ProgressTracker() as tracker:
            await tracker.initialize_batch(total_tasks=5)
            await tracker.update_overall_progress(completed=5, total=5)

        # Should exit cleanly

    def test_progress_tracker_get_current_status(self):
        """Test getting current status without batch initialization."""
        tracker = ProgressTracker()

        status = tracker.get_current_status()

        # Should have these keys even without batch init
        assert "completed_tasks" in status
        assert "total_tasks" in status
        assert status["completed_tasks"] == 0
        assert status["total_tasks"] == 0


# =============================================================================
# TEST: Integration Tests
# =============================================================================


class TestGenerationIntegration:
    """Integration tests for generation components."""

    @pytest.mark.asyncio
    async def test_rate_limiter_with_variations(self):
        """Test rate limiter working with variation generation."""
        limiter = RateLimiter(max_requests=100, period_seconds=60)
        params = VariationParameters(
            text_variations=["test"],
            voice_ids=["voice-1"],
        )
        engine = VariationEngine(params)

        # Generate some variations with rate limiting
        count = 0
        for variation in engine.generate_variations(max_combinations=5):
            await limiter.wait_for_token()
            count += 1

        assert count == 5
        assert limiter.get_available_tokens() < limiter.max_requests

    @pytest.mark.asyncio
    async def test_progress_tracker_with_variation_engine(self):
        """Test progress tracking during variation generation."""
        tracker = ProgressTracker()
        params = VariationParameters(
            text_variations=["hey assistant"],
            voice_ids=["voice-1", "voice-2"],
        )
        engine = VariationEngine(params)

        variations = list(engine.generate_variations(max_combinations=10))
        await tracker.initialize_batch(total_tasks=len(variations))

        for i, var in enumerate(variations):
            await tracker.update_task_status(f"task_{i}", "completed")
            await tracker.update_overall_progress(
                completed=i + 1, total=len(variations)
            )

        await tracker.finalize_batch()

        status = tracker.get_current_status()
        assert status["completed_tasks"] == len(variations)

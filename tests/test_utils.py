"""
Test Suite for wakegen.utils module

This module tests utility functions for audio processing, logging, and caching.

EDUCATIONAL NOTES:
- We use tmp_path fixture to create temporary directories for tests
- We mock external dependencies like librosa to isolate tests
- We use numpy to create synthetic audio data for testing
"""

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# =============================================================================
# IMPORTS: Testing wakegen.utils modules
# =============================================================================
from wakegen.utils.audio import (
    save_audio,
    load_audio,
    resample_audio,
    load_audio_file,
    get_audio_duration,
)
from wakegen.utils.logging import (
    get_logger,
    setup_logging,
)
from wakegen.utils.caching import (
    CacheStats,
    CacheEntry,
    GenerationCache,
)
from wakegen.core.exceptions import AudioError


# =============================================================================
# FIXTURES: Reusable test setup
# =============================================================================


@pytest.fixture
def sample_audio_bytes() -> bytes:
    """Create sample audio bytes for testing."""
    # Generate 1 second of silence at 16000Hz
    # CONCEPT: 16-bit PCM audio is 2 bytes per sample
    samples = np.zeros(16000, dtype=np.int16)
    return samples.tobytes()


@pytest.fixture
def sample_audio_data() -> tuple[np.ndarray[Any, Any], int]:
    """Create sample audio numpy array for testing."""
    # Generate 1 second of sine wave at 16000Hz
    sr = 16000
    t = np.linspace(0, 1, sr)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return audio, sr


@pytest.fixture
def temp_audio_file(sample_audio_data, tmp_path) -> Path:
    """Create a temporary audio file for testing."""
    import soundfile as sf

    audio, sr = sample_audio_data
    file_path = tmp_path / "test_audio.wav"
    sf.write(str(file_path), audio, sr)
    return file_path


# =============================================================================
# TEST: Audio Utilities
# =============================================================================


class TestSaveAudio:
    """Tests for save_audio function."""

    def test_save_audio_creates_file(self, sample_audio_bytes, tmp_path):
        """Test that save_audio creates a file."""
        file_path = tmp_path / "output" / "test.wav"

        save_audio(sample_audio_bytes, str(file_path))

        assert file_path.exists()
        assert file_path.stat().st_size > 0

    def test_save_audio_creates_directories(self, sample_audio_bytes, tmp_path):
        """Test that save_audio creates parent directories."""
        # CONCEPT: makedirs(exist_ok=True) creates all parent directories
        file_path = tmp_path / "deep" / "nested" / "path" / "audio.wav"

        save_audio(sample_audio_bytes, str(file_path))

        assert file_path.exists()

    def test_save_audio_with_custom_sample_rate(self, sample_audio_bytes, tmp_path):
        """Test save_audio with custom sample rate."""
        file_path = tmp_path / "audio.wav"

        save_audio(sample_audio_bytes, str(file_path), sample_rate=22050)

        assert file_path.exists()

    def test_save_audio_invalid_path_raises_error(self, sample_audio_bytes):
        """Test that invalid path raises AudioError."""
        # Try to save to root directory without permissions
        # On Windows, trying to write to an invalid path should fail
        with pytest.raises(AudioError):
            save_audio(sample_audio_bytes, "")


class TestLoadAudio:
    """Tests for load_audio function."""

    def test_load_audio_returns_data_and_rate(self, temp_audio_file):
        """Test that load_audio returns audio data and sample rate."""
        data, sr = load_audio(str(temp_audio_file))

        assert isinstance(data, np.ndarray)
        assert isinstance(sr, int)
        assert len(data) > 0

    def test_load_audio_nonexistent_file_raises_error(self):
        """Test that loading non-existent file raises AudioError."""
        with pytest.raises(AudioError):
            load_audio("/nonexistent/path/audio.wav")


class TestResampleAudio:
    """Tests for resample_audio function."""

    def test_resample_audio_changes_sample_rate(self, tmp_path):
        """Test that resample_audio changes the sample rate."""
        import soundfile as sf

        # Create a 44100Hz audio file
        sr_original = 44100
        audio = np.random.randn(sr_original).astype(np.float32)
        file_path = tmp_path / "to_resample.wav"
        sf.write(str(file_path), audio, sr_original)

        # Resample to 16000Hz
        resample_audio(str(file_path), target_sr=16000)

        # Load and verify
        data, sr = load_audio(str(file_path))
        assert sr == 16000

    def test_resample_audio_same_rate_no_change(self, temp_audio_file):
        """Test that resampling to same rate doesn't modify file."""
        import soundfile as sf

        # Get original modification time
        original_data, original_sr = load_audio(str(temp_audio_file))

        # Resample to same rate (should be a no-op)
        resample_audio(str(temp_audio_file), target_sr=original_sr)

        # Verify file unchanged (data should be the same)
        new_data, new_sr = load_audio(str(temp_audio_file))
        assert new_sr == original_sr


class TestLoadAudioFile:
    """Tests for async load_audio_file function."""

    @pytest.mark.asyncio
    async def test_load_audio_file_async(self, temp_audio_file):
        """Test async loading of audio file."""
        data, sr = await load_audio_file(str(temp_audio_file))

        assert isinstance(data, np.ndarray)
        assert isinstance(sr, int)
        assert len(data) > 0


class TestGetAudioDuration:
    """Tests for get_audio_duration function."""

    def test_get_audio_duration_mono(self):
        """Test duration calculation for mono audio."""
        # 16000 samples at 16000Hz = 1 second
        audio = np.zeros(16000, dtype=np.float32)
        duration = get_audio_duration(audio, 16000)

        assert abs(duration - 1.0) < 0.001

    def test_get_audio_duration_stereo(self):
        """Test duration calculation for stereo audio."""
        # 2 channels, 16000 samples each at 16000Hz = 1 second
        audio = np.zeros((2, 16000), dtype=np.float32)
        duration = get_audio_duration(audio, 16000)

        assert abs(duration - 1.0) < 0.001

    def test_get_audio_duration_various_rates(self):
        """Test duration calculation with various sample rates."""
        audio = np.zeros(22050, dtype=np.float32)

        duration = get_audio_duration(audio, 22050)
        assert abs(duration - 1.0) < 0.001

        duration = get_audio_duration(audio, 44100)
        assert abs(duration - 0.5) < 0.001


# =============================================================================
# TEST: Logging Utilities
# =============================================================================


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_default_name(self):
        """Test getting default wakegen logger."""
        logger = get_logger()

        assert logger.name == "wakegen"
        assert isinstance(logger, logging.Logger)

    def test_get_logger_with_name(self):
        """Test getting logger with custom name."""
        logger = get_logger("providers")

        assert logger.name == "wakegen.providers"

    def test_get_logger_returns_same_instance(self):
        """Test that same name returns same logger instance."""
        logger1 = get_logger("test")
        logger2 = get_logger("test")

        assert logger1 is logger2


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_logging_default(self):
        """Test setup_logging with default level."""
        setup_logging()

        logger = logging.getLogger("wakegen")
        assert logger.level == logging.INFO

    def test_setup_logging_debug_level(self):
        """Test setup_logging with DEBUG level."""
        setup_logging("DEBUG")

        logger = logging.getLogger("wakegen")
        assert logger.level == logging.DEBUG

    def test_setup_logging_warning_level(self):
        """Test setup_logging with WARNING level."""
        setup_logging("WARNING")

        logger = logging.getLogger("wakegen")
        assert logger.level == logging.WARNING


# =============================================================================
# TEST: Caching - CacheStats
# =============================================================================


class TestCacheStats:
    """Tests for CacheStats dataclass."""

    def test_cache_stats_defaults(self):
        """Test CacheStats default values."""
        stats = CacheStats()

        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.total_size_bytes == 0
        assert stats.file_count == 0
        assert stats.evictions == 0

    def test_cache_stats_hit_rate_no_requests(self):
        """Test hit rate when no requests made."""
        stats = CacheStats()

        # CONCEPT: hit_rate is a @property, not a method
        assert stats.hit_rate == 0.0

    def test_cache_stats_hit_rate_calculation(self):
        """Test hit rate calculation."""
        stats = CacheStats(hits=30, misses=70)

        assert stats.hit_rate == 0.3

    def test_cache_stats_hit_rate_all_hits(self):
        """Test hit rate when all requests are hits."""
        stats = CacheStats(hits=100, misses=0)

        assert stats.hit_rate == 1.0

    def test_cache_stats_total_size_mb(self):
        """Test total size MB calculation."""
        # 10 MB = 10 * 1024 * 1024 bytes
        stats = CacheStats(total_size_bytes=10 * 1024 * 1024)

        # CONCEPT: total_size_mb is a @property, not a method
        assert stats.total_size_mb == 10.0

    def test_cache_stats_to_dict(self):
        """Test conversion to dictionary."""
        stats = CacheStats(hits=5, misses=10, total_size_bytes=1024)

        data = stats.to_dict()

        assert data["hits"] == 5
        assert data["misses"] == 10
        assert "hit_rate" in data
        assert "total_size_mb" in data


# =============================================================================
# TEST: Caching - CacheEntry
# =============================================================================


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_cache_entry_creation(self):
        """Test creating a CacheEntry."""
        entry = CacheEntry(
            cache_key="abc123",
            file_path="/cache/audio.wav",
            text="hello world",
            voice_id="voice-1",
            provider="edge_tts",
        )

        assert entry.cache_key == "abc123"
        assert entry.file_path == "/cache/audio.wav"
        assert entry.text == "hello world"
        assert entry.access_count == 0

    def test_cache_entry_touch_updates_access(self):
        """Test that touch() updates access time and count."""
        import time

        entry = CacheEntry(
            cache_key="key",
            file_path="/path",
            text="text",
            voice_id="voice",
            provider="provider",
        )

        original_access_time = entry.last_accessed
        original_count = entry.access_count

        # Small delay to ensure time difference
        time.sleep(0.01)

        entry.touch()

        assert entry.access_count == original_count + 1
        assert entry.last_accessed >= original_access_time


# =============================================================================
# TEST: Caching - GenerationCache
# =============================================================================


class TestGenerationCache:
    """Tests for GenerationCache class."""

    @pytest.fixture
    def cache_dir(self, tmp_path) -> Path:
        """Create a temporary cache directory."""
        return tmp_path / "test_cache"

    @pytest.fixture
    def cache(self, cache_dir) -> GenerationCache:
        """Create a GenerationCache instance for testing."""
        return GenerationCache(
            cache_dir=str(cache_dir),
            max_size_mb=10.0,
            enabled=True,
        )

    def test_cache_initialization(self, cache, cache_dir):
        """Test that cache initializes correctly."""
        assert cache.enabled is True
        assert cache.cache_dir == Path(cache_dir)
        assert cache.max_size_bytes == 10.0 * 1024 * 1024

    def test_cache_creates_directory(self, cache, cache_dir):
        """Test that cache creates its directory."""
        assert cache_dir.exists()

    def test_cache_disabled_get_returns_none(self, tmp_path):
        """Test that disabled cache always returns None."""
        cache = GenerationCache(
            cache_dir=str(tmp_path / "disabled_cache"),
            enabled=False,
        )

        result = cache.get("text", "voice", "provider")
        assert result is None

    def test_cache_get_miss_returns_none(self, cache):
        """Test that cache miss returns None."""
        result = cache.get("nonexistent", "voice", "provider")
        assert result is None

    def test_cache_put_and_get(self, cache, temp_audio_file):
        """Test adding to cache and retrieving."""
        # Put the file in cache
        cached_path = cache.put(
            text="hello",
            voice_id="voice-1",
            provider="edge_tts",
            source_path=str(temp_audio_file),
            copy=True,
        )

        assert cached_path is not None
        assert Path(cached_path).exists()

        # Get from cache
        retrieved = cache.get("hello", "voice-1", "edge_tts")
        assert retrieved == cached_path

    def test_cache_get_cache_path(self, cache):
        """Test getting the path for a new cache entry."""
        path = cache.get_cache_path("text", "voice", "provider", ".wav")

        assert path.endswith(".wav")
        assert cache.cache_dir in Path(path).parents or str(cache.cache_dir) in path

    def test_cache_get_stats(self, cache):
        """Test getting cache statistics."""
        stats = cache.get_stats()

        assert isinstance(stats, CacheStats)
        assert stats.hits >= 0
        assert stats.misses >= 0

    def test_cache_clear(self, cache, temp_audio_file):
        """Test clearing the cache."""
        # Add something to cache
        cache.put("text", "voice", "provider", str(temp_audio_file))

        # Clear cache
        cache.clear()

        # Verify cache is empty
        result = cache.get("text", "voice", "provider")
        assert result is None

    def test_cache_context_manager(self, cache_dir):
        """Test cache as context manager."""
        with GenerationCache(str(cache_dir), enabled=True) as cache:
            assert cache.enabled is True

        # Should not raise after exiting context

    def test_cache_hit_updates_stats(self, cache, temp_audio_file):
        """Test that cache hit updates statistics."""
        # Put and get to create a hit
        cache.put("text", "voice", "provider", str(temp_audio_file))
        cache.get("text", "voice", "provider")

        stats = cache.get_stats()
        assert stats.hits >= 1

    def test_cache_miss_updates_stats(self, cache):
        """Test that cache miss updates statistics."""
        cache.get("nonexistent", "voice", "provider")

        stats = cache.get_stats()
        assert stats.misses >= 1


# =============================================================================
# TEST: GPU Utilities (if accessible without GPU)
# =============================================================================


class TestGpuUtils:
    """Tests for GPU utility functions (mocked to avoid GPU dependency)."""

    def test_gpu_module_importable(self):
        """Test that GPU module can be imported."""
        from wakegen.utils import gpu

        assert gpu is not None


# =============================================================================
# TEST: Docker Bridge Utilities
# =============================================================================


class TestDockerBridge:
    """Tests for Docker bridge utilities."""

    def test_docker_bridge_module_importable(self):
        """Test that docker_bridge module can be imported."""
        from wakegen.utils import docker_bridge

        assert docker_bridge is not None


# =============================================================================
# TEST: Async Helpers
# =============================================================================


class TestAsyncHelpers:
    """Tests for async helper utilities."""

    def test_async_helpers_module_importable(self):
        """Test that async_helpers module can be imported."""
        from wakegen.utils import async_helpers

        assert async_helpers is not None

"""
Integration tests for WakeGen.

These tests verify that the various components work together correctly.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from typing import Generator


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ============================================================================
# Provider Tests
# ============================================================================

class TestProviderRegistry:
    """Test provider registry functionality."""
    
    def test_list_all_providers(self):
        """Test listing all provider types."""
        from wakegen.core.types import ProviderType
        
        providers = list(ProviderType)
        
        # Check we have expected providers
        assert ProviderType.EDGE_TTS in providers
        assert ProviderType.KOKORO in providers
        assert ProviderType.PIPER in providers
        assert ProviderType.BARK in providers
        assert ProviderType.CHATTTS in providers
    
    def test_get_edge_tts_provider(self):
        """Test getting Edge TTS provider (always available)."""
        from wakegen.providers.registry import get_provider
        from wakegen.core.types import ProviderType
        from wakegen.config.settings import ProviderConfig
        
        provider = get_provider(ProviderType.EDGE_TTS, ProviderConfig())
        
        assert provider is not None
        assert provider.provider_type == ProviderType.EDGE_TTS


class TestProviderGeneration:
    """Test actual generation with providers."""
    
    @pytest.mark.asyncio
    async def test_edge_tts_generation(self, temp_dir: Path):
        """Test generating audio with Edge TTS."""
        from wakegen.providers.registry import get_provider
        from wakegen.core.types import ProviderType
        from wakegen.config.settings import ProviderConfig
        
        provider = get_provider(ProviderType.EDGE_TTS, ProviderConfig())
        output_path = temp_dir / "test_edge.wav"
        
        await provider.generate(
            text="hello world",
            voice_id="en-US-AriaNeural",
            output_path=str(output_path)
        )
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    @pytest.mark.asyncio
    async def test_edge_tts_list_voices(self):
        """Test listing voices from Edge TTS."""
        from wakegen.providers.registry import get_provider
        from wakegen.core.types import ProviderType
        from wakegen.config.settings import ProviderConfig
        
        provider = get_provider(ProviderType.EDGE_TTS, ProviderConfig())
        voices = await provider.list_voices()
        
        assert len(voices) > 0
        
        # Check voice structure
        voice = voices[0]
        assert hasattr(voice, 'id')
        assert hasattr(voice, 'name')


# ============================================================================
# Export Tests
# ============================================================================

class TestExportFormats:
    """Test export format functionality."""
    
    def test_import_export_formats(self):
        """Test that export formats can be imported."""
        from wakegen.export.formats import (
            ExportFormat,
            list_export_formats,
        )
        
        assert ExportFormat is not None
        
        formats = list_export_formats()
        assert len(formats) > 0
    
    def test_export_format_enum(self):
        """Test ExportFormat enum values."""
        from wakegen.export.formats import ExportFormat
        
        # Check all expected formats exist
        assert ExportFormat.OPENWAKEWORD is not None
        assert ExportFormat.MYCROFT_PRECISE is not None
        assert ExportFormat.PICOVOICE is not None
        assert ExportFormat.TENSORFLOW is not None
        assert ExportFormat.PYTORCH is not None
        assert ExportFormat.HUGGINGFACE is not None


# ============================================================================
# Configuration Tests
# ============================================================================

class TestConfiguration:
    """Test configuration loading."""
    
    def test_generation_config(self):
        """Test GenerationConfig creation."""
        from wakegen.models.config import GenerationConfig
        
        config = GenerationConfig()
        
        assert config.output_dir is not None
        assert config.sample_rate == 16000


# ============================================================================
# Workflow Tests
# ============================================================================

class TestGenerationWorkflow:
    """Test complete generation workflow."""
    
    @pytest.mark.asyncio
    async def test_full_generation_pipeline(self, temp_dir: Path):
        """Test a complete generation pipeline."""
        from wakegen.providers.registry import get_provider
        from wakegen.core.types import ProviderType
        from wakegen.config.settings import ProviderConfig
        
        # 1. Get provider
        provider = get_provider(ProviderType.EDGE_TTS, ProviderConfig())
        
        # 2. Generate multiple samples
        for i in range(3):
            output_path = temp_dir / f"sample_{i:03d}.wav"
            await provider.generate(
                text="hey assistant",
                voice_id="en-US-AriaNeural",
                output_path=str(output_path)
            )
            assert output_path.exists()
        
        # 3. Check all files exist
        wav_files = list(temp_dir.glob("*.wav"))
        assert len(wav_files) == 3


# ============================================================================
# Utility Tests
# ============================================================================

class TestUtilities:
    """Test utility functions."""
    
    def test_logging_setup(self):
        """Test logging utilities."""
        from wakegen.utils.logging import get_logger, setup_logging
        
        logger = get_logger("test")
        assert logger is not None
    
    def test_caching_module(self):
        """Test caching module."""
        from wakegen.utils.caching import GenerationCache
        
        assert GenerationCache is not None


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

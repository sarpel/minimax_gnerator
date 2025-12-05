"""
Unit tests for TTS providers.

These tests verify each provider can be imported and has the correct type.
Some tests require specific dependencies to be installed and will be skipped if not available.
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


@pytest.fixture
def provider_config():
    """Create a provider configuration."""
    from wakegen.config.settings import ProviderConfig
    return ProviderConfig()


# ============================================================================
# Edge TTS Provider Tests
# ============================================================================

class TestEdgeTTSProvider:
    """Tests for Edge TTS provider."""
    
    def test_provider_import(self):
        """Test provider can be imported."""
        from wakegen.providers.free.edge_tts import EdgeTTSProvider
        assert EdgeTTSProvider is not None
    
    def test_provider_type(self, provider_config):
        """Test provider type is correct."""
        from wakegen.providers.free.edge_tts import EdgeTTSProvider
        from wakegen.core.types import ProviderType
        
        provider = EdgeTTSProvider(provider_config)
        assert provider.provider_type == ProviderType.EDGE_TTS
    
    @pytest.mark.asyncio
    async def test_list_voices(self, provider_config):
        """Test listing voices."""
        from wakegen.providers.free.edge_tts import EdgeTTSProvider
        
        provider = EdgeTTSProvider(provider_config)
        voices = await provider.list_voices()
        
        assert len(voices) > 0
        
        # Check for common English voices
        voice_ids = [v.id for v in voices]
        assert any("en-US" in vid for vid in voice_ids)
    
    @pytest.mark.asyncio
    async def test_generate(self, provider_config, temp_dir):
        """Test generating audio."""
        from wakegen.providers.free.edge_tts import EdgeTTSProvider
        
        provider = EdgeTTSProvider(provider_config)
        output_path = temp_dir / "test.wav"
        
        await provider.generate(
            text="Hello world",
            voice_id="en-US-AriaNeural",
            output_path=str(output_path)
        )
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0


# ============================================================================
# Kokoro Provider Tests
# ============================================================================

class TestKokoroProvider:
    """Tests for Kokoro TTS provider."""
    
    def test_provider_import(self):
        """Test that provider can be imported."""
        from wakegen.providers.opensource.kokoro import KokoroTTSProvider
        assert KokoroTTSProvider is not None
    
    def test_provider_type(self, provider_config):
        """Test provider type is correct."""
        from wakegen.providers.opensource.kokoro import KokoroTTSProvider
        from wakegen.core.types import ProviderType
        
        provider = KokoroTTSProvider(provider_config)
        assert provider.provider_type == ProviderType.KOKORO


# ============================================================================
# Piper Provider Tests
# ============================================================================

class TestPiperProvider:
    """Tests for Piper TTS provider."""
    
    def test_provider_import(self):
        """Test that provider can be imported."""
        from wakegen.providers.opensource.piper import PiperTTSProvider
        assert PiperTTSProvider is not None
    
    def test_provider_type(self, provider_config):
        """Test provider type is correct."""
        from wakegen.providers.opensource.piper import PiperTTSProvider
        from wakegen.core.types import ProviderType
        
        provider = PiperTTSProvider(provider_config)
        assert provider.provider_type == ProviderType.PIPER


# ============================================================================
# Mimic3 Provider Tests
# ============================================================================

class TestMimic3Provider:
    """Tests for Mimic3 TTS provider."""
    
    def test_provider_import(self):
        """Test that provider can be imported."""
        from wakegen.providers.opensource.mimic3 import Mimic3Provider
        assert Mimic3Provider is not None
    
    def test_provider_type(self, provider_config):
        """Test provider type is correct."""
        from wakegen.providers.opensource.mimic3 import Mimic3Provider
        from wakegen.core.types import ProviderType
        
        provider = Mimic3Provider(provider_config)
        assert provider.provider_type == ProviderType.MIMIC3


# ============================================================================
# Bark Provider Tests
# ============================================================================

class TestBarkProvider:
    """Tests for Bark TTS provider."""
    
    def test_provider_import(self):
        """Test that provider can be imported."""
        from wakegen.providers.opensource.bark import BarkProvider
        assert BarkProvider is not None
    
    def test_has_speaker_presets(self):
        """Test speaker presets are defined on the class."""
        from wakegen.providers.opensource.bark import BarkProvider
        
        # SPEAKER_PRESETS is a class attribute
        assert hasattr(BarkProvider, 'SPEAKER_PRESETS')


# ============================================================================
# ChatTTS Provider Tests
# ============================================================================

class TestChatTTSProvider:
    """Tests for ChatTTS provider."""
    
    def test_provider_import(self):
        """Test that provider can be imported."""
        from wakegen.providers.opensource.chattts import ChatTTSProvider
        assert ChatTTSProvider is not None


# ============================================================================
# StyleTTS2 Provider Tests
# ============================================================================

class TestStyleTTS2Provider:
    """Tests for StyleTTS2 provider."""
    
    def test_provider_import(self):
        """Test that provider can be imported."""
        from wakegen.providers.opensource.styletts2 import StyleTTS2Provider
        assert StyleTTS2Provider is not None
    
    def test_provider_type(self, provider_config):
        """Test provider type is correct."""
        from wakegen.providers.opensource.styletts2 import StyleTTS2Provider
        from wakegen.core.types import ProviderType
        
        provider = StyleTTS2Provider(provider_config)
        assert provider.provider_type == ProviderType.STYLETTS2


# ============================================================================
# Orpheus Provider Tests
# ============================================================================

class TestOrpheusProvider:
    """Tests for Orpheus TTS provider."""
    
    def test_provider_import(self):
        """Test that provider can be imported."""
        from wakegen.providers.opensource.orpheus import OrpheusTTSProvider
        assert OrpheusTTSProvider is not None
    
    def test_provider_type(self, provider_config):
        """Test provider type is correct."""
        from wakegen.providers.opensource.orpheus import OrpheusTTSProvider
        from wakegen.core.types import ProviderType
        
        provider = OrpheusTTSProvider(provider_config)
        assert provider.provider_type == ProviderType.ORPHEUS


# ============================================================================
# Coqui XTTS Provider Tests
# ============================================================================

class TestCoquiXTTSProvider:
    """Tests for Coqui XTTS provider."""
    
    def test_provider_import(self):
        """Test that provider can be imported."""
        from wakegen.providers.opensource.coqui_xtts import CoquiXTTSProvider
        assert CoquiXTTSProvider is not None
    
    def test_provider_type(self, provider_config):
        """Test provider type is correct."""
        from wakegen.providers.opensource.coqui_xtts import CoquiXTTSProvider
        from wakegen.core.types import ProviderType
        
        provider = CoquiXTTSProvider(provider_config)
        assert provider.provider_type == ProviderType.COQUI_XTTS


# ============================================================================
# MiniMax Provider Tests
# ============================================================================

class TestMiniMaxProvider:
    """Tests for MiniMax commercial provider."""
    
    def test_provider_import(self):
        """Test that provider can be imported."""
        from wakegen.providers.commercial.minimax import MiniMaxProvider
        assert MiniMaxProvider is not None


# ============================================================================
# Provider Registry Tests
# ============================================================================

class TestProviderRegistry:
    """Tests for provider registry."""
    
    def test_get_all_provider_types(self):
        """Test getting all provider types."""
        from wakegen.core.types import ProviderType
        
        all_types = list(ProviderType)
        
        # Check we have expected providers
        expected = [
            ProviderType.EDGE_TTS,
            ProviderType.KOKORO,
            ProviderType.PIPER,
            ProviderType.MIMIC3,
            ProviderType.COQUI_XTTS,
            ProviderType.STYLETTS2,
            ProviderType.ORPHEUS,
            ProviderType.BARK,
            ProviderType.CHATTTS,
            ProviderType.MINIMAX,
        ]
        
        for provider_type in expected:
            assert provider_type in all_types
    
    def test_get_provider_function(self):
        """Test get_provider function exists and works for edge_tts."""
        from wakegen.providers.registry import get_provider
        from wakegen.core.types import ProviderType
        from wakegen.config.settings import ProviderConfig
        
        provider = get_provider(ProviderType.EDGE_TTS, ProviderConfig())
        assert provider is not None
        assert provider.provider_type == ProviderType.EDGE_TTS


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

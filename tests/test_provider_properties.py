"""
Property-based tests for TTS provider implementations.

**Feature: critical-bug-fixes, Property 2: Provider voice listing returns valid Voice objects**
**Validates: Requirements 2.3, 3.3**
"""

import pytest
from hypothesis import given, strategies as st, settings

from wakegen.core.types import ProviderType, Gender
from wakegen.models.audio import Voice
from wakegen.models.config import ProviderConfig


class TestBarkProviderVoiceListingProperty:
    """
    Property-based tests for BarkProvider voice listing.
    
    **Feature: critical-bug-fixes, Property 2: Provider voice listing returns valid Voice objects**
    **Validates: Requirements 2.3**
    
    Property: For any provider (BarkProvider), when list_voices is called, 
    all returned Voice objects should have a non-empty `id` attribute that 
    matches the expected voice identifier format.
    """

    @pytest.mark.asyncio
    async def test_bark_provider_instantiation(self):
        """Test that BarkProvider can be instantiated without errors."""
        from wakegen.providers.opensource.bark import BarkProvider
        
        # Test instantiation without config (should use default)
        provider = BarkProvider()
        assert provider is not None
        assert provider.config is not None
        assert isinstance(provider.config, ProviderConfig)
        
        # Test instantiation with explicit config
        config = ProviderConfig()
        provider_with_config = BarkProvider(config=config)
        assert provider_with_config.config is config

    @pytest.mark.asyncio
    @given(language=st.sampled_from(["en", "zh", "de", "es", "fr", "hi", "it", "ja", "ko", "pl", "pt", "ru", "tr", None]))
    @settings(max_examples=100, deadline=None)
    async def test_bark_voice_listing_returns_valid_voices(self, language):
        """
        **Feature: critical-bug-fixes, Property 2: Provider voice listing returns valid Voice objects**
        **Validates: Requirements 2.3**
        
        Property: For any language filter (including None for all languages),
        all returned Voice objects should have:
        - A non-empty `id` attribute
        - The `id` should match the expected format (v2/{lang}_speaker_{num})
        - A valid `provider` attribute set to ProviderType.BARK
        - Valid `gender` and `language` attributes
        """
        from wakegen.providers.opensource.bark import BarkProvider
        
        provider = BarkProvider()
        voices = await provider.list_voices(language=language)
        
        # All voices should be Voice instances with valid attributes
        for voice in voices:
            # Check it's a Voice instance
            assert isinstance(voice, Voice), f"Expected Voice instance, got {type(voice)}"
            
            # Check id is non-empty string
            assert voice.id is not None, "Voice id should not be None"
            assert isinstance(voice.id, str), f"Voice id should be string, got {type(voice.id)}"
            assert len(voice.id) > 0, "Voice id should not be empty"
            
            # Check id matches expected format: v2/{lang}_speaker_{num}
            assert voice.id.startswith("v2/"), f"Voice id should start with 'v2/', got {voice.id}"
            assert "_speaker_" in voice.id, f"Voice id should contain '_speaker_', got {voice.id}"
            
            # Check provider is set correctly
            assert voice.provider == ProviderType.BARK, f"Voice provider should be BARK, got {voice.provider}"
            
            # Check gender is valid
            assert isinstance(voice.gender, Gender), f"Voice gender should be Gender enum, got {type(voice.gender)}"
            
            # Check language is non-empty
            assert voice.language is not None, "Voice language should not be None"
            assert len(voice.language) > 0, "Voice language should not be empty"
            
            # If language filter was provided, verify it matches
            if language is not None:
                assert voice.language == language, f"Voice language {voice.language} should match filter {language}"

    @pytest.mark.asyncio
    async def test_bark_voice_count_per_language(self):
        """Test that each language has exactly 10 speaker presets."""
        from wakegen.providers.opensource.bark import BarkProvider
        
        provider = BarkProvider()
        
        # Test each language individually
        for lang in ["en", "zh", "de", "es", "fr", "hi", "it", "ja", "ko", "pl", "pt", "ru", "tr"]:
            voices = await provider.list_voices(language=lang)
            assert len(voices) == 10, f"Expected 10 voices for {lang}, got {len(voices)}"

    @pytest.mark.asyncio
    async def test_bark_all_voices_count(self):
        """Test that listing all voices returns expected total count."""
        from wakegen.providers.opensource.bark import BarkProvider
        
        provider = BarkProvider()
        voices = await provider.list_voices()
        
        # 13 languages * 10 speakers = 130 total voices
        expected_count = 13 * 10
        assert len(voices) == expected_count, f"Expected {expected_count} total voices, got {len(voices)}"

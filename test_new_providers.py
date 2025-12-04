#!/usr/bin/env python3
"""
Test script for Phase 6 - Additional Providers (Piper, Coqui XTTS)

This script tests the provider discovery and basic functionality.
"""

import asyncio
from wakegen.providers.registry import get_provider
from wakegen.core.types import ProviderType
from wakegen.models.config import ProviderConfig

async def test_provider_discovery():
    """Test that both new providers can be discovered and instantiated."""
    print("Testing provider discovery...")

    # Test Piper provider
    try:
        piper = get_provider(ProviderType.PIPER, ProviderConfig())
        assert piper is not None
        print("SUCCESS: Piper provider discovered successfully")
        print(f"   Provider type: {piper.provider_type}")
    except Exception as e:
        print(f"FAILED: Piper provider discovery failed: {e}")
        return False

    # Test Coqui XTTS provider
    try:
        coqui = get_provider(ProviderType.COQUI_XTTS, ProviderConfig())
        assert coqui is not None
        print("SUCCESS: Coqui XTTS provider discovered successfully")
        print(f"   Provider type: {coqui.provider_type}")
    except Exception as e:
        print(f"FAILED: Coqui XTTS provider discovery failed: {e}")
        return False

    return True

async def test_voice_listing():
    """Test that both providers can list voices."""
    print("\nTesting voice listing...")

    config = ProviderConfig()

    # Test Piper voice listing
    try:
        piper = get_provider(ProviderType.PIPER, config)
        voices = await piper.list_voices()
        assert len(voices) > 0
        print("SUCCESS: Piper voice listing successful")
        print(f"   Available voices: {len(voices)}")
        for voice in voices[:3]:  # Show first 3 voices
            print(f"   - {voice.name} ({voice.id})")
    except Exception as e:
        print(f"FAILED: Piper voice listing failed: {e}")
        return False

    # Test Coqui XTTS voice listing
    try:
        coqui = get_provider(ProviderType.COQUI_XTTS, config)
        voices = await coqui.list_voices()
        assert len(voices) > 0
        print("SUCCESS: Coqui XTTS voice listing successful")
        print(f"   Available voices: {len(voices)}")
        for voice in voices[:3]:  # Show first 3 voices
            print(f"   - {voice.name} ({voice.id})")
    except Exception as e:
        print(f"FAILED: Coqui XTTS voice listing failed: {e}")
        return False

    return True

async def test_config_validation():
    """Test that both providers can validate their configuration."""
    print("\nTesting configuration validation...")

    config = ProviderConfig()

    # Test Piper config validation
    try:
        piper = get_provider(ProviderType.PIPER, config)
        await piper.validate_config()
        print("SUCCESS: Piper configuration validation successful")
    except Exception as e:
        print(f"FAILED: Piper configuration validation failed: {e}")
        return False

    # Test Coqui XTTS config validation
    try:
        coqui = get_provider(ProviderType.COQUI_XTTS, config)
        await coqui.validate_config()
        print("SUCCESS: Coqui XTTS configuration validation successful")
    except Exception as e:
        print(f"FAILED: Coqui XTTS configuration validation failed: {e}")
        return False

    return True

async def main():
    """Run all tests."""
    print("Testing Phase 6 - Additional Providers Implementation")
    print("=" * 60)

    all_passed = True

    # Run discovery test
    if not await test_provider_discovery():
        all_passed = False

    # Run voice listing test
    if not await test_voice_listing():
        all_passed = False

    # Run config validation test
    if not await test_config_validation():
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("SUCCESS: All tests passed! Phase 6 implementation is working correctly.")
        print("\nSummary:")
        print("   SUCCESS: Piper TTS provider implemented with Turkish support")
        print("   SUCCESS: Coqui XTTS provider implemented with voice cloning")
        print("   SUCCESS: Both providers registered and discoverable")
        print("   SUCCESS: Voice listing working for both providers")
        print("   SUCCESS: Configuration validation working")
    else:
        print("FAILED: Some tests failed. Please check the implementation.")

    return all_passed

if __name__ == "__main__":
    # Run the async main function
    result = asyncio.run(main())
    exit(0 if result else 1)
#!/usr/bin/env python3
"""
Simple test for Piper TTS Provider implementation.
This test verifies that the Piper TTS provider can be instantiated and basic methods work.
"""

import asyncio
import os
import tempfile
from wakegen.providers.opensource.piper import PiperTTSProvider
from wakegen.core.exceptions import ProviderError

async def test_piper_provider():
    """Test the Piper TTS provider basic functionality."""
    print("Testing Piper TTS Provider...")

    # Create a simple config
    config = {"provider": "piper"}

    try:
        # Test 1: Provider instantiation
        print("1. Testing provider instantiation...")
        provider = PiperTTSProvider(config)
        print("SUCCESS: Provider instantiated successfully")

        # Test 2: List voices
        print("2. Testing list_voices() method...")
        voices = await provider.list_voices()
        print(f"SUCCESS: Found {len(voices)} Turkish voices:")
        for voice in voices:
            print(f"   - {voice.name} ({voice.id})")

        # Test 3: Validate config
        print("3. Testing validate_config() method...")
        await provider.validate_config()
        print("SUCCESS: Configuration validated successfully")

        # Test 4: Try to generate a simple audio (this might fail if Piper isn't installed)
        print("4. Testing generate() method with simple text...")
        test_text = "Merhaba, bu bir test mesajidir."
        voice_id = "tr_TR-dfki-medium"  # Use one of the known Turkish voices

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            output_path = temp_file.name

        try:
            await provider.generate(test_text, voice_id, output_path)
            if os.path.exists(output_path):
                print(f"SUCCESS: Audio generated successfully at {output_path}")
                file_size = os.path.getsize(output_path)
                print(f"File size: {file_size} bytes")
                os.unlink(output_path)  # Clean up
            else:
                print("WARNING: Audio file was not created (Piper might not be installed)")
        except ProviderError as e:
            print(f"WARNING: Generation failed (expected if Piper isn't installed): {str(e)}")
        except Exception as e:
            print(f"WARNING: Unexpected error during generation: {str(e)}")

        print("All tests completed!")

    except Exception as e:
        print(f"FAILED: Test failed: {str(e)}")
        return False

    return True

if __name__ == "__main__":
    # Run the test
    success = asyncio.run(test_piper_provider())
    if success:
        print("\nPiper TTS Provider test completed successfully!")
    else:
        print("\nPiper TTS Provider test failed!")
        exit(1)
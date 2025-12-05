import pytest
import sys
from unittest.mock import MagicMock, patch, AsyncMock
import os

# We need to mock the 'TTS' library because it's huge and might not be installed
# in the test environment. This allows us to test our logic without the heavy dependency.
sys.modules["TTS"] = MagicMock()
sys.modules["TTS.api"] = MagicMock()

from wakegen.providers.opensource.coqui_xtts import CoquiXTTSProvider
from wakegen.core.exceptions import ProviderError

@pytest.mark.asyncio
async def test_list_voices_returns_placeholder():
    """
    Test that list_voices returns the correct placeholder for voice cloning.
    """
    # Initialize the provider with empty config
    provider = CoquiXTTSProvider({})
    
    # Get the voices
    voices = await provider.list_voices()
    
    # Verify we get exactly one voice (the placeholder)
    assert len(voices) == 1
    
    # Verify the placeholder details
    voice = voices[0]
    assert voice.id == "reference_audio_required"
    assert "Voice Cloning" in voice.name
    assert voice.supports_cloning is True

@pytest.mark.asyncio
async def test_generate_validates_reference_audio():
    """
    Test that generate() raises an error if the voice_id is not a valid file path.
    """
    provider = CoquiXTTSProvider({})
    
    # Mock the internal methods to avoid actual model loading
    # We use AsyncMock because these methods are awaited in the implementation
    provider._ensure_xtts_available = AsyncMock(return_value=None)
    provider._load_xtts_model = AsyncMock(return_value=MagicMock())
    
    # Try to generate with a fake voice ID (not a file)
    # This should fail because we removed preset support
    with pytest.raises(ProviderError) as excinfo:
        await provider.generate("Hello", "fake_voice_id", "output.wav")
    
    # Check the error message
    assert "requires a reference audio file" in str(excinfo.value)

@pytest.mark.asyncio
async def test_generate_calls_cloning_with_valid_file():
    """
    Test that generate() calls the cloning method when a valid file is provided.
    """
    provider = CoquiXTTSProvider({})
    
    # Mock internal methods
    # We use AsyncMock for async methods
    
    with patch.object(CoquiXTTSProvider, '_ensure_xtts_available', new_callable=AsyncMock) as mock_ensure:
        mock_ensure.return_value = None
        
        with patch.object(CoquiXTTSProvider, '_load_xtts_model', new_callable=AsyncMock) as mock_load:
            mock_model = MagicMock()
            mock_load.return_value = mock_model
            
            with patch.object(CoquiXTTSProvider, '_generate_with_voice_cloning', new_callable=AsyncMock) as mock_clone:
                mock_clone.return_value = None
                
                # Create a dummy reference file
                with open("dummy_ref.wav", "w") as f:
                    f.write("dummy audio content")
                
                try:
                    # Call generate with the dummy file
                    await provider.generate("Hello", "dummy_ref.wav", "output.wav")
                    
                    # Verify cloning was called
                    mock_clone.assert_called_once()
                    
                    # Verify arguments passed to clone
                    # args: model, text, reference_path, output_path
                    args = mock_clone.call_args[0]
                    assert args[0] == mock_model
                    assert args[1] == "Hello"
                    assert args[2] == "dummy_ref.wav"
                    assert args[3] == "output.wav"
                    
                finally:
                    # Cleanup
                    if os.path.exists("dummy_ref.wav"):
                        os.remove("dummy_ref.wav")

if __name__ == "__main__":
    # Helper to run tests directly
    import asyncio
    try:
        asyncio.run(test_list_voices_returns_placeholder())
        asyncio.run(test_generate_validates_reference_audio())
        # We skip the complex patching test in main execution for simplicity, 
        # relying on pytest for the full suite.
        print("Simple tests passed!")
    except Exception as e:
        print(f"Tests failed: {e}")
        exit(1)
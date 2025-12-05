from __future__ import annotations
import os
import asyncio
import soundfile as sf
from typing import List, Any, Optional, Tuple
from pathlib import Path

from wakegen.core.types import ProviderType, Gender
from wakegen.core.exceptions import ProviderError
from wakegen.providers.base import BaseProvider
from wakegen.providers.registry import register_provider
from wakegen.models.audio import Voice

# We implement the 'BaseProvider' class to create our Kokoro TTS provider.
# Kokoro is a very lightweight (82M params) TTS model that runs fast on CPU.
# It uses ONNX (Open Neural Network Exchange) for efficient inference.

class KokoroTTSProvider(BaseProvider):
    """
    Provider implementation for Kokoro TTS (Open Source).
    Kokoro is an extremely efficient, high-quality TTS system that runs well on CPU.
    It's perfect for devices like Raspberry Pi due to its small size (82M parameters).
    """

    def __init__(self, config: Any):
        """
        Initialize the Kokoro TTS provider.
        We don't load the model here (lazy loading) to save resources until needed.
        """
        super().__init__(config)
        # We'll store the Kokoro instance here once initialized
        self._kokoro: Optional[Any] = None
        # Define the model filename - this will be downloaded automatically by the library
        self._model_filename = "kokoro-v0_19.onnx"
        self._voices_filename = "voices.json"

    @property
    def provider_type(self) -> ProviderType:
        """
        Returns the type of this provider.
        This is used by the registry to identify this provider.
        """
        return ProviderType.KOKORO

    async def _ensure_kokoro_available(self) -> None:
        """
        Check if the kokoro-onnx library is installed.
        If not, we raise a helpful error message telling the user what to install.
        """
        try:
            # Try to import the library to make sure it's installed
            import kokoro_onnx  # noqa: F401
        except ImportError:
            raise ProviderError(
                "Kokoro ONNX library is not installed. "
                "Please install with: pip install kokoro-onnx soundfile"
            )

    async def _get_kokoro_instance(self) -> Any:
        """
        Get or create the Kokoro instance (Lazy Loading).
        
        Lazy loading means we don't load the heavy model into memory until
        we actually need to generate speech. This saves RAM on startup.
        """
        # If we already loaded it, just return it
        if self._kokoro is not None:
            return self._kokoro

        try:
            # Import inside the method to avoid errors if library is missing at top level
            from kokoro_onnx import Kokoro

            # Initialize the model
            # The library handles downloading the model files if they don't exist
            # This might take a moment on the first run
            self._kokoro = Kokoro(self._model_filename, self._voices_filename)
            
            return self._kokoro

        except Exception as e:
            raise ProviderError(f"Failed to initialize Kokoro model: {str(e)}") from e

    async def generate(self, text: str, voice_id: str, output_path: str) -> None:
        """
        Generates audio using Kokoro TTS.

        Args:
            text: The text to speak (e.g., "Hello world").
            voice_id: The ID of the voice to use (e.g., "af_bella").
            output_path: The full path where the audio file should be saved.
        """
        try:
            # 1. Ensure the library is installed
            await self._ensure_kokoro_available()

            # 2. Get the model instance (loads it if not already loaded)
            kokoro = await self._get_kokoro_instance()

            # 3. Generate the audio
            # The create method returns raw audio samples and the sample rate
            # We run this in a thread executor because it's a CPU-bound blocking operation
            # and we don't want to freeze the whole application while generating.
            loop = asyncio.get_running_loop()
            
            # We define a helper function to run in the thread
            def _run_inference():
                return kokoro.create(
                    text=text,
                    voice=voice_id,
                    speed=1.0,
                    lang="en-us"
                )

            # Run the inference in a separate thread
            samples, sample_rate = await loop.run_in_executor(None, _run_inference)

            # 4. Save the audio to a file
            # We use soundfile to write the numpy array to a WAV file
            # This is also a blocking I/O operation, so we run it in a thread
            def _save_audio():
                sf.write(output_path, samples, sample_rate)

            await loop.run_in_executor(None, _save_audio)

        except Exception as e:
            raise ProviderError(f"Kokoro TTS generation failed: {str(e)}") from e

    async def list_voices(self) -> List[Voice]:
        """
        Lists available voices for Kokoro TTS.
        Returns a list of high-quality voices available in the model.
        """
        try:
            # These are the officially supported voices for Kokoro v0.19
            # We hardcode them here because they are part of the model definition
            available_voices = [
                ("af_bella", "American Female - Bella", Gender.FEMALE, "en-US"),
                ("af_nicole", "American Female - Nicole", Gender.FEMALE, "en-US"),
                ("af_sarah", "American Female - Sarah", Gender.FEMALE, "en-US"),
                ("am_adam", "American Male - Adam", Gender.MALE, "en-US"),
                ("am_michael", "American Male - Michael", Gender.MALE, "en-US"),
                ("bf_emma", "British Female - Emma", Gender.FEMALE, "en-GB"),
                ("bf_isabella", "British Female - Isabella", Gender.FEMALE, "en-GB"),
                ("bm_george", "British Male - George", Gender.MALE, "en-GB"),
                ("bm_lewis", "British Male - Lewis", Gender.MALE, "en-GB"),
            ]

            # Convert the raw tuples into our standardized Voice objects
            voice_list = []
            for v_id, v_name, v_gender, v_lang in available_voices:
                voice_list.append(Voice(
                    id=v_id,
                    name=v_name,
                    gender=v_gender,
                    language=v_lang,
                    provider=self.provider_type
                ))

            return voice_list

        except Exception as e:
            raise ProviderError(f"Failed to list Kokoro voices: {str(e)}") from e

    async def validate_config(self) -> None:
        """
        Validate that the provider is correctly configured.
        For Kokoro, we just need to check if the library is installed.
        """
        try:
            await self._ensure_kokoro_available()
        except Exception as e:
            raise ProviderError(f"Kokoro TTS configuration validation failed: {str(e)}") from e

# Register this provider so the factory knows about it
# This line is crucial - without it, the system won't know 'kokoro' exists!
register_provider(ProviderType.KOKORO, KokoroTTSProvider)

async def test_kokoro_provider():
    """
    Simple test function to verify the Kokoro provider works correctly.
    This can be used for manual testing or integration testing.
    """
    try:
        # Create a mock config (Kokoro doesn't need any special config)
        class MockConfig:
            pass

        # Create the provider instance
        provider = KokoroTTSProvider(MockConfig())

        # Test 1: Validate configuration
        print("Testing configuration validation...")
        await provider.validate_config()
        print("✓ Configuration validation passed")

        # Test 2: List available voices
        print("Testing voice listing...")
        voices = await provider.list_voices()
        print(f"✓ Found {len(voices)} voices:")
        for voice in voices:
            print(f"  - {voice.id}: {voice.name} ({voice.language})")

        # Test 3: Generate a simple audio file
        print("Testing audio generation...")
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            await provider.generate("Hello, this is a test of the Kokoro TTS provider.", "af_bella", temp_path)
            print(f"✓ Audio generation successful. File saved to: {temp_path}")

            # Verify the file was created and has content
            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                print("✓ Audio file is valid")
            else:
                print("✗ Audio file is empty or missing")

        except Exception as e:
            print(f"✗ Audio generation failed: {e}")
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)

        print("✓ Kokoro provider test completed successfully!")

    except Exception as e:
        print(f"✗ Kokoro provider test failed: {e}")
        raise

# This allows running the test directly: python -m wakegen.providers.opensource.kokoro
if __name__ == "__main__":
    import asyncio
    asyncio.run(test_kokoro_provider())
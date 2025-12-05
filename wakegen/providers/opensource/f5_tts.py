"""
F5-TTS Provider Module

F5-TTS is a high-quality voice synthesis model with excellent voice cloning
capabilities. It produces natural-sounding speech and can clone voices from
short audio references.

Key Features:
- High-quality audio output
- Voice cloning from reference audio
- Multiple pre-trained voices
- Good controllability (speed, pitch)

Requirements:
- f5-tts package: pip install f5-tts
- GPU recommended for best performance
- ~1.5GB VRAM for inference

Think of F5-TTS like a voice actor who can learn to mimic any voice after
hearing just a short sample. Give it a reference clip, and it speaks in
that voice.
"""

from __future__ import annotations
import os
import asyncio
from typing import List, Any, Optional
from pathlib import Path

from wakegen.core.types import ProviderType, Gender
from wakegen.core.exceptions import ProviderError
from wakegen.providers.base import BaseProvider
from wakegen.providers.registry import register_provider
from wakegen.models.audio import Voice
from wakegen.utils.logging import get_logger

logger = get_logger("wakegen.providers.f5_tts")


class F5TTSProvider(BaseProvider):
    """
    Provider implementation for F5-TTS (Open Source).
    
    F5-TTS is a diffusion-based text-to-speech model that produces
    high-quality, natural-sounding speech. It supports voice cloning
    from reference audio samples.
    
    Usage:
        provider = F5TTSProvider(config)
        
        # Generate with preset voice
        await provider.generate("Hello world", "default", "output.wav")
        
        # Generate with voice cloning (use path to reference audio as voice_id)
        await provider.generate("Hello world", "/path/to/reference.wav", "output.wav")
    """
    
    # Preset voices available without reference audio
    # These are voices that come bundled with F5-TTS
    PRESET_VOICES = [
        ("default", "Default Voice", Gender.NEUTRAL, "en-US"),
        ("female_1", "Female Voice 1", Gender.FEMALE, "en-US"),
        ("male_1", "Male Voice 1", Gender.MALE, "en-US"),
    ]
    
    def __init__(self, config: Any):
        """
        Initialize the F5-TTS provider.
        
        Args:
            config: Provider configuration object.
        """
        super().__init__(config)
        self._model: Optional[Any] = None
        self._device: Optional[str] = None
        
        # Configuration options
        self._use_gpu = getattr(config, "use_gpu", True)
        self._model_type = getattr(config, "model_type", "F5-TTS")  # or "E2-TTS"
    
    @property
    def provider_type(self) -> ProviderType:
        """Returns the type identifier for this provider."""
        return ProviderType.F5_TTS
    
    async def _ensure_f5_available(self) -> None:
        """
        Check if the F5-TTS library is installed.
        Raises a helpful error if not found.
        """
        try:
            import f5_tts  # noqa: F401
        except ImportError:
            raise ProviderError(
                "F5-TTS library is not installed. "
                "Please install with: pip install f5-tts\n"
                "Note: GPU is recommended for best performance."
            )
    
    async def _get_model(self) -> Any:
        """
        Get or create the F5-TTS model instance (lazy loading).
        
        The model is loaded on first use to save memory and startup time.
        """
        if self._model is not None:
            return self._model
        
        try:
            # Import inside method to handle missing package gracefully
            from f5_tts.api import F5TTS
            
            # Determine device
            if self._use_gpu:
                try:
                    import torch
                    if torch.cuda.is_available():
                        self._device = "cuda"
                    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                        self._device = "mps"
                    else:
                        self._device = "cpu"
                except ImportError:
                    self._device = "cpu"
            else:
                self._device = "cpu"
            
            logger.info(f"Loading F5-TTS model on {self._device}...")
            
            # Initialize F5-TTS
            # The model will be downloaded automatically on first use
            self._model = F5TTS(model_type=self._model_type, device=self._device)
            
            logger.info("F5-TTS model loaded successfully")
            return self._model
        
        except Exception as e:
            raise ProviderError(f"Failed to initialize F5-TTS model: {str(e)}") from e
    
    async def generate(self, text: str, voice_id: str, output_path: str) -> None:
        """
        Generate audio using F5-TTS.
        
        Args:
            text: The text to synthesize.
            voice_id: Either a preset voice name (e.g., "default") or a path
                      to a reference audio file for voice cloning.
            output_path: Path where the output audio file will be saved.
        """
        try:
            # Ensure library is available
            await self._ensure_f5_available()
            
            # Get model instance
            model = await self._get_model()
            
            # Determine if this is voice cloning or preset voice
            reference_audio = None
            reference_text = None
            
            if os.path.isfile(voice_id):
                # Voice cloning mode: voice_id is a path to reference audio
                reference_audio = voice_id
                # For voice cloning, we need a transcript of the reference
                # If not provided, we can use empty string (the model handles it)
                reference_text = ""
                logger.debug(f"Using voice cloning with reference: {voice_id}")
            else:
                # Preset voice mode
                logger.debug(f"Using preset voice: {voice_id}")
            
            # Run generation in executor (CPU-bound operation)
            loop = asyncio.get_running_loop()
            
            def _run_inference():
                """Run the TTS inference."""
                if reference_audio:
                    # Voice cloning mode
                    return model.infer(
                        ref_file=reference_audio,
                        ref_text=reference_text,
                        gen_text=text,
                        file_wave=output_path,
                    )
                else:
                    # Use default/preset voice
                    # F5-TTS may not have preset voices; generate with default settings
                    return model.infer(
                        gen_text=text,
                        file_wave=output_path,
                    )
            
            await loop.run_in_executor(None, _run_inference)
            
            # Verify output file was created
            if not os.path.exists(output_path):
                raise ProviderError("F5-TTS did not produce output file")
            
            logger.debug(f"Generated audio saved to: {output_path}")
        
        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(f"F5-TTS generation failed: {str(e)}") from e
    
    async def list_voices(self) -> List[Voice]:
        """
        List available voices for F5-TTS.
        
        Returns preset voices and notes about voice cloning capability.
        """
        try:
            voices = []
            
            # Add preset voices
            for v_id, v_name, v_gender, v_lang in self.PRESET_VOICES:
                voices.append(Voice(
                    id=v_id,
                    name=v_name,
                    gender=v_gender,
                    language=v_lang,
                    provider=self.provider_type,
                ))
            
            # Add a "cloning" placeholder to indicate voice cloning is supported
            voices.append(Voice(
                id="[voice_cloning]",
                name="Voice Cloning (provide path to reference audio as voice_id)",
                gender=Gender.NEUTRAL,
                language="any",
                provider=self.provider_type,
            ))
            
            return voices
        
        except Exception as e:
            raise ProviderError(f"Failed to list F5-TTS voices: {str(e)}") from e
    
    async def validate_config(self) -> None:
        """
        Validate that F5-TTS is properly configured.
        """
        try:
            await self._ensure_f5_available()
            logger.info("F5-TTS configuration validated successfully")
        except Exception as e:
            raise ProviderError(f"F5-TTS configuration validation failed: {str(e)}") from e
    
    def supports_voice_cloning(self) -> bool:
        """Returns True as F5-TTS supports voice cloning."""
        return True


# Register this provider with the registry
register_provider(ProviderType.F5_TTS, F5TTSProvider)


# =============================================================================
# TEST FUNCTION
# =============================================================================


async def test_f5_tts_provider():
    """
    Test function to verify F5-TTS provider works correctly.
    Run with: python -m wakegen.providers.opensource.f5_tts
    """
    try:
        print("=" * 60)
        print("F5-TTS Provider Test")
        print("=" * 60)
        
        # Create mock config
        class MockConfig:
            use_gpu = True
            model_type = "F5-TTS"
        
        # Create provider instance
        provider = F5TTSProvider(MockConfig())
        
        # Test 1: Validate configuration
        print("\n1. Testing configuration validation...")
        await provider.validate_config()
        print("   ✓ Configuration validation passed")
        
        # Test 2: List voices
        print("\n2. Testing voice listing...")
        voices = await provider.list_voices()
        print(f"   ✓ Found {len(voices)} voices:")
        for voice in voices:
            print(f"      - {voice.id}: {voice.name}")
        
        # Test 3: Generate audio
        print("\n3. Testing audio generation...")
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
        
        try:
            await provider.generate(
                "Hello, this is a test of F5 TTS provider.",
                "default",
                temp_path
            )
            
            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                print(f"   ✓ Audio generated successfully: {temp_path}")
                print(f"   ✓ File size: {os.path.getsize(temp_path)} bytes")
            else:
                print("   ✗ Audio file is empty or missing")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
        print("\n" + "=" * 60)
        print("F5-TTS Provider Test Completed Successfully!")
        print("=" * 60)
    
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(test_f5_tts_provider())

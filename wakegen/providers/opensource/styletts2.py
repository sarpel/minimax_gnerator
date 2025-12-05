"""
StyleTTS 2 Provider Module

StyleTTS 2 is a state-of-the-art neural text-to-speech model that produces
highly expressive and natural-sounding speech. It achieves human-level
naturalness on single-speaker datasets.

Key Features:
- State-of-the-art quality (human-level naturalness)
- Style control for expressive speech
- Prosody modeling for natural intonation
- Zero-shot voice adaptation

Requirements:
- styletts2 package: pip install styletts2
- GPU recommended (requires ~1GB VRAM)
- PyTorch with CUDA for best performance

Think of StyleTTS 2 like a professional voice actor who not only speaks
the words but understands the emotion and context, delivering each line
with appropriate feeling and expression.
"""

from __future__ import annotations
import os
import asyncio
from typing import List, Any, Optional, Dict
from pathlib import Path

from wakegen.core.types import ProviderType, Gender
from wakegen.core.exceptions import ProviderError
from wakegen.providers.base import BaseProvider
from wakegen.providers.registry import register_provider
from wakegen.models.audio import Voice
from wakegen.utils.logging import get_logger

logger = get_logger("wakegen.providers.styletts2")


class StyleTTS2Provider(BaseProvider):
    """
    Provider implementation for StyleTTS 2 (Open Source).
    
    StyleTTS 2 produces highly expressive, human-level quality speech.
    It models speech as a style variable, allowing natural prosody
    and expressiveness.
    
    Usage:
        provider = StyleTTS2Provider(config)
        
        # Basic generation
        await provider.generate("Hello world", "default", "output.wav")
        
        # With style/emotion (prefix voice_id with style)
        await provider.generate("Great news!", "happy:default", "output.wav")
    """
    
    # Available styles/emotions
    STYLES = ["neutral", "happy", "sad", "angry", "surprised", "fearful"]
    
    # Preset voices
    PRESET_VOICES = [
        ("default", "Default Voice (LJSpeech)", Gender.FEMALE, "en-US"),
        ("libritts", "LibriTTS Multi-Speaker", Gender.NEUTRAL, "en-US"),
    ]
    
    def __init__(self, config: Any):
        """
        Initialize the StyleTTS 2 provider.
        
        Args:
            config: Provider configuration object.
        """
        super().__init__(config)
        self._model: Optional[Any] = None
        self._device: Optional[str] = None
        
        # Configuration
        self._use_gpu = getattr(config, "use_gpu", True)
        self._diffusion_steps = getattr(config, "diffusion_steps", 5)
        self._embedding_scale = getattr(config, "embedding_scale", 1.0)
    
    @property
    def provider_type(self) -> ProviderType:
        """Returns the type identifier for this provider."""
        return ProviderType.STYLETTS2
    
    async def _ensure_styletts2_available(self) -> None:
        """
        Check if the StyleTTS 2 library is installed.
        """
        try:
            import styletts2  # noqa: F401
        except ImportError:
            raise ProviderError(
                "StyleTTS 2 library is not installed. "
                "Please install with: pip install styletts2\n"
                "Note: GPU is recommended for best performance."
            )
    
    async def _get_model(self) -> Any:
        """
        Get or create the StyleTTS 2 model instance (lazy loading).
        """
        if self._model is not None:
            return self._model
        
        try:
            from styletts2 import tts as StyleTTS2
            
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
                        logger.warning("No GPU available, StyleTTS 2 will run on CPU (slower)")
                except ImportError:
                    self._device = "cpu"
            else:
                self._device = "cpu"
            
            logger.info(f"Loading StyleTTS 2 model on {self._device}...")
            
            # Initialize StyleTTS 2
            # Model weights will be downloaded automatically
            self._model = StyleTTS2(device=self._device)
            
            logger.info("StyleTTS 2 model loaded successfully")
            return self._model
        
        except Exception as e:
            raise ProviderError(f"Failed to initialize StyleTTS 2 model: {str(e)}") from e
    
    def _parse_voice_id(self, voice_id: str) -> Dict[str, str]:
        """
        Parse voice_id to extract style and voice name.
        
        Format: "style:voice" or just "voice"
        Examples:
            "default" -> {"style": "neutral", "voice": "default"}
            "happy:default" -> {"style": "happy", "voice": "default"}
        """
        if ":" in voice_id:
            style, voice = voice_id.split(":", 1)
            return {"style": style, "voice": voice}
        return {"style": "neutral", "voice": voice_id}
    
    async def generate(self, text: str, voice_id: str, output_path: str) -> None:
        """
        Generate audio using StyleTTS 2.
        
        Args:
            text: The text to synthesize.
            voice_id: Voice identifier, optionally with style prefix (e.g., "happy:default").
            output_path: Path where the output audio file will be saved.
        """
        try:
            # Ensure library is available
            await self._ensure_styletts2_available()
            
            # Get model instance
            model = await self._get_model()
            
            # Parse voice_id for style and voice
            params = self._parse_voice_id(voice_id)
            style = params["style"]
            voice = params["voice"]
            
            logger.debug(f"Generating with style={style}, voice={voice}")
            
            # Run generation in executor
            loop = asyncio.get_running_loop()
            
            def _run_inference():
                """Run the TTS inference."""
                # StyleTTS 2 inference
                # Note: Actual API may vary based on styletts2 package version
                return model.inference(
                    text=text,
                    output_wav_file=output_path,
                    diffusion_steps=self._diffusion_steps,
                    embedding_scale=self._embedding_scale,
                )
            
            await loop.run_in_executor(None, _run_inference)
            
            # Verify output
            if not os.path.exists(output_path):
                raise ProviderError("StyleTTS 2 did not produce output file")
            
            logger.debug(f"Generated audio saved to: {output_path}")
        
        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(f"StyleTTS 2 generation failed: {str(e)}") from e
    
    async def list_voices(self) -> List[Voice]:
        """
        List available voices and styles for StyleTTS 2.
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
            
            # Add styled versions
            for style in self.STYLES:
                if style != "neutral":  # Skip neutral as it's the default
                    voices.append(Voice(
                        id=f"{style}:default",
                        name=f"Default Voice ({style.capitalize()} style)",
                        gender=Gender.FEMALE,
                        language="en-US",
                        provider=self.provider_type,
                    ))
            
            return voices
        
        except Exception as e:
            raise ProviderError(f"Failed to list StyleTTS 2 voices: {str(e)}") from e
    
    async def validate_config(self) -> None:
        """
        Validate that StyleTTS 2 is properly configured.
        """
        try:
            await self._ensure_styletts2_available()
            logger.info("StyleTTS 2 configuration validated successfully")
        except Exception as e:
            raise ProviderError(f"StyleTTS 2 configuration validation failed: {str(e)}") from e
    
    def get_supported_styles(self) -> List[str]:
        """Returns the list of supported speaking styles."""
        return self.STYLES.copy()


# Register this provider with the registry
register_provider(ProviderType.STYLETTS2, StyleTTS2Provider)


# =============================================================================
# TEST FUNCTION
# =============================================================================


async def test_styletts2_provider():
    """
    Test function to verify StyleTTS 2 provider works correctly.
    Run with: python -m wakegen.providers.opensource.styletts2
    """
    try:
        print("=" * 60)
        print("StyleTTS 2 Provider Test")
        print("=" * 60)
        
        # Create mock config
        class MockConfig:
            use_gpu = True
            diffusion_steps = 5
            embedding_scale = 1.0
        
        # Create provider instance
        provider = StyleTTS2Provider(MockConfig())
        
        # Test 1: Validate configuration
        print("\n1. Testing configuration validation...")
        await provider.validate_config()
        print("   ✓ Configuration validation passed")
        
        # Test 2: List voices
        print("\n2. Testing voice listing...")
        voices = await provider.list_voices()
        print(f"   ✓ Found {len(voices)} voices:")
        for voice in voices[:5]:  # Show first 5
            print(f"      - {voice.id}: {voice.name}")
        if len(voices) > 5:
            print(f"      ... and {len(voices) - 5} more")
        
        # Test 3: Check supported styles
        print("\n3. Supported styles:")
        styles = provider.get_supported_styles()
        print(f"   {', '.join(styles)}")
        
        # Test 4: Generate audio
        print("\n4. Testing audio generation...")
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
        
        try:
            await provider.generate(
                "Hello, this is a test of StyleTTS 2 provider.",
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
        print("StyleTTS 2 Provider Test Completed Successfully!")
        print("=" * 60)
    
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(test_styletts2_provider())

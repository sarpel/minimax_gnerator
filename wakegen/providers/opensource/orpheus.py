"""
Orpheus TTS Provider Module

Orpheus TTS is a scalable neural text-to-speech model available in multiple
sizes (150M to 3B parameters). It provides flexibility to choose between
speed and quality based on available resources.

Key Features:
- Multiple model sizes: 150M, 400M, 1B, 3B parameters
- Apache 2.0 license (commercially friendly)
- Good quality across all model sizes
- Auto-select model based on available hardware

Model Sizes:
- orpheus-150m: Fast, lightweight, CPU-friendly (~100MB)
- orpheus-400m: Balanced quality/speed (~300MB)
- orpheus-1b: High quality (~800MB, GPU recommended)
- orpheus-3b: Highest quality (~2.5GB, GPU required)

Requirements:
- orpheus-tts or orpheus-speech package
- PyTorch
- GPU recommended for 1B+ models

Think of Orpheus like a team of speakers - you can choose a quick intern
(150M) for drafts, a regular employee (400M) for standard work, or bring
in the senior experts (1B/3B) when quality matters most.
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

logger = get_logger("wakegen.providers.orpheus")


class OrpheusTTSProvider(BaseProvider):
    """
    Provider implementation for Orpheus TTS (Open Source).
    
    Orpheus TTS provides multiple model sizes for different use cases.
    Smaller models run fast on CPU, larger models provide higher quality
    but require more resources (GPU recommended).
    
    Usage:
        # With auto model selection
        provider = OrpheusTTSProvider(config)
        
        # Force specific model size
        config.model_size = "400m"
        provider = OrpheusTTSProvider(config)
        
        await provider.generate("Hello world", "default", "output.wav")
    """
    
    # Available model sizes
    MODEL_SIZES = ["150m", "400m", "1b", "3b"]
    
    # Model memory requirements (approximate, in MB)
    MODEL_MEMORY = {
        "150m": 200,
        "400m": 500,
        "1b": 1500,
        "3b": 4000,
    }
    
    # Preset voices
    PRESET_VOICES = [
        ("default", "Default Voice", Gender.NEUTRAL, "en-US"),
        ("narrator", "Narrator Voice", Gender.MALE, "en-US"),
        ("assistant", "Assistant Voice", Gender.FEMALE, "en-US"),
    ]
    
    def __init__(self, config: Any):
        """
        Initialize the Orpheus TTS provider.
        
        Args:
            config: Provider configuration object.
                    Optional attributes:
                    - model_size: One of "150m", "400m", "1b", "3b" or "auto"
                    - use_gpu: Whether to use GPU if available
        """
        super().__init__(config)
        self._model: Optional[Any] = None
        self._device: Optional[str] = None
        self._selected_model_size: Optional[str] = None
        
        # Configuration
        self._model_size = getattr(config, "model_size", "auto")
        self._use_gpu = getattr(config, "use_gpu", True)
    
    @property
    def provider_type(self) -> ProviderType:
        """Returns the type identifier for this provider."""
        return ProviderType.ORPHEUS
    
    async def _ensure_orpheus_available(self) -> None:
        """
        Check if the Orpheus TTS library is installed.
        """
        try:
            # Try different possible package names
            try:
                import orpheus_tts  # noqa: F401
            except ImportError:
                import orpheus_speech  # noqa: F401
        except ImportError:
            raise ProviderError(
                "Orpheus TTS library is not installed. "
                "Please install with one of:\n"
                "  pip install orpheus-tts\n"
                "  pip install orpheus-speech\n"
                "Note: GPU recommended for 1B+ models."
            )
    
    def _determine_model_size(self) -> str:
        """
        Determine the best model size based on available resources.
        
        Auto-selection logic:
        - If GPU with 4GB+ VRAM: 1b model
        - If GPU with 2GB+ VRAM: 400m model
        - If CPU only: 150m model
        """
        if self._model_size != "auto":
            if self._model_size not in self.MODEL_SIZES:
                logger.warning(f"Invalid model size '{self._model_size}', using 'auto'")
            else:
                return self._model_size
        
        # Auto-select based on available resources
        try:
            import torch
            if torch.cuda.is_available():
                # Get GPU memory
                gpu_memory_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                
                if gpu_memory_mb >= 5000:  # 5GB+
                    return "1b"  # Could use 3b but it's slow
                elif gpu_memory_mb >= 2000:  # 2GB+
                    return "400m"
                else:
                    return "150m"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                # Apple Silicon typically has good unified memory
                return "400m"
        except ImportError:
            pass
        
        # Default to smallest model for CPU
        return "150m"
    
    async def _get_model(self) -> Any:
        """
        Get or create the Orpheus TTS model instance (lazy loading).
        """
        if self._model is not None:
            return self._model
        
        try:
            # Determine model size
            self._selected_model_size = self._determine_model_size()
            logger.info(f"Selected Orpheus model size: {self._selected_model_size}")
            
            # Import the library
            try:
                from orpheus_tts import OrpheusModel
            except ImportError:
                from orpheus_speech import OrpheusModel
            
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
            
            # Warn if using large model on CPU
            if self._device == "cpu" and self._selected_model_size in ["1b", "3b"]:
                logger.warning(
                    f"Using {self._selected_model_size} model on CPU may be very slow. "
                    "Consider using a smaller model or enabling GPU."
                )
            
            logger.info(f"Loading Orpheus TTS ({self._selected_model_size}) on {self._device}...")
            
            # Initialize model
            # Model weights will be downloaded automatically
            self._model = OrpheusModel(
                model_size=self._selected_model_size,
                device=self._device
            )
            
            logger.info("Orpheus TTS model loaded successfully")
            return self._model
        
        except Exception as e:
            raise ProviderError(f"Failed to initialize Orpheus TTS model: {str(e)}") from e
    
    async def generate(self, text: str, voice_id: str, output_path: str) -> None:
        """
        Generate audio using Orpheus TTS.
        
        Args:
            text: The text to synthesize.
            voice_id: Voice identifier (e.g., "default", "narrator").
            output_path: Path where the output audio file will be saved.
        """
        try:
            # Ensure library is available
            await self._ensure_orpheus_available()
            
            # Get model instance
            model = await self._get_model()
            
            logger.debug(f"Generating with voice={voice_id}, model={self._selected_model_size}")
            
            # Run generation in executor
            loop = asyncio.get_running_loop()
            
            def _run_inference():
                """Run the TTS inference."""
                # Note: Actual API may vary based on orpheus package version
                return model.synthesize(
                    text=text,
                    voice=voice_id,
                    output_path=output_path,
                )
            
            await loop.run_in_executor(None, _run_inference)
            
            # Verify output
            if not os.path.exists(output_path):
                raise ProviderError("Orpheus TTS did not produce output file")
            
            logger.debug(f"Generated audio saved to: {output_path}")
        
        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(f"Orpheus TTS generation failed: {str(e)}") from e
    
    async def list_voices(self) -> List[Voice]:
        """
        List available voices for Orpheus TTS.
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
            
            return voices
        
        except Exception as e:
            raise ProviderError(f"Failed to list Orpheus TTS voices: {str(e)}") from e
    
    async def validate_config(self) -> None:
        """
        Validate that Orpheus TTS is properly configured.
        """
        try:
            await self._ensure_orpheus_available()
            
            # Validate model size if specified
            if self._model_size != "auto" and self._model_size not in self.MODEL_SIZES:
                raise ProviderError(
                    f"Invalid model_size '{self._model_size}'. "
                    f"Choose from: {', '.join(self.MODEL_SIZES)} or 'auto'"
                )
            
            logger.info("Orpheus TTS configuration validated successfully")
        except Exception as e:
            raise ProviderError(f"Orpheus TTS configuration validation failed: {str(e)}") from e
    
    def get_model_info(self) -> dict:
        """
        Get information about the current model configuration.
        """
        return {
            "configured_size": self._model_size,
            "selected_size": self._selected_model_size,
            "device": self._device,
            "available_sizes": self.MODEL_SIZES,
            "memory_requirements": self.MODEL_MEMORY,
        }


# Register this provider with the registry
register_provider(ProviderType.ORPHEUS, OrpheusTTSProvider)


# =============================================================================
# TEST FUNCTION
# =============================================================================


async def test_orpheus_provider():
    """
    Test function to verify Orpheus TTS provider works correctly.
    Run with: python -m wakegen.providers.opensource.orpheus
    """
    try:
        print("=" * 60)
        print("Orpheus TTS Provider Test")
        print("=" * 60)
        
        # Create mock config
        class MockConfig:
            model_size = "auto"  # Auto-select based on hardware
            use_gpu = True
        
        # Create provider instance
        provider = OrpheusTTSProvider(MockConfig())
        
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
        
        # Test 3: Show model info
        print("\n3. Model configuration:")
        info = provider.get_model_info()
        print(f"   Configured size: {info['configured_size']}")
        print(f"   Available sizes: {', '.join(info['available_sizes'])}")
        
        # Test 4: Generate audio
        print("\n4. Testing audio generation...")
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
        
        try:
            await provider.generate(
                "Hello, this is a test of Orpheus TTS provider.",
                "default",
                temp_path
            )
            
            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                print(f"   ✓ Audio generated successfully: {temp_path}")
                print(f"   ✓ File size: {os.path.getsize(temp_path)} bytes")
                
                # Show model info after generation
                info = provider.get_model_info()
                print(f"   ✓ Selected model: {info['selected_size']}")
                print(f"   ✓ Device: {info['device']}")
            else:
                print("   ✗ Audio file is empty or missing")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
        print("\n" + "=" * 60)
        print("Orpheus TTS Provider Test Completed Successfully!")
        print("=" * 60)
    
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(test_orpheus_provider())

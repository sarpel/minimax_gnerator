"""
Bark TTS Provider

Bark is an expressive text-to-speech model from Suno that can generate
highly realistic, multilingual speech as well as other audio including
music, background noise, and simple sound effects.

Features:
- Very expressive speech synthesis
- Can generate non-speech audio (laughter, sighs, music)
- Multiple speaker presets
- Multilingual support
- Can add expressions via text markers [laughs], [sighs], etc.

Requirements:
- pip install bark
- GPU recommended for reasonable speed
- ~5GB VRAM for GPU inference
- CPU inference is slow but works

Reference: https://github.com/suno-ai/bark
"""

from __future__ import annotations
import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from wakegen.core.exceptions import ProviderError
from wakegen.core.types import Gender, ProviderType
from wakegen.providers.base import BaseProvider
from wakegen.models.audio import Voice


class BarkProvider(BaseProvider):
    """
    Bark TTS Provider - Expressive speech synthesis from Suno.
    
    Bark can generate highly expressive speech including emotions,
    non-speech sounds (laughter, sighs), and even music. It's particularly
    good for natural-sounding wake words with varied expressions.
    
    Speaker presets are identified by language and speaker number:
    - v2/en_speaker_0 through v2/en_speaker_9 (English)
    - v2/zh_speaker_0 through v2/zh_speaker_9 (Chinese)
    - etc.
    
    You can also add expressions using text markers:
    - [laughs], [clears throat], [sighs], [music]
    - ♪ for singing, ... for hesitation
    """
    
    # Speaker presets organized by language
    SPEAKER_PRESETS: Dict[str, List[str]] = {
        "en": [f"v2/en_speaker_{i}" for i in range(10)],
        "zh": [f"v2/zh_speaker_{i}" for i in range(10)],
        "de": [f"v2/de_speaker_{i}" for i in range(10)],
        "es": [f"v2/es_speaker_{i}" for i in range(10)],
        "fr": [f"v2/fr_speaker_{i}" for i in range(10)],
        "hi": [f"v2/hi_speaker_{i}" for i in range(10)],
        "it": [f"v2/it_speaker_{i}" for i in range(10)],
        "ja": [f"v2/ja_speaker_{i}" for i in range(10)],
        "ko": [f"v2/ko_speaker_{i}" for i in range(10)],
        "pl": [f"v2/pl_speaker_{i}" for i in range(10)],
        "pt": [f"v2/pt_speaker_{i}" for i in range(10)],
        "ru": [f"v2/ru_speaker_{i}" for i in range(10)],
        "tr": [f"v2/tr_speaker_{i}" for i in range(10)],
    }
    
    # Known speaker characteristics (approximate - Bark speakers vary)
    SPEAKER_GENDERS: Dict[str, Gender] = {
        "v2/en_speaker_0": Gender.MALE,
        "v2/en_speaker_1": Gender.MALE,
        "v2/en_speaker_2": Gender.FEMALE,
        "v2/en_speaker_3": Gender.MALE,
        "v2/en_speaker_4": Gender.FEMALE,
        "v2/en_speaker_5": Gender.FEMALE,
        "v2/en_speaker_6": Gender.MALE,
        "v2/en_speaker_7": Gender.FEMALE,
        "v2/en_speaker_8": Gender.MALE,
        "v2/en_speaker_9": Gender.FEMALE,
    }
    
    # Expression markers that can be added to text
    EXPRESSIONS = {
        "laugh": "[laughs]",
        "sigh": "[sighs]",
        "clear_throat": "[clears throat]",
        "gasp": "[gasps]",
        "music": "[music]",
        "hesitate": "...",
        "sing": "♪",
    }
    
    def __init__(
        self,
        use_gpu: bool = True,
        use_small_models: bool = False,
        text_use_gpu: bool = True,
        coarse_use_gpu: bool = True,
        fine_use_gpu: bool = True,
    ) -> None:
        """
        Initialize the Bark provider.
        
        Args:
            use_gpu: Whether to use GPU for inference.
            use_small_models: Use smaller models for faster inference.
            text_use_gpu: Use GPU for text encoding.
            coarse_use_gpu: Use GPU for coarse audio generation.
            fine_use_gpu: Use GPU for fine audio generation.
        """
        super().__init__()
        self._model = None
        self._use_gpu = use_gpu
        self._use_small_models = use_small_models
        self._text_use_gpu = text_use_gpu
        self._coarse_use_gpu = coarse_use_gpu
        self._fine_use_gpu = fine_use_gpu
        self._sample_rate = 24000  # Bark outputs 24kHz audio
    
    @property
    def provider_type(self) -> ProviderType:
        """Return the provider type."""
        return ProviderType.BARK
    
    @property
    def requires_api_key(self) -> bool:
        """Bark does not require an API key."""
        return False
    
    @property
    def supports_streaming(self) -> bool:
        """Bark does not support streaming."""
        return False
    
    async def initialize(self) -> None:
        """Initialize the Bark model."""
        if self._model is not None:
            return
        
        try:
            from bark import preload_models, SAMPLE_RATE
            from bark.generation import (
                COARSE_MODEL,
                FINE_MODEL,
                TEXT_MODEL,
            )
            
            # Set environment variables for model configuration
            os.environ["SUNO_USE_SMALL_MODELS"] = "1" if self._use_small_models else "0"
            
            # Preload models
            await asyncio.to_thread(
                preload_models,
                text_use_gpu=self._text_use_gpu and self._use_gpu,
                coarse_use_gpu=self._coarse_use_gpu and self._use_gpu,
                fine_use_gpu=self._fine_use_gpu and self._use_gpu,
            )
            
            self._sample_rate = SAMPLE_RATE
            self._model = True  # Mark as initialized
            
        except ImportError as e:
            raise ProviderError(
                f"Bark is not installed. Install with: pip install git+https://github.com/suno-ai/bark.git\n"
                f"Original error: {e}"
            )
        except Exception as e:
            raise ProviderError(f"Failed to initialize Bark: {e}")
    
    async def generate(
        self,
        text: str,
        voice_id: str,
        output_path: str,
        **kwargs: Any,
    ) -> None:
        """
        Generate speech using Bark.
        
        Args:
            text: Text to synthesize.
            voice_id: Speaker preset (e.g., "v2/en_speaker_0").
            output_path: Path to save the audio file.
            **kwargs: Additional parameters:
                - expression: Add expression marker (laugh, sigh, etc.)
                - text_temp: Text generation temperature (default: 0.7)
                - waveform_temp: Waveform generation temperature (default: 0.7)
                
        Raises:
            ProviderError: If generation fails.
        """
        await self.initialize()
        
        try:
            from bark import generate_audio
            from scipy.io.wavfile import write as write_wav
            
            # Handle expression marker
            expression = kwargs.get("expression")
            if expression and expression in self.EXPRESSIONS:
                text = f"{self.EXPRESSIONS[expression]} {text}"
            
            # Get generation parameters
            text_temp = kwargs.get("text_temp", 0.7)
            waveform_temp = kwargs.get("waveform_temp", 0.7)
            
            # Generate audio
            audio_array = await asyncio.to_thread(
                generate_audio,
                text,
                history_prompt=voice_id if voice_id else None,
                text_temp=text_temp,
                waveform_temp=waveform_temp,
            )
            
            # Convert to int16 for WAV output
            audio_int16 = (audio_array * 32767).astype(np.int16)
            
            # Save audio
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            write_wav(str(output_path), self._sample_rate, audio_int16)
            
        except Exception as e:
            raise ProviderError(f"Bark generation failed: {e}")
    
    async def list_voices(self, language: Optional[str] = None) -> List[Voice]:
        """
        List available speaker presets.
        
        Args:
            language: Filter by language code (e.g., "en", "zh").
            
        Returns:
            List of available voices.
        """
        voices = []
        
        languages = [language] if language else self.SPEAKER_PRESETS.keys()
        
        for lang in languages:
            if lang not in self.SPEAKER_PRESETS:
                continue
                
            for preset in self.SPEAKER_PRESETS[lang]:
                # Determine gender (default to neutral if unknown)
                gender = self.SPEAKER_GENDERS.get(preset, Gender.NEUTRAL)
                
                # Extract speaker number for name
                speaker_num = preset.split("_")[-1]
                
                voices.append(Voice(
                    voice_id=preset,
                    name=f"Bark {lang.upper()} Speaker {speaker_num}",
                    gender=gender,
                    language=lang,
                    description=f"Bark expressive voice for {lang.upper()}",
                ))
        
        return voices
    
    async def check_availability(self) -> bool:
        """Check if Bark is available."""
        try:
            import bark
            return True
        except ImportError:
            return False
    
    def get_expressions(self) -> Dict[str, str]:
        """
        Get available expression markers.
        
        Returns:
            Dictionary of expression names to markers.
        """
        return self.EXPRESSIONS.copy()
    
    def add_expression_to_text(self, text: str, expression: str) -> str:
        """
        Add an expression marker to text.
        
        Args:
            text: Original text.
            expression: Expression name (laugh, sigh, etc.).
            
        Returns:
            Text with expression marker prepended.
        """
        if expression in self.EXPRESSIONS:
            return f"{self.EXPRESSIONS[expression]} {text}"
        return text

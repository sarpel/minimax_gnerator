from __future__ import annotations
import os
import tempfile
import asyncio
from typing import List, Any, Optional, Dict
from pathlib import Path

from wakegen.core.types import ProviderType, Gender
from wakegen.core.exceptions import ProviderError
from wakegen.providers.base import BaseProvider
from wakegen.providers.registry import register_provider
from wakegen.models.audio import Voice

# We implement the 'BaseProvider' class to create our Coqui XTTS provider.
# Coqui XTTS is a zero-shot voice cloning TTS system with multilingual support including Turkish.

class CoquiXTTSProvider(BaseProvider):
    """
    Provider implementation for Coqui XTTS (Open Source).
    XTTS is a zero-shot voice cloning TTS system that supports multiple languages including Turkish.

    WARNING: HARDWARE REQUIREMENTS
    ------------------------------
    XTTS is a large model and requires significant resources:
    - RAM: Minimum 4GB (8GB+ recommended)
    - GPU: Highly recommended (NVIDIA CUDA). CPU generation is very slow.
    - Storage: ~2GB for model weights.
    
    This provider is NOT recommended for Raspberry Pi Zero or older hardware.
    """

    def __init__(self, config: Any):
        """
        Initialize the Coqui XTTS provider.
        """
        super().__init__(config)
        # We'll store the XTTS model and voice cache
        self._xtts_model: Optional[Any] = None
        self._voice_cache: Dict[str, Any] = {}
        self._model_loaded: bool = False

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.COQUI_XTTS

    async def _ensure_xtts_available(self) -> None:
        """
        Check if Coqui XTTS is available and load the model if needed.
        """
        try:
            # Try to import TTS to ensure it's installed
            import TTS  # noqa: F401
        except ImportError:
            raise ProviderError("Coqui TTS library is not installed. Please install with: pip install TTS")

    async def _load_xtts_model(self) -> Any:
        """
        Load the XTTS model if not already loaded.
        This is resource-intensive, so we cache it.
        """
        if self._model_loaded and self._xtts_model is not None:
            return self._xtts_model

        try:
            from TTS.api import TTS as TTSAPI

            # Load the XTTS model (this may take time and memory)
            # We use the multilingual model which supports Turkish
            self._xtts_model = TTSAPI(model_name="tts_models/multilingual/multi-dataset/xtts_v2")

            self._model_loaded = True
            return self._xtts_model

        except Exception as e:
            raise ProviderError(f"Failed to load XTTS model: {str(e)}") from e

    async def generate(self, text: str, voice_id: str, output_path: str) -> None:
        """
        Generates audio using Coqui XTTS with voice cloning.

        XTTS VOICE CLONING WORKFLOW:
        =============================
        Coqui XTTS is a zero-shot voice cloning system. The 'voice_id' parameter
        must be a PATH to a reference audio file, not a preset name.

        Args:
            text: The text to speak (e.g., "Hey Katya").
            voice_id: REQUIRED - File path to reference audio for voice cloning.
                      Example: "/path/to/reference_voice.wav"
                      
                      The reference audio should be:
                      - Clear speech (minimal background noise)
                      - 5-30 seconds long (optimal)
                      - The language/accent will be preserved in output
                      
                      NOTE: This is NOT a preset name like "tr_female_1".
                      You provide an actual audio file to clone.
                      
            output_path: The full path where the generated audio file should be saved.
                        Will be created as WAV format by XTTS internally.

        Raises:
            ProviderError: If voice_id is not a valid file path or generation fails.

        Example:
            provider = CoquiXTTSProvider(config)
            
            # Clone a Turkish voice
            await provider.generate(
                text="Merhaba, hoşgeldiniz",
                voice_id="/samples/turkish_speaker.wav",
                output_path="/output/generated_tr.wav"
            )
        """
        try:
            # Ensure XTTS is available
            await self._ensure_xtts_available()

            # Load the model
            model = await self._load_xtts_model()

            # VALIDATION: Check if voice_id is a valid file path
            # XTTS is a voice cloning model. It MUST have a reference audio file.
            # It does not support "preset" names like "tr_female_1".
            if not os.path.isfile(voice_id):
                raise ProviderError(
                    f"Coqui XTTS requires a reference audio file for voice cloning.\n"
                    f"The provided voice_id '{voice_id}' is not a valid file path.\n"
                    f"Please provide the full path to an audio file (e.g., '/path/to/voice.wav')."
                )

            # Voice cloning mode - use the reference audio
            await self._generate_with_voice_cloning(model, text, voice_id, output_path)

        except Exception as e:
            raise ProviderError(f"Coqui XTTS generation failed: {str(e)}") from e

    async def _generate_with_voice_cloning(self, model: Any, text: str, reference_audio_path: str, output_path: str) -> None:
        """
        Generate audio using voice cloning from a reference audio file.

        Args:
            model: The loaded XTTS model
            text: The text to speak
            reference_audio_path: Path to reference audio for voice cloning
            output_path: Where to save the generated audio
        """
        try:
            # Create a temporary file for the audio
            # We use a temp file first to ensure the generation completes before writing to the final path
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_path = temp_audio.name

            try:
                # Use XTTS voice cloning
                # The reference audio provides the voice characteristics to clone
                # We explicitly set language to Turkish ('tr') as per requirements
                model.tts_to_file(
                    text=text,
                    speaker_wav=reference_audio_path,
                    language="tr",
                    file_path=temp_path
                )

                # Move the temporary file to the final location
                # os.replace is atomic on POSIX systems
                os.replace(temp_path, output_path)

            except Exception as synth_error:
                # Clean up temp file if synthesis failed to avoid disk clutter
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise ProviderError(f"XTTS voice cloning failed: {str(synth_error)}") from synth_error

        except Exception as e:
            raise ProviderError(f"Voice cloning generation failed: {str(e)}") from e

    async def list_voices(self) -> List[Voice]:
        """
        Lists available voices from Coqui XTTS.
        
        IMPORTANT: Coqui XTTS is a VOICE CLONING model, NOT a preset-based model.
        
        Unlike other TTS providers (e.g., Edge TTS, Piper), XTTS does NOT have:
        - Predefined speaker presets (e.g., "tr_female", "tr_male")
        - Named voice profiles
        - Built-in voice selection
        
        Instead, XTTS works by:
        1. Taking a reference audio file (any voice sample)
        2. Analyzing the speaker characteristics
        3. Cloning that voice for new text
        
        This means you can generate ANY voice you have an audio sample for.
        It's like "voice morphing" - provide a reference, get that voice speaking new text.
        
        Usage:
            # First, provide a reference audio file
            voice_id = "/path/to/turkish_speaker.wav"
            await provider.generate("Hey Katya", voice_id, "output.wav")
        
        Returns:
            A placeholder voice object indicating voice cloning mode.
        """
        try:
            # Return a single placeholder to inform the UI/User that this provider
            # works by cloning, not by selecting a preset.
            return [
                Voice(
                    id="reference_audio_required",
                    name="Voice Cloning (Provide Reference Audio Path)",
                    gender=Gender.NEUTRAL,
                    language="tr-TR",
                    provider=self.provider_type,
                    supports_cloning=True
                )
            ]

        except Exception as e:
            raise ProviderError(f"Failed to list XTTS voices: {str(e)}") from e

    async def validate_config(self) -> None:
        """
        Validate that Coqui XTTS is properly configured.
        """
        try:
            await self._ensure_xtts_available()
            # Try to load the model to ensure everything works
            await self._load_xtts_model()
        except Exception as e:
            raise ProviderError(f"Coqui XTTS configuration validation failed: {str(e)}") from e

    async def clone_voice(self, reference_audio_path: str, output_embedding_path: str) -> str:
        """
        Create a voice embedding from reference audio for voice cloning.

        Args:
            reference_audio_path: Path to reference audio file
            output_embedding_path: Where to save the voice embedding

        Returns:
            Path to the created voice embedding file
        """
        try:
            # Ensure the model is loaded
            model = await self._load_xtts_model()

            # XTTS automatically handles voice embedding when you provide speaker_wav
            # For explicit embedding creation, we can use the model's speaker manager
            # But for simplicity, we'll just verify the reference audio exists
            if not os.path.isfile(reference_audio_path):
                raise ProviderError(f"Reference audio file not found: {reference_audio_path}")

            # Create a simple embedding file (XTTS handles this internally)
            # For our purposes, we'll just copy the reference audio path as the "embedding"
            # In a real implementation, this would extract speaker characteristics
            if not os.path.exists(os.path.dirname(output_embedding_path)):
                os.makedirs(os.path.dirname(output_embedding_path), exist_ok=True)

            # Copy the reference audio as the embedding (simplified approach)
            import shutil
            shutil.copy2(reference_audio_path, output_embedding_path)

            return output_embedding_path

        except Exception as e:
            raise ProviderError(f"Voice cloning failed: {str(e)}") from e

# Register this provider so the factory knows about it
register_provider(ProviderType.COQUI_XTTS, CoquiXTTSProvider)
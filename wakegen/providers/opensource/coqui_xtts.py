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

        Args:
            text: The text to speak (e.g., "Hey Katya").
            voice_id: The ID of the voice to use or path to reference audio for cloning.
            output_path: The full path where the audio file should be saved.
        """
        try:
            # Ensure XTTS is available
            await self._ensure_xtts_available()

            # Load the model
            model = await self._load_xtts_model()

            # Check if voice_id is a path to a reference audio file (voice cloning)
            if os.path.isfile(voice_id):
                # Voice cloning mode - use the reference audio
                await self._generate_with_voice_cloning(model, text, voice_id, output_path)
            else:
                # Use a predefined voice
                await self._generate_with_preset_voice(model, text, voice_id, output_path)

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
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_path = temp_audio.name

            try:
                # Use XTTS voice cloning
                # The reference audio provides the voice characteristics to clone
                model.tts_to_file(
                    text=text,
                    speaker_wav=reference_audio_path,
                    language="tr",  # Turkish
                    file_path=temp_path
                )

                # Move the temporary file to the final location
                os.replace(temp_path, output_path)

            except Exception as synth_error:
                # Clean up temp file if synthesis failed
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise ProviderError(f"XTTS voice cloning failed: {str(synth_error)}") from synth_error

        except Exception as e:
            raise ProviderError(f"Voice cloning generation failed: {str(e)}") from e

    async def _generate_with_preset_voice(self, model: Any, text: str, voice_id: str, output_path: str) -> None:
        """
        Generate audio using a preset voice.

        Args:
            model: The loaded XTTS model
            text: The text to speak
            voice_id: The voice ID to use
            output_path: Where to save the generated audio
        """
        try:
            # For preset voices, we use a generic speaker or try to find the voice
            # XTTS doesn't have traditional voice IDs, so we use language and generic speaker
            speaker = self._get_preset_speaker(voice_id)

            # Create a temporary file for the audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_path = temp_audio.name

            try:
                # Generate with the preset speaker
                model.tts_to_file(
                    text=text,
                    speaker=speaker,
                    language="tr",  # Turkish
                    file_path=temp_path
                )

                # Move the temporary file to the final location
                os.replace(temp_path, output_path)

            except Exception as synth_error:
                # Clean up temp file if synthesis failed
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise ProviderError(f"XTTS preset voice generation failed: {str(synth_error)}") from synth_error

        except Exception as e:
            raise ProviderError(f"Preset voice generation failed: {str(e)}") from e

    def _get_preset_speaker(self, voice_id: str) -> str:
        """
        Map voice IDs to XTTS speaker configurations.
        XTTS uses speaker embeddings rather than traditional voice IDs.
        """
        # For XTTS, we use generic speakers or try to map known configurations
        # The speaker parameter can be a path to a speaker embedding or a preset
        speaker_mapping = {
            "tr_female_1": "tr_female",
            "tr_male_1": "tr_male",
            "tr_neutral_1": "tr_neutral",
            # Default fallback
            "default": "tr_female"
        }

        return speaker_mapping.get(voice_id, speaker_mapping["default"])

    async def list_voices(self) -> List[Voice]:
        """
        Lists available voices from Coqui XTTS.
        Returns a list of voices with Turkish support.
        """
        try:
            # XTTS supports voice cloning, so we provide some preset options
            # plus the ability to clone from any reference audio
            voices = [
                {
                    "id": "tr_female_1",
                    "name": "Turkish Female 1",
                    "gender": "female",
                    "language": "tr-TR",
                    "provider": self.provider_type,
                    "supports_cloning": False
                },
                {
                    "id": "tr_male_1",
                    "name": "Turkish Male 1",
                    "gender": "male",
                    "language": "tr-TR",
                    "provider": self.provider_type,
                    "supports_cloning": False
                },
                {
                    "id": "tr_neutral_1",
                    "name": "Turkish Neutral 1",
                    "gender": "neutral",
                    "language": "tr-TR",
                    "provider": self.provider_type,
                    "supports_cloning": False
                },
                {
                    "id": "voice_cloning",
                    "name": "Voice Cloning (Reference Audio)",
                    "gender": "neutral",
                    "language": "tr-TR",
                    "provider": self.provider_type,
                    "supports_cloning": True
                }
            ]

            # Convert to our Voice model
            voice_list = []
            for v in voices:
                gender = Gender[v["gender"].upper()]  # Convert string to Gender enum
                voice_list.append(Voice(
                    id=v["id"],
                    name=v["name"],
                    gender=gender,
                    language=v["language"],
                    provider=self.provider_type
                ))

            return voice_list

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
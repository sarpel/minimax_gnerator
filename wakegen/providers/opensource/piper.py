from __future__ import annotations

import asyncio
import logging
import os
import wave
from functools import lru_cache
from pathlib import Path
from typing import Any

from wakegen.core.exceptions import ProviderError
from wakegen.core.types import Gender, ProviderType
from wakegen.models.audio import Voice
from wakegen.providers.base import BaseProvider
from wakegen.providers.registry import register_provider

# Logger for debugging Piper operations
logger = logging.getLogger(__name__)

# ============================================================================
# PIPER TTS PROVIDER
# ============================================================================
# Piper is a fast, local, CPU-friendly TTS engine.
# It requires downloading ONNX model files from Hugging Face.
#
# IMPORTANT: PiperVoice.load() requires an ACTUAL FILE PATH to the .onnx model,
# NOT a voice ID string! The voice ID must first be resolved to a downloaded
# model file path.
# ============================================================================

# Default model storage directory
PIPER_MODELS_DIR = Path.home() / ".piper_models"


class PiperTTSProvider(BaseProvider):
    """
    Provider implementation for Piper TTS (Open Source).

    Piper is a fast, local TTS system that works well on CPU and supports Turkish.
    Models are automatically downloaded from Hugging Face on first use.

    Reference: https://github.com/OHF-Voice/piper1-gpl
    """

    def __init__(self, config: Any):
        """
        Initialize the Piper TTS provider.
        Piper doesn't require API keys - just downloads voice models on demand.
        """
        super().__init__(config)
        # Ensure models directory exists
        PIPER_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.PIPER

    async def _ensure_piper_available(self) -> None:
        """
        Check if Piper library is installed.

        Raises:
            ProviderError: If piper-tts is not installed
        """
        try:
            # The package is 'piper-tts' but the import is 'piper'
            import piper  # noqa: F401
        except ImportError:
            raise ProviderError(
                "Piper TTS library is not installed. Please install with: pip install piper-tts"
            )

    def _get_model_path(self, voice_id: str) -> Path:
        """
        Get the local path for a voice model.

        CONCEPT: Voice IDs like 'en_US-lessac-medium' are converted to file paths
        like '~/.piper_models/en_US-lessac-medium.onnx'

        Args:
            voice_id: The voice ID (e.g., "en_US-lessac-medium")

        Returns:
            Path to the .onnx model file
        """
        return PIPER_MODELS_DIR / f"{voice_id}.onnx"

    def _get_config_path(self, voice_id: str) -> Path:
        """
        Get the local path for a voice config file.

        Args:
            voice_id: The voice ID

        Returns:
            Path to the .onnx.json config file
        """
        return PIPER_MODELS_DIR / f"{voice_id}.onnx.json"

    async def _download_model(self, voice_id: str) -> Path:
        """
        Download a Piper voice model from Hugging Face if not already cached.

        CONCEPT: Piper models are hosted on HuggingFace under 'rhasspy/piper-voices'.
        Each voice has two files:
          - {voice_id}.onnx      (the neural network model)
          - {voice_id}.onnx.json (configuration file)

        Args:
            voice_id: The voice ID (e.g., "en_US-lessac-medium")

        Returns:
            Path to the downloaded .onnx model file

        Raises:
            ProviderError: If download fails
        """
        model_path = self._get_model_path(voice_id)
        config_path = self._get_config_path(voice_id)

        # If both files exist, skip download
        if model_path.exists() and config_path.exists():
            logger.debug(f"Piper model already cached: {voice_id}")
            return model_path

        logger.info(f"Downloading Piper voice model: {voice_id}")

        try:
            # Use huggingface_hub for downloading (it's a dependency of piper-tts)
            from huggingface_hub import hf_hub_download

            # Parse voice_id to construct HuggingFace path
            # Format: {language}_{country}-{name}-{quality}
            # Example: en_US-lessac-medium -> en/en_US/lessac/medium/
            parts = voice_id.split("-")
            if len(parts) < 2:
                raise ProviderError(
                    f"Invalid voice ID format: {voice_id}. "
                    f"Expected format: 'language_COUNTRY-name-quality' (e.g., 'en_US-lessac-medium')"
                )

            lang_country = parts[0]  # e.g., "en_US"
            lang = lang_country.split("_")[0]  # e.g., "en"
            name = parts[1] if len(parts) > 1 else "unknown"
            quality = parts[2] if len(parts) > 2 else "medium"

            # Construct the subfolder path for HuggingFace
            # e.g., "en/en_US/lessac/medium/"
            subfolder = f"{lang}/{lang_country}/{name}/{quality}"

            # Download the .onnx model file
            onnx_filename = f"{lang_country}-{name}-{quality}.onnx"
            downloaded_model = hf_hub_download(
                repo_id="rhasspy/piper-voices",
                filename=onnx_filename,
                subfolder=subfolder,
                local_dir=PIPER_MODELS_DIR,
                local_dir_use_symlinks=False,
            )

            # Download the .onnx.json config file
            json_filename = f"{lang_country}-{name}-{quality}.onnx.json"
            downloaded_config = hf_hub_download(
                repo_id="rhasspy/piper-voices",
                filename=json_filename,
                subfolder=subfolder,
                local_dir=PIPER_MODELS_DIR,
                local_dir_use_symlinks=False,
            )

            # Move files to standard locations (flatten the directory structure)
            # huggingface_hub downloads to: {local_dir}/{subfolder}/{filename}
            # We want: {local_dir}/{voice_id}.onnx
            src_model = Path(downloaded_model)
            src_config = Path(downloaded_config)

            if src_model != model_path:
                src_model.rename(model_path)
            if src_config != config_path:
                src_config.rename(config_path)

            logger.info(f"Successfully downloaded Piper model: {voice_id}")
            return model_path

        except ImportError:
            raise ProviderError(
                "huggingface_hub is required for downloading Piper models. "
                "Install with: pip install huggingface_hub"
            )
        except Exception as e:
            raise ProviderError(
                f"Failed to download Piper voice model '{voice_id}': {e!s}"
            ) from e

    async def generate(self, text: str, voice_id: str, output_path: str) -> None:
        """
        Generates audio using Piper TTS.

        HOW IT WORKS:
        1. Ensure Piper library is installed
        2. Download the voice model from HuggingFace if not cached
        3. Load the model using PiperVoice.load(model_path)
        4. Synthesize audio using voice.synthesize_wav()

        Args:
            text: The text to speak (e.g., "Hey Katya").
            voice_id: The ID of the voice to use (e.g., "en_US-lessac-medium").
            output_path: The full path where the audio file should be saved.

        Raises:
            ProviderError: If generation fails
        """
        try:
            # Step 1: Ensure Piper is installed
            await self._ensure_piper_available()

            # Step 2: Download model if needed
            model_path = await self._download_model(voice_id)

            # Step 3: Generate audio using Python API
            await self._generate_with_python_api(text, str(model_path), output_path)

        except ProviderError:
            # Re-raise ProviderErrors as-is
            raise
        except Exception as e:
            raise ProviderError(f"Piper TTS generation failed: {e!s}") from e

    async def _generate_with_python_api(
        self, text: str, model_path: str, output_path: str
    ) -> None:
        """
        Generate audio using Piper Python API.

        IMPORTANT: The official Piper API usage is:
            voice = PiperVoice.load("/path/to/model.onnx")
            with wave.open("output.wav", "wb") as wav_file:
                voice.synthesize_wav("Hello!", wav_file)

        Args:
            text: The text to speak
            model_path: Full path to the .onnx model file
            output_path: Where to save the audio file
        """
        try:
            from piper import PiperVoice

            # Load the voice model from the ONNX file
            # NOTE: This expects a FILE PATH, not a voice ID!
            voice = PiperVoice.load(model_path)

            # Synthesize using the correct API: synthesize_wav with wave.open
            # This is the official way per the documentation
            with wave.open(output_path, "wb") as wav_file:
                voice.synthesize_wav(text, wav_file)

            logger.debug(f"Piper generated audio: {output_path}")

        except Exception as e:
            raise ProviderError(f"Piper Python API generation failed: {e!s}") from e

    async def list_voices(self) -> list[Voice]:
        """
        Lists available voices from Piper TTS.

        CONCEPT: Piper has 100+ voices available on HuggingFace.
        We list popular ones here, especially those supporting Turkish.
        Full list: https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/VOICES.md
        """
        try:
            # Popular Piper voices across languages
            # Format: voice_id maps to HuggingFace path structure
            voices_data = [
                # Turkish voices
                {
                    "id": "tr_TR-dfki-medium",
                    "name": "Turkish Female (DFKI Medium)",
                    "gender": "female",
                    "language": "tr-TR",
                },
                # English voices (popular)
                {
                    "id": "en_US-lessac-medium",
                    "name": "English US Male (Lessac Medium)",
                    "gender": "male",
                    "language": "en-US",
                },
                {
                    "id": "en_US-amy-medium",
                    "name": "English US Female (Amy Medium)",
                    "gender": "female",
                    "language": "en-US",
                },
                {
                    "id": "en_GB-alba-medium",
                    "name": "English GB Female (Alba Medium)",
                    "gender": "female",
                    "language": "en-GB",
                },
                # German voices
                {
                    "id": "de_DE-thorsten-medium",
                    "name": "German Male (Thorsten Medium)",
                    "gender": "male",
                    "language": "de-DE",
                },
                # French voices
                {
                    "id": "fr_FR-siwis-medium",
                    "name": "French Female (Siwis Medium)",
                    "gender": "female",
                    "language": "fr-FR",
                },
                # Spanish voices
                {
                    "id": "es_ES-sharvard-medium",
                    "name": "Spanish Male (Sharvard Medium)",
                    "gender": "male",
                    "language": "es-ES",
                },
            ]

            # Convert to our Voice model
            voice_list = []
            for v in voices_data:
                gender = Gender.FEMALE if v["gender"] == "female" else Gender.MALE
                voice_list.append(
                    Voice(
                        id=v["id"],
                        name=v["name"],
                        gender=gender,
                        language=v["language"],
                        provider=self.provider_type,
                        supports_cloning=False,
                    )
                )

            return voice_list

        except Exception as e:
            raise ProviderError(f"Failed to list Piper voices: {e!s}") from e

    async def validate_config(self) -> None:
        """
        Validate Piper TTS configuration.
        Piper doesn't require API keys, just check the library is installed.
        """
        await self._ensure_piper_available()


# Register this provider so the factory knows about it
register_provider(ProviderType.PIPER, PiperTTSProvider)

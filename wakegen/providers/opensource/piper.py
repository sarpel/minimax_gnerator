from __future__ import annotations
import os
import tempfile
import subprocess
import asyncio
from typing import List, Any, Optional
from pathlib import Path

from wakegen.core.types import ProviderType, Gender
from wakegen.core.exceptions import ProviderError
from wakegen.providers.base import BaseProvider
from wakegen.providers.registry import register_provider
from wakegen.models.audio import Voice

# We implement the 'BaseProvider' class to create our Piper TTS provider.
# Piper is a CPU-friendly, fast inference TTS engine with Turkish language support.

class PiperTTSProvider(BaseProvider):
    """
    Provider implementation for Piper TTS (Open Source).
    Piper is a fast, local TTS system that works well on CPU and supports Turkish.
    """

    def __init__(self, config: Any):
        """
        Initialize the Piper TTS provider.
        Piper doesn't require API keys, so we just need basic configuration.
        """
        super().__init__(config)
        # We'll store the Piper executable path and model cache
        self._piper_executable: Optional[str] = None
        self._model_cache: dict = {}

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.PIPER

    async def _ensure_piper_available(self) -> None:
        """
        Check if Piper is available and download the executable if needed.
        This handles the setup automatically.
        """
        try:
            # Try to import piper-tts to ensure it's installed
            import piper_tts  # noqa: F401
        except ImportError:
            raise ProviderError("Piper TTS library is not installed. Please install with: pip install piper-tts")

        # Check if we can find the piper executable
        try:
            result = subprocess.run(
                ["piper", "--help"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                # Try to download Piper if not available
                self._download_piper_executable()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self._download_piper_executable()

    def _download_piper_executable(self) -> None:
        """
        Download the Piper executable if it's not available.
        This is a fallback method to ensure Piper works.
        """
        try:
            import piper_tts.download
            # Use the official download method
            piper_tts.download.ensure_piper_installed()
        except Exception as e:
            raise ProviderError(f"Failed to download Piper executable: {str(e)}") from e

    async def generate(self, text: str, voice_id: str, output_path: str) -> None:
        """
        Generates audio using Piper TTS.

        Args:
            text: The text to speak (e.g., "Hey Katya").
            voice_id: The ID of the voice to use (e.g., "tr_TR-dfki-medium").
            output_path: The full path where the audio file should be saved.
        """
        try:
            # Ensure Piper is available
            await self._ensure_piper_available()

            # First, try the subprocess-based approach (more reliable)
            try:
                await self._generate_with_subprocess(text, voice_id, output_path)
                return
            except ProviderError as subprocess_error:
                # If subprocess fails, try the Python API as fallback
                try:
                    await self._generate_with_python_api(text, voice_id, output_path)
                    return
                except ProviderError as python_api_error:
                    # If both methods fail, raise the subprocess error as it's more likely to be the primary issue
                    raise ProviderError(f"Piper TTS generation failed with both methods. Subprocess error: {str(subprocess_error)}, Python API error: {str(python_api_error)}")

        except Exception as e:
            raise ProviderError(f"Piper TTS generation failed: {str(e)}") from e

    async def _generate_with_subprocess(self, text: str, voice_id: str, output_path: str) -> None:
        """
        Generate audio using Piper CLI directly (primary method).

        Args:
            text: The text to speak
            voice_id: The voice model ID
            output_path: Where to save the audio file
        """
        try:
            # Build the Piper CLI command - exactly as requested
            cmd = [
                "piper",
                "--model", voice_id,
                "--output_file", output_path
            ]

            # Run Piper CLI as a subprocess
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            # Send the text to Piper via stdin and get the result
            stdout, stderr = await process.communicate(input=text.encode())

            # Check if Piper succeeded - exactly as requested
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                raise ProviderError(f"Piper failed: {error_msg}")

        except FileNotFoundError:
            raise ProviderError("Piper executable not found. Please ensure Piper is installed correctly.")
        except Exception as e:
            raise ProviderError(f"Piper subprocess generation failed: {str(e)}") from e

    async def _generate_with_python_api(self, text: str, voice_id: str, output_path: str) -> None:
        """
        Generate audio using Piper Python API (fallback method).

        Args:
            text: The text to speak
            voice_id: The voice model ID
            output_path: Where to save the audio file
        """
        try:
            # Import Piper modules
            from piper_tts import PiperVoice, synthesize

            # Get or download the voice model
            voice = await self._get_piper_voice(voice_id)

            # Create a temporary file for the audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_path = temp_audio.name

            try:
                # Synthesize the audio
                synthesize(
                    text,
                    temp_path,
                    voice,
                    speaker_id=0,  # Default speaker
                    length_scale=1.0,  # Normal speed
                    noise_scale=0.667,  # Default noise
                    noise_w=0.8  # Default noise weight
                )

                # Move the temporary file to the final location
                os.replace(temp_path, output_path)

            except Exception as synth_error:
                # Clean up temp file if synthesis failed
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise ProviderError(f"Piper TTS synthesis failed: {str(synth_error)}") from synth_error

        except Exception as e:
            raise ProviderError(f"Piper Python API generation failed: {str(e)}") from e

    async def _get_piper_voice(self, voice_id: str) -> Any:
        """
        Get or download a Piper voice model.

        Args:
            voice_id: The voice ID (e.g., "tr_TR-dfki-medium")

        Returns:
            The PiperVoice object ready for synthesis
        """
        # Check if we have this voice cached
        if voice_id in self._model_cache:
            return self._model_cache[voice_id]

        try:
            from piper_tts import PiperVoice

            # Create the voice object (this will download if needed)
            voice = PiperVoice.load(voice_id)

            # Cache the voice for future use
            self._model_cache[voice_id] = voice
            return voice

        except Exception as e:
            raise ProviderError(f"Failed to load Piper voice {voice_id}: {str(e)}") from e

    async def list_voices(self) -> List[Voice]:
        """
        Lists available voices from Piper TTS.
        Returns a list of voices with Turkish support highlighted.
        """
        try:
            # These are known Piper voices that support Turkish
            # Piper doesn't have a built-in voice listing API, so we provide known voices
            turkish_voices = [
                {
                    "id": "tr_TR-dfki-medium",
                    "name": "Turkish Female (DFKI Medium)",
                    "gender": "female",
                    "language": "tr-TR",
                    "provider": self.provider_type
                },
                {
                    "id": "tr_TR-dfki-x_low",
                    "name": "Turkish Female (DFKI X-Low)",
                    "gender": "female",
                    "language": "tr-TR",
                    "provider": self.provider_type
                },
                {
                    "id": "tr_TR-dfki-x_high",
                    "name": "Turkish Female (DFKI X-High)",
                    "gender": "female",
                    "language": "tr-TR",
                    "provider": self.provider_type
                },
                {
                    "id": "tr_TR-dfki-low",
                    "name": "Turkish Female (DFKI Low)",
                    "gender": "female",
                    "language": "tr-TR",
                    "provider": self.provider_type
                },
                {
                    "id": "tr_TR-dfki-high",
                    "name": "Turkish Female (DFKI High)",
                    "gender": "female",
                    "language": "tr-TR",
                    "provider": self.provider_type
                },
                # Additional Turkish voice models added for enhanced support
                {
                    "id": "tr_TR-dfki-fast",
                    "name": "Turkish Female (DFKI Fast)",
                    "gender": "female",
                    "language": "tr-TR",
                    "provider": self.provider_type
                },
                {
                    "id": "tr_TR-dfki-slow",
                    "name": "Turkish Female (DFKI Slow)",
                    "gender": "female",
                    "language": "tr-TR",
                    "provider": self.provider_type
                }
            ]

            # Convert to our Voice model
            voice_list = []
            for v in turkish_voices:
                gender = Gender.FEMALE if v["gender"] == "female" else Gender.MALE
                voice_list.append(Voice(
                    id=v["id"],
                    name=v["name"],
                    gender=gender,
                    language=v["language"],
                    provider=self.provider_type
                ))

            return voice_list

        except Exception as e:
            raise ProviderError(f"Failed to list Piper voices: {str(e)}") from e

    async def validate_config(self) -> None:
        """
        Piper TTS doesn't require API keys, so validation is always successful.
        We just check that the library is available.
        """
        try:
            await self._ensure_piper_available()
        except Exception as e:
            raise ProviderError(f"Piper TTS configuration validation failed: {str(e)}") from e

# Register this provider so the factory knows about it
register_provider(ProviderType.PIPER, PiperTTSProvider)
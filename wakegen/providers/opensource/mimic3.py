from __future__ import annotations
import os
import asyncio
import subprocess
from typing import List, Any, Optional
from pathlib import Path

from wakegen.core.types import ProviderType, Gender
from wakegen.core.exceptions import ProviderError
from wakegen.providers.base import BaseProvider
from wakegen.providers.registry import register_provider
from wakegen.models.audio import Voice

# We implement the 'BaseProvider' class to create our Mimic3 TTS provider.
# Mimic3 is a privacy-friendly, fully offline TTS from Mycroft AI.
# It's designed to work well on embedded systems like Raspberry Pi.

class Mimic3Provider(BaseProvider):
    """
    Provider implementation for Mimic3 TTS (Open Source).
    Mimic3 is a privacy-friendly, fully offline TTS system from Mycroft AI.
    
    Key Features:
    - Privacy-friendly: Runs completely offline, no cloud dependency
    - Raspberry Pi optimized: Works on embedded systems with limited resources
    - Good voice quality: Multiple high-quality voices available
    - Multi-language support: Supports many languages including English
    
    Installation:
        pip install mycroft-mimic3-tts
    
    Ideal for:
    - Privacy-sensitive applications
    - Offline deployments
    - Raspberry Pi projects (Zero/4/5)
    - Edge computing scenarios
    """

    def __init__(self, config: Any):
        """
        Initialize the Mimic3 TTS provider.
        Mimic3 doesn't require API keys, just the CLI tool installed.
        
        Args:
            config: Provider configuration (minimal requirements for Mimic3)
        """
        super().__init__(config)
        # We store whether we've verified mimic3 is available
        self._verified: bool = False

    @property
    def provider_type(self) -> ProviderType:
        """
        Returns the type of this provider.
        This is used by the registry to identify this provider.
        """
        return ProviderType.MIMIC3

    async def _ensure_mimic3_available(self) -> None:
        """
        Check if Mimic3 CLI is available on the system.
        
        This method verifies that the 'mimic3' command can be executed.
        We only check once to avoid repeated subprocess calls.
        
        Raises:
            ProviderError: If mimic3 is not installed or not accessible
        """
        # If we already verified, skip the check
        if self._verified:
            return

        try:
            # Try to run mimic3 with --version flag to verify it's installed
            # We use asyncio.create_subprocess_exec for async compatibility
            process = await asyncio.create_subprocess_exec(
                "mimic3",
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Wait for the process to complete with a timeout
            # If it takes more than 5 seconds, something is wrong
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=5.0
            )
            
            # Check if the command succeeded
            if process.returncode != 0:
                raise ProviderError(
                    "Mimic3 is installed but returned an error. "
                    "Please ensure it's properly configured."
                )
            
            # Mark as verified so we don't check again
            self._verified = True
            
        except FileNotFoundError:
            # The 'mimic3' command doesn't exist on the system
            raise ProviderError(
                "Mimic3 is not installed. "
                "Please install with: pip install mycroft-mimic3-tts"
            )
        except asyncio.TimeoutError:
            raise ProviderError(
                "Mimic3 verification timed out. "
                "Please check your installation."
            )
        except Exception as e:
            raise ProviderError(f"Failed to verify Mimic3 installation: {str(e)}") from e

    async def generate(self, text: str, voice_id: str, output_path: str) -> None:
        """
        Generates audio using Mimic3 TTS via subprocess.
        
        This uses the reliable subprocess approach (like Piper) rather than
        a Python API, ensuring maximum compatibility and stability.
        
        Command format:
            mimic3 --voice <voice_id> "<text>" > output.wav
        
        Args:
            text: The text to speak (e.g., "Hey Katya")
            voice_id: The ID of the voice to use (e.g., "en_US/vctk_low#p226")
            output_path: The full path where the audio file should be saved
            
        Raises:
            ProviderError: If generation fails for any reason
        """
        try:
            # 1. Ensure Mimic3 is available
            await self._ensure_mimic3_available()

            # 2. Sanitize the text to prevent command injection
            # We remove any characters that could cause issues
            safe_text = text.replace('"', '\\"').replace('`', '\\`')

            # 3. Build the Mimic3 CLI command
            # We use --voice to specify the voice model
            # The text is passed via stdin for safety
            cmd = [
                "mimic3",
                "--voice", voice_id,
                "--output-file", output_path  # Direct output to file
            ]

            # 4. Run Mimic3 CLI as a subprocess
            # We create an async subprocess for non-blocking execution
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,   # We'll send text here
                stdout=asyncio.subprocess.PIPE,  # Capture output
                stderr=asyncio.subprocess.PIPE   # Capture errors
            )

            # 5. Send the text to Mimic3 via stdin and wait for completion
            # We encode the text to bytes because subprocess expects bytes
            stdout, stderr = await process.communicate(input=safe_text.encode('utf-8'))

            # 6. Check if Mimic3 succeeded
            # returncode 0 means success, anything else is an error
            if process.returncode != 0:
                error_msg = stderr.decode('utf-8') if stderr else "Unknown error"
                raise ProviderError(f"Mimic3 generation failed: {error_msg}")

            # 7. Verify the output file was created
            # This is a safety check to ensure we actually got audio
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                raise ProviderError(
                    "Mimic3 completed but no audio file was created. "
                    "Please check your voice_id and text."
                )

        except FileNotFoundError:
            raise ProviderError(
                "Mimic3 executable not found. "
                "Ensure it's in your PATH after installation."
            )
        except ProviderError:
            # Re-raise ProviderErrors as-is
            raise
        except Exception as e:
            raise ProviderError(f"Mimic3 TTS generation failed: {str(e)}") from e

    async def list_voices(self) -> List[Voice]:
        """
        Lists available voices for Mimic3 TTS.
        
        Mimic3 supports many voices across multiple languages. We provide
        a curated list of high-quality English voices here.
        
        Voice naming format: <language>/<dataset>_<quality>#<speaker_id>
        Example: en_US/vctk_low#p226
        
        Available quality levels:
        - low: Smallest size, fastest inference (best for Raspberry Pi)
        - medium: Balanced quality and speed
        - high: Best quality, slower inference
        
        Returns:
            List of Voice objects with English voice options
        """
        try:
            # These are known high-quality English voices from Mimic3
            # We focus on en_US voices with various speaker options
            english_voices = [
                # VCTK Dataset - Multiple speakers (low quality for Raspberry Pi)
                ("en_US/vctk_low#p226", "US English Female - P226 (Low)", Gender.FEMALE, "en-US"),
                ("en_US/vctk_low#p227", "US English Male - P227 (Low)", Gender.MALE, "en-US"),
                ("en_US/vctk_low#p228", "US English Female - P228 (Low)", Gender.FEMALE, "en-US"),
                ("en_US/vctk_low#p229", "US English Female - P229 (Low)", Gender.FEMALE, "en-US"),
                ("en_US/vctk_low#p230", "US English Female - P230 (Low)", Gender.FEMALE, "en-US"),
                ("en_US/vctk_low#p231", "US English Female - P231 (Low)", Gender.FEMALE, "en-US"),
                ("en_US/vctk_low#p232", "US English Male - P232 (Low)", Gender.MALE, "en-US"),
                ("en_US/vctk_low#p233", "US English Female - P233 (Low)", Gender.FEMALE, "en-US"),
                ("en_US/vctk_low#p234", "US English Male - P234 (Low)", Gender.MALE, "en-US"),
                ("en_US/vctk_low#p236", "US English Female - P236 (Low)", Gender.FEMALE, "en-US"),
                
                # LJSpeech Dataset - Single female voice (medium quality)
                ("en_US/ljspeech_low", "US English Female - LJSpeech (Low)", Gender.FEMALE, "en-US"),
                
                # CMU Arctic Dataset - Multiple speakers
                ("en_US/cmu-arctic_low#aew", "US English Male - Arctic AEW (Low)", Gender.MALE, "en-US"),
                ("en_US/cmu-arctic_low#ahw", "US English Male - Arctic AHW (Low)", Gender.MALE, "en-US"),
                ("en_US/cmu-arctic_low#aup", "US English Male - Arctic AUP (Low)", Gender.MALE, "en-US"),
                ("en_US/cmu-arctic_low#awb", "US English Male - Arctic AWB (Low)", Gender.MALE, "en-US"),
                ("en_US/cmu-arctic_low#axb", "US English Female - Arctic AXB (Low)", Gender.FEMALE, "en-US"),
                ("en_US/cmu-arctic_low#bdl", "US English Male - Arctic BDL (Low)", Gender.MALE, "en-US"),
                ("en_US/cmu-arctic_low#clb", "US English Female - Arctic CLB (Low)", Gender.FEMALE, "en-US"),
                ("en_US/cmu-arctic_low#eey", "US English Female - Arctic EEY (Low)", Gender.FEMALE, "en-US"),
                ("en_US/cmu-arctic_low#fem", "US English Male - Arctic FEM (Low)", Gender.MALE, "en-US"),
                ("en_US/cmu-arctic_low#gka", "US English Male - Arctic GKA (Low)", Gender.MALE, "en-US"),
                ("en_US/cmu-arctic_low#jmk", "US English Male - Arctic JMK (Low)", Gender.MALE, "en-US"),
                ("en_US/cmu-arctic_low#ksp", "US English Male - Arctic KSP (Low)", Gender.MALE, "en-US"),
                ("en_US/cmu-arctic_low#ljm", "US English Female - Arctic LJM (Low)", Gender.FEMALE, "en-US"),
                ("en_US/cmu-arctic_low#rms", "US English Male - Arctic RMS (Low)", Gender.MALE, "en-US"),
                ("en_US/cmu-arctic_low#rxr", "US English Male - Arctic RXR (Low)", Gender.MALE, "en-US"),
                ("en_US/cmu-arctic_low#slp", "US English Female - Arctic SLP (Low)", Gender.FEMALE, "en-US"),
                ("en_US/cmu-arctic_low#slt", "US English Female - Arctic SLT (Low)", Gender.FEMALE, "en-US"),
            ]

            # Convert the raw tuples into our standardized Voice objects
            voice_list = []
            for v_id, v_name, v_gender, v_lang in english_voices:
                voice_list.append(Voice(
                    id=v_id,
                    name=v_name,
                    gender=v_gender,
                    language=v_lang,
                    provider=self.provider_type
                ))

            return voice_list

        except Exception as e:
            raise ProviderError(f"Failed to list Mimic3 voices: {str(e)}") from e

    async def validate_config(self) -> None:
        """
        Validate that the provider is correctly configured.
        
        For Mimic3, we just need to verify that the CLI tool is available.
        No API keys or other configuration is needed.
        
        Raises:
            ProviderError: If Mimic3 is not installed or accessible
        """
        try:
            await self._ensure_mimic3_available()
        except Exception as e:
            raise ProviderError(f"Mimic3 TTS configuration validation failed: {str(e)}") from e

# Register this provider so the factory knows about it
# This line is crucial - without it, the system won't know 'mimic3' exists!
register_provider(ProviderType.MIMIC3, Mimic3Provider)
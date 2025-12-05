from __future__ import annotations
import os
import asyncio
import logging
import time  # Added import for time module
from typing import List, Dict, Any, Optional
from typing import TYPE_CHECKING

import httpx
from pydantic import BaseModel, Field, field_validator

from wakegen.core.types import ProviderType, Gender
from wakegen.core.exceptions import ProviderError, ConfigError
from wakegen.providers.base import BaseProvider
from wakegen.providers.registry import register_provider
from wakegen.models.audio import Voice

# Set up logging for this module
logger = logging.getLogger("wakegen.minimax")

# We define the MiniMax API configuration and voice settings using Pydantic models.
# This ensures type safety and validation of all parameters.

class MiniMaxVoiceSetting(BaseModel):
    """
    Voice settings for MiniMax TTS API.
    These control the basic characteristics of the generated speech.
    """
    speed: float = Field(
        1.0,
        description="Speech speed (0.5 to 2.0, where 1.0 is normal)",
        ge=0.5,
        le=2.0
    )
    volume: float = Field(
        1.0,
        description="Volume level (0.1 to 10.0, where 1.0 is normal)",
        ge=0.1,
        le=10.0
    )
    pitch: float = Field(
        0.0,
        description="Pitch adjustment in semitones (-12 to +12)",
        ge=-12.0,
        le=12.0
    )

class MiniMaxVoiceModify(BaseModel):
    """
    Advanced voice modification settings for MiniMax TTS API.
    These allow fine-tuning of the voice characteristics.
    """
    pitch: Optional[float] = Field(
        None,
        description="Additional pitch adjustment (semitones)"
    )
    intensity: Optional[float] = Field(
        None,
        description="Voice intensity (emotional strength)"
    )
    timbre: Optional[float] = Field(
        None,
        description="Voice timbre (tone color)"
    )
    sound_effects: Optional[str] = Field(
        None,
        description="Sound effects like 'spacious_echo'"
    )

class MiniMaxAudioSetting(BaseModel):
    """
    Audio output settings for MiniMax TTS API.
    These control the technical characteristics of the generated audio file.
    """
    sample_rate: int = Field(
        16000,
        description="Sample rate in Hz",
        ge=8000,
        le=48000
    )
    format: str = Field(
        "wav",
        description="Audio format (wav, mp3, flac)",
        pattern="^(wav|mp3|flac)$"
    )
    channel: int = Field(
        1,
        description="Number of audio channels (1=mono, 2=stereo)",
        ge=1,
        le=2
    )

class MiniMaxTTSRequest(BaseModel):
    """
    Complete request model for MiniMax TTS API.
    This represents the full payload sent to the MiniMax API endpoint.
    """
    text: str = Field(..., description="Text to synthesize")
    voice_id: str = Field(..., description="Voice identifier")
    voice_setting: MiniMaxVoiceSetting = Field(
        default_factory=MiniMaxVoiceSetting,
        description="Basic voice settings"
    )
    voice_modify: Optional[MiniMaxVoiceModify] = Field(
        None,
        description="Advanced voice modifications"
    )
    audio_setting: MiniMaxAudioSetting = Field(
        default_factory=MiniMaxAudioSetting,
        description="Audio output settings"
    )
    language_boost: Optional[str] = Field(
        None,
        description="Language to boost (e.g., 'Turkish' for better Turkish pronunciation)"
    )

class MiniMaxTTSResponse(BaseModel):
    """
    Response model for MiniMax TTS API.
    This represents the expected response from the MiniMax API.
    """
    success: bool = Field(..., description="Whether the request was successful")
    audio_data: Optional[str] = Field(
        None,
        description="Base64 encoded audio data (if success=True)"
    )
    error: Optional[str] = Field(
        None,
        description="Error message (if success=False)"
    )
    request_id: Optional[str] = Field(
        None,
        description="Request identifier for tracking"
    )

class MiniMaxProvider(BaseProvider):
    """
    MiniMax TTS Provider implementation.
    This is a commercial TTS provider that supports Turkish voices and advanced features.
    """

    def __init__(self, config: Any):
        """
        Initialize the MiniMax provider with configuration.
        """
        super().__init__(config)

        # Validate that we have the required API key
        if not self.config.minimax_api_key:
            raise ConfigError(
                "MiniMax API key is required. "
                "Please set MINIMAX_API_KEY in your environment."
            )

        # Set up HTTP client for async requests
        self.api_key = self.config.minimax_api_key
        self.group_id = self.config.minimax_group_id
        self.base_url = "https://api.minimaxi.chat"  # Fixed: Changed from https://api.minimax.ai to official endpoint
        self.endpoint = "/v1/t2a_v2"

        # Rate limiting configuration - using official MiniMax rate limits
        self.max_requests_per_minute = 60  # Official MiniMax API limit
        self.current_requests = 0
        self.last_reset_time = time.time()

        # Turkish voices configuration
        self.turkish_voices = {
            "Turkish_CalmWoman": {
                "gender": Gender.FEMALE,
                "language": "tr-TR",
                "description": "Calm female Turkish voice"
            },
            "Turkish_Trustworthyman": {
                "gender": Gender.MALE,
                "language": "tr-TR",
                "description": "Trustworthy male Turkish voice"
            }
        }

        # English voices that support Turkish language boost
        self.english_with_turkish_boost = {
            "en-US-Woman": {
                "gender": Gender.FEMALE,
                "language": "en-US",
                "description": "English female voice with Turkish boost support"
            },
            "en-US-Man": {
                "gender": Gender.MALE,
                "language": "en-US",
                "description": "English male voice with Turkish boost support"
            }
        }

    @property
    def provider_type(self) -> ProviderType:
        """
        Returns the type of this provider.
        """
        return ProviderType.MINIMAX

    async def _check_rate_limit(self) -> None:
        """
        Check and enforce rate limiting to avoid hitting API limits.
        This uses a simple token bucket algorithm.
        """
        current_time = time.time()
        time_since_reset = current_time - self.last_reset_time

        # Reset counter if more than 1 minute has passed
        if time_since_reset > 60:
            self.current_requests = 0
            self.last_reset_time = current_time

        # Check if we've hit the limit
        if self.current_requests >= self.max_requests_per_minute:
            wait_time = 60 - time_since_reset
            logger.warning(f"Rate limit reached. Waiting {wait_time:.1f} seconds...")
            await asyncio.sleep(wait_time)
            self.current_requests = 0
            self.last_reset_time = time.time()

        self.current_requests += 1

    async def _make_api_request(self, request_data: MiniMaxTTSRequest) -> MiniMaxTTSResponse:
        """
        Make an HTTP request to the MiniMax TTS API.
        Handles authentication, headers, and error responses.
        """
        try:
            # Enforce rate limiting
            await self._check_rate_limit()

            # Build the full URL
            url = f"{self.base_url}{self.endpoint}"

            # Prepare headers with API key
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }

            # Add group ID if available
            if self.group_id:
                headers["X-Minimax-Group-Id"] = self.group_id

            # Convert request data to dict for JSON serialization
            request_dict = request_data.model_dump(
                exclude_none=True,
                by_alias=True
            )

            logger.debug(f"Making MiniMax API request to {url}")

            # Make the async HTTP request
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    url,
                    headers=headers,
                    json=request_dict
                )

                # Check for HTTP errors
                response.raise_for_status()

                # Parse the JSON response
                response_data = response.json()

                # Validate and convert to our response model
                return MiniMaxTTSResponse(**response_data)

        except httpx.HTTPStatusError as e:
            error_msg = f"MiniMax API HTTP error: {e.response.status_code} - {e.response.text}"
            logger.error(error_msg)
            raise ProviderError(error_msg) from e

        except httpx.RequestError as e:
            error_msg = f"MiniMax API request failed: {str(e)}"
            logger.error(error_msg)
            raise ProviderError(error_msg) from e

        except Exception as e:
            error_msg = f"MiniMax API unexpected error: {str(e)}"
            logger.error(error_msg)
            raise ProviderError(error_msg) from e

    async def generate(self, text: str, voice_id: str, output_path: str) -> None:
        """
        Generates audio from text using MiniMax TTS and saves it to a file.

        Args:
            text: The text to synthesize (e.g., "Hey Katya")
            voice_id: The ID of the voice to use (e.g., "Turkish_CalmWoman")
            output_path: The full path where the audio file should be saved

        Raises:
            ProviderError: If the API request fails or audio generation fails
            ConfigError: If the voice_id is not supported
        """
        try:
            # Validate that the voice is supported
            if voice_id not in self.turkish_voices and voice_id not in self.english_with_turkish_boost:
                available_voices = list(self.turkish_voices.keys()) + list(self.english_with_turkish_boost.keys())
                raise ConfigError(
                    f"Unsupported voice_id: {voice_id}. "
                    f"Available voices: {', '.join(available_voices)}"
                )

            # Determine if we need Turkish language boost
            language_boost = "Turkish" if any(
                voice_id in turkish_voices
                for turkish_voices in [self.turkish_voices, self.english_with_turkish_boost]
                if voice_id in turkish_voices
            ) else None

            # Create the request with default settings
            request = MiniMaxTTSRequest(
                text=text,
                voice_id=voice_id,
                voice_setting=MiniMaxVoiceSetting(),  # Use defaults
                audio_setting=MiniMaxAudioSetting(),  # Use defaults
                language_boost=language_boost
            )

            # Make the API request
            response = await self._make_api_request(request)

            # Check if the request was successful
            if not response.success:
                error_msg = response.error or "Unknown MiniMax API error"
                raise ProviderError(f"MiniMax TTS failed: {error_msg}")

            if not response.audio_data:
                raise ProviderError("No audio data returned from MiniMax API")

            # Decode the base64 audio data and save to file
            import base64
            audio_bytes = base64.b64decode(response.audio_data)

            # Ensure the output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Write the audio data to file
            with open(output_path, "wb") as f:
                f.write(audio_bytes)

            logger.info(f"Successfully generated MiniMax TTS audio: {output_path}")

        except Exception as e:
            # Log the error and re-raise as ProviderError
            error_msg = f"MiniMax TTS generation failed: {str(e)}"
            logger.error(error_msg)
            raise ProviderError(error_msg) from e

    async def list_voices(self) -> List[Voice]:
        """
        Lists available voices from MiniMax TTS.
        Returns both Turkish voices and English voices with Turkish boost support.
        """
        try:
            voice_list = []

            # Add Turkish voices
            for voice_id, voice_info in self.turkish_voices.items():
                voice_list.append(Voice(
                    id=voice_id,
                    name=voice_info["description"],
                    gender=voice_info["gender"],
                    language=voice_info["language"],
                    provider=self.provider_type
                ))

            # Add English voices with Turkish boost
            for voice_id, voice_info in self.english_with_turkish_boost.items():
                voice_list.append(Voice(
                    id=voice_id,
                    name=voice_info["description"] + " (Turkish boost)",
                    gender=voice_info["gender"],
                    language=voice_info["language"],
                    provider=self.provider_type
                ))

            logger.info(f"Listed {len(voice_list)} MiniMax voices")
            return voice_list

        except Exception as e:
            error_msg = f"Failed to list MiniMax voices: {str(e)}"
            logger.error(error_msg)
            raise ProviderError(error_msg) from e

    async def validate_config(self) -> None:
        """
        Validates that the MiniMax provider is correctly configured.
        Checks for API key and tests API connectivity.
        """
        try:
            # Check that API key is present
            if not self.api_key:
                raise ConfigError("MiniMax API key is missing")

            # Test the API connection with a simple request
            # We'll use a minimal request to test connectivity
            test_request = MiniMaxTTSRequest(
                text="Test",
                voice_id="Turkish_CalmWoman",
                voice_setting=MiniMaxVoiceSetting(),
                audio_setting=MiniMaxAudioSetting(),
                language_boost="Turkish"
            )

            # Make a test request (this will also test rate limiting)
            response = await self._make_api_request(test_request)

            if not response.success:
                error_msg = response.error or "MiniMax API validation failed"
                raise ConfigError(f"MiniMax API validation error: {error_msg}")

            logger.info("MiniMax provider configuration validated successfully")

        except Exception as e:
            error_msg = f"MiniMax configuration validation failed: {str(e)}"
            logger.error(error_msg)
            raise ConfigError(error_msg) from e

# Register the MiniMax provider so the factory knows about it
register_provider(ProviderType.MINIMAX, MiniMaxProvider)
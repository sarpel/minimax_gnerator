import edge_tts
from typing import List, Any
from wakegen.core.types import ProviderType, Gender
from wakegen.core.exceptions import ProviderError
from wakegen.providers.base import BaseProvider
from wakegen.providers.registry import register_provider
from wakegen.models.audio import Voice

# We implement the 'BaseProvider' class to create our Edge TTS provider.
# This class handles the communication with the Microsoft Edge TTS service.

class EdgeTTSProvider(BaseProvider):
    """
    Provider implementation for Microsoft Edge TTS (Free).
    """

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.EDGE_TTS

    async def generate(self, text: str, voice_id: str, output_path: str) -> None:
        """
        Generates audio using Edge TTS.
        """
        try:
            # Create the Communicate object with text and voice
            communicate = edge_tts.Communicate(text, voice_id)
            
            # Save the audio to the specified path
            await communicate.save(output_path)
            
        except Exception as e:
            raise ProviderError(f"Edge TTS generation failed: {str(e)}") from e

    async def list_voices(self) -> List[Voice]:
        """
        Lists available voices from Edge TTS.
        """
        try:
            # Get all available voices
            voices = await edge_tts.list_voices()
            
            # Convert them to our internal 'Voice' model
            voice_list = []
            for v in voices:
                # Determine gender (Edge TTS returns "Male" or "Female")
                gender = Gender.MALE if v["Gender"] == "Male" else Gender.FEMALE
                
                voice_list.append(Voice(
                    id=v["ShortName"],
                    name=v.get("FriendlyName", v.get("DisplayName", v["ShortName"])), # Fallback to DisplayName or ShortName
                    gender=gender,
                    language=v["Locale"],
                    provider=self.provider_type
                ))
            return voice_list
            
        except Exception as e:
            raise ProviderError(f"Failed to list Edge TTS voices: {str(e)}") from e

    async def validate_config(self) -> None:
        """
        Edge TTS doesn't require API keys, so validation is always successful.
        """
        pass

# Register this provider so the factory knows about it
register_provider(ProviderType.EDGE_TTS, EdgeTTSProvider)
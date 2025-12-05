from __future__ import annotations
from typing import Protocol, List, TYPE_CHECKING
from wakegen.core.types import ProviderType

if TYPE_CHECKING:
    from wakegen.models.audio import Voice

# We use 'Protocol' to define an interface.
# Think of this as a contract. Any class that claims to be a 'TTSProvider'
# MUST implement the methods defined here.
# This allows us to swap different TTS engines easily.

class TTSProvider(Protocol):
    """
    Interface that all Text-to-Speech providers must implement.
    """

    @property
    def provider_type(self) -> ProviderType:
        """
        Returns the type of this provider (e.g., EDGE_TTS).
        """
        ...

    async def generate(self, text: str, voice_id: str, output_path: str) -> None:
        """
        Generates audio from text and saves it to a file.

        Args:
            text: The text to speak (e.g., "Hey Katya").
            voice_id: The ID of the voice to use.
            output_path: The full path where the audio file should be saved.
        """
        ...

    async def list_voices(self) -> List["Voice"]:
        """
        Returns a list of available voices for this provider.
        """
        ...

    async def validate_config(self) -> None:
        """
        Checks if the provider is correctly configured (e.g., API keys are valid).
        Raises ConfigError if something is wrong.
        """
        ...
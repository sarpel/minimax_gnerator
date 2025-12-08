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

    async def cleanup(self) -> None:
        """
        Release resources held by the provider.
        
        Issue 18: Cleanup protocol for releasing heavy ML models, GPU memory,
        and other resources. Providers with large models should implement this
        to free memory when the provider is no longer needed.
        
        Example:
            provider = get_provider(ProviderType.COQUI_XTTS, config)
            try:
                await provider.generate(...)
            finally:
                await provider.cleanup()  # Release model from memory
        """
        ...
    
    async def health_check(self) -> bool:
        """
        Check if the provider is operational and ready to generate audio.
        
        Issue 19: Health check protocol for monitoring provider status.
        Returns True if the provider can successfully generate audio,
        False otherwise.
        
        This can be used by:
        - Web UI to show provider status
        - Monitoring systems to track availability
        - Load balancers to route requests
        
        Returns:
            True if provider is healthy and operational, False otherwise
            
        Example:
            if await provider.health_check():
                await provider.generate(...)
            else:
                logger.error("Provider is not healthy")
        """
        ...

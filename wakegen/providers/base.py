from abc import ABC, abstractmethod
from typing import List, Any
from wakegen.core.types import ProviderType
from wakegen.models.config import ProviderConfig

# We use 'ABC' (Abstract Base Class) to define a common structure.
# While 'Protocol' defines the interface (what methods must exist),
# 'BaseProvider' can provide some default implementations or shared logic.

class BaseProvider(ABC):
    """
    Abstract base class for TTS providers.
    """
    def __init__(self, config: ProviderConfig):
        """
        Initialize the provider with configuration.
        """
        self.config = config

    @property
    @abstractmethod
    def provider_type(self) -> ProviderType:
        """
        The type of this provider. Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    async def generate(self, text: str, voice_id: str, output_path: str) -> None:
        """
        Generate audio. Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    async def list_voices(self) -> List[Any]:
        """
        List voices. Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    async def validate_config(self) -> None:
        """
        Validate config. Must be implemented by subclasses.
        """
        pass

    async def cleanup(self) -> None:
        """
        Release resources held by the provider.
        
        Issue 18: Default implementation does nothing. Providers with heavy
        models (XTTS, Bark, ChatTTS) should override this to release memory.
        """
        # Default: no cleanup needed for lightweight providers
        pass
    
    async def health_check(self) -> bool:
        """
        Check if provider is operational.
        
        Issue 19: Default implementation validates config. Providers can
        override for more sophisticated checks.
        
        Returns:
            True if provider is healthy, False otherwise
        """
        try:
            await self.validate_config()
            return True
        except Exception:
            return False

from typing import Dict, Type, List
from wakegen.core.types import ProviderType
from wakegen.core.protocols import TTSProvider
from wakegen.core.exceptions import ConfigError
from wakegen.models.config import ProviderConfig

# The registry is a dictionary that maps ProviderType (e.g., "edge_tts")
# to the class that implements it (e.g., EdgeTTSProvider).
_PROVIDER_REGISTRY: Dict[ProviderType, Type[TTSProvider]] = {}

def register_provider(provider_type: ProviderType, provider_class: Type[TTSProvider]) -> None:
    """
    Registers a provider class for a given type.
    """
    _PROVIDER_REGISTRY[provider_type] = provider_class

def get_provider(provider_type: ProviderType, config: ProviderConfig) -> TTSProvider:
    """
    Creates and returns an instance of the requested provider.

    Args:
        provider_type: The type of provider to create.
        config: The configuration to pass to the provider.

    Raises:
        ConfigError: If the provider type is unknown.
    """
    if provider_type not in _PROVIDER_REGISTRY:
        raise ConfigError(f"Unknown provider type: {provider_type}")
    
    provider_class = _PROVIDER_REGISTRY[provider_type]
    # We assume the provider class accepts 'config' in its constructor
    return provider_class(config) # type: ignore

def list_available_providers() -> List[ProviderType]:
    """
    Returns a list of all registered provider types.
    
    This allows the CLI to dynamically show which providers are supported
    without hardcoding the list.
    """
    return list(_PROVIDER_REGISTRY.keys())
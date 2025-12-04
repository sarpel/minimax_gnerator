import pytest
from wakegen.providers.registry import get_provider
from wakegen.core.types import ProviderType
from wakegen.config.settings import ProviderConfig
from wakegen.core.exceptions import ConfigError

# We use 'pytest' for testing. It finds functions starting with 'test_'.

def test_get_provider_edge_tts():
    """
    Tests if we can successfully retrieve the Edge TTS provider.
    """
    # Create a dummy configuration
    config = ProviderConfig()
    
    # Ask the registry for the Edge TTS provider
    provider = get_provider(ProviderType.EDGE_TTS, config)
    
    # Check if we got something back
    assert provider is not None
    
    # Check if it's the right type (we check the class name string to avoid circular imports)
    assert provider.__class__.__name__ == "EdgeTTSProvider"

def test_get_provider_invalid():
    """
    Tests if the registry correctly raises an error for an invalid provider.
    """
    config = ProviderConfig()
    
    # We expect a ConfigError when asking for a non-existent provider
    with pytest.raises(ConfigError):
        get_provider("INVALID_PROVIDER", config) # type: ignore
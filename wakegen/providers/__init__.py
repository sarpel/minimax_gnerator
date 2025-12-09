# This file initializes the 'providers' module.
# This module contains the implementations for different TTS services (like Edge TTS).

# We import the providers here so that they are registered when the package is imported.
from wakegen.providers.free.edge_tts import EdgeTTSProvider

# Import opensource providers
from wakegen.providers.opensource import (
    CoquiXTTSProvider,
    KokoroTTSProvider,
    Mimic3Provider,
    PiperTTSProvider,
)

# Export registry functions for convenience
from wakegen.providers.registry import (
    ProviderInfo,
    check_provider_availability,
    discover_available_providers,
    get_available_provider_types,
    get_provider,
    list_available_providers,
)

# Import commercial providers
try:
    from wakegen.providers.commercial.minimax import MiniMaxProvider
except ImportError:
    # Commercial providers are optional
    pass


__all__ = [
    # Registry functions
    "get_provider",
    "list_available_providers",
    "discover_available_providers",
    "check_provider_availability",
    "get_available_provider_types",
    "ProviderInfo",
    # Provider classes
    "EdgeTTSProvider",
    "PiperTTSProvider",
    "CoquiXTTSProvider",
    "KokoroTTSProvider",
    "Mimic3Provider",
]

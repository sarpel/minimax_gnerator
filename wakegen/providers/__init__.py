# This file initializes the 'providers' module.
# This module contains the implementations for different TTS services (like Edge TTS).

# We import the providers here so that they are registered when the package is imported.
from wakegen.providers.free.edge_tts import EdgeTTSProvider

# Import opensource providers
from wakegen.providers.opensource import PiperTTSProvider, CoquiXTTSProvider, KokoroTTSProvider, Mimic3Provider

# Import commercial providers
try:
    from wakegen.providers.commercial.minimax import MiniMaxProvider
except ImportError:
    # Commercial providers are optional
    pass
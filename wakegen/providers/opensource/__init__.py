# OpenSource TTS Providers Package
#
# This package contains open-source TTS providers like Piper and Coqui XTTS.
# These providers are free to use and don't require API keys.

# Import the provider classes
from .piper import PiperTTSProvider
from .coqui_xtts import CoquiXTTSProvider
from .kokoro import KokoroTTSProvider
from .mimic3 import Mimic3Provider

# Export the providers so they can be imported from this package
__all__ = ["PiperTTSProvider", "CoquiXTTSProvider", "KokoroTTSProvider", "Mimic3Provider"]
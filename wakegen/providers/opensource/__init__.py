# OpenSource TTS Providers Package
#
# This package contains open-source TTS providers like Piper, Coqui XTTS,
# Kokoro, and others. These providers are free to use and don't require API keys.

# Import the provider classes
from .piper import PiperTTSProvider
from .coqui_xtts import CoquiXTTSProvider
from .kokoro import KokoroTTSProvider
from .mimic3 import Mimic3Provider
from .f5_tts import F5TTSProvider
from .styletts2 import StyleTTS2Provider
from .orpheus import OrpheusTTSProvider
from .bark import BarkProvider
from .chattts import ChatTTSProvider

# Export the providers so they can be imported from this package
__all__ = [
    "PiperTTSProvider",
    "CoquiXTTSProvider",
    "KokoroTTSProvider",
    "Mimic3Provider",
    "F5TTSProvider",
    "StyleTTS2Provider",
    "OrpheusTTSProvider",
    "BarkProvider",
    "ChatTTSProvider",
]
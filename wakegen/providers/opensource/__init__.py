# OpenSource TTS Providers Package
#
# This package contains open-source TTS providers like Piper, Coqui XTTS,
# Kokoro, and others. These providers are free to use and don't require API keys.

from .bark import BarkProvider
from .chattts import ChatTTSProvider
from .coqui_docker import CoquiDockerProvider
from .f5_tts import F5TTSProvider
from .kokoro import KokoroTTSProvider
from .mimic3 import Mimic3Provider
from .orpheus import OrpheusTTSProvider

# Import the provider classes
from .piper import PiperTTSProvider
from .styletts2 import StyleTTS2Provider

# Export the providers so they can be imported from this package
__all__ = [
    "BarkProvider",
    "ChatTTSProvider",
    "CoquiDockerProvider",
    "F5TTSProvider",
    "KokoroTTSProvider",
    "Mimic3Provider",
    "OrpheusTTSProvider",
    "PiperTTSProvider",
    "StyleTTS2Provider",
]

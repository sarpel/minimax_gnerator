from enum import Enum, auto

# We use Enums (Enumerations) to define a fixed set of options.
# This prevents typos (like typing "wav" instead of "WAV") and makes the code clearer.

class ProviderType(str, Enum):
    """
    Defines the available Text-to-Speech (TTS) providers.
    Inheriting from 'str' allows these to be used directly as strings.
    """
    EDGE_TTS = "edge_tts"
    MINIMAX = "minimax"  # MiniMax commercial TTS provider
    # Future providers can be added here:
    # ELEVENLABS = "elevenlabs"
    # OPENAI = "openai"
    PIPER = "piper"        # Piper TTS - CPU-friendly, fast inference
    COQUI_XTTS = "coqui_xtts"  # Coqui XTTS - Zero-shot voice cloning
    KOKORO = "kokoro"      # Kokoro TTS - Lightweight (82M), high quality, CPU-friendly
    MIMIC3 = "mimic3"      # Mimic3 TTS - Privacy-friendly, offline, Raspberry Pi compatible
    F5_TTS = "f5_tts"      # F5-TTS - High quality voice synthesis with cloning
    STYLETTS2 = "styletts2"  # StyleTTS 2 - State-of-the-art expressive TTS
    ORPHEUS = "orpheus"    # Orpheus TTS - Scalable (150M-3B), Apache 2.0
    BARK = "bark"          # Bark - Expressive TTS from Suno with non-speech sounds
    CHATTTS = "chattts"    # ChatTTS - Conversational speech optimized

class AudioFormat(str, Enum):
    """
    Defines the supported audio file formats.
    """
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"

class QualityLevel(str, Enum):
    """
    Defines the quality levels for audio generation.
    """
    LOW = "low"       # Good for quick testing
    MEDIUM = "medium" # Standard quality
    HIGH = "high"     # Best quality (might be slower or larger files)

class Gender(str, Enum):
    """
    Defines the gender of the voice.
    """
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"

class AugmentationType(str, Enum):
    """
    Defines the types of audio augmentation that can be applied.
    """
    BACKGROUND_NOISE = "background_noise"
    ROOM_SIMULATION = "room_simulation"
    MICROPHONE_SIMULATION = "microphone_simulation"
    TIME_STRETCH = "time_stretch"
    PITCH_SHIFT = "pitch_shift"
    COMPRESSION = "compression"
    DEGRADATION = "degradation"

class EnvironmentProfile(str, Enum):
    """
    Defines pre-built environment profiles for common scenarios.
    """
    MORNING_KITCHEN = "morning_kitchen"
    EVENING_LIVING_ROOM = "evening_living_room"
    OFFICE_SPACE = "office_space"
    CAR_INTERIOR = "car_interior"
    OUTDOOR_PARK = "outdoor_park"
    BEDROOM_NIGHT = "bedroom_night"
class WakeGenError(Exception):
    """
    Base exception for all errors in the WakeGen application.
    All other custom exceptions will inherit from this one.
    This allows us to catch ANY application-specific error with `except WakeGenError:`.
    """
    pass

class ProviderError(WakeGenError):
    """
    Raised when a TTS provider (like Edge TTS) fails.
    Examples: Network error, API limit reached, invalid voice ID.
    """
    pass

class ConfigError(WakeGenError):
    """
    Raised when there is a problem with the configuration.
    Examples: Missing API key, invalid output directory, unsupported format.
    """
    pass

class AudioError(WakeGenError):
    """
    Raised when there is a problem processing audio.
    Examples: Failed to save file, failed to resample, corrupt audio data.
    """
    pass

class GenerationError(WakeGenError):
    """
    Raised when there is a problem with audio generation.
    Examples: Failed to generate variations, invalid generation parameters,
    checkpoint failure, batch processing error.
    """
    pass

class AugmentationError(WakeGenError):
    """
    Raised when there is a problem with audio augmentation.
    Examples: Failed to apply effect, invalid augmentation parameters,
    unsupported augmentation type.
    """
    pass

class NoiseError(AugmentationError):
    """
    Raised when there is a problem with background noise processing.
    Examples: Failed to load noise file, invalid SNR value.
    """
    pass

class RoomSimulationError(AugmentationError):
    """
    Raised when there is a problem with room simulation.
    Examples: Failed to generate impulse response, invalid room parameters.
    """
    pass

class MicrophoneSimulationError(AugmentationError):
    """
    Raised when there is a problem with microphone simulation.
    Examples: Invalid frequency response, failed to apply EQ.
    """
    pass

class QualityAssuranceError(WakeGenError):
    """
    Raised when there is a problem with quality assurance operations.
    Examples: Validation failure, scoring error, ASR verification failure,
    duplicate detection error, statistics calculation error.
    """
    pass
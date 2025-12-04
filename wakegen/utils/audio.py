import os
import soundfile as sf
import librosa
import numpy as np
from wakegen.core.exceptions import AudioError

# We use 'soundfile' to read and write audio files because it's fast and reliable.
# We use 'librosa' for more complex operations like resampling (changing the speed/pitch).

def save_audio(data: bytes, file_path: str, sample_rate: int = 24000) -> None:
    """
    Saves raw audio bytes to a file.

    Args:
        data: The raw audio data (bytes).
        file_path: The full path where the file should be saved.
        sample_rate: The sample rate of the audio (default 24000Hz for Edge TTS).

    Raises:
        AudioError: If the file cannot be saved.
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Write the bytes to the file
        with open(file_path, "wb") as f:
            f.write(data)
            
    except Exception as e:
        # Wrap any error in our custom AudioError
        raise AudioError(f"Failed to save audio to {file_path}: {str(e)}") from e

def load_audio(file_path: str) -> tuple[np.ndarray, int]:
    """
    Loads an audio file into a numpy array.

    Args:
        file_path: The path to the audio file.

    Returns:
        A tuple containing:
        - The audio data as a numpy array.
        - The sample rate of the audio.

    Raises:
        AudioError: If the file cannot be loaded.
    """
    try:
        # librosa.load returns (data, sample_rate)
        # sr=None means "use the native sample rate of the file"
        data, sr = librosa.load(file_path, sr=None)
        return data, int(sr)
    except Exception as e:
        raise AudioError(f"Failed to load audio from {file_path}: {str(e)}") from e

def resample_audio(file_path: str, target_sr: int = 16000) -> None:
    """
    Resamples an audio file to a target sample rate and overwrites it.
    16000Hz is the standard for most speech recognition models.

    Args:
        file_path: The path to the audio file.
        target_sr: The desired sample rate (default 16000Hz).

    Raises:
        AudioError: If resampling fails.
    """
    try:
        # 1. Load the audio
        y, sr = load_audio(file_path)

        # 2. If the sample rate is already correct, do nothing
        if sr == target_sr:
            return

        # 3. Resample the audio
        y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

        # 4. Save the resampled audio back to the same file
        # soundfile expects the data and the sample rate
        sf.write(file_path, y_resampled, target_sr)

    except Exception as e:
        raise AudioError(f"Failed to resample audio {file_path}: {str(e)}") from e

# Additional functions for quality assurance system
async def load_audio_file(file_path: str) -> tuple[np.ndarray, int]:
    """
    Async wrapper for load_audio function.

    Args:
        file_path: The path to the audio file.

    Returns:
        A tuple containing:
        - The audio data as a numpy array.
        - The sample rate of the audio.
    """
    return load_audio(file_path)

def get_audio_duration(audio_data: np.ndarray, sample_rate: int) -> float:
    """
    Calculate the duration of audio data in seconds.

    Args:
        audio_data: Audio data as numpy array.
        sample_rate: Sample rate in Hz.

    Returns:
        Duration in seconds.
    """
    if len(audio_data.shape) > 1:
        # For multi-channel audio, use first channel for duration calculation
        return len(audio_data[0]) / sample_rate
    else:
        return len(audio_data) / sample_rate
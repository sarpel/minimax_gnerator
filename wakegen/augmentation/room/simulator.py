"""
Room Simulation

This module simulates different room acoustics using impulse responses
and convolution. It creates realistic reverberation effects that mimic
how sound behaves in different physical spaces.

Key Features:
- Room impulse response generation
- Convolution-based reverberation
- Configurable room parameters (size, materials)
- Different room type presets
- Efficient processing for limited hardware
"""

from __future__ import annotations
import numpy as np
import pyroomacoustics as pra
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from wakegen.core.exceptions import RoomSimulationError
from wakegen.utils.audio import load_audio
import soundfile as sf
import warnings

# Suppress pyroomacoustics warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="pyroomacoustics")

@dataclass
class RoomParameters:
    """
    Defines the acoustic parameters of a room.

    Attributes:
        length: Room length in meters.
        width: Room width in meters.
        height: Room height in meters.
        rt60: Reverberation time (T60) in seconds.
        absorption: Wall absorption coefficient (0.0 to 1.0).
        max_order: Maximum reflection order for image method.
        mic_position: Microphone position as (x, y, z).
        source_position: Sound source position as (x, y, z).
    """
    length: float
    width: float
    height: float
    rt60: float
    absorption: float
    max_order: int
    mic_position: Tuple[float, float, float]
    source_position: Tuple[float, float, float]

class RoomSimulator:
    """
    Simulates room acoustics using impulse responses and convolution.

    The simulator uses the image method to generate room impulse responses
    and then applies convolution to create realistic reverberation effects.
    """

    def __init__(self, sample_rate: int = 24000):
        """
        Initialize the room simulator.

        Args:
            sample_rate: Target sample rate for audio processing.
        """
        self.sample_rate = sample_rate
        self._validate_sample_rate(sample_rate)

    def _validate_sample_rate(self, sample_rate: int) -> None:
        """Validate that the sample rate is suitable for room simulation."""
        if not (16000 <= sample_rate <= 48000):
            raise RoomSimulationError(f"Sample rate {sample_rate}Hz is out of valid range for room simulation (16000-48000Hz)")

    def create_room_impulse_response(
        self,
        room_params: RoomParameters
    ) -> np.ndarray:
        """
        Generate a room impulse response using the image method.

        Args:
            room_params: RoomParameters object defining the room characteristics.

        Returns:
            Room impulse response as numpy array.

        Raises:
            RoomSimulationError: If room parameters are invalid or generation fails.
        """
        try:
            # Validate room parameters
            self._validate_room_parameters(room_params)

            # Create room corners array [[x1, y1], [x2, y2], ...]
            corners = np.array([
                [0, 0],
                [room_params.length, 0],
                [room_params.length, room_params.width],
                [0, room_params.width]
            ]).T  # Transpose to get shape (2, 4)

            # Create the room
            room = pra.ShoeBox(
                corners,
                fs=self.sample_rate,
                materials=pra.Material(room_params.absorption),
                max_order=room_params.max_order
            )

            # Add microphone and source
            room.add_microphone_array(
                pra.MicrophoneArray(
                    np.array([room_params.mic_position])[:, np.newaxis],
                    room.fs
                )
            )

            room.add_source(
                room_params.source_position,
                signal=np.array([1.0])  # Impulse signal
            )

            # Compute RIR (Room Impulse Response)
            room.compute_rir()

            # Get the impulse response
            rir = room.rir[0][0]  # Get first microphone, first source

            return rir

        except Exception as e:
            raise RoomSimulationError(f"Failed to generate room impulse response: {str(e)}") from e

    def _validate_room_parameters(self, room_params: RoomParameters) -> None:
        """Validate room parameters are physically reasonable."""
        if room_params.length <= 0 or room_params.width <= 0 or room_params.height <= 0:
            raise RoomSimulationError("Room dimensions must be positive")

        if room_params.rt60 <= 0 or room_params.rt60 > 5.0:
            raise RoomSimulationError("RT60 must be between 0 and 5 seconds")

        if not (0.0 <= room_params.absorption <= 1.0):
            raise RoomSimulationError("Absorption coefficient must be between 0.0 and 1.0")

        if room_params.max_order < 1 or room_params.max_order > 10:
            raise RoomSimulationError("Max reflection order must be between 1 and 10")

        # Check positions are within room bounds
        for pos in [room_params.mic_position, room_params.source_position]:
            if (not 0 <= pos[0] <= room_params.length or
                not 0 <= pos[1] <= room_params.width or
                not 0 <= pos[2] <= room_params.height):
                raise RoomSimulationError(f"Position {pos} is outside room bounds")

    def apply_room_simulation(
        self,
        audio: np.ndarray,
        rir: np.ndarray,
        wet_dry_mix: float = 0.5
    ) -> np.ndarray:
        """
        Apply room simulation to audio using convolution with impulse response.

        Args:
            audio: Input audio signal.
            rir: Room impulse response.
            wet_dry_mix: Mix between wet (processed) and dry (original) signal (0.0 to 1.0).

        Returns:
            Processed audio with room effects.

        Raises:
            RoomSimulationError: If convolution fails or parameters are invalid.
        """
        if not (0.0 <= wet_dry_mix <= 1.0):
            raise RoomSimulationError("Wet/dry mix must be between 0.0 and 1.0")

        try:
            # Perform convolution (this is the computationally intensive part)
            # We use FFT-based convolution for efficiency
            processed = self._fft_convolve(audio, rir)

            # Mix wet and dry signals
            dry_level = np.sqrt(1.0 - wet_dry_mix)
            wet_level = np.sqrt(wet_dry_mix)

            mixed = (audio * dry_level) + (processed * wet_level)

            # Normalize to prevent clipping
            max_val = np.max(np.abs(mixed))
            if max_val > 0:
                mixed = mixed / max_val * 0.95  # 5% headroom

            return mixed

        except Exception as e:
            raise RoomSimulationError(f"Failed to apply room simulation: {str(e)}") from e

    def _fft_convolve(self, signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Perform FFT-based convolution for efficient processing.

        This is much faster than direct convolution for long signals.

        Args:
            signal: Input signal.
            kernel: Impulse response kernel.

        Returns:
            Convolved result.
        """
        # Pad both signals to avoid circular convolution artifacts
        len_signal = len(signal)
        len_kernel = len(kernel)
        total_length = len_signal + len_kernel - 1

        # Next power of 2 for efficient FFT
        fft_size = 1
        while fft_size < total_length:
            fft_size <<= 1

        # FFT of both signals
        fft_signal = np.fft.rfft(signal, n=fft_size)
        fft_kernel = np.fft.rfft(kernel, n=fft_size)

        # Multiply in frequency domain
        fft_result = fft_signal * fft_kernel

        # Inverse FFT
        result = np.fft.irfft(fft_result, n=fft_size)

        # Return only the valid part
        return result[:total_length]

    async def simulate_room_effects(
        self,
        input_path: str,
        output_path: str,
        room_params: RoomParameters,
        wet_dry_mix: float = 0.5
    ) -> None:
        """
        Apply room simulation to an audio file and save the result.

        Args:
            input_path: Path to input audio file.
            output_path: Path to save processed audio.
            room_params: RoomParameters defining the room characteristics.
            wet_dry_mix: Mix between wet and dry signal.

        Raises:
            RoomSimulationError: If room simulation fails.
        """
        try:
            # Load input audio
            audio, sr = load_audio(input_path)

            # Resample if needed
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)

            # Generate room impulse response
            rir = self.create_room_impulse_response(room_params)

            # Apply room simulation
            processed_audio = self.apply_room_simulation(audio, rir, wet_dry_mix)

            # Save result
            sf.write(output_path, processed_audio, self.sample_rate)

        except Exception as e:
            raise RoomSimulationError(f"Failed to simulate room effects: {str(e)}") from e

    def get_preset_room(self, preset_name: str) -> RoomParameters:
        """
        Get pre-configured room parameters for common room types.

        Args:
            preset_name: Name of preset ('small_room', 'medium_room', 'large_room',
                        'bathroom', 'living_room', 'office').

        Returns:
            RoomParameters object.

        Raises:
            RoomSimulationError: If preset name is unknown.
        """
        presets = self._get_room_presets()

        if preset_name not in presets:
            available = list(presets.keys())
            raise RoomSimulationError(f"Unknown room preset '{preset_name}'. Available: {available}")

        return presets[preset_name]

    def _get_room_presets(self) -> Dict[str, RoomParameters]:
        """Define standard room presets with typical parameters."""
        return {
            "small_room": RoomParameters(
                length=3.0, width=2.5, height=2.4,
                rt60=0.3, absorption=0.2,
                max_order=3,
                mic_position=(1.5, 1.25, 1.2),
                source_position=(0.5, 0.5, 1.0)
            ),
            "medium_room": RoomParameters(
                length=5.0, width=4.0, height=2.8,
                rt60=0.5, absorption=0.15,
                max_order=4,
                mic_position=(2.5, 2.0, 1.4),
                source_position=(1.0, 1.0, 1.0)
            ),
            "large_room": RoomParameters(
                length=8.0, width=6.0, height=3.0,
                rt60=0.8, absorption=0.1,
                max_order=5,
                mic_position=(4.0, 3.0, 1.5),
                source_position=(2.0, 2.0, 1.0)
            ),
            "bathroom": RoomParameters(
                length=2.5, width=2.0, height=2.4,
                rt60=1.2, absorption=0.05,  # Hard surfaces, very reverberant
                max_order=4,
                mic_position=(1.25, 1.0, 1.2),
                source_position=(0.5, 0.5, 1.0)
            ),
            "living_room": RoomParameters(
                length=6.0, width=4.5, height=2.6,
                rt60=0.4, absorption=0.2,
                max_order=4,
                mic_position=(3.0, 2.25, 1.3),
                source_position=(1.0, 1.0, 1.0)
            ),
            "office": RoomParameters(
                length=10.0, width=8.0, height=2.8,
                rt60=0.3, absorption=0.3,  # Carpeted, acoustic treatment
                max_order=3,
                mic_position=(5.0, 4.0, 1.4),
                source_position=(2.0, 2.0, 1.0)
            )
        }
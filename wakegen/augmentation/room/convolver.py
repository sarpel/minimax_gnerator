"""
Room Convolver

This module provides efficient convolution operations for applying
room impulse responses to audio signals. It includes optimized
algorithms for real-time capable processing on limited hardware.

Key Features:
- Optimized FFT-based convolution
- Overlap-add processing for long signals
- Memory-efficient processing
- Batch processing capabilities
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple
from wakegen.core.exceptions import RoomSimulationError

class RoomConvolver:
    """
    Efficient convolver for applying room impulse responses to audio.

    Uses optimized FFT-based algorithms with overlap-add processing
    to handle long audio signals efficiently.
    """

    def __init__(self, block_size: int = 4096):
        """
        Initialize the convolver.

        Args:
            block_size: Processing block size for overlap-add method.
        """
        self.block_size = block_size
        self._validate_block_size(block_size)

    def _validate_block_size(self, block_size: int) -> None:
        """Validate that block size is reasonable."""
        if block_size < 256:
            raise RoomSimulationError("Block size too small (minimum 256)")
        if block_size > 16384:
            raise RoomSimulationError("Block size too large (maximum 16384)")
        if not (block_size & (block_size - 1)) == 0:
            raise RoomSimulationError("Block size must be a power of 2 for FFT efficiency")

    def convolve(
        self,
        signal: np.ndarray,
        kernel: np.ndarray,
        wet_dry_mix: float = 1.0
    ) -> np.ndarray:
        """
        Apply convolution using overlap-add method for efficiency.

        Args:
            signal: Input audio signal.
            kernel: Room impulse response kernel.
            wet_dry_mix: Mix between wet (convolved) and dry (original) signal.

        Returns:
            Processed audio signal.

        Raises:
            RoomSimulationError: If convolution fails.
        """
        if not (0.0 <= wet_dry_mix <= 1.0):
            raise RoomSimulationError("Wet/dry mix must be between 0.0 and 1.0")

        try:
            # If kernel is very short, use direct convolution
            if len(kernel) < 128:
                return self._direct_convolve(signal, kernel, wet_dry_mix)
            else:
                return self._overlap_add_convolve(signal, kernel, wet_dry_mix)

        except Exception as e:
            raise RoomSimulationError(f"Convolution failed: {str(e)}") from e

    def _direct_convolve(
        self,
        signal: np.ndarray,
        kernel: np.ndarray,
        wet_dry_mix: float
    ) -> np.ndarray:
        """
        Direct convolution for short kernels.

        Args:
            signal: Input signal.
            kernel: Impulse response kernel.
            wet_dry_mix: Wet/dry mix ratio.

        Returns:
            Convolved signal.
        """
        # Use numpy's convolve for short kernels
        result = np.convolve(signal, kernel, mode='full')

        # Apply wet/dry mix
        dry_level = np.sqrt(1.0 - wet_dry_mix)
        wet_level = np.sqrt(wet_dry_mix)

        # Ensure result is same length as input
        result = result[:len(signal)]

        mixed = (signal * dry_level) + (result * wet_level)

        # Normalize
        max_val = np.max(np.abs(mixed))
        if max_val > 0:
            mixed = mixed / max_val * 0.95

        return mixed

    def _overlap_add_convolve(
        self,
        signal: np.ndarray,
        kernel: np.ndarray,
        wet_dry_mix: float
    ) -> np.ndarray:
        """
        Overlap-add convolution for long kernels.

        This processes the signal in blocks to reduce memory usage.

        Args:
            signal: Input signal.
            kernel: Impulse response kernel.
            wet_dry_mix: Wet/dry mix ratio.

        Returns:
            Convolved signal.
        """
        # Pad kernel to block size for FFT efficiency
        kernel_padded = self._pad_to_length(kernel, self.block_size)

        # Pre-compute kernel FFT
        kernel_fft = np.fft.rfft(kernel_padded)

        # Initialize output
        output = np.zeros_like(signal, dtype=np.float64)
        dry_level = np.sqrt(1.0 - wet_dry_mix)
        wet_level = np.sqrt(wet_dry_mix)

        # Process in blocks
        for i in range(0, len(signal), self.block_size // 2):
            # Get current block
            block_start = i
            block_end = min(i + self.block_size, len(signal))
            block = signal[block_start:block_end]

            # Pad block to block size
            block_padded = self._pad_to_length(block, self.block_size)

            # FFT of block
            block_fft = np.fft.rfft(block_padded)

            # Multiply in frequency domain
            result_fft = block_fft * kernel_fft

            # Inverse FFT
            result = np.fft.irfft(result_fft)

            # Overlap-add
            output_start = block_start
            output_end = min(block_start + len(result), len(output))

            if output_end > output_start:
                output[output_start:output_end] += result[:output_end - output_start]

        # Apply wet/dry mix
        mixed = (signal * dry_level) + (output * wet_level)

        # Normalize
        max_val = np.max(np.abs(mixed))
        if max_val > 0:
            mixed = mixed / max_val * 0.95

        return mixed

    def _pad_to_length(self, array: np.ndarray, target_length: int) -> np.ndarray:
        """
        Pad array to target length with zeros.

        Args:
            array: Input array.
            target_length: Desired length.

        Returns:
            Padded array.
        """
        if len(array) >= target_length:
            return array[:target_length]

        padded = np.zeros(target_length, dtype=array.dtype)
        padded[:len(array)] = array
        return padded

    def batch_convolve(
        self,
        signals: list[np.ndarray],
        kernel: np.ndarray,
        wet_dry_mix: float = 1.0
    ) -> list[np.ndarray]:
        """
        Apply convolution to multiple signals efficiently.

        Args:
            signals: List of input signals.
            kernel: Room impulse response kernel.
            wet_dry_mix: Wet/dry mix ratio.

        Returns:
            List of processed signals.

        Raises:
            RoomSimulationError: If batch processing fails.
        """
        try:
            return [self.convolve(signal, kernel, wet_dry_mix) for signal in signals]
        except Exception as e:
            raise RoomSimulationError(f"Batch convolution failed: {str(e)}") from e

    def get_optimal_block_size(self, signal_length: int, kernel_length: int) -> int:
        """
        Calculate optimal block size for overlap-add processing.

        Args:
            signal_length: Length of input signal.
            kernel_length: Length of impulse response.

        Returns:
            Recommended block size.
        """
        # Block size should be at least kernel length
        min_block = kernel_length

        # Find next power of 2 >= min_block, but <= 16384
        block_size = 256
        while block_size < min_block:
            block_size *= 2
            if block_size > 16384:
                block_size = 16384
                break

        return block_size
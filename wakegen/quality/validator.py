"""Sample Validator Module

This module provides comprehensive validation for audio samples including:
- File integrity checks
- Audio format validation
- Quality metrics analysis
- Technical specifications verification
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field

from wakegen.core.exceptions import QualityAssuranceError
from wakegen.utils.audio import load_audio_file, get_audio_duration

class ValidationError(QualityAssuranceError):
    """Custom exception for validation failures."""

@dataclass
class SampleValidationResult:
    """Result of sample validation with detailed metrics."""

    is_valid: bool
    file_size_bytes: int
    duration_seconds: float
    sample_rate_hz: int
    channels: int
    bit_depth: int
    file_hash: str
    signal_to_noise_ratio_db: Optional[float] = None
    peak_amplitude: Optional[float] = None
    rms_amplitude: Optional[float] = None
    zero_crossing_rate: Optional[float] = None
    error_message: Optional[str] = None

class SampleValidationConfig(BaseModel):
    """Configuration for sample validation."""

    min_duration: float = Field(0.5, description="Minimum duration in seconds")
    max_duration: float = Field(10.0, description="Maximum duration in seconds")
    required_sample_rate: int = Field(16000, description="Required sample rate in Hz")
    min_snr_db: float = Field(15.0, description="Minimum signal-to-noise ratio in dB")
    max_peak_amplitude: float = Field(0.95, description="Maximum peak amplitude (0-1)")
    min_rms_amplitude: float = Field(0.01, description="Minimum RMS amplitude")

async def validate_sample(
    file_path: str | Path,
    config: Optional[SampleValidationConfig] = None
) -> SampleValidationResult:
    """Validate an audio sample with comprehensive quality checks.

    Args:
        file_path: Path to the audio file to validate
        config: Optional validation configuration

    Returns:
        SampleValidationResult with validation metrics

    Raises:
        ValidationError: If validation fails catastrophically
    """
    if config is None:
        config = SampleValidationConfig()

    file_path = Path(file_path)

    # Basic file existence and type validation
    if not file_path.exists():
        return SampleValidationResult(
            is_valid=False,
            file_size_bytes=0,
            duration_seconds=0.0,
            sample_rate_hz=0,
            channels=0,
            bit_depth=0,
            file_hash="",
            error_message=f"File does not exist: {file_path}"
        )

    if not file_path.is_file():
        return SampleValidationResult(
            is_valid=False,
            file_size_bytes=0,
            duration_seconds=0.0,
            sample_rate_hz=0,
            channels=0,
            bit_depth=0,
            file_hash="",
            error_message=f"Path is not a file: {file_path}"
        )

    # File size validation
    file_size = file_path.stat().st_size
    if file_size == 0:
        return SampleValidationResult(
            is_valid=False,
            file_size_bytes=file_size,
            duration_seconds=0.0,
            sample_rate_hz=0,
            channels=0,
            bit_depth=0,
            file_hash="",
            error_message="File is empty"
        )

    # Calculate file hash for integrity
    file_hash = await _calculate_file_hash(file_path)

    try:
        # Load audio file and get basic metadata
        audio_data, sample_rate = await load_audio_file(str(file_path))
        duration = get_audio_duration(audio_data, sample_rate)
        channels = 1 if len(audio_data.shape) == 1 else audio_data.shape[0]
        bit_depth = 16  # Default for most audio formats

        # Duration validation
        if duration < config.min_duration:
            return SampleValidationResult(
                is_valid=False,
                file_size_bytes=file_size,
                duration_seconds=duration,
                sample_rate_hz=sample_rate,
                channels=channels,
                bit_depth=bit_depth,
                file_hash=file_hash,
                error_message=f"Duration {duration:.2f}s is below minimum {config.min_duration}s"
            )

        if duration > config.max_duration:
            return SampleValidationResult(
                is_valid=False,
                file_size_bytes=file_size,
                duration_seconds=duration,
                sample_rate_hz=sample_rate,
                channels=channels,
                bit_depth=bit_depth,
                file_hash=file_hash,
                error_message=f"Duration {duration:.2f}s exceeds maximum {config.max_duration}s"
            )

        # Sample rate validation
        if sample_rate != config.required_sample_rate:
            return SampleValidationResult(
                is_valid=False,
                file_size_bytes=file_size,
                duration_seconds=duration,
                sample_rate_hz=sample_rate,
                channels=channels,
                bit_depth=bit_depth,
                file_hash=file_hash,
                error_message=f"Sample rate {sample_rate}Hz != required {config.required_sample_rate}Hz"
            )

        # Audio quality metrics
        peak_amplitude = np.max(np.abs(audio_data))
        rms_amplitude = np.sqrt(np.mean(audio_data**2))
        zero_crossing_rate = _calculate_zero_crossing_rate(audio_data)
        snr_db = _calculate_signal_to_noise_ratio(audio_data)

        # Quality threshold checks
        validation_errors = []

        if peak_amplitude > config.max_peak_amplitude:
            validation_errors.append(f"Peak amplitude {peak_amplitude:.3f} > {config.max_peak_amplitude}")

        if rms_amplitude < config.min_rms_amplitude:
            validation_errors.append(f"RMS amplitude {rms_amplitude:.6f} < {config.min_rms_amplitude}")

        if snr_db < config.min_snr_db:
            validation_errors.append(f"SNR {snr_db:.2f}dB < {config.min_snr_db}dB")

        # Determine overall validity
        is_valid = len(validation_errors) == 0
        error_message = "; ".join(validation_errors) if validation_errors else None

        return SampleValidationResult(
            is_valid=is_valid,
            file_size_bytes=file_size,
            duration_seconds=duration,
            sample_rate_hz=sample_rate,
            channels=channels,
            bit_depth=bit_depth,
            file_hash=file_hash,
            signal_to_noise_ratio_db=snr_db,
            peak_amplitude=peak_amplitude,
            rms_amplitude=rms_amplitude,
            zero_crossing_rate=zero_crossing_rate,
            error_message=error_message
        )

    except Exception as e:
        raise ValidationError(f"Validation failed for {file_path}: {str(e)}") from e

async def _calculate_file_hash(file_path: Path) -> str:
    """Calculate SHA256 hash of file for integrity verification.

    Args:
        file_path: Path to file to hash

    Returns:
        SHA256 hash as hex string
    """
    hash_sha256 = hashlib.sha256()

    # Read file in chunks to avoid memory issues
    chunk_size = 8192
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hash_sha256.update(chunk)

    return hash_sha256.hexdigest()

def _calculate_zero_crossing_rate(audio_data: np.ndarray) -> float:
    """Calculate zero crossing rate for audio signal.

    Zero crossing rate measures how often the signal changes sign,
    which can indicate speech vs noise characteristics.

    Args:
        audio_data: Audio signal as numpy array

    Returns:
        Zero crossing rate (crossings per second)
    """
    if len(audio_data.shape) > 1:
        # For multi-channel audio, use first channel
        audio_data = audio_data[0]

    # Count sign changes
    sign_changes = np.sum(np.abs(np.diff(np.sign(audio_data)))) / 2
    return sign_changes / len(audio_data)

def _calculate_signal_to_noise_ratio(audio_data: np.ndarray) -> float:
    """Calculate signal-to-noise ratio in decibels.

    Args:
        audio_data: Audio signal as numpy array

    Returns:
        SNR in decibels
    """
    if len(audio_data.shape) > 1:
        # For multi-channel audio, use first channel
        audio_data = audio_data[0]

    # Calculate signal power (RMS)
    signal_power = np.mean(audio_data**2)

    # Estimate noise by taking minimum power in small windows
    window_size = 1024
    num_windows = len(audio_data) // window_size
    noise_power = np.inf

    for i in range(num_windows):
        window = audio_data[i*window_size : (i+1)*window_size]
        window_power = np.mean(window**2)
        if window_power < noise_power:
            noise_power = window_power

    # Avoid division by zero and calculate SNR in dB
    if noise_power == 0:
        return float('inf')

    snr = signal_power / noise_power
    return 10 * np.log10(snr)
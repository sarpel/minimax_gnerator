"""Duplicate Detection Module

This module provides multiple methods for detecting duplicate audio samples:
- Audio fingerprinting
- Spectrogram hashing
- Embedding similarity
"""

from __future__ import annotations

import asyncio
import hashlib
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity

from wakegen.core.exceptions import QualityAssuranceError
from wakegen.utils.audio import load_audio_file

# Suppress scikit-learn warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

class DeduplicationError(QualityAssuranceError):
    """Custom exception for deduplication failures."""

@dataclass
class DuplicateDetectionResult:
    """Result of duplicate detection analysis."""

    is_duplicate: bool
    similarity_score: float
    method_used: str
    reference_file: Optional[str] = None
    comparison_metrics: Optional[Dict[str, float]] = None
    error_message: Optional[str] = None

class DeduplicationConfig(BaseModel):
    """Configuration for duplicate detection."""

    # Similarity thresholds (0-1)
    fingerprint_threshold: float = Field(0.95, description="Fingerprint similarity threshold")
    spectrogram_threshold: float = Field(0.90, description="Spectrogram hash similarity threshold")
    embedding_threshold: float = Field(0.85, description="Embedding similarity threshold")

    # Processing parameters
    fingerprint_window_size: int = Field(1024, description="Window size for fingerprinting")
    spectrogram_bins: int = Field(128, description="Spectrogram frequency bins")
    embedding_dimension: int = Field(64, description="Embedding dimension for similarity")

    # Performance
    max_files_in_memory: int = Field(1000, description="Maximum files to keep in memory cache")

async def detect_duplicates(
    target_file_path: str | Path,
    reference_files: List[str | Path],
    config: Optional[DeduplicationConfig] = None
) -> List[DuplicateDetectionResult]:
    """Detect duplicates of target file among reference files using multiple methods.

    Args:
        target_file_path: Path to target audio file
        reference_files: List of reference audio files to compare against
        config: Optional deduplication configuration

    Returns:
        List of DuplicateDetectionResult for each reference file

    Raises:
        DeduplicationError: If duplicate detection fails catastrophically
    """
    if config is None:
        config = DeduplicationConfig()

    target_file_path = Path(target_file_path)

    if not target_file_path.exists():
        raise DeduplicationError(f"Target file does not exist: {target_file_path}")

    # Load target file
    try:
        target_audio, target_sample_rate = await load_audio_file(str(target_file_path))
    except Exception as e:
        raise DeduplicationError(f"Failed to load target file: {str(e)}") from e

    results = []

    for ref_file in reference_files:
        ref_file = Path(ref_file)

        if not ref_file.exists():
            results.append(DuplicateDetectionResult(
                is_duplicate=False,
                similarity_score=0.0,
                method_used="file_check",
                reference_file=str(ref_file),
                error_message="Reference file does not exist"
            ))
            continue

        try:
            # Try multiple detection methods in order of increasing computational cost
            result = await _detect_duplicates_single(
                target_audio, target_sample_rate,
                ref_file, config
            )
            results.append(result)

        except Exception as e:
            results.append(DuplicateDetectionResult(
                is_duplicate=False,
                similarity_score=0.0,
                method_used="error",
                reference_file=str(ref_file),
                error_message=f"Detection failed: {str(e)}"
            ))

    return results

async def _detect_duplicates_single(
    target_audio: np.ndarray,
    target_sample_rate: int,
    reference_file: Path,
    config: DeduplicationConfig
) -> DuplicateDetectionResult:
    """Detect duplicates between target and single reference file.

    Args:
        target_audio: Target audio data
        target_sample_rate: Target sample rate
        reference_file: Reference file path
        config: Deduplication configuration

    Returns:
        DuplicateDetectionResult
    """
    # Load reference file
    ref_audio, ref_sample_rate = await load_audio_file(str(reference_file))

    # Ensure same sample rate for comparison
    if target_sample_rate != ref_sample_rate:
        # Simple resampling for comparison (not production quality)
        if target_sample_rate > ref_sample_rate:
            # Upsample reference
            ratio = target_sample_rate / ref_sample_rate
            ref_audio = _simple_resample(ref_audio, ratio)
        else:
            # Downsample target
            ratio = ref_sample_rate / target_sample_rate
            target_audio = _simple_resample(target_audio, ratio)

    # Try methods in order of computational efficiency
    methods = [
        ("fingerprint", _detect_fingerprint_duplicates),
        ("spectrogram", _detect_spectrogram_duplicates),
        ("embedding", _detect_embedding_duplicates)
    ]

    for method_name, method_func in methods:
        try:
            result = await method_func(target_audio, ref_audio, config)
            return result
        except Exception:
            # Try next method if this one fails
            continue

    # All methods failed
    return DuplicateDetectionResult(
        is_duplicate=False,
        similarity_score=0.0,
        method_used="all_failed",
        reference_file=str(reference_file),
        error_message="All detection methods failed"
    )

async def _detect_fingerprint_duplicates(
    target_audio: np.ndarray,
    ref_audio: np.ndarray,
    config: DeduplicationConfig
) -> DuplicateDetectionResult:
    """Detect duplicates using audio fingerprinting.

    Audio fingerprinting creates robust hashes of audio content.

    Args:
        target_audio: Target audio data
        ref_audio: Reference audio data
        config: Deduplication configuration

    Returns:
        DuplicateDetectionResult
    """
    # Generate fingerprints
    target_fingerprint = _generate_audio_fingerprint(target_audio, config)
    ref_fingerprint = _generate_audio_fingerprint(ref_audio, config)

    # Calculate similarity
    similarity = _calculate_fingerprint_similarity(target_fingerprint, ref_fingerprint)

    # Determine if duplicate
    is_duplicate = similarity >= config.fingerprint_threshold

    return DuplicateDetectionResult(
        is_duplicate=is_duplicate,
        similarity_score=similarity,
        method_used="fingerprint",
        comparison_metrics={
            "fingerprint_similarity": similarity,
            "fingerprint_length": len(target_fingerprint)
        }
    )

def _generate_audio_fingerprint(audio_data: np.ndarray, config: DeduplicationConfig) -> List[str]:
    """Generate robust audio fingerprint from audio data.

    Uses spectro-temporal features with hashing for robustness.

    Args:
        audio_data: Audio signal
        config: Deduplication configuration

    Returns:
        List of fingerprint hashes
    """
    if len(audio_data.shape) > 1:
        # Use first channel for mono fingerprinting
        audio_data = audio_data[0]

    fingerprints = []
    window_size = config.fingerprint_window_size
    step_size = window_size // 2

    for i in range(0, len(audio_data) - window_size, step_size):
        window = audio_data[i:i + window_size]

        # Calculate spectral features
        fft_result = np.fft.rfft(window)
        magnitudes = np.abs(fft_result)

        # Create frequency bins
        num_bins = min(config.spectrogram_bins, len(magnitudes))
        binned = magnitudes[:num_bins]

        # Normalize and create hash
        normalized = (binned - np.min(binned)) / (np.max(binned) - np.min(binned) + 1e-10)
        hash_input = normalized.tobytes()
        fingerprint = hashlib.sha256(hash_input).hexdigest()

        fingerprints.append(fingerprint)

    return fingerprints

def _calculate_fingerprint_similarity(
    fingerprint1: List[str],
    fingerprint2: List[str]
) -> float:
    """Calculate similarity between two fingerprints.

    Uses Jaccard similarity for hash comparison.

    Args:
        fingerprint1: First fingerprint
        fingerprint2: Second fingerprint

    Returns:
        Similarity score (0-1)
    """
    set1 = set(fingerprint1)
    set2 = set(fingerprint2)

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    if union == 0:
        return 0.0

    return intersection / union

async def _detect_spectrogram_duplicates(
    target_audio: np.ndarray,
    ref_audio: np.ndarray,
    config: DeduplicationConfig
) -> DuplicateDetectionResult:
    """Detect duplicates using spectrogram hashing.

    Creates visual hashes of spectrograms for comparison.

    Args:
        target_audio: Target audio data
        ref_audio: Reference audio data
        config: Deduplication configuration

    Returns:
        DuplicateDetectionResult
    """
    # Generate spectrogram hashes
    target_hash = _generate_spectrogram_hash(target_audio, config)
    ref_hash = _generate_spectrogram_hash(ref_audio, config)

    # Calculate similarity
    similarity = _calculate_hash_similarity(target_hash, ref_hash)

    # Determine if duplicate
    is_duplicate = similarity >= config.spectrogram_threshold

    return DuplicateDetectionResult(
        is_duplicate=is_duplicate,
        similarity_score=similarity,
        method_used="spectrogram",
        comparison_metrics={
            "spectrogram_similarity": similarity,
            "hash_length": len(target_hash)
        }
    )

def _generate_spectrogram_hash(audio_data: np.ndarray, config: DeduplicationConfig) -> str:
    """Generate hash from spectrogram representation.

    Args:
        audio_data: Audio signal
        config: Deduplication configuration

    Returns:
        Spectrogram hash string
    """
    if len(audio_data.shape) > 1:
        # Use first channel
        audio_data = audio_data[0]

    # Calculate STFT
    window_size = config.fingerprint_window_size
    fft_result = np.fft.rfft(audio_data[:window_size * 10])  # Use first 10 windows
    magnitudes = np.abs(fft_result)

    # Create frequency bins
    num_bins = min(config.spectrogram_bins, len(magnitudes))
    binned = magnitudes[:num_bins]

    # Normalize and create hash
    normalized = (binned - np.min(binned)) / (np.max(binned) - np.min(binned) + 1e-10)
    hash_input = normalized.tobytes()

    return hashlib.sha256(hash_input).hexdigest()

def _calculate_hash_similarity(hash1: str, hash2: str) -> float:
    """Calculate similarity between two hashes.

    Uses bit-level comparison for hash similarity.

    Args:
        hash1: First hash
        hash2: Second hash

    Returns:
        Similarity score (0-1)
    """
    # Convert hex strings to bytes
    bytes1 = bytes.fromhex(hash1)
    bytes2 = bytes.fromhex(hash2)

    # Calculate bit similarity
    matching_bits = 0
    total_bits = len(bytes1) * 8

    for b1, b2 in zip(bytes1, bytes2):
        # XOR to find differing bits
        xor = b1 ^ b2
        # Count matching bits (1s in XOR = differing bits)
        matching_bits += (8 - bin(xor).count("1"))

    return matching_bits / total_bits

async def _detect_embedding_duplicates(
    target_audio: np.ndarray,
    ref_audio: np.ndarray,
    config: DeduplicationConfig
) -> DuplicateDetectionResult:
    """Detect duplicates using audio embeddings.

    Creates semantic embeddings of audio content for similarity comparison.

    Args:
        target_audio: Target audio data
        ref_audio: Reference audio data
        config: Deduplication configuration

    Returns:
        DuplicateDetectionResult
    """
    # Generate embeddings
    target_embedding = _generate_audio_embedding(target_audio, config)
    ref_embedding = _generate_audio_embedding(ref_audio, config)

    # Calculate cosine similarity
    similarity = float(cosine_similarity(
        [target_embedding],
        [ref_embedding]
    )[0][0])

    # Determine if duplicate
    is_duplicate = similarity >= config.embedding_threshold

    return DuplicateDetectionResult(
        is_duplicate=is_duplicate,
        similarity_score=similarity,
        method_used="embedding",
        comparison_metrics={
            "embedding_similarity": similarity,
            "embedding_dimension": len(target_embedding)
        }
    )

def _generate_audio_embedding(audio_data: np.ndarray, config: DeduplicationConfig) -> np.ndarray:
    """Generate semantic embedding from audio data.

    Uses spectral and temporal features to create semantic representation.

    Args:
        audio_data: Audio signal
        config: Deduplication configuration

    Returns:
        Audio embedding vector
    """
    if len(audio_data.shape) > 1:
        # Use first channel
        audio_data = audio_data[0]

    # Calculate various audio features
    features = []

    # 1. Spectral features
    fft_result = np.fft.rfft(audio_data[:config.fingerprint_window_size * 4])
    magnitudes = np.abs(fft_result)
    spectral_centroid = np.sum(np.arange(len(magnitudes)) * magnitudes) / np.sum(magnitudes)
    spectral_rolloff = np.sum(np.cumsum(magnitudes) < 0.85 * np.sum(magnitudes))
    spectral_flatness = _calculate_spectral_flatness(magnitudes)

    features.extend([spectral_centroid, spectral_rolloff, spectral_flatness])

    # 2. Temporal features
    rms = np.sqrt(np.mean(audio_data**2))
    zcr = np.sum(np.abs(np.diff(np.sign(audio_data)))) / (2 * len(audio_data))
    peak = np.max(np.abs(audio_data))

    features.extend([rms, zcr, peak])

    # 3. Statistical features
    mean = np.mean(audio_data)
    std = np.std(audio_data)
    skewness = _calculate_skewness(audio_data)
    kurtosis = _calculate_kurtosis(audio_data)

    features.extend([mean, std, skewness, kurtosis])

    # Pad or truncate to desired dimension
    embedding = np.array(features[:config.embedding_dimension])

    if len(embedding) < config.embedding_dimension:
        # Pad with zeros
        padding = np.zeros(config.embedding_dimension - len(embedding))
        embedding = np.concatenate([embedding, padding])

    # Normalize embedding
    embedding = (embedding - np.min(embedding)) / (np.max(embedding) - np.min(embedding) + 1e-10)

    return embedding

def _calculate_spectral_flatness(magnitudes: np.ndarray) -> float:
    """Calculate spectral flatness measure.

    Args:
        magnitudes: Spectral magnitudes

    Returns:
        Spectral flatness (0-1)
    """
    magnitudes = magnitudes + 1e-10  # Avoid log(0)
    geometric_mean = np.exp(np.mean(np.log(magnitudes)))
    arithmetic_mean = np.mean(magnitudes)
    return geometric_mean / arithmetic_mean

def _calculate_skewness(data: np.ndarray) -> float:
    """Calculate skewness of data distribution.

    Args:
        data: Input data

    Returns:
        Skewness value
    """
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0.0
    return np.mean(((data - mean) / std) ** 3)

def _calculate_kurtosis(data: np.ndarray) -> float:
    """Calculate kurtosis of data distribution.

    Args:
        data: Input data

    Returns:
        Kurtosis value
    """
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0.0
    return np.mean(((data - mean) / std) ** 4) - 3  # Excess kurtosis

def _simple_resample(audio_data: np.ndarray, ratio: float) -> np.ndarray:
    """Simple audio resampling using linear interpolation.

    For basic comparison purposes only.

    Args:
        audio_data: Audio signal
        ratio: Resampling ratio

    Returns:
        Resampled audio
    """
    if len(audio_data.shape) > 1:
        # Handle multi-channel
        resampled = []
        for channel in audio_data:
            resampled_channel = _simple_resample_channel(channel, ratio)
            resampled.append(resampled_channel)
        return np.array(resampled)
    else:
        return _simple_resample_channel(audio_data, ratio)

def _simple_resample_channel(channel_data: np.ndarray, ratio: float) -> np.ndarray:
    """Resample single channel using linear interpolation.

    Args:
        channel_data: Single channel audio
        ratio: Resampling ratio

    Returns:
        Resampled channel
    """
    original_length = len(channel_data)
    new_length = int(original_length * ratio)

    # Create new indices
    new_indices = np.linspace(0, original_length - 1, new_length)

    # Linear interpolation
    return np.interp(new_indices, np.arange(original_length), channel_data)
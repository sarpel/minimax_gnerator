"""Quality Scorer Module

This module implements a comprehensive quality scoring system for audio samples
using weighted composite scoring across multiple dimensions.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

from wakegen.core.exceptions import QualityAssuranceError
from wakegen.quality.validator import validate_sample, SampleValidationResult
from wakegen.utils.audio import load_audio_file

class QualityScoringError(QualityAssuranceError):
    """Custom exception for quality scoring failures."""

@dataclass
class QualityScoreResult:
    """Result of quality scoring with detailed component scores."""

    overall_score: float
    clarity_score: float
    snr_score: float
    naturalness_score: float
    diversity_score: float
    technical_score: float
    validation_result: SampleValidationResult
    error_message: Optional[str] = None

class QualityScoringConfig(BaseModel):
    """Configuration for quality scoring."""

    # Weight factors for composite scoring (must sum to 1.0)
    clarity_weight: float = Field(0.25, description="Weight for clarity score")
    snr_weight: float = Field(0.20, description="Weight for SNR score")
    naturalness_weight: float = Field(0.20, description="Weight for naturalness score")
    diversity_weight: float = Field(0.15, description="Weight for diversity score")
    technical_weight: float = Field(0.20, description="Weight for technical score")

    # Scoring thresholds
    min_clarity: float = Field(0.7, description="Minimum clarity score (0-1)")
    min_snr: float = Field(0.6, description="Minimum SNR score (0-1)")
    min_naturalness: float = Field(0.6, description="Minimum naturalness score (0-1)")
    min_diversity: float = Field(0.5, description="Minimum diversity score (0-1)")
    min_technical: float = Field(0.8, description="Minimum technical score (0-1)")

    def __post_init__(self):
        """Validate that weights sum to 1.0."""
        total_weight = (
            self.clarity_weight +
            self.snr_weight +
            self.naturalness_weight +
            self.diversity_weight +
            self.technical_weight
        )
        if not math.isclose(total_weight, 1.0, rel_tol=1e-6):
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

async def calculate_quality_score(
    file_path: str,
    config: Optional[QualityScoringConfig] = None
) -> QualityScoreResult:
    """Calculate comprehensive quality score for an audio sample.

    Uses weighted composite scoring across multiple quality dimensions:
    - Clarity: Speech intelligibility and lack of distortion
    - SNR: Signal-to-noise ratio quality
    - Naturalness: How natural the speech sounds
    - Diversity: Spectral diversity and richness
    - Technical: Technical compliance with specifications

    Args:
        file_path: Path to audio file to score
        config: Optional scoring configuration

    Returns:
        QualityScoreResult with detailed scores

    Raises:
        QualityScoringError: If scoring fails catastrophically
    """
    if config is None:
        config = QualityScoringConfig()

    try:
        # First validate the sample
        validation_result = await validate_sample(file_path)

        if not validation_result.is_valid:
            return QualityScoreResult(
                overall_score=0.0,
                clarity_score=0.0,
                snr_score=0.0,
                naturalness_score=0.0,
                diversity_score=0.0,
                technical_score=0.0,
                validation_result=validation_result,
                error_message=f"Sample failed validation: {validation_result.error_message}"
            )

        # Load audio data for analysis
        audio_data, sample_rate = await load_audio_file(file_path)

        # Calculate individual component scores
        clarity_score = _calculate_clarity_score(audio_data, sample_rate)
        snr_score = _calculate_snr_score(validation_result.signal_to_noise_ratio_db)
        naturalness_score = _calculate_naturalness_score(audio_data, sample_rate)
        diversity_score = _calculate_diversity_score(audio_data)
        technical_score = _calculate_technical_score(validation_result)

        # Apply minimum thresholds
        clarity_score = max(clarity_score, config.min_clarity)
        snr_score = max(snr_score, config.min_snr)
        naturalness_score = max(naturalness_score, config.min_naturalness)
        diversity_score = max(diversity_score, config.min_diversity)
        technical_score = max(technical_score, config.min_technical)

        # Calculate weighted composite score
        overall_score = (
            clarity_score * config.clarity_weight +
            snr_score * config.snr_weight +
            naturalness_score * config.naturalness_weight +
            diversity_score * config.diversity_weight +
            technical_score * config.technical_weight
        )

        return QualityScoreResult(
            overall_score=overall_score,
            clarity_score=clarity_score,
            snr_score=snr_score,
            naturalness_score=naturalness_score,
            diversity_score=diversity_score,
            technical_score=technical_score,
            validation_result=validation_result,
            error_message=None
        )

    except Exception as e:
        raise QualityScoringError(f"Quality scoring failed for {file_path}: {str(e)}") from e

def _calculate_clarity_score(audio_data: np.ndarray, sample_rate: int) -> float:
    """Calculate clarity score based on spectral characteristics.

    Clarity measures speech intelligibility and lack of distortion.
    Higher clarity indicates cleaner, more understandable speech.

    Args:
        audio_data: Audio signal
        sample_rate: Sample rate in Hz

    Returns:
        Clarity score (0-1)
    """
    if len(audio_data.shape) > 1:
        # Use first channel for mono analysis
        audio_data = audio_data[0]

    # Calculate spectral centroid (measure of brightness)
    fft_result = np.fft.rfft(audio_data)
    frequencies = np.fft.rfftfreq(len(audio_data), 1.0/sample_rate)
    magnitudes = np.abs(fft_result)

    # Weighted average frequency (spectral centroid)
    spectral_centroid = np.sum(frequencies * magnitudes) / np.sum(magnitudes)

    # Normalize to 0-1 range (typical speech is 500-3000 Hz)
    # Lower centroid = more bass = potentially clearer for speech
    normalized_centroid = 1.0 - min(spectral_centroid / 4000.0, 1.0)

    # Calculate spectral flatness (measure of tone-like vs noise-like)
    geometric_mean = np.exp(np.mean(np.log(magnitudes + 1e-10)))
    arithmetic_mean = np.mean(magnitudes)
    spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)

    # Higher flatness = more noise-like = less clear
    flatness_score = 1.0 - spectral_flatness

    # Combine metrics (equal weighting)
    clarity_score = (normalized_centroid * 0.6 + flatness_score * 0.4)

    return max(0.0, min(1.0, clarity_score))

def _calculate_snr_score(snr_db: Optional[float]) -> float:
    """Calculate SNR score from SNR in dB.

    Converts SNR dB to normalized score.

    Args:
        snr_db: Signal-to-noise ratio in decibels

    Returns:
        SNR score (0-1)
    """
    if snr_db is None:
        return 0.0

    # Normalize SNR to 0-1 range
    # 0dB = 0.0, 20dB = 0.5, 40dB = 1.0, >40dB = 1.0
    normalized_snr = min(snr_db / 40.0, 1.0)

    # Apply sigmoid to emphasize middle range
    return 1.0 / (1.0 + np.exp(-5.0 * (normalized_snr - 0.5)))

def _calculate_naturalness_score(audio_data: np.ndarray, sample_rate: int) -> float:
    """Calculate naturalness score based on temporal characteristics.

    Naturalness measures how human-like and smooth the speech sounds.

    Args:
        audio_data: Audio signal
        sample_rate: Sample rate in Hz

    Returns:
        Naturalness score (0-1)
    """
    if len(audio_data.shape) > 1:
        # Use first channel for mono analysis
        audio_data = audio_data[0]

    # Calculate temporal envelope smoothness
    window_size = int(sample_rate * 0.02)  # 20ms windows
    num_windows = len(audio_data) // window_size

    if num_windows < 2:
        return 0.5  # Not enough data

    # Calculate RMS in each window
    rms_values = []
    for i in range(num_windows):
        window = audio_data[i*window_size : (i+1)*window_size]
        rms = np.sqrt(np.mean(window**2))
        rms_values.append(rms)

    # Calculate smoothness (inverse of variance)
    rms_variance = np.var(rms_values)
    smoothness = 1.0 / (1.0 + rms_variance * 100.0)  # Normalize

    # Calculate zero-crossing rate consistency
    zcr_values = []
    for i in range(num_windows):
        window = audio_data[i*window_size : (i+1)*window_size]
        zcr = np.sum(np.abs(np.diff(np.sign(window)))) / (2 * len(window))
        zcr_values.append(zcr)

    zcr_variance = np.var(zcr_values)
    zcr_consistency = 1.0 / (1.0 + zcr_variance * 50.0)  # Normalize

    # Combine metrics
    naturalness_score = (smoothness * 0.6 + zcr_consistency * 0.4)

    return max(0.0, min(1.0, naturalness_score))

def _calculate_diversity_score(audio_data: np.ndarray) -> float:
    """Calculate spectral diversity score.

    Diversity measures the richness and variety of frequency content.

    Args:
        audio_data: Audio signal

    Returns:
        Diversity score (0-1)
    """
    if len(audio_data.shape) > 1:
        # Use first channel for mono analysis
        audio_data = audio_data[0]

    # Calculate spectral entropy (measure of frequency distribution)
    fft_result = np.fft.rfft(audio_data)
    magnitudes = np.abs(fft_result)

    # Normalize magnitudes to probability distribution
    magnitudes = magnitudes + 1e-10  # Avoid log(0)
    prob_dist = magnitudes / np.sum(magnitudes)

    # Calculate entropy
    entropy = -np.sum(prob_dist * np.log2(prob_dist))

    # Normalize entropy to 0-1 range
    max_entropy = np.log2(len(prob_dist))  # Maximum possible entropy
    normalized_entropy = entropy / max_entropy

    # Higher entropy = more diverse = better
    return max(0.0, min(1.0, normalized_entropy))

def _calculate_technical_score(validation_result: SampleValidationResult) -> float:
    """Calculate technical compliance score.

    Technical score measures compliance with technical specifications.

    Args:
        validation_result: Validation result with metrics

    Returns:
        Technical score (0-1)
    """
    # Start with perfect score
    score = 1.0

    # Penalize for validation errors
    if validation_result.error_message:
        score *= 0.5  # Significant penalty for validation errors

    # Reward for good SNR
    if validation_result.signal_to_noise_ratio_db:
        snr_bonus = min(validation_result.signal_to_noise_ratio_db / 50.0, 0.2)
        score = min(1.0, score + snr_bonus)

    # Penalize for high peak amplitude (clipping risk)
    if validation_result.peak_amplitude:
        peak_penalty = max(0.0, validation_result.peak_amplitude - 0.9) * 2.0
        score = max(0.0, score - peak_penalty)

    # Reward for good RMS amplitude (not too quiet)
    if validation_result.rms_amplitude:
        rms_bonus = min(validation_result.rms_amplitude * 50.0, 0.1)
        score = min(1.0, score + rms_bonus)

    return max(0.0, min(1.0, score))
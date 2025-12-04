"""ASR Verification Module

This module provides ASR-based pronunciation verification using Whisper
to validate that generated audio samples contain the correct wake words.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

from wakegen.core.exceptions import QualityAssuranceError

# Suppress Whisper warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="whisper")

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    whisper = None

class ASRVerificationError(QualityAssuranceError):
    """Custom exception for ASR verification failures."""

@dataclass
class ASRVerificationResult:
    """Result of ASR pronunciation verification."""

    is_correct: bool
    transcribed_text: str
    expected_text: str
    confidence: float
    word_error_rate: float
    processing_time_seconds: float
    error_message: Optional[str] = None

class ASRVerificationConfig(BaseModel):
    """Configuration for ASR verification."""

    model_size: str = Field("tiny", description="Whisper model size (tiny, base, small, medium, large)")
    language: Optional[str] = Field(None, description="Language code for transcription")
    min_confidence: float = Field(0.7, description="Minimum confidence threshold (0-1)")
    max_word_error_rate: float = Field(0.3, description="Maximum acceptable word error rate")

    # Performance optimization
    use_gpu: bool = Field(False, description="Use GPU acceleration if available")
    beam_size: int = Field(5, description="Beam size for decoding")
    temperature: float = Field(0.0, description="Temperature for sampling (0.0 = greedy)")

async def verify_pronunciation(
    audio_file_path: str | Path,
    expected_text: str,
    config: Optional[ASRVerificationConfig] = None
) -> ASRVerificationResult:
    """Verify pronunciation using ASR transcription.

    Uses Whisper to transcribe audio and compare against expected text.

    Args:
        audio_file_path: Path to audio file to verify
        expected_text: Expected transcription text
        config: Optional ASR configuration

    Returns:
        ASRVerificationResult with verification metrics

    Raises:
        ASRVerificationError: If ASR processing fails catastrophically
    """
    if config is None:
        config = ASRVerificationConfig()

    if not WHISPER_AVAILABLE:
        raise ASRVerificationError(
            "Whisper is not available. Please install with: pip install openai-whisper"
        )

    audio_file_path = Path(audio_file_path)

    if not audio_file_path.exists():
        return ASRVerificationResult(
            is_correct=False,
            transcribed_text="",
            expected_text=expected_text,
            confidence=0.0,
            word_error_rate=1.0,
            processing_time_seconds=0.0,
            error_message=f"Audio file does not exist: {audio_file_path}"
        )

    if not audio_file_path.is_file():
        return ASRVerificationResult(
            is_correct=False,
            transcribed_text="",
            expected_text=expected_text,
            confidence=0.0,
            word_error_rate=1.0,
            processing_time_seconds=0.0,
            error_message=f"Path is not a file: {audio_file_path}"
        )

    try:
        import time
        start_time = time.time()

        # Load Whisper model (this is cached after first load)
        model = await _load_whisper_model(config)

        # Transcribe audio
        result = await asyncio.to_thread(
            model.transcribe,
            str(audio_file_path),
            language=config.language,
            beam_size=config.beam_size,
            temperature=config.temperature,
            fp16=config.use_gpu
        )

        processing_time = time.time() - start_time

        transcribed_text = result["text"].strip()
        confidence = _calculate_confidence(result)
        word_error_rate = _calculate_word_error_rate(transcribed_text, expected_text)

        # Determine if pronunciation is correct
        is_correct = (
            word_error_rate <= config.max_word_error_rate and
            confidence >= config.min_confidence
        )

        error_message = None
        if not is_correct:
            if word_error_rate > config.max_word_error_rate:
                error_message = f"High WER: {word_error_rate:.3f} > {config.max_word_error_rate}"
            if confidence < config.min_confidence:
                error_message = f"Low confidence: {confidence:.3f} < {config.min_confidence}"

        return ASRVerificationResult(
            is_correct=is_correct,
            transcribed_text=transcribed_text,
            expected_text=expected_text,
            confidence=confidence,
            word_error_rate=word_error_rate,
            processing_time_seconds=processing_time,
            error_message=error_message
        )

    except Exception as e:
        raise ASRVerificationError(f"ASR verification failed: {str(e)}") from e

async def _load_whisper_model(config: ASRVerificationConfig) -> whisper.Whisper:
    """Load Whisper model with caching and resource management.

    Args:
        config: ASR configuration

    Returns:
        Loaded Whisper model
    """
    # Use global cache to avoid reloading models
    if not hasattr(_load_whisper_model, "_model_cache"):
        _load_whisper_model._model_cache = {}

    cache_key = f"{config.model_size}_{config.language}_{config.use_gpu}"

    if cache_key in _load_whisper_model._model_cache:
        return _load_whisper_model._model_cache[cache_key]

    # Load model (this can take time and memory)
    model = whisper.load_model(
        config.model_size,
        device="cuda" if config.use_gpu else "cpu"
    )

    _load_whisper_model._model_cache[cache_key] = model
    return model

def _calculate_confidence(transcription_result: dict) -> float:
    """Calculate overall confidence score from Whisper transcription.

    Args:
        transcription_result: Whisper transcription result

    Returns:
        Confidence score (0-1)
    """
    if "segments" not in transcription_result or not transcription_result["segments"]:
        return 0.5  # Default confidence if no segment info

    # Calculate average confidence from segments
    segment_confidences = []
    for segment in transcription_result["segments"]:
        if "avg_logprob" in segment:
            # Convert log probability to confidence (0-1)
            # Higher logprob = higher confidence
            logprob = segment["avg_logprob"]
            confidence = 1.0 / (1.0 + np.exp(-logprob))
            segment_confidences.append(confidence)

    if not segment_confidences:
        return 0.5

    # Return average confidence
    return float(np.mean(segment_confidences))

def _calculate_word_error_rate(transcribed: str, reference: str) -> float:
    """Calculate Word Error Rate (WER) between transcribed and reference text.

    WER = (S + D + I) / N
    Where:
    S = substitutions
    D = deletions
    I = insertions
    N = number of words in reference

    Args:
        transcribed: Transcribed text from ASR
        reference: Reference/expected text

    Returns:
        Word Error Rate (0-1)
    """
    # Tokenize into words
    ref_words = reference.lower().split()
    trans_words = transcribed.lower().split()

    # Create distance matrix for dynamic programming
    d = np.zeros((len(ref_words) + 1, len(trans_words) + 1))

    # Initialize first row and column
    for i in range(len(ref_words) + 1):
        d[i, 0] = i  # Deletions
    for j in range(len(trans_words) + 1):
        d[0, j] = j  # Insertions

    # Fill distance matrix
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(trans_words) + 1):
            if ref_words[i-1] == trans_words[j-1]:
                cost = 0  # Match
            else:
                cost = 1  # Substitution

            d[i, j] = min(
                d[i-1, j] + 1,      # Deletion
                d[i, j-1] + 1,      # Insertion
                d[i-1, j-1] + cost  # Substitution or match
            )

    # Calculate WER
    wer = d[len(ref_words), len(trans_words)] / len(ref_words) if ref_words else 1.0

    return float(wer)

async def batch_verify_pronunciation(
    audio_file_paths: list[str | Path],
    expected_texts: list[str],
    config: Optional[ASRVerificationConfig] = None
) -> list[ASRVerificationResult]:
    """Batch verify multiple audio files for efficiency.

    Args:
        audio_file_paths: List of audio file paths
        expected_texts: List of expected texts (must match file paths length)
        config: Optional ASR configuration

    Returns:
        List of ASRVerificationResult objects

    Raises:
        ASRVerificationError: If batch processing fails
    """
    if config is None:
        config = ASRVerificationConfig()

    if len(audio_file_paths) != len(expected_texts):
        raise ASRVerificationError(
            "audio_file_paths and expected_texts must have same length"
        )

    # Load model once for all files
    if WHISPER_AVAILABLE:
        model = await _load_whisper_model(config)
    else:
        raise ASRVerificationError(
            "Whisper is not available for batch processing"
        )

    results = []
    for audio_path, expected_text in zip(audio_file_paths, expected_texts):
        try:
            result = await verify_pronunciation(audio_path, expected_text, config)
            results.append(result)
        except Exception as e:
            results.append(ASRVerificationResult(
                is_correct=False,
                transcribed_text="",
                expected_text=expected_text,
                confidence=0.0,
                word_error_rate=1.0,
                processing_time_seconds=0.0,
                error_message=f"Processing error: {str(e)}"
            ))

    return results
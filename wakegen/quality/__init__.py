"""Quality Assurance & Analysis System for Wake Word Dataset Generator.

This module provides comprehensive quality assurance and analysis capabilities
for validating, scoring, and analyzing generated audio samples.
"""

from .asr_check import ASRVerificationError, ASRVerificationResult, verify_pronunciation
from .deduplication import (
    DeduplicationError,
    DuplicateDetectionResult,
    detect_duplicates,
)
from .report_generator import ReportGenerationError, generate_report
from .scorer import QualityScoreResult, QualityScoringError, calculate_quality_score
from .statistics import (
    DatasetStatisticsResult,
    StatisticsError,
    calculate_dataset_statistics,
)
from .validator import SampleValidationError, SampleValidationResult, validate_sample

__all__ = [
    # Validator exports
    "validate_sample",
    "SampleValidationResult",
    "SampleValidationError",
    # Scorer exports
    "calculate_quality_score",
    "QualityScoreResult",
    "QualityScoringError",
    # ASR Check exports
    "verify_pronunciation",
    "ASRVerificationResult",
    "ASRVerificationError",
    # Deduplication exports
    "detect_duplicates",
    "DuplicateDetectionResult",
    "DeduplicationError",
    # Statistics exports
    "calculate_dataset_statistics",
    "DatasetStatisticsResult",
    "StatisticsError",
    # Report Generator exports
    "generate_report",
    "ReportGenerationError",
]

"""Quality Assurance & Analysis System for Wake Word Dataset Generator.

This module provides comprehensive quality assurance and analysis capabilities
for validating, scoring, and analyzing generated audio samples.
"""

from .validator import (
    validate_sample,
    SampleValidationResult,
    ValidationError,
)
from .scorer import (
    calculate_quality_score,
    QualityScoreResult,
    QualityScoringError,
)
from .asr_check import (
    verify_pronunciation,
    ASRVerificationResult,
    ASRVerificationError,
)
from .deduplication import (
    detect_duplicates,
    DuplicateDetectionResult,
    DeduplicationError,
)
from .statistics import (
    calculate_dataset_statistics,
    DatasetStatisticsResult,
    StatisticsError,
)
from .report_generator import (
    generate_report,
    ReportGenerationError,
)

__all__ = [
    # Validator exports
    "validate_sample",
    "SampleValidationResult",
    "ValidationError",
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
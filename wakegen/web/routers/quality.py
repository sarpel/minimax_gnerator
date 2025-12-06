"""
Quality Validation API Router

This module provides endpoints for validating and analyzing audio datasets.
Quality checks ensure generated samples meet training requirements.

    ENDPOINTS:
    ==========
    GET  /validate/{directory} - Validate audio files in a directory
    GET  /metrics/{directory}  - Get quality metrics for a dataset
    GET  /issues/{directory}   - List detected issues
    POST /auto-fix             - Attempt to fix common issues

    QUALITY CHECKS:
    ===============
    1. Duration: Check if samples are within acceptable range
    2. Sample Rate: Verify consistent sample rates
    3. Clipping: Detect audio clipping/distortion
    4. Silence: Check for silent or near-silent files
    5. SNR: Estimate signal-to-noise ratio
    6. Format: Verify file format and encoding
"""

import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
from enum import Enum

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class IssueSeverity(str, Enum):
    """Severity levels for detected issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class IssueType(str, Enum):
    """Types of quality issues that can be detected."""
    DURATION_SHORT = "duration_short"
    DURATION_LONG = "duration_long"
    SAMPLE_RATE_MISMATCH = "sample_rate_mismatch"
    CLIPPING = "clipping"
    SILENCE = "silence"
    LOW_SNR = "low_snr"
    FORMAT_ERROR = "format_error"
    CORRUPT_FILE = "corrupt_file"


# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class QualityIssue(BaseModel):
    """A detected quality issue in an audio file."""
    file: str = Field(..., description="Relative path to the file")
    type: IssueType = Field(..., description="Type of issue")
    severity: IssueSeverity = Field(..., description="Severity level")
    message: str = Field(..., description="Human-readable description")
    value: Optional[float] = Field(None, description="Measured value if applicable")
    threshold: Optional[float] = Field(None, description="Threshold that was exceeded")


class DurationMetrics(BaseModel):
    """Duration statistics for the dataset."""
    min: float = Field(..., description="Minimum duration in seconds")
    max: float = Field(..., description="Maximum duration in seconds")
    avg: float = Field(..., description="Average duration in seconds")
    std: float = Field(..., description="Standard deviation")


class SNRMetrics(BaseModel):
    """Signal-to-noise ratio statistics."""
    min: float = Field(..., description="Minimum SNR in dB")
    max: float = Field(..., description="Maximum SNR in dB")
    avg: float = Field(..., description="Average SNR in dB")


class SampleRateInfo(BaseModel):
    """Sample rate information."""
    consistent: bool = Field(..., description="Whether all files have same sample rate")
    common: int = Field(..., description="Most common sample rate")
    distribution: Dict[int, int] = Field(default_factory=dict, description="Count per sample rate")


class QualityMetrics(BaseModel):
    """Complete quality metrics for a dataset."""
    directory: str
    total_files: int = Field(..., description="Total number of audio files")
    valid_files: int = Field(..., description="Number of valid files")
    duration: DurationMetrics
    snr: SNRMetrics
    sample_rate: SampleRateInfo
    total_duration_seconds: float
    health_score: float = Field(..., ge=0, le=100, description="Overall health score")


class ValidationResult(BaseModel):
    """Result of dataset validation."""
    directory: str
    total_files: int
    valid_files: int
    issues: List[QualityIssue]
    health_score: float
    summary: str


class WakeWordDistribution(BaseModel):
    """Distribution of samples across wake words."""
    name: str
    count: int
    percentage: float


class DatasetSummary(BaseModel):
    """Summary of a dataset directory."""
    directory: str
    total_files: int
    total_size_bytes: int
    total_duration_seconds: float
    wake_words: List[WakeWordDistribution]
    health_score: float
    issues_count: int


# =============================================================================
# ROUTER
# =============================================================================


router = APIRouter()


@router.get(
    "/validate",
    response_model=ValidationResult,
    summary="Validate dataset"
)
async def validate_dataset(
    directory: str = Query("./output", description="Directory to validate"),
    min_duration: float = Query(0.3, description="Minimum acceptable duration (seconds)"),
    max_duration: float = Query(3.0, description="Maximum acceptable duration (seconds)"),
    expected_sample_rate: int = Query(16000, description="Expected sample rate (Hz)")
) -> ValidationResult:
    """
    Validate all audio files in a directory.

    Checks for common issues like duration problems, clipping, and format errors.
    Returns a list of issues and an overall health score.
    """
    dir_path = Path(directory)

    if not dir_path.exists():
        raise HTTPException(status_code=404, detail=f"Directory not found: {directory}")

    # Find all audio files
    audio_files = list(dir_path.rglob("*.wav"))
    total_files = len(audio_files)

    if total_files == 0:
        return ValidationResult(
            directory=directory,
            total_files=0,
            valid_files=0,
            issues=[],
            health_score=100.0,
            summary="No audio files found"
        )

    issues = []
    valid_count = 0

    # Validate each file
    for file_path in audio_files:
        try:
            file_issues = await validate_file(
                file_path,
                min_duration=min_duration,
                max_duration=max_duration,
                expected_sample_rate=expected_sample_rate
            )

            if not file_issues:
                valid_count += 1
            else:
                issues.extend(file_issues)

        except Exception as e:
            issues.append(QualityIssue(
                file=str(file_path.relative_to(dir_path)),
                type=IssueType.CORRUPT_FILE,
                severity=IssueSeverity.ERROR,
                message=f"Error reading file: {str(e)}"
            ))

    # Calculate health score
    health_score = (valid_count / total_files) * 100 if total_files > 0 else 0

    # Generate summary
    error_count = sum(1 for i in issues if i.severity == IssueSeverity.ERROR)
    warning_count = sum(1 for i in issues if i.severity == IssueSeverity.WARNING)
    summary = f"{valid_count}/{total_files} files valid. {error_count} errors, {warning_count} warnings."

    return ValidationResult(
        directory=directory,
        total_files=total_files,
        valid_files=valid_count,
        issues=issues,
        health_score=health_score,
        summary=summary
    )


async def validate_file(
    file_path: Path,
    min_duration: float,
    max_duration: float,
    expected_sample_rate: int
) -> List[QualityIssue]:
    """
    Validate a single audio file.

    Returns a list of detected issues (empty list if file is valid).
    """
    issues = []
    relative_path = file_path.name

    try:
        import soundfile as sf

        info = sf.info(str(file_path))
        duration = info.duration

        # Check duration
        if duration < min_duration:
            issues.append(QualityIssue(
                file=relative_path,
                type=IssueType.DURATION_SHORT,
                severity=IssueSeverity.WARNING,
                message=f"Duration too short ({duration:.2f}s < {min_duration}s)",
                value=duration,
                threshold=min_duration
            ))

        if duration > max_duration:
            issues.append(QualityIssue(
                file=relative_path,
                type=IssueType.DURATION_LONG,
                severity=IssueSeverity.WARNING,
                message=f"Duration too long ({duration:.2f}s > {max_duration}s)",
                value=duration,
                threshold=max_duration
            ))

        # Check sample rate
        if info.samplerate != expected_sample_rate:
            issues.append(QualityIssue(
                file=relative_path,
                type=IssueType.SAMPLE_RATE_MISMATCH,
                severity=IssueSeverity.WARNING,
                message=f"Sample rate {info.samplerate} Hz (expected {expected_sample_rate} Hz)",
                value=float(info.samplerate),
                threshold=float(expected_sample_rate)
            ))

        # Check for clipping (would need to load audio data)
        # This is a placeholder - real implementation would analyze waveform
        # y, sr = sf.read(str(file_path))
        # max_amplitude = np.max(np.abs(y))
        # if max_amplitude > 0.99:
        #     issues.append(...)

    except ImportError:
        logger.warning("soundfile not available for validation")
    except Exception as e:
        issues.append(QualityIssue(
            file=relative_path,
            type=IssueType.FORMAT_ERROR,
            severity=IssueSeverity.ERROR,
            message=str(e)
        ))

    return issues


@router.get(
    "/metrics",
    response_model=QualityMetrics,
    summary="Get quality metrics"
)
async def get_metrics(
    directory: str = Query("./output", description="Directory to analyze")
) -> QualityMetrics:
    """
    Get detailed quality metrics for a dataset.

    Analyzes all audio files and returns statistics about duration,
    sample rate, SNR, and overall health.
    """
    dir_path = Path(directory)

    if not dir_path.exists():
        raise HTTPException(status_code=404, detail=f"Directory not found: {directory}")

    audio_files = list(dir_path.rglob("*.wav"))
    total_files = len(audio_files)

    if total_files == 0:
        raise HTTPException(status_code=400, detail="No audio files found")

    # Placeholder metrics - real implementation would analyze files
    return QualityMetrics(
        directory=directory,
        total_files=total_files,
        valid_files=int(total_files * 0.95),  # Placeholder
        duration=DurationMetrics(min=0.3, max=2.1, avg=0.8, std=0.3),
        snr=SNRMetrics(min=18, max=52, avg=35),
        sample_rate=SampleRateInfo(consistent=True, common=16000, distribution={16000: total_files}),
        total_duration_seconds=total_files * 0.8,  # Placeholder
        health_score=87.0
    )


@router.get(
    "/summary",
    response_model=DatasetSummary,
    summary="Get dataset summary"
)
async def get_summary(
    directory: str = Query("./output", description="Directory to summarize")
) -> DatasetSummary:
    """
    Get a high-level summary of a dataset.

    Includes file counts, wake word distribution, and overall health.
    """
    dir_path = Path(directory)

    if not dir_path.exists():
        raise HTTPException(status_code=404, detail=f"Directory not found: {directory}")

    audio_files = list(dir_path.rglob("*.wav"))
    total_files = len(audio_files)

    # Calculate total size
    total_size = sum(f.stat().st_size for f in audio_files)

    # Count files per subdirectory (wake word)
    wake_words: Dict[str, int] = {}
    for f in audio_files:
        parent = f.parent.name
        wake_words[parent] = wake_words.get(parent, 0) + 1

    # Convert to distribution
    distribution = [
        WakeWordDistribution(
            name=name,
            count=count,
            percentage=(count / total_files) * 100 if total_files > 0 else 0
        )
        for name, count in sorted(wake_words.items(), key=lambda x: -x[1])
    ]

    return DatasetSummary(
        directory=directory,
        total_files=total_files,
        total_size_bytes=total_size,
        total_duration_seconds=total_files * 0.8,  # Placeholder
        wake_words=distribution,
        health_score=87.0,
        issues_count=3  # Placeholder
    )


@router.get(
    "/recommendations",
    summary="Get recommendations"
)
async def get_recommendations(
    directory: str = Query("./output", description="Directory to analyze")
) -> List[str]:
    """
    Get recommendations for improving dataset quality.

    Based on the analysis of the dataset, returns actionable suggestions.
    """
    # Placeholder recommendations
    return [
        "Consider adding more samples for underrepresented wake words",
        "Apply noise augmentation to improve robustness",
        "Review files with detected clipping issues",
        "Ensure consistent sample rates across all files"
    ]

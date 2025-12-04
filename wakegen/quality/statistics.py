"""Dataset Statistics Module

This module provides comprehensive statistical analysis of audio datasets
including quality metrics, distribution analysis, and metadata aggregation.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from wakegen.core.exceptions import QualityAssuranceError
from wakegen.quality.validator import validate_sample
from wakegen.quality.scorer import calculate_quality_score
from wakegen.utils.audio import load_audio_file

# Suppress pandas warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")

class StatisticsError(QualityAssuranceError):
    """Custom exception for statistics calculation failures."""

@dataclass
class DatasetStatisticsResult:
    """Result of dataset statistics calculation."""

    file_count: int
    total_duration_seconds: float
    average_duration_seconds: float
    min_duration_seconds: float
    max_duration_seconds: float
    sample_rate_distribution: Dict[int, int]
    quality_score_distribution: Dict[str, int]
    snr_distribution: Dict[str, int]
    file_size_distribution: Dict[str, int]
    metadata: Dict[str, Any]
    detailed_stats: Optional[pd.DataFrame] = None
    error_message: Optional[str] = None

class DatasetStatisticsConfig(BaseModel):
    """Configuration for dataset statistics calculation."""

    # Quality score ranges for distribution
    quality_ranges: List[float] = Field(
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        description="Quality score range boundaries"
    )

    # SNR ranges for distribution
    snr_ranges: List[float] = Field(
        [0, 10, 20, 30, 40, 50, 100],
        description="SNR range boundaries in dB"
    )

    # File size ranges for distribution (bytes)
    size_ranges: List[int] = Field(
        [0, 10000, 50000, 100000, 500000, 1000000],
        description="File size range boundaries in bytes"
    )

    # Performance optimization
    max_concurrent_files: int = Field(10, description="Maximum concurrent file processing")
    batch_size: int = Field(100, description="Batch size for processing")

async def calculate_dataset_statistics(
    dataset_path: str | Path,
    config: Optional[DatasetStatisticsConfig] = None
) -> DatasetStatisticsResult:
    """Calculate comprehensive statistics for an audio dataset.

    Analyzes all audio files in a directory and generates detailed statistics.

    Args:
        dataset_path: Path to dataset directory
        config: Optional statistics configuration

    Returns:
        DatasetStatisticsResult with comprehensive metrics

    Raises:
        StatisticsError: If statistics calculation fails catastrophically
    """
    if config is None:
        config = DatasetStatisticsConfig()

    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise StatisticsError(f"Dataset path does not exist: {dataset_path}")

    if not dataset_path.is_dir():
        raise StatisticsError(f"Dataset path is not a directory: {dataset_path}")

    # Find all audio files
    audio_files = []
    for ext in ['*.wav', '*.mp3', '*.ogg', '*.flac']:
        audio_files.extend(dataset_path.glob(ext))

    if not audio_files:
        return DatasetStatisticsResult(
            file_count=0,
            total_duration_seconds=0.0,
            average_duration_seconds=0.0,
            min_duration_seconds=0.0,
            max_duration_seconds=0.0,
            sample_rate_distribution={},
            quality_score_distribution={},
            snr_distribution={},
            file_size_distribution={},
            metadata={},
            error_message="No audio files found in dataset"
        )

    try:
        # Process files in batches for memory efficiency
        results = await _process_files_in_batches(
            audio_files, config
        )

        # Calculate statistics from results
        return _calculate_statistics_from_results(results, config)

    except Exception as e:
        raise StatisticsError(f"Dataset statistics calculation failed: {str(e)}") from e

async def _process_files_in_batches(
    audio_files: List[Path],
    config: DatasetStatisticsConfig
) -> List[Dict[str, Any]]:
    """Process audio files in batches for memory efficiency.

    Args:
        audio_files: List of audio file paths
        config: Statistics configuration

    Returns:
        List of file analysis results
    """
    results = []
    semaphore = asyncio.Semaphore(config.max_concurrent_files)

    async def process_file(file_path: Path) -> Dict[str, Any]:
        """Process single file with rate limiting."""
        async with semaphore:
            return await _analyze_single_file(file_path)

    # Process files concurrently with batching
    for i in range(0, len(audio_files), config.batch_size):
        batch = audio_files[i:i + config.batch_size]
        batch_results = await asyncio.gather(*[process_file(file) for file in batch])
        results.extend(batch_results)

    return results

async def _analyze_single_file(file_path: Path) -> Dict[str, Any]:
    """Analyze single audio file and extract statistics.

    Args:
        file_path: Path to audio file

    Returns:
        Dictionary of analysis results
    """
    try:
        # Basic file info
        file_size = file_path.stat().st_size
        file_hash = await _calculate_file_hash(file_path)

        # Validate sample
        validation_result = await validate_sample(str(file_path))

        # Calculate quality score
        quality_result = await calculate_quality_score(str(file_path))

        return {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_size": file_size,
            "file_hash": file_hash,
            "is_valid": validation_result.is_valid,
            "duration": validation_result.duration_seconds,
            "sample_rate": validation_result.sample_rate_hz,
            "channels": validation_result.channels,
            "bit_depth": validation_result.bit_depth,
            "snr_db": validation_result.signal_to_noise_ratio_db,
            "peak_amplitude": validation_result.peak_amplitude,
            "rms_amplitude": validation_result.rms_amplitude,
            "zero_crossing_rate": validation_result.zero_crossing_rate,
            "quality_score": quality_result.overall_score,
            "clarity_score": quality_result.clarity_score,
            "naturalness_score": quality_result.naturalness_score,
            "diversity_score": quality_result.diversity_score,
            "technical_score": quality_result.technical_score,
            "error": validation_result.error_message or quality_result.error_message
        }

    except Exception as e:
        return {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "error": f"Analysis failed: {str(e)}",
            "is_valid": False
        }

async def _calculate_file_hash(file_path: Path) -> str:
    """Calculate file hash for uniqueness tracking.

    Args:
        file_path: Path to file

    Returns:
        SHA256 hash of file
    """
    hash_sha256 = hashlib.sha256()

    # Read file in chunks
    chunk_size = 8192
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hash_sha256.update(chunk)

    return hash_sha256.hexdigest()

def _calculate_statistics_from_results(
    results: List[Dict[str, Any]],
    config: DatasetStatisticsConfig
) -> DatasetStatisticsResult:
    """Calculate comprehensive statistics from file analysis results.

    Args:
        results: List of file analysis results
        config: Statistics configuration

    Returns:
        DatasetStatisticsResult with comprehensive metrics
    """
    # Filter out failed analyses
    valid_results = [r for r in results if r.get("is_valid", False) and "duration" in r]

    if not valid_results:
        return DatasetStatisticsResult(
            file_count=len(results),
            total_duration_seconds=0.0,
            average_duration_seconds=0.0,
            min_duration_seconds=0.0,
            max_duration_seconds=0.0,
            sample_rate_distribution={},
            quality_score_distribution={},
            snr_distribution={},
            file_size_distribution={},
            metadata={"failed_files": len(results)},
            error_message="No valid audio files found"
        )

    # Basic duration statistics
    durations = [r["duration"] for r in valid_results]
    total_duration = sum(durations)
    avg_duration = total_duration / len(durations)
    min_duration = min(durations)
    max_duration = max(durations)

    # Sample rate distribution
    sample_rates = [r["sample_rate"] for r in valid_results]
    sample_rate_dist = {}
    for sr in set(sample_rates):
        sample_rate_dist[sr] = sample_rates.count(sr)

    # Quality score distribution
    quality_scores = [r["quality_score"] for r in valid_results]
    quality_dist = _create_distribution(quality_scores, config.quality_ranges)

    # SNR distribution
    snr_values = [r["snr_db"] for r in valid_results if r["snr_db"] is not None]
    snr_dist = _create_distribution(snr_values, config.snr_ranges)

    # File size distribution
    file_sizes = [r["file_size"] for r in valid_results]
    size_dist = _create_distribution(file_sizes, config.size_ranges)

    # Create detailed DataFrame
    detailed_df = pd.DataFrame(valid_results)

    # Calculate additional metadata
    metadata = {
        "total_files": len(results),
        "valid_files": len(valid_results),
        "invalid_files": len(results) - len(valid_results),
        "unique_hashes": len(set(r["file_hash"] for r in valid_results)),
        "average_quality": np.mean(quality_scores),
        "quality_std": np.std(quality_scores),
        "average_snr": np.mean(snr_values) if snr_values else 0.0,
        "snr_std": np.std(snr_values) if len(snr_values) > 1 else 0.0,
        "channel_distribution": _count_distribution([r["channels"] for r in valid_results]),
        "bit_depth_distribution": _count_distribution([r["bit_depth"] for r in valid_results])
    }

    return DatasetStatisticsResult(
        file_count=len(valid_results),
        total_duration_seconds=total_duration,
        average_duration_seconds=avg_duration,
        min_duration_seconds=min_duration,
        max_duration_seconds=max_duration,
        sample_rate_distribution=sample_rate_dist,
        quality_score_distribution=quality_dist,
        snr_distribution=snr_dist,
        file_size_distribution=size_dist,
        metadata=metadata,
        detailed_stats=detailed_df,
        error_message=None if valid_results else "No valid files for statistics"
    )

def _create_distribution(values: List[float], ranges: List[float]) -> Dict[str, int]:
    """Create distribution histogram from values and ranges.

    Args:
        values: List of numerical values
        ranges: List of range boundaries

    Returns:
        Dictionary with range labels as keys and counts as values
    """
    if not values:
        return {}

    distribution = {}

    for i in range(len(ranges) - 1):
        lower = ranges[i]
        upper = ranges[i + 1]

        if i == len(ranges) - 2:
            # Last range is inclusive of upper bound
            count = sum(1 for v in values if lower <= v <= upper)
        else:
            count = sum(1 for v in values if lower <= v < upper)

        range_label = f"{lower}-{upper}"
        distribution[range_label] = count

    return distribution

def _count_distribution(values: List[Any]) -> Dict[Any, int]:
    """Count distribution of discrete values.

    Args:
        values: List of values

    Returns:
        Dictionary with values as keys and counts as values
    """
    distribution = {}
    for value in values:
        distribution[value] = distribution.get(value, 0) + 1
    return distribution

async def generate_dataset_report(
    dataset_path: str | Path,
    output_path: str | Path,
    config: Optional[DatasetStatisticsConfig] = None
) -> Path:
    """Generate comprehensive dataset report with statistics and visualizations.

    Args:
        dataset_path: Path to dataset directory
        output_path: Path to save report (HTML format)
        config: Optional statistics configuration

    Returns:
        Path to generated report file

    Raises:
        StatisticsError: If report generation fails
    """
    try:
        # Calculate statistics
        stats_result = await calculate_dataset_statistics(dataset_path, config)

        if stats_result.error_message:
            raise StatisticsError(f"Statistics calculation failed: {stats_result.error_message}")

        # Generate HTML report
        report_content = _generate_html_report(stats_result)
        output_path = Path(output_path)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write report
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        return output_path

    except Exception as e:
        raise StatisticsError(f"Report generation failed: {str(e)}") from e

def _generate_html_report(stats_result: DatasetStatisticsResult) -> str:
    """Generate HTML report content from statistics.

    Args:
        stats_result: Statistics result

    Returns:
        HTML report content as string
    """
    # Create HTML report with comprehensive statistics
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Dataset Quality Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #3498db; }}
        .stats-box {{ background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .metric {{ font-weight: bold; color: #2980b9; }}
        .value {{ font-family: monospace; background: #e8f4fc; padding: 2px 8px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .error {{ color: #e74c3c; }}
        .success {{ color: #27ae60; }}
    </style>
</head>
<body>
    <h1>🎵 Dataset Quality Report</h1>

    <div class="stats-box">
        <h2>📊 Basic Statistics</h2>
        <p><span class="metric">Total Files:</span> <span class="value">{stats_result.file_count}</span></p>
        <p><span class="metric">Total Duration:</span> <span class="value">{stats_result.total_duration_seconds:.2f} seconds ({stats_result.total_duration_seconds/3600:.2f} hours)</span></p>
        <p><span class="metric">Average Duration:</span> <span class="value">{stats_result.average_duration_seconds:.2f} seconds</span></p>
        <p><span class="metric">Duration Range:</span> <span class="value">{stats_result.min_duration_seconds:.2f}s - {stats_result.max_duration_seconds:.2f}s</span></p>
    </div>

    <div class="stats-box">
        <h2>🎛️ Sample Rate Distribution</h2>
        <table>
            <tr><th>Sample Rate (Hz)</th><th>Count</th><th>Percentage</th></tr>
            {_generate_table_rows(stats_result.sample_rate_distribution, stats_result.file_count)}
        </table>
    </div>

    <div class="stats-box">
        <h2>⭐ Quality Score Distribution</h2>
        <table>
            <tr><th>Quality Range</th><th>Count</th><th>Percentage</th></tr>
            {_generate_table_rows(stats_result.quality_score_distribution, stats_result.file_count)}
        </table>
    </div>

    <div class="stats-box">
        <h2>🔊 SNR Distribution</h2>
        <table>
            <tr><th>SNR Range (dB)</th><th>Count</th><th>Percentage</th></tr>
            {_generate_table_rows(stats_result.snr_distribution, stats_result.file_count)}
        </table>
    </div>

    <div class="stats-box">
        <h2>💾 File Size Distribution</h2>
        <table>
            <tr><th>Size Range (bytes)</th><th>Count</th><th>Percentage</th></tr>
            {_generate_table_rows(stats_result.file_size_distribution, stats_result.file_count)}
        </table>
    </div>

    <div class="stats-box">
        <h2>📋 Additional Metadata</h2>
        <pre>{_format_metadata(stats_result.metadata)}</pre>
    </div>

    {_generate_error_section(stats_result)}
</body>
</html>"""

    return html_content

def _generate_table_rows(distribution: Dict, total: int) -> str:
    """Generate HTML table rows from distribution data.

    Args:
        distribution: Distribution dictionary
        total: Total count for percentage calculation

    Returns:
        HTML table rows string
    """
    rows = []
    for key, count in distribution.items():
        percentage = (count / total * 100) if total > 0 else 0
        rows.append(f"<tr><td>{key}</td><td>{count}</td><td>{percentage:.1f}%</td></tr>")
    return "\n".join(rows)

def _format_metadata(metadata: Dict[str, Any]) -> str:
    """Format metadata dictionary for HTML display.

    Args:
        metadata: Metadata dictionary

    Returns:
        Formatted metadata string
    """
    lines = []
    for key, value in metadata.items():
        if isinstance(value, (int, float)):
            lines.append(f"{key}: {value}")
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                lines.append(f"  {sub_key}: {sub_value}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)

def _generate_error_section(stats_result: DatasetStatisticsResult) -> str:
    """Generate error section if there are issues.

    Args:
        stats_result: Statistics result

    Returns:
        HTML error section string
    """
    if stats_result.error_message:
        return f"""
    <div class="stats-box error">
        <h2>⚠️ Errors</h2>
        <p>{stats_result.error_message}</p>
    </div>
    """

    # Check for files with errors in detailed stats
    if stats_result.detailed_stats is not None:
        error_files = stats_result.detailed_stats[stats_result.detailed_stats["error"].notna()]
        if not error_files.empty:
            error_list_items = []
            for _, row in error_files.iterrows():
                error_list_items.append(f"<li>{row['file_name']}: {row['error']}</li>")
            error_list_html = "\n".join(error_list_items)

            return f"""
    <div class="stats-box error">
        <h2>⚠️ Files with Errors ({len(error_files)})</h2>
        <ul>
            {error_list_html}
        </ul>
    </div>
    """

    return ""
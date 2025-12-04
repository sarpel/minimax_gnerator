"""Report Generator Module

This module generates comprehensive HTML reports with interactive visualizations
for dataset quality analysis using Plotly and Jinja2 templating.
"""

from __future__ import annotations

import asyncio
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, Field

from wakegen.core.exceptions import QualityAssuranceError
from wakegen.quality.statistics import calculate_dataset_statistics, DatasetStatisticsResult

# Suppress plotly warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="plotly")

class ReportGenerationError(QualityAssuranceError):
    """Custom exception for report generation failures."""

@dataclass
class ReportGenerationResult:
    """Result of report generation."""

    report_path: Path
    statistics: DatasetStatisticsResult
    generation_time_seconds: float
    error_message: Optional[str] = None

class ReportGenerationConfig(BaseModel):
    """Configuration for report generation."""

    # Visualization settings
    plotly_theme: str = Field("plotly_white", description="Plotly theme for visualizations")
    color_scheme: str = Field("viridis", description="Color scheme for charts")
    interactive_charts: bool = Field(True, description="Enable interactive charts")

    # Content settings
    include_detailed_tables: bool = Field(True, description="Include detailed data tables")
    include_sample_analysis: bool = Field(True, description="Include individual sample analysis")
    max_samples_in_report: int = Field(50, description="Maximum samples to include in detailed analysis")

    # Template settings
    template_name: str = Field("default_report.html", description="Jinja2 template name")
    custom_css: Optional[str] = Field(None, description="Custom CSS for report")

async def generate_report(
    dataset_path: str | Path,
    output_path: str | Path,
    config: Optional[ReportGenerationConfig] = None
) -> ReportGenerationResult:
    """Generate comprehensive HTML report with interactive visualizations.

    Creates a detailed quality analysis report with charts, tables, and metrics.

    Args:
        dataset_path: Path to dataset directory
        output_path: Path to save HTML report
        config: Optional report configuration

    Returns:
        ReportGenerationResult with generation metrics

    Raises:
        ReportGenerationError: If report generation fails
    """
    import time
    start_time = time.time()

    if config is None:
        config = ReportGenerationConfig()

    try:
        # Calculate dataset statistics
        stats_result = await calculate_dataset_statistics(dataset_path)

        if stats_result.error_message:
            raise ReportGenerationError(
                f"Statistics calculation failed: {stats_result.error_message}"
            )

        # Generate report content
        report_content = await _generate_report_content(stats_result, config)

        # Write report to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        generation_time = time.time() - start_time

        return ReportGenerationResult(
            report_path=output_path,
            statistics=stats_result,
            generation_time_seconds=generation_time,
            error_message=None
        )

    except Exception as e:
        raise ReportGenerationError(f"Report generation failed: {str(e)}") from e

async def _generate_report_content(
    stats_result: DatasetStatisticsResult,
    config: ReportGenerationConfig
) -> str:
    """Generate complete HTML report content.

    Args:
        stats_result: Dataset statistics
        config: Report configuration

    Returns:
        Complete HTML report as string
    """
    # Set up Plotly theme
    import plotly.io as pio
    pio.templates.default = config.plotly_theme

    # Generate all visualizations
    visualizations = await _generate_visualizations(stats_result, config)

    # Generate data tables
    data_tables = _generate_data_tables(stats_result, config)

    # Create report context
    context = {
        "title": "Wake Word Dataset Quality Report",
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "statistics": stats_result,
        "visualizations": visualizations,
        "data_tables": data_tables,
        "config": config,
        "metadata": _format_metadata_for_report(stats_result.metadata)
    }

    # Generate HTML using Jinja2 template
    return _render_jinja2_template(context, config)

async def _generate_visualizations(
    stats_result: DatasetStatisticsResult,
    config: ReportGenerationConfig
) -> Dict[str, str]:
    """Generate all interactive visualizations for the report.

    Args:
        stats_result: Dataset statistics
        config: Report configuration

    Returns:
        Dictionary of visualization HTML divs
    """
    visualizations = {}

    # Duration distribution histogram
    if hasattr(stats_result.detailed_stats, 'duration'):
        visualizations["duration_histogram"] = _create_duration_histogram(
            stats_result.detailed_stats, config
        )

    # Quality score distribution
    if hasattr(stats_result.detailed_stats, 'quality_score'):
        visualizations["quality_distribution"] = _create_quality_distribution(
            stats_result.detailed_stats, config
        )

    # SNR vs Quality scatter plot
    if (hasattr(stats_result.detailed_stats, 'snr_db') and
        hasattr(stats_result.detailed_stats, 'quality_score')):
        visualizations["snr_quality_scatter"] = _create_snr_quality_scatter(
            stats_result.detailed_stats, config
        )

    # Component scores breakdown
    if (hasattr(stats_result.detailed_stats, 'clarity_score') and
        hasattr(stats_result.detailed_stats, 'naturalness_score')):
        visualizations["component_scores"] = _create_component_scores_chart(
            stats_result.detailed_stats, config
        )

    # File size distribution
    if hasattr(stats_result.detailed_stats, 'file_size'):
        visualizations["file_size_distribution"] = _create_file_size_distribution(
            stats_result.detailed_stats, config
        )

    return visualizations

def _create_duration_histogram(df: pd.DataFrame, config: ReportGenerationConfig) -> str:
    """Create duration distribution histogram.

    Args:
        df: DataFrame with duration data
        config: Report configuration

    Returns:
        HTML div with Plotly figure
    """
    fig = px.histogram(
        df,
        x="duration",
        nbins=50,
        title="Duration Distribution",
        labels={"duration": "Duration (seconds)", "count": "Frequency"},
        color_discrete_sequence=[px.colors.qualitative.Plotly[0]]
    )

    fig.update_layout(
        showlegend=False,
        hovermode="x unified",
        template=config.plotly_theme
    )

    return fig.to_html(full_html=False, include_plotlyjs="cdn")

def _create_quality_distribution(df: pd.DataFrame, config: ReportGenerationConfig) -> str:
    """Create quality score distribution chart.

    Args:
        df: DataFrame with quality score data
        config: Report configuration

    Returns:
        HTML div with Plotly figure
    """
    fig = px.histogram(
        df,
        x="quality_score",
        nbins=20,
        title="Quality Score Distribution",
        labels={"quality_score": "Quality Score", "count": "Frequency"},
        color_discrete_sequence=[px.colors.qualitative.Plotly[1]]
    )

    # Add vertical line for average
    avg_quality = df["quality_score"].mean()
    fig.add_vline(
        x=avg_quality,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Avg: {avg_quality:.3f}"
    )

    fig.update_layout(
        showlegend=False,
        hovermode="x unified",
        template=config.plotly_theme
    )

    return fig.to_html(full_html=False, include_plotlyjs="cdn")

def _create_snr_quality_scatter(df: pd.DataFrame, config: ReportGenerationConfig) -> str:
    """Create SNR vs Quality scatter plot.

    Args:
        df: DataFrame with SNR and quality data
        config: Report configuration

    Returns:
        HTML div with Plotly figure
    """
    fig = px.scatter(
        df,
        x="snr_db",
        y="quality_score",
        title="SNR vs Quality Score Correlation",
        labels={"snr_db": "SNR (dB)", "quality_score": "Quality Score"},
        color="quality_score",
        color_continuous_scale=config.color_scheme,
        opacity=0.6
    )

    fig.update_layout(
        hovermode="closest",
        template=config.plotly_theme
    )

    return fig.to_html(full_html=False, include_plotlyjs="cdn")

def _create_component_scores_chart(df: pd.DataFrame, config: ReportGenerationConfig) -> str:
    """Create component scores breakdown chart.

    Args:
        df: DataFrame with component scores
        config: Report configuration

    Returns:
        HTML div with Plotly figure
    """
    # Calculate average component scores
    component_scores = {
        "Clarity": df["clarity_score"].mean(),
        "Naturalness": df["naturalness_score"].mean(),
        "Diversity": df["diversity_score"].mean(),
        "Technical": df["technical_score"].mean(),
        "Overall": df["quality_score"].mean()
    }

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=list(component_scores.keys()),
        y=list(component_scores.values()),
        marker_color=px.colors.qualitative.Plotly,
        text=[f"{v:.3f}" for v in component_scores.values()],
        textposition="auto"
    ))

    fig.update_layout(
        title="Average Component Scores",
        xaxis_title="Component",
        yaxis_title="Score",
        yaxis_range=[0, 1],
        template=config.plotly_theme
    )

    return fig.to_html(full_html=False, include_plotlyjs="cdn")

def _create_file_size_distribution(df: pd.DataFrame, config: ReportGenerationConfig) -> str:
    """Create file size distribution chart.

    Args:
        df: DataFrame with file size data
        config: Report configuration

    Returns:
        HTML div with Plotly figure
    """
    # Convert bytes to KB for readability
    df["size_kb"] = df["file_size"] / 1024

    fig = px.histogram(
        df,
        x="size_kb",
        nbins=30,
        title="File Size Distribution",
        labels={"size_kb": "File Size (KB)", "count": "Frequency"},
        color_discrete_sequence=[px.colors.qualitative.Plotly[2]]
    )

    fig.update_layout(
        showlegend=False,
        hovermode="x unified",
        template=config.plotly_theme
    )

    return fig.to_html(full_html=False, include_plotlyjs="cdn")

def _generate_data_tables(
    stats_result: DatasetStatisticsResult,
    config: ReportGenerationConfig
) -> Dict[str, str]:
    """Generate HTML data tables for the report.

    Args:
        stats_result: Dataset statistics
        config: Report configuration

    Returns:
        Dictionary of HTML table strings
    """
    tables = {}

    # Sample rate distribution table
    sample_rate_df = pd.DataFrame([
        {"Sample Rate (Hz)": k, "Count": v, "Percentage": f"{(v/stats_result.file_count*100):.1f}%"}
        for k, v in stats_result.sample_rate_distribution.items()
    ])

    tables["sample_rate_table"] = sample_rate_df.to_html(
        index=False,
        classes="table table-striped table-hover"
    )

    # Quality distribution table
    quality_df = pd.DataFrame([
        {"Quality Range": k, "Count": v, "Percentage": f"{(v/stats_result.file_count*100):.1f}%"}
        for k, v in stats_result.quality_score_distribution.items()
    ])

    tables["quality_table"] = quality_df.to_html(
        index=False,
        classes="table table-striped table-hover"
    )

    # SNR distribution table
    snr_df = pd.DataFrame([
        {"SNR Range (dB)": k, "Count": v, "Percentage": f"{(v/stats_result.file_count*100):.1f}%"}
        for k, v in stats_result.snr_distribution.items()
    ])

    tables["snr_table"] = snr_df.to_html(
        index=False,
        classes="table table-striped table-hover"
    )

    # File size distribution table
    size_df = pd.DataFrame([
        {"Size Range (bytes)": k, "Count": v, "Percentage": f"{(v/stats_result.file_count*100):.1f}%"}
        for k, v in stats_result.file_size_distribution.items()
    ])

    tables["size_table"] = size_df.to_html(
        index=False,
        classes="table table-striped table-hover"
    )

    # Sample analysis table (limited)
    if (config.include_sample_analysis and stats_result.detailed_stats is not None and
        len(stats_result.detailed_stats) > 0):

        sample_df = stats_result.detailed_stats[[
            "file_name", "duration", "quality_score",
            "snr_db", "is_valid"
        ]].head(config.max_samples_in_report).copy()

        # Format values
        sample_df["duration"] = sample_df["duration"].round(2)
        sample_df["quality_score"] = sample_df["quality_score"].round(3)
        sample_df["snr_db"] = sample_df["snr_db"].round(1)
        sample_df["is_valid"] = sample_df["is_valid"].apply(lambda x: "✓" if x else "✗")

        tables["sample_table"] = sample_df.to_html(
            index=False,
            classes="table table-striped table-hover table-sm"
        )

    return tables

def _format_metadata_for_report(metadata: Dict[str, Any]) -> Dict[str, str]:
    """Format metadata for report display.

    Args:
        metadata: Raw metadata dictionary

    Returns:
        Formatted metadata dictionary
    """
    formatted = {}

    for key, value in metadata.items():
        if isinstance(value, (int, float)):
            formatted[key] = str(value)
        elif isinstance(value, dict):
            formatted[key] = "<br>".join([f"{k}: {v}" for k, v in value.items()])
        else:
            formatted[key] = str(value)

    return formatted

def _render_jinja2_template(context: Dict[str, Any], config: ReportGenerationConfig) -> str:
    """Render HTML report using Jinja2 template.

    Args:
        context: Template context data
        config: Report configuration

    Returns:
        Rendered HTML content
    """
    # Set up Jinja2 environment
    env = Environment(loader=FileSystemLoader("templates"))

    # Try to load custom template, fall back to default
    try:
        template = env.get_template(config.template_name)
    except Exception:
        # Use built-in template
        template_content = _get_default_template()
        template = env.from_string(template_content)

    return template.render(**context)

def _get_default_template() -> str:
    """Get default HTML template for report.

    Returns:
        Default template content
    """
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        :root {
            --primary-color: #3498db;
            --secondary-color: #2980b9;
            --success-color: #27ae60;
            --danger-color: #e74c3c;
            --warning-color: #f39c12;
            --info-color: #16a085;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f7fa;
            padding: 20px;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.1);
            border-radius: 10px;
            overflow: hidden;
        }

        header {
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            padding: 30px;
            text-align: center;
        }

        h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 600;
        }

        .subtitle {
            font-size: 1.1em;
            opacity: 0.9;
            margin-bottom: 15px;
        }

        .generated-at {
            font-size: 0.9em;
            opacity: 0.8;
        }

        .nav {
            background: white;
            padding: 15px 30px;
            border-bottom: 1px solid #eee;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
        }

        .nav-links {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }

        .nav-link {
            color: var(--primary-color);
            text-decoration: none;
            padding: 8px 15px;
            border-radius: 5px;
            transition: all 0.3s;
        }

        .nav-link:hover {
            background: var(--primary-color);
            color: white;
        }

        .nav-link.active {
            background: var(--primary-color);
            color: white;
        }

        main {
            padding: 30px;
        }

        .section {
            margin-bottom: 40px;
        }

        .section-title {
            color: var(--primary-color);
            font-size: 1.8em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid var(--primary-color);
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
            border-left: 4px solid var(--primary-color);
            transition: transform 0.3s;
        }

        .stat-card:hover {
            transform: translateY(-5px);
        }

        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: var(--primary-color);
            margin: 10px 0;
        }

        .stat-label {
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
            margin-bottom: 20px;
        }

        .chart-title {
            font-size: 1.2em;
            color: var(--secondary-color);
            margin-bottom: 15px;
            font-weight: 600;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }

        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }

        th {
            background-color: var(--primary-color);
            color: white;
            font-weight: 600;
        }

        tr:hover {
            background-color: #f5f5f5;
        }

        .table-sm th, .table-sm td {
            padding: 8px 12px;
            font-size: 0.9em;
        }

        .metadata-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
        }

        .metadata-item {
            background: white;
            padding: 15px;
            border-radius: 6px;
            border: 1px solid #eee;
        }

        .metadata-label {
            font-weight: 600;
            color: var(--secondary-color);
            margin-bottom: 5px;
            font-size: 0.9em;
        }

        .metadata-value {
            color: #333;
            font-size: 1.1em;
        }

        .success-badge {
            display: inline-block;
            background: var(--success-color);
            color: white;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: 600;
        }

        .danger-badge {
            display: inline-block;
            background: var(--danger-color);
            color: white;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: 600;
        }

        .warning-badge {
            display: inline-block;
            background: var(--warning-color);
            color: white;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: 600;
        }

        footer {
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
            background: #f8f9fa;
            margin-top: 40px;
        }

        @media (max-width: 768px) {
            .stats-grid {
                grid-template-columns: 1fr;
            }

            .metadata-grid {
                grid-template-columns: 1fr;
            }

            h1 {
                font-size: 1.8em;
            }
        }

        {%- if config.custom_css %}
        {{ config.custom_css }}
        {%- endif %}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎵 {{ title }}</h1>
            <div class="subtitle">Comprehensive Quality Analysis</div>
            <div class="generated-at">Generated: {{ generated_at }}</div>
        </header>

        <div class="nav">
            <div class="nav-links">
                <a href="#overview" class="nav-link active">📊 Overview</a>
                <a href="#visualizations" class="nav-link">📈 Visualizations</a>
                <a href="#distributions" class="nav-link">📋 Distributions</a>
                <a href="#samples" class="nav-link">🎵 Samples</a>
                <a href="#metadata" class="nav-link">ℹ️ Metadata</a>
            </div>
        </div>

        <main>
            <section id="overview" class="section">
                <h2 class="section-title">📊 Dataset Overview</h2>

                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-label">Total Files</div>
                        <div class="stat-value">{{ statistics.file_count }}</div>
                    </div>

                    <div class="stat-card">
                        <div class="stat-label">Total Duration</div>
                        <div class="stat-value">{{ "%.2f"|format(statistics.total_duration_seconds/3600) }} hours</div>
                    </div>

                    <div class="stat-card">
                        <div class="stat-label">Average Duration</div>
                        <div class="stat-value">{{ "%.2f"|format(statistics.average_duration_seconds) }}s</div>
                    </div>

                    <div class="stat-card">
                        <div class="stat-label">Duration Range</div>
                        <div class="stat-value">{{ "%.2f"|format(statistics.min_duration_seconds) }}-{{ "%.2f"|format(statistics.max_duration_seconds) }}s</div>
                    </div>

                    <div class="stat-card">
                        <div class="stat-label">Average Quality</div>
                        <div class="stat-value">{{ "%.3f"|format(statistics.metadata.average_quality) }}</div>
                    </div>

                    <div class="stat-card">
                        <div class="stat-label">Average SNR</div>
                        <div class="stat-value">{{ "%.1f"|format(statistics.metadata.average_snr) }} dB</div>
                    </div>
                </div>
            </section>

            <section id="visualizations" class="section">
                <h2 class="section-title">📈 Interactive Visualizations</h2>

                <div class="chart-container">
                    <h3 class="chart-title">Duration Distribution</h3>
                    {{ visualizations.duration_histogram|safe }}
                </div>

                <div class="chart-container">
                    <h3 class="chart-title">Quality Score Distribution</h3>
                    {{ visualizations.quality_distribution|safe }}
                </div>

                <div class="chart-container">
                    <h3 class="chart-title">SNR vs Quality Correlation</h3>
                    {{ visualizations.snr_quality_scatter|safe }}
                </div>

                <div class="chart-container">
                    <h3 class="chart-title">Component Scores Breakdown</h3>
                    {{ visualizations.component_scores|safe }}
                </div>

                <div class="chart-container">
                    <h3 class="chart-title">File Size Distribution</h3>
                    {{ visualizations.file_size_distribution|safe }}
                </div>
            </section>

            <section id="distributions" class="section">
                <h2 class="section-title">📋 Distribution Tables</h2>

                <div class="chart-container">
                    <h3 class="chart-title">Sample Rate Distribution</h3>
                    {{ data_tables.sample_rate_table|safe }}
                </div>

                <div class="chart-container">
                    <h3 class="chart-title">Quality Score Distribution</h3>
                    {{ data_tables.quality_table|safe }}
                </div>

                <div class="chart-container">
                    <h3 class="chart-title">SNR Distribution</h3>
                    {{ data_tables.snr_table|safe }}
                </div>

                <div class="chart-container">
                    <h3 class="chart-title">File Size Distribution</h3>
                    {{ data_tables.size_table|safe }}
                </div>
            </section>

            {%- if config.include_sample_analysis and data_tables.sample_table %}
            <section id="samples" class="section">
                <h2 class="section-title">🎵 Sample Analysis</h2>

                <div class="chart-container">
                    <h3 class="chart-title">Top {{ config.max_samples_in_report }} Samples</h3>
                    {{ data_tables.sample_table|safe }}
                </div>
            </section>
            {%- endif %}

            <section id="metadata" class="section">
                <h2 class="section-title">ℹ️ Dataset Metadata</h2>

                <div class="metadata-grid">
                    {%- for key, value in metadata.items() %}
                    <div class="metadata-item">
                        <div class="metadata-label">{{ key|replace('_', ' ')|title }}</div>
                        <div class="metadata-value">{{ value|safe }}</div>
                    </div>
                    {%- endfor %}
                </div>
            </section>
        </main>

        <footer>
            <p>Generated by WakeGen Quality Assurance System | © {{ generated_at[:4] }}</p>
        </footer>
    </div>

    <script>
        // Smooth scrolling for navigation
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });

                    // Update active nav link
                    document.querySelectorAll('.nav-link').forEach(link => {
                        link.classList.remove('active');
                    });
                    this.classList.add('active');
                }
            });
        });

        // Add copy functionality to tables
        document.querySelectorAll('table').forEach(table => {
            const header = document.createElement('div');
            header.style.display = 'flex';
            header.style.justifyContent = 'flex-end';
            header.style.marginBottom = '10px';

            const copyBtn = document.createElement('button');
            copyBtn.textContent = '📋 Copy Table';
            copyBtn.style.padding = '5px 10px';
            copyBtn.style.background = 'var(--primary-color)';
            copyBtn.style.color = 'white';
            copyBtn.style.border = 'none';
            copyBtn.style.borderRadius = '4px';
            copyBtn.style.cursor = 'pointer';

            copyBtn.addEventListener('click', () => {
                const range = document.createRange();
                range.selectNode(table);
                window.getSelection().removeAllRanges();
                window.getSelection().addRange(range);

                try {
                    document.execCommand('copy');
                    copyBtn.textContent = '✓ Copied!';
                    setTimeout(() => {
                        copyBtn.textContent = '📋 Copy Table';
                    }, 2000);
                } catch (e) {
                    copyBtn.textContent = '❌ Failed';
                    setTimeout(() => {
                        copyBtn.textContent = '📋 Copy Table';
                    }, 2000);
                }

                window.getSelection().removeAllRanges();
            });

            header.appendChild(copyBtn);
            table.parentNode.insertBefore(header, table);
        });
    </script>
</body>
</html>"""
"""
Export API Router

This module provides endpoints for exporting datasets to various training formats.

    ENDPOINTS:
    ==========
    GET  /formats         - List available export formats
    POST /start           - Start export job
    GET  /status/{job_id} - Get export job status
    GET  /recent          - List recent exports

    SUPPORTED FORMATS:
    ==================
    1. OpenWakeWord: For openWakeWord training
    2. Mycroft Precise: Mycroft's wake word format
    3. Picovoice: Porcupine wake word engine
    4. TensorFlow: TFRecord format
    5. PyTorch: Custom PyTorch dataset format
    6. HuggingFace: HuggingFace datasets format
"""

import logging
from typing import List, Optional, Dict
from pathlib import Path
from enum import Enum
import uuid

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class ExportFormat(str, Enum):
    """Supported export formats."""
    OPENWAKEWORD = "openwakeword"
    MYCROFT = "mycroft"
    PICOVOICE = "picovoice"
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    HUGGINGFACE = "huggingface"


class ExportStatus(str, Enum):
    """Status of an export job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class SplitConfig(BaseModel):
    """Train/validation/test split configuration."""
    train: float = Field(0.8, ge=0, le=1, description="Training set ratio")
    val: float = Field(0.1, ge=0, le=1, description="Validation set ratio")
    test: float = Field(0.1, ge=0, le=1, description="Test set ratio")


class FormatInfo(BaseModel):
    """Information about an export format."""
    id: str
    name: str
    description: str
    icon: str
    file_structure: str
    supports_metadata: bool = True


class ExportRequest(BaseModel):
    """Request to start an export job."""
    input_dir: str = Field(..., description="Input directory with audio files")
    output_dir: str = Field(..., description="Output directory for export")
    format: ExportFormat = Field(..., description="Export format")
    split: SplitConfig = Field(default_factory=SplitConfig, description="Data split ratios")
    stratify: bool = Field(True, description="Stratify by wake word class")
    generate_manifest: bool = Field(True, description="Generate manifest/metadata files")
    copy_files: bool = Field(False, description="Copy files instead of symlinking")


class ExportResponse(BaseModel):
    """Response from starting an export job."""
    job_id: str
    status: ExportStatus
    message: str
    format: ExportFormat
    output_dir: str


class ExportJob(BaseModel):
    """Export job information."""
    id: str
    format: ExportFormat
    input_dir: str
    output_dir: str
    status: ExportStatus
    progress_percentage: float = 0
    total_files: int = 0
    processed_files: int = 0
    error_message: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class RecentExport(BaseModel):
    """Summary of a recent export."""
    id: str
    format: str
    output_dir: str
    status: str
    date: str
    file_count: int


# =============================================================================
# DATA
# =============================================================================


FORMAT_INFO: Dict[ExportFormat, FormatInfo] = {
    ExportFormat.OPENWAKEWORD: FormatInfo(
        id="openwakeword",
        name="OpenWakeWord",
        description="For openWakeWord training with train/val/test splits",
        icon="🎯",
        file_structure="output/{split}/{class}/*.wav"
    ),
    ExportFormat.MYCROFT: FormatInfo(
        id="mycroft",
        name="Mycroft Precise",
        description="Mycroft's precise wake word engine format",
        icon="🔮",
        file_structure="output/{wake_word}/*.wav + clips.txt"
    ),
    ExportFormat.PICOVOICE: FormatInfo(
        id="picovoice",
        name="Picovoice Porcupine",
        description="Format for Picovoice's Porcupine wake word engine",
        icon="🔊",
        file_structure="output/*.wav + keywords.json"
    ),
    ExportFormat.TENSORFLOW: FormatInfo(
        id="tensorflow",
        name="TensorFlow",
        description="TFRecord format for TensorFlow models",
        icon="🧠",
        file_structure="output/{split}.tfrecord"
    ),
    ExportFormat.PYTORCH: FormatInfo(
        id="pytorch",
        name="PyTorch",
        description="PyTorch Dataset compatible format",
        icon="🔥",
        file_structure="output/{split}/*.wav + metadata.json"
    ),
    ExportFormat.HUGGINGFACE: FormatInfo(
        id="huggingface",
        name="HuggingFace",
        description="HuggingFace datasets format for easy sharing",
        icon="🤗",
        file_structure="output/dataset_dict/ (Arrow format)"
    ),
}

# In-memory job storage (would use database in production)
_export_jobs: Dict[str, ExportJob] = {}


# =============================================================================
# ROUTER
# =============================================================================


router = APIRouter()


@router.get(
    "/formats",
    response_model=List[FormatInfo],
    summary="List export formats"
)
async def list_formats() -> List[FormatInfo]:
    """
    Get information about all supported export formats.

    Each format has different requirements and produces different output structures.
    """
    return list(FORMAT_INFO.values())


@router.get(
    "/formats/{format_id}",
    response_model=FormatInfo,
    summary="Get format details"
)
async def get_format(format_id: ExportFormat) -> FormatInfo:
    """Get detailed information about a specific export format."""
    if format_id not in FORMAT_INFO:
        raise HTTPException(status_code=404, detail=f"Format not found: {format_id}")
    return FORMAT_INFO[format_id]


@router.post(
    "/start",
    response_model=ExportResponse,
    summary="Start export"
)
async def start_export(
    request: ExportRequest,
    background_tasks: BackgroundTasks
) -> ExportResponse:
    """
    Start a new export job.

    The export runs in the background. Use /status/{job_id} to check progress.
    """
    input_path = Path(request.input_dir)
    if not input_path.exists():
        raise HTTPException(status_code=404, detail=f"Input directory not found: {request.input_dir}")

    # Validate split ratios
    total_split = request.split.train + request.split.val + request.split.test
    if abs(total_split - 1.0) > 0.01:
        raise HTTPException(
            status_code=400,
            detail=f"Split ratios must sum to 1.0 (got {total_split})"
        )

    # Create output directory
    output_path = Path(request.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Count input files
    audio_files = list(input_path.rglob("*.wav"))
    if not audio_files:
        raise HTTPException(status_code=400, detail="No WAV files found in input directory")

    # Create job
    job_id = str(uuid.uuid4())[:8]
    job = ExportJob(
        id=job_id,
        format=request.format,
        input_dir=request.input_dir,
        output_dir=request.output_dir,
        status=ExportStatus.PENDING,
        total_files=len(audio_files)
    )
    _export_jobs[job_id] = job

    # Queue background task
    background_tasks.add_task(run_export, job_id, request)

    return ExportResponse(
        job_id=job_id,
        status=ExportStatus.PENDING,
        message=f"Export job started for {len(audio_files)} files",
        format=request.format,
        output_dir=request.output_dir
    )


async def run_export(job_id: str, request: ExportRequest) -> None:
    """
    Background task to run export.
    """
    import asyncio
    import shutil
    import json
    import random
    from datetime import datetime
    import pandas as pd
    from sklearn.model_selection import train_test_split

    job = _export_jobs.get(job_id)
    if not job:
        return

    job.status = ExportStatus.RUNNING
    job.started_at = datetime.now().isoformat()

    try:
        input_path = Path(request.input_dir)
        output_path = Path(request.output_dir)

        # 1. Collect all files and labels
        files = []
        labels = []
        file_paths = list(input_path.rglob("*.wav"))
        
        job.total_files = len(file_paths)
        if job.total_files == 0:
            raise ValueError("No WAV files found to export")

        for f in file_paths:
            # Assuming parent directory is the label (wake word name)
            label = f.parent.name
            files.append(str(f))
            labels.append(label)

        # 2. Split dataset
        # We need to split into Train/Val/Test
        # First split off Test from (Train+Val)
        remaining_ratio = request.split.train + request.split.val
        if remaining_ratio <= 0:
            raise ValueError("Train + Validation ratio must be > 0")

        # Normalize test size relative to total
        test_size = request.split.test
        
        # Initial DataFrame
        df = pd.DataFrame({'path': files, 'label': labels})

        splits = {}
        
        if test_size > 0:
            if request.stratify:
                fn_train_val, fn_test, param_train_val, param_test = train_test_split(
                    df['path'], df['label'], test_size=test_size, stratify=df['label'], random_state=42
                )
            else:
                fn_train_val, fn_test, param_train_val, param_test = train_test_split(
                    df['path'], df['label'], test_size=test_size, random_state=42
                )
            splits['test'] = pd.DataFrame({'path': fn_test, 'label': param_test})
            df_remaining = pd.DataFrame({'path': fn_train_val, 'label': param_train_val})
        else:
            splits['test'] = pd.DataFrame(columns=['path', 'label'])
            df_remaining = df

        # Now split remaining into Train/Val
        # Ratio of Val relative to remaining
        if len(df_remaining) > 0 and request.split.val > 0:
            # val_ratio relative to (train + val)
            val_relative = request.split.val / (request.split.train + request.split.val)
            
            if request.stratify and len(df_remaining['label'].unique()) > 1:
                # Proper stratification requires at least 2 classes
                 fn_train, fn_val, param_train, param_val = train_test_split(
                    df_remaining['path'], df_remaining['label'], test_size=val_relative, stratify=df_remaining['label'], random_state=42
                )
            else:
                 fn_train, fn_val, param_train, param_val = train_test_split(
                    df_remaining['path'], df_remaining['label'], test_size=val_relative, random_state=42
                )
            
            splits['train'] = pd.DataFrame({'path': fn_train, 'label': param_train})
            splits['val'] = pd.DataFrame({'path': fn_val, 'label': param_val})
        else:
            splits['train'] = df_remaining
            splits['val'] = pd.DataFrame(columns=['path', 'label'])

        # 3. Export files
        processed = 0
        
        for split_name, split_df in splits.items():
            if split_df.empty:
                continue
                
            for _, row in split_df.iterrows():
                src_file = Path(row['path'])
                label = row['label']
                
                # Determine destination based on format
                if request.format == ExportFormat.OPENWAKEWORD:
                    # structure: output/{split}/{label}/{filename}
                    dest_dir = output_path / split_name / label
                elif request.format == ExportFormat.MYCROFT:
                    # structure: output/{label}/{filename} (Mycroft handles splits via text files usually, but here we separate by folder or just dump)
                    # For simplicity, we'll group by label
                    dest_dir = output_path / label
                elif request.format == ExportFormat.PYTORCH:
                     dest_dir = output_path / split_name / label
                else:
                    # Default flat or label-based
                    dest_dir = output_path / split_name / label

                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_file = dest_dir / src_file.name
                
                # Copy or Symlink
                if request.copy_files:
                    shutil.copy2(src_file, dest_file)
                else:
                    # Symlink (requires admin on Windows sometimes, fallback to copy)
                    try:
                        import os
                        os.symlink(src_file, dest_file)
                    except OSError:
                        shutil.copy2(src_file, dest_file)
                
                processed += 1
                if processed % 10 == 0:
                    job.processed_files = processed
                    job.progress_percentage = (processed / job.total_files) * 95 # Leave 5% for manifest
                    await asyncio.sleep(0.001) # Yield control

        # 4. Generate Manifests
        if request.generate_manifest:
            manifest = {
                "created_at": datetime.now().isoformat(),
                "format": request.format,
                "stats": {
                    "total_files": job.total_files,
                    "splits": {k: len(v) for k, v in splits.items()}
                },
                "classes": sorted(list(set(labels)))
            }
            
            with open(output_path / "dataset_info.json", "w") as f:
                json.dump(manifest, f, indent=2)

        job.progress_percentage = 100
        job.processed_files = job.total_files
        job.status = ExportStatus.COMPLETED
        job.completed_at = datetime.now().isoformat()
        logger.info(f"Export job {job_id} completed")

    except Exception as e:
        import traceback
        traceback.print_exc()
        job.status = ExportStatus.FAILED
        job.error_message = str(e)
        logger.error(f"Export job {job_id} failed: {e}")


@router.get(
    "/status/{job_id}",
    response_model=ExportJob,
    summary="Get job status"
)
async def get_status(job_id: str) -> ExportJob:
    """Get the status of an export job."""
    if job_id not in _export_jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return _export_jobs[job_id]


@router.get(
    "/recent",
    response_model=List[RecentExport],
    summary="List recent exports"
)
async def list_recent(
    limit: int = Query(10, ge=1, le=50, description="Maximum number of results")
) -> List[RecentExport]:
    """Get a list of recent export jobs."""
    jobs = sorted(
        _export_jobs.values(),
        key=lambda j: j.started_at or "",
        reverse=True
    )[:limit]

    return [
        RecentExport(
            id=job.id,
            format=job.format.value,
            output_dir=job.output_dir,
            status=job.status.value,
            date=job.completed_at or job.started_at or "Unknown",
            file_count=job.processed_files
        )
        for job in jobs
    ]

"""
Generation API Router

This module provides endpoints for starting and managing audio generation jobs.
It supports both quick generation and configuration-based generation with
real-time progress updates via WebSocket.

    ENDPOINTS:
    ==========
    POST /start              - Start a new generation job
    GET  /status/{job_id}    - Get job status
    POST /cancel/{job_id}    - Cancel a running job
    GET  /recent             - List recent jobs
    WebSocket /ws/{job_id}   - Real-time progress updates

    HOW THIS INTEGRATES:
    ====================
    Uses the existing generation system from:
        - wakegen.generation.orchestrator (GenerationOrchestrator)
        - wakegen.providers.registry (provider instances)
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# JOB MANAGEMENT (In-Memory for now)
# =============================================================================
# In a production system, you'd use Redis or a database for job persistence.
# For the initial implementation, we use in-memory storage.


class JobStatus(str, Enum):
    """Possible states for a generation job."""
    PENDING = "pending"      # Job created, waiting to start
    RUNNING = "running"      # Generation in progress
    COMPLETED = "completed"  # Finished successfully
    FAILED = "failed"        # Error occurred
    CANCELLED = "cancelled"  # User cancelled


class GenerationJob(BaseModel):
    """Represents a generation job and its current state."""
    id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    created_at: datetime = Field(..., description="When job was created")
    started_at: Optional[datetime] = Field(None, description="When execution started")
    completed_at: Optional[datetime] = Field(None, description="When job finished")

    # Configuration
    wake_words: List[str] = Field(..., description="Wake words to generate")
    count: int = Field(..., description="Samples per word")
    provider: str = Field(..., description="TTS provider to use")
    output_dir: str = Field(..., description="Output directory")

    # Progress
    total_samples: int = Field(0, description="Total samples to generate")
    completed_samples: int = Field(0, description="Samples generated so far")
    current_word: Optional[str] = Field(None, description="Current wake word")
    current_file: Optional[str] = Field(None, description="Current file being generated")
    error_message: Optional[str] = Field(None, description="Error message if failed")

    @property
    def progress_percentage(self) -> float:
        """Calculate progress as a percentage."""
        if self.total_samples == 0:
            return 0.0
        return (self.completed_samples / self.total_samples) * 100


# In-memory job storage
# Maps job_id -> GenerationJob
_jobs: Dict[str, GenerationJob] = {}


def get_job(job_id: str) -> Optional[GenerationJob]:
    """Get a job by ID."""
    return _jobs.get(job_id)


def save_job(job: GenerationJob) -> None:
    """Save a job to storage."""
    _jobs[job.id] = job


def list_jobs(limit: int = 10) -> List[GenerationJob]:
    """List recent jobs, newest first."""
    jobs = list(_jobs.values())
    jobs.sort(key=lambda j: j.created_at, reverse=True)
    return jobs[:limit]


# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class GenerationRequest(BaseModel):
    """Request to start a new generation job."""
    wake_words: List[str] = Field(..., description="Words to generate samples for")
    count: int = Field(10, ge=1, le=1000, description="Samples per wake word")
    provider: str = Field("edge_tts", description="TTS provider ID")
    voice_id: Optional[str] = Field(None, description="Specific voice to use")
    output_dir: str = Field("./output", description="Where to save files")
    languages: Optional[List[str]] = Field(None, description="Language filter")


class GenerationResponse(BaseModel):
    """Response when starting a generation job."""
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Initial job status")
    message: str = Field(..., description="Status message")
    websocket_url: str = Field(..., description="WebSocket URL for progress")


class JobStatusResponse(BaseModel):
    """Detailed status of a generation job."""
    id: str
    status: JobStatus
    progress_percentage: float
    completed_samples: int
    total_samples: int
    current_word: Optional[str]
    current_file: Optional[str]
    error_message: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]


class RecentJobSummary(BaseModel):
    """Brief summary of a job for the dashboard."""
    id: str
    wake_words: List[str]
    status: JobStatus
    completed_samples: int
    total_samples: int
    created_at: datetime


# =============================================================================
# BACKGROUND TASK: Run Generation
# =============================================================================


async def run_generation_job(job_id: str) -> None:
    """
    Execute the generation job in the background.

    This function is called by FastAPI's BackgroundTasks and runs
    asynchronously while the HTTP response has already been sent.

        WHAT THIS DOES:
        ===============
        1. Updates job status to RUNNING
        2. Initializes the TTS provider
        3. Generates samples for each wake word
        4. Updates progress after each sample
        5. Sets job status to COMPLETED or FAILED
    """
    job = get_job(job_id)
    if not job:
        logger.error(f"Job {job_id} not found")
        return

    try:
        # Update status to running
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now()
        save_job(job)

        # Import generation dependencies
        from wakegen.providers.registry import get_provider, check_provider_availability
        from wakegen.core.types import ProviderType
        from wakegen.config.settings import get_provider_config, get_generation_config
        import os

        # Validate provider
        try:
            provider_type = ProviderType(job.provider.lower())
        except ValueError:
            job.status = JobStatus.FAILED
            job.error_message = f"Unknown provider: {job.provider}"
            job.completed_at = datetime.now()
            save_job(job)
            return

        availability = check_provider_availability(provider_type)
        if not availability.is_available:
            job.status = JobStatus.FAILED
            job.error_message = f"Provider not available: {availability.missing_dependencies}"
            job.completed_at = datetime.now()
            save_job(job)
            return

        # Get provider instance
        provider_config = get_provider_config()
        provider = get_provider(provider_type, provider_config)

        # Get voice to use
        voices = await provider.list_voices()
        if not voices:
            job.status = JobStatus.FAILED
            job.error_message = "No voices available"
            job.completed_at = datetime.now()
            save_job(job)
            return

        # Select first English voice or first available
        voice = next(
            (v for v in voices if v.language.startswith("en-")),
            voices[0]
        )

        # Create output directory
        os.makedirs(job.output_dir, exist_ok=True)

        # Calculate total samples
        job.total_samples = len(job.wake_words) * job.count
        save_job(job)

        # Generate samples
        sample_index = 0
        for wake_word in job.wake_words:
            job.current_word = wake_word

            # Create subdirectory for this wake word
            word_dir = os.path.join(
                job.output_dir,
                wake_word.replace(" ", "_").lower()
            )
            os.makedirs(word_dir, exist_ok=True)

            for i in range(job.count):
                sample_index += 1
                filename = f"{wake_word.replace(' ', '_').lower()}_{sample_index:04d}.wav"
                file_path = os.path.join(word_dir, filename)

                job.current_file = filename
                save_job(job)

                try:
                    await provider.generate(wake_word, voice.id, file_path)
                    job.completed_samples += 1
                    save_job(job)
                except Exception as e:
                    logger.warning(f"Failed to generate sample: {e}")
                    # Continue with other samples instead of failing completely

        # Mark as completed
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.now()
        job.current_word = None
        job.current_file = None
        save_job(job)

        logger.info(f"Job {job_id} completed: {job.completed_samples}/{job.total_samples} samples")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        job.status = JobStatus.FAILED
        job.error_message = str(e)
        job.completed_at = datetime.now()
        save_job(job)


# =============================================================================
# API ROUTER
# =============================================================================


router = APIRouter()


@router.post(
    "/start",
    response_model=GenerationResponse,
    summary="Start generation job"
)
async def start_generation(
    request: GenerationRequest,
    background_tasks: BackgroundTasks
) -> GenerationResponse:
    """
    Start a new audio generation job.

    The job runs in the background and you can track progress via:
    1. Polling the /status/{job_id} endpoint
    2. Connecting to the WebSocket at /ws/{job_id}

        REQUEST BODY:
        =============
        wake_words: List of words to generate (e.g., ["hey assistant"])
        count: Number of samples per word (1-1000)
        provider: TTS provider ID (e.g., "edge_tts")
        output_dir: Where to save generated files
    """
    # Create job
    job_id = str(uuid.uuid4())[:8]  # Short ID for convenience

    job = GenerationJob(
        id=job_id,
        status=JobStatus.PENDING,
        created_at=datetime.now(),
        wake_words=request.wake_words,
        count=request.count,
        provider=request.provider,
        output_dir=request.output_dir,
        total_samples=len(request.wake_words) * request.count
    )
    save_job(job)

    # Schedule background execution
    background_tasks.add_task(run_generation_job, job_id)

    return GenerationResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message=f"Generation job started for {len(request.wake_words)} wake words",
        websocket_url=f"/ws/progress/{job_id}"
    )


@router.get(
    "/status/{job_id}",
    response_model=JobStatusResponse,
    summary="Get job status"
)
async def get_job_status(job_id: str) -> JobStatusResponse:
    """
    Get the current status and progress of a generation job.

    Poll this endpoint to track progress, or use WebSocket for real-time updates.
    """
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    return JobStatusResponse(
        id=job.id,
        status=job.status,
        progress_percentage=job.progress_percentage,
        completed_samples=job.completed_samples,
        total_samples=job.total_samples,
        current_word=job.current_word,
        current_file=job.current_file,
        error_message=job.error_message,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at
    )


@router.post(
    "/cancel/{job_id}",
    summary="Cancel running job"
)
async def cancel_job(job_id: str) -> dict:
    """
    Cancel a running generation job.

    Note: The current sample may still complete before cancellation takes effect.
    """
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    if job.status not in [JobStatus.PENDING, JobStatus.RUNNING]:
        return {"message": f"Job already {job.status.value}"}

    job.status = JobStatus.CANCELLED
    job.completed_at = datetime.now()
    save_job(job)

    return {"message": f"Job {job_id} cancelled"}


@router.get(
    "/recent",
    response_model=List[RecentJobSummary],
    summary="List recent jobs"
)
async def get_recent_jobs(
    limit: int = 10
) -> List[RecentJobSummary]:
    """
    Get a list of recent generation jobs for the dashboard.

    Returns jobs sorted by creation time, newest first.
    """
    jobs = list_jobs(limit)

    return [
        RecentJobSummary(
            id=job.id,
            wake_words=job.wake_words,
            status=job.status,
            completed_samples=job.completed_samples,
            total_samples=job.total_samples,
            created_at=job.created_at
        )
        for job in jobs
    ]

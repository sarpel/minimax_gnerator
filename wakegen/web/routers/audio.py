"""
Audio Serving API Router

This module provides endpoints for serving and managing audio files.
It enables audio playback, waveform data generation, and file management.

    ENDPOINTS:
    ==========
    GET  /files              - List audio files in a directory
    GET  /play/{path}        - Stream an audio file for playback
    GET  /waveform/{path}    - Get waveform data for visualization
    GET  /info/{path}        - Get audio file metadata
    DELETE /{path}           - Delete an audio file

    WHY A DEDICATED AUDIO ROUTER?
    =============================
    1. Audio files need proper Content-Type headers for browser playback
    2. Waveform data requires audio processing (librosa)
    3. We want to add caching and validation
    4. Separation of concerns from generation/config endpoints
"""

import logging
import os
from pathlib import Path
from typing import List, Optional
import base64

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class AudioFileInfo(BaseModel):
    """Information about an audio file."""
    filename: str = Field(..., description="File name")
    path: str = Field(..., description="Relative path")
    size_bytes: int = Field(..., description="File size in bytes")
    duration_seconds: Optional[float] = Field(None, description="Duration if available")
    sample_rate: Optional[int] = Field(None, description="Sample rate if available")


class AudioFileList(BaseModel):
    """List of audio files in a directory."""
    directory: str
    files: List[AudioFileInfo]
    total_count: int


class WaveformData(BaseModel):
    """Waveform data for visualization."""
    filename: str
    samples: List[float] = Field(..., description="Normalized amplitude values (-1 to 1)")
    duration_seconds: float
    sample_rate: int


class AudioMetadata(BaseModel):
    """Detailed metadata for an audio file."""
    filename: str
    path: str
    format: str
    duration_seconds: float
    sample_rate: int
    channels: int
    bit_depth: Optional[int] = None
    size_bytes: int


# =============================================================================
# ROUTER
# =============================================================================


router = APIRouter()


@router.get(
    "/files",
    response_model=AudioFileList,
    summary="List audio files"
)
async def list_audio_files(
    directory: str = Query("./output", description="Directory to list"),
    extension: str = Query("wav", description="File extension to filter")
) -> AudioFileList:
    """
    List all audio files in the specified directory.

    Scans the directory for audio files and returns basic info about each.
    """
    dir_path = Path(directory)

    if not dir_path.exists():
        raise HTTPException(status_code=404, detail=f"Directory not found: {directory}")

    if not dir_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {directory}")

    files = []
    for file_path in dir_path.rglob(f"*.{extension}"):
        try:
            stat = file_path.stat()
            files.append(AudioFileInfo(
                filename=file_path.name,
                path=str(file_path.relative_to(dir_path)),
                size_bytes=stat.st_size
            ))
        except Exception as e:
            logger.warning(f"Error reading file {file_path}: {e}")

    return AudioFileList(
        directory=directory,
        files=files,
        total_count=len(files)
    )


@router.get(
    "/play/{file_path:path}",
    summary="Stream audio file",
    response_class=FileResponse
)
async def play_audio(file_path: str) -> FileResponse:
    """
    Stream an audio file for browser playback.

    Returns the audio file with proper Content-Type headers so browsers
    can play it in an <audio> element.

        PATH PARAMETERS:
        ================
        file_path: Path to the audio file (can include subdirectories)

        RETURNS:
        ========
        FileResponse with audio/wav or appropriate content type
    """
    # Resolve the path (prevents directory traversal attacks)
    path = Path(file_path).resolve()

    # Basic security check - ensure it's a real file
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    if not path.is_file():
        raise HTTPException(status_code=400, detail="Path is not a file")

    # Determine content type based on extension
    extension = path.suffix.lower()
    content_types = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
        ".m4a": "audio/mp4"
    }
    content_type = content_types.get(extension, "application/octet-stream")

    return FileResponse(
        path=str(path),
        media_type=content_type,
        filename=path.name
    )


@router.get(
    "/waveform/{file_path:path}",
    response_model=WaveformData,
    summary="Get waveform data"
)
async def get_waveform(
    file_path: str,
    num_samples: int = Query(200, ge=50, le=1000, description="Number of waveform points")
) -> WaveformData:
    """
    Generate waveform data for audio visualization.

    Processes the audio file and returns downsampled amplitude data
    suitable for drawing a waveform in the browser.

        PATH PARAMETERS:
        ================
        file_path: Path to the audio file

        QUERY PARAMETERS:
        =================
        num_samples: Number of points in the waveform (default 200)

        RETURNS:
        ========
        WaveformData with normalized amplitude values

        HOW IT WORKS:
        =============
        1. Load audio file with librosa
        2. Calculate RMS energy for chunks
        3. Normalize to -1..1 range
        4. Downsample to requested number of points
    """
    path = Path(file_path).resolve()

    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    try:
        import librosa
        import numpy as np

        # Load audio file
        y, sr = librosa.load(str(path), sr=None, mono=True)

        # Calculate duration
        duration = len(y) / sr

        # Calculate chunk size for downsampling
        chunk_size = max(1, len(y) // num_samples)

        # Calculate RMS for each chunk
        waveform_samples = []
        for i in range(0, len(y), chunk_size):
            chunk = y[i:i + chunk_size]
            rms = np.sqrt(np.mean(chunk ** 2))
            waveform_samples.append(float(rms))

        # Truncate to exact number of samples
        waveform_samples = waveform_samples[:num_samples]

        # Normalize to 0..1 range
        max_val = max(waveform_samples) if waveform_samples else 1
        if max_val > 0:
            waveform_samples = [s / max_val for s in waveform_samples]

        return WaveformData(
            filename=path.name,
            samples=waveform_samples,
            duration_seconds=duration,
            sample_rate=sr
        )

    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="librosa not installed. Install with: pip install librosa"
        )
    except Exception as e:
        logger.error(f"Error generating waveform: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")


@router.get(
    "/info/{file_path:path}",
    response_model=AudioMetadata,
    summary="Get audio metadata"
)
async def get_audio_info(file_path: str) -> AudioMetadata:
    """
    Get detailed metadata for an audio file.

    Returns information like duration, sample rate, channels, etc.
    """
    path = Path(file_path).resolve()

    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    try:
        import soundfile as sf

        info = sf.info(str(path))

        return AudioMetadata(
            filename=path.name,
            path=file_path,
            format=info.format,
            duration_seconds=info.duration,
            sample_rate=info.samplerate,
            channels=info.channels,
            size_bytes=path.stat().st_size
        )

    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="soundfile not installed"
        )
    except Exception as e:
        logger.error(f"Error reading audio info: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")


@router.delete(
    "/{file_path:path}",
    summary="Delete audio file"
)
async def delete_audio(file_path: str) -> dict:
    """
    Delete an audio file.

    Use with caution - this permanently removes the file!
    """
    path = Path(file_path).resolve()

    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    if not path.is_file():
        raise HTTPException(status_code=400, detail="Path is not a file")

    # Only allow deleting audio files
    if path.suffix.lower() not in [".wav", ".mp3", ".ogg", ".flac", ".m4a"]:
        raise HTTPException(status_code=400, detail="Not an audio file")

    try:
        os.remove(str(path))
        return {"message": f"Deleted {path.name}", "path": file_path}
    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting: {str(e)}")

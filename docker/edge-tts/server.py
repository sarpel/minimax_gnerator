"""
Edge TTS HTTP API Server

CONCEPT: This FastAPI application provides a REST API wrapper around Microsoft's
Edge TTS (Text-to-Speech) service. Edge TTS is a free, cloud-based service that
doesn't require API keys - it uses the same voices as Microsoft Edge browser.

ENDPOINTS:
    POST /api/tts          - Generate speech from text
    GET  /api/voices       - List all available voices
    GET  /health           - Health check for container orchestration

USAGE EXAMPLE:
    curl -X POST http://localhost:5000/api/tts \
        -H "Content-Type: application/json" \
        -d '{"text": "Hello world", "voice_id": "en-US-AriaNeural"}'
"""

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import edge_tts
import uuid
import os
from pathlib import Path
from typing import Optional

# =============================================================================
# FASTAPI APP CONFIGURATION
# =============================================================================
# CONCEPT: FastAPI automatically generates OpenAPI documentation at /docs

app = FastAPI(
    title="Edge TTS API",
    description="REST API wrapper for Microsoft Edge Text-to-Speech",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI available at /docs
    redoc_url="/redoc",  # ReDoc available at /redoc
)

# Output directory for generated audio files
OUTPUT_DIR = Path("/app/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================
# CONCEPT: Pydantic models define the shape of request/response data.
# FastAPI uses these for automatic validation and documentation.


class TTSRequest(BaseModel):
    """Request model for text-to-speech generation."""

    # The text to convert to speech
    text: str = Field(
        ...,  # ... means required
        description="Text to synthesize into speech",
        min_length=1,
        max_length=5000,
        examples=["Hello, this is a test."],
    )

    # Voice ID in format "locale-VoiceName" (e.g., "en-US-AriaNeural")
    voice_id: str = Field(
        default="en-US-AriaNeural",
        description="Voice ID to use for synthesis",
        examples=["en-US-AriaNeural", "tr-TR-EmelNeural", "en-GB-SoniaNeural"],
    )

    # Optional: Rate adjustment (-50% to +100%)
    rate: Optional[str] = Field(
        default=None,
        description="Speech rate adjustment (e.g., '+20%', '-10%')",
        examples=["+20%", "-10%", "+50%"],
    )

    # Optional: Pitch adjustment
    pitch: Optional[str] = Field(
        default=None,
        description="Pitch adjustment (e.g., '+10Hz', '-5Hz')",
        examples=["+10Hz", "-5Hz"],
    )


class TTSResponse(BaseModel):
    """Response model for successful TTS generation."""

    status: str = "success"
    output_path: str
    filename: str
    download_url: str


class VoiceInfo(BaseModel):
    """Information about an available voice."""

    id: str
    name: str
    locale: str
    gender: str


# =============================================================================
# API ENDPOINTS
# =============================================================================


@app.post("/api/tts", response_model=TTSResponse)
async def generate_speech(request: TTSRequest):
    """
    Generate speech from text using Edge TTS.

    HOW IT WORKS:
    1. Validate the request (FastAPI does this automatically via Pydantic)
    2. Create a Communicate object with the text and voice
    3. Generate audio and save to file
    4. Return the file path/URL

    Args:
        request: TTSRequest with text and voice_id

    Returns:
        TTSResponse with path to generated audio file

    Raises:
        HTTPException 400: If voice_id is invalid
        HTTPException 500: If generation fails
    """
    try:
        # Generate unique filename to avoid collisions
        filename = f"{uuid.uuid4()}.mp3"
        output_path = OUTPUT_DIR / filename

        # Create the Edge TTS communicator
        # SYNTAX: edge_tts.Communicate(text, voice, rate=None, pitch=None)
        communicate = edge_tts.Communicate(
            text=request.text,
            voice=request.voice_id,
            rate=request.rate,
            pitch=request.pitch,
        )

        # Generate and save the audio
        # CONCEPT: save() is an async method that downloads the audio from
        # Microsoft's servers and writes it to the specified path
        await communicate.save(str(output_path))

        return TTSResponse(
            status="success",
            output_path=str(output_path),
            filename=filename,
            download_url=f"/api/download/{filename}",
        )

    except Exception as e:
        error_msg = str(e)

        # Provide helpful error messages for common issues
        if "No audio was received" in error_msg:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid voice_id '{request.voice_id}'. "
                f"Use /api/voices to see available voices.",
            )

        raise HTTPException(
            status_code=500, detail=f"TTS generation failed: {error_msg}"
        )


@app.get("/api/download/{filename}")
async def download_audio(filename: str):
    """
    Download a generated audio file.

    Args:
        filename: Name of the audio file to download

    Returns:
        The audio file as a downloadable response
    """
    file_path = OUTPUT_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(path=str(file_path), media_type="audio/mpeg", filename=filename)


@app.get("/api/voices")
async def list_voices(locale: Optional[str] = None):
    """
    List all available Edge TTS voices.

    CONCEPT: Edge TTS provides 300+ voices across many languages.
    This endpoint returns all of them, optionally filtered by locale.

    Args:
        locale: Optional filter by locale (e.g., "en-US", "tr-TR")

    Returns:
        List of available voices with their IDs and metadata
    """
    try:
        # Fetch the list of voices from Edge TTS
        voices = await edge_tts.list_voices()

        # Filter by locale if specified
        if locale:
            voices = [v for v in voices if v["Locale"].startswith(locale)]

        # Transform to our response format
        voice_list = []
        for v in voices:
            voice_list.append(
                {
                    "id": v["ShortName"],  # e.g., "en-US-AriaNeural"
                    "name": v.get("FriendlyName", v["ShortName"]),
                    "locale": v["Locale"],  # e.g., "en-US"
                    "gender": v["Gender"],  # "Male" or "Female"
                }
            )

        return {"count": len(voice_list), "voices": voice_list}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list voices: {str(e)}")


@app.get("/health")
async def health_check():
    """
    Health check endpoint for container orchestration.

    CONCEPT: Docker Compose and Kubernetes use this to verify the
    container is running correctly. Returns 200 OK if healthy.
    """
    return {"status": "healthy", "service": "edge-tts"}


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Edge TTS API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "generate": "POST /api/tts",
            "voices": "GET /api/voices",
            "download": "GET /api/download/{filename}",
        },
    }

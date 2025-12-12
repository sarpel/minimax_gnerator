"""
Bark TTS HTTP API Server

CONCEPT: Bark is Suno's transformer-based text-to-audio model. It can generate
highly realistic, multilingual speech along with other audio elements like
music, background noise, and sound effects. It can even produce non-verbal
communications like laughing, sighing, and crying.

FEATURES:
    - Multilingual speech synthesis
    - Emotional/expressive speech
    - Non-verbal sounds ([laughs], [sighs], etc.)
    - Music generation
    - Multiple speaker presets

ENDPOINTS:
    POST /api/tts          - Generate speech from text
    GET  /api/voices       - List available speaker presets
    GET  /api/expressions  - List available expression markers
    GET  /health           - Health check

USAGE EXAMPLE:
    curl -X POST http://localhost:5003/api/tts \
        -H "Content-Type: application/json" \
        -d '{"text": "[laughs] Hello there!", "voice_id": "v2/en_speaker_0"}'
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import uuid
import os
from pathlib import Path
from typing import Optional
import numpy as np
from scipy.io.wavfile import write as write_wav

# =============================================================================
# FASTAPI APP CONFIGURATION
# =============================================================================

app = FastAPI(
    title="Bark TTS API",
    description="REST API for Suno's Bark expressive text-to-speech model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Output directory for generated audio files
OUTPUT_DIR = Path("/app/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Bark model instance (loaded on first use)
_bark_loaded = False


def _ensure_bark_loaded():
    """
    Lazy load Bark models on first use.

    CONCEPT: Bark models are large (~5GB). We don't load them until the first
    TTS request, which speeds up container startup.
    """
    global _bark_loaded
    if not _bark_loaded:
        from bark import preload_models

        preload_models()
        _bark_loaded = True


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================


class TTSRequest(BaseModel):
    """Request model for text-to-speech generation."""

    text: str = Field(
        ...,
        description="Text to synthesize. Can include emotion markers like [laughs], [sighs]",
        min_length=1,
        max_length=1000,  # Bark works best with shorter texts
        examples=["Hello, how are you?", "[laughs] That's funny!", "♪ La la la ♪"],
    )

    voice_id: str = Field(
        default="v2/en_speaker_0",
        description="Speaker preset ID (see /api/voices for options)",
        examples=["v2/en_speaker_0", "v2/en_speaker_6", "v2/zh_speaker_0"],
    )

    text_temp: Optional[float] = Field(
        default=0.7,
        description="Temperature for text generation (higher = more variation)",
        ge=0.0,
        le=1.0,
    )

    waveform_temp: Optional[float] = Field(
        default=0.7, description="Temperature for waveform generation", ge=0.0, le=1.0
    )


class TTSResponse(BaseModel):
    """Response model for successful TTS generation."""

    status: str = "success"
    output_path: str
    filename: str
    download_url: str
    duration_hint: str


# =============================================================================
# AVAILABLE VOICES AND EXPRESSIONS
# =============================================================================

# CONCEPT: Bark has speaker presets for different languages and voices
SPEAKER_PRESETS = {
    # English speakers
    "v2/en_speaker_0": {
        "name": "English Speaker 0",
        "language": "en",
        "gender": "neutral",
    },
    "v2/en_speaker_1": {
        "name": "English Speaker 1",
        "language": "en",
        "gender": "neutral",
    },
    "v2/en_speaker_2": {
        "name": "English Speaker 2",
        "language": "en",
        "gender": "neutral",
    },
    "v2/en_speaker_3": {
        "name": "English Speaker 3",
        "language": "en",
        "gender": "neutral",
    },
    "v2/en_speaker_4": {
        "name": "English Speaker 4",
        "language": "en",
        "gender": "neutral",
    },
    "v2/en_speaker_5": {
        "name": "English Speaker 5",
        "language": "en",
        "gender": "neutral",
    },
    "v2/en_speaker_6": {
        "name": "English Speaker 6",
        "language": "en",
        "gender": "neutral",
    },
    "v2/en_speaker_7": {
        "name": "English Speaker 7",
        "language": "en",
        "gender": "neutral",
    },
    "v2/en_speaker_8": {
        "name": "English Speaker 8",
        "language": "en",
        "gender": "neutral",
    },
    "v2/en_speaker_9": {
        "name": "English Speaker 9",
        "language": "en",
        "gender": "neutral",
    },
    # Chinese speakers
    "v2/zh_speaker_0": {
        "name": "Chinese Speaker 0",
        "language": "zh",
        "gender": "neutral",
    },
    "v2/zh_speaker_1": {
        "name": "Chinese Speaker 1",
        "language": "zh",
        "gender": "neutral",
    },
    # German speakers
    "v2/de_speaker_0": {
        "name": "German Speaker 0",
        "language": "de",
        "gender": "neutral",
    },
    "v2/de_speaker_1": {
        "name": "German Speaker 1",
        "language": "de",
        "gender": "neutral",
    },
    # Turkish speakers
    "v2/tr_speaker_0": {
        "name": "Turkish Speaker 0",
        "language": "tr",
        "gender": "neutral",
    },
    "v2/tr_speaker_1": {
        "name": "Turkish Speaker 1",
        "language": "tr",
        "gender": "neutral",
    },
}

# CONCEPT: Bark supports expression markers that add emotions to speech
EXPRESSION_MARKERS = {
    "laughs": "[laughs]",
    "sighs": "[sighs]",
    "clears_throat": "[clears throat]",
    "gasps": "[gasps]",
    "music": "♪",
    "emphasis": "...",
}


# =============================================================================
# API ENDPOINTS
# =============================================================================


@app.post("/api/tts", response_model=TTSResponse)
async def generate_speech(request: TTSRequest):
    """
    Generate expressive speech from text using Bark.

    HOW IT WORKS:
    1. Load Bark models if not already loaded
    2. Generate audio array using Bark's generate_audio function
    3. Save as WAV file
    4. Return file path/URL

    TIPS:
    - Use [laughs], [sighs], etc. for expressions
    - Use ♪ for singing/music
    - Keep text under 200 characters for best results
    """
    try:
        # Lazy load models
        _ensure_bark_loaded()

        from bark import generate_audio, SAMPLE_RATE

        # Generate unique filename
        filename = f"{uuid.uuid4()}.wav"
        output_path = OUTPUT_DIR / filename

        # Generate audio
        # CONCEPT: generate_audio returns a numpy array of audio samples
        audio_array = generate_audio(
            request.text,
            history_prompt=request.voice_id,
            text_temp=request.text_temp,
            waveform_temp=request.waveform_temp,
        )

        # Save as WAV file
        # SYNTAX: write_wav(filename, sample_rate, audio_data)
        write_wav(str(output_path), SAMPLE_RATE, audio_array)

        # Estimate duration
        duration_seconds = len(audio_array) / SAMPLE_RATE

        return TTSResponse(
            status="success",
            output_path=str(output_path),
            filename=filename,
            download_url=f"/api/download/{filename}",
            duration_hint=f"{duration_seconds:.1f}s",
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Bark TTS generation failed: {str(e)}"
        )


@app.get("/api/download/{filename}")
async def download_audio(filename: str):
    """Download a generated audio file."""
    file_path = OUTPUT_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(path=str(file_path), media_type="audio/wav", filename=filename)


@app.get("/api/voices")
async def list_voices(language: Optional[str] = None):
    """
    List available speaker presets.

    Args:
        language: Optional filter by language code (en, zh, de, tr)
    """
    voices = []
    for voice_id, info in SPEAKER_PRESETS.items():
        if language is None or info["language"] == language:
            voices.append({"id": voice_id, **info})

    return {"count": len(voices), "voices": voices}


@app.get("/api/expressions")
async def list_expressions():
    """List available expression markers that can be used in text."""
    return {
        "expressions": EXPRESSION_MARKERS,
        "usage_hint": "Include these in your text, e.g., '[laughs] That's so funny!'",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "bark-tts", "models_loaded": _bark_loaded}


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Bark TTS API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "generate": "POST /api/tts",
            "voices": "GET /api/voices",
            "expressions": "GET /api/expressions",
            "download": "GET /api/download/{filename}",
        },
    }

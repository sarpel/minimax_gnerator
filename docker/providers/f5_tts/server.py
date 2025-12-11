import os
import shutil
import tempfile
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse, JSONResponse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("f5-tts-sidecar")

app = FastAPI(title="F5-TTS Sidecar")

# Global model cache
model = None

class GenerateRequest(BaseModel):
    text: str
    voice_id: str  # Path to reference audio
    reference_text: str = ""

@app.on_event("startup")
async def load_model():
    global model
    try:
        from f5_tts.api import F5TTS
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading F5-TTS on {device}...")
        model = F5TTS(model_type="F5-TTS", device=device)
        logger.info("Model loaded.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Don't crash startup, might be a build environment without GPU

@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/generate")
async def generate(request: GenerateRequest):
    global model
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        logger.info(f"Generating audio for text: {request.text[:50]}...")
        
        # Verify reference audio exists inside the container
        # Note: In a real deployment, you'd mount a volume or upload the file.
        # For this sidecar, we assume volumes are mounted at runtime.
        if not os.path.exists(request.voice_id):
             raise HTTPException(status_code=400, detail=f"Reference audio not found: {request.voice_id}")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = f.name
        
        # Run inference
        model.infer(
            ref_file=request.voice_id,
            ref_text=request.reference_text,
            gen_text=request.text,
            file_wave=output_path
        )
        
        return FileResponse(output_path, media_type="audio/wav")

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

import os
import tempfile
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("coqui-xtts-sidecar")

app = FastAPI(title="Coqui XTTS Sidecar")

model = None

class GenerateRequest(BaseModel):
    text: str
    voice_id: str  # Path to reference audio
    language: str = "en"

@app.on_event("startup")
async def load_model():
    global model
    try:
        from TTS.api import TTS
        import torch
        
        # Check GPU
        has_gpu = torch.cuda.is_available()
        logger.info(f"Loading Coqui XTTS (GPU: {has_gpu})...")
        
        model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
        if has_gpu:
            model.to("cuda")
            
        logger.info("Model loaded.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")

@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/generate")
async def generate(request: GenerateRequest):
    global model
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    try:
        if not os.path.exists(request.voice_id):
             raise HTTPException(status_code=400, detail=f"Reference audio not found: {request.voice_id}")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = f.name
            
        model.tts_to_file(
            text=request.text,
            speaker_wav=request.voice_id,
            language=request.language,
            file_path=output_path
        )
        
        return FileResponse(output_path, media_type="audio/wav")

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

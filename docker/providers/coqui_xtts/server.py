import os
import tempfile
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("coqui-xtts-sidecar")

app = FastAPI(title="Coqui XTTS Sidecar")

model = None

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
async def generate(
    text: str = Form(...),
    language: str = Form("en"),
    reference_audio: UploadFile = File(...)
):
    global model
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    temp_ref_path = None
    output_path = None
    
    try:
        # Save uploaded reference audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_ref_path = f.name
        
        with open(temp_ref_path, "wb") as f:
            shutil.copyfileobj(reference_audio.file, f)

        # Prepare output path
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = f.name
            
        model.tts_to_file(
            text=text,
            speaker_wav=temp_ref_path,
            language=language,
            file_path=output_path
        )
        
        return FileResponse(output_path, media_type="audio/wav")

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup temp ref
        if temp_ref_path and os.path.exists(temp_ref_path):
            os.unlink(temp_ref_path)
        # Note: output_path is cleaned up by FileResponse background task usually, 
        # or we rely on temp dir cleanup. For now this is fine.

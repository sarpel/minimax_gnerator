# WakeGen TTS Docker Stack - README

A comprehensive Docker Compose stack for running 10 different TTS (Text-to-Speech) providers, each in its own container with dedicated ports and output directories.

## 🎯 Quick Start

### Start All Providers
```bash
cd docker
docker compose --profile full up -d
```

### Start Specific Provider
```bash
# Start only Piper TTS
docker compose --profile piper up -d

# Start only Edge TTS
docker compose --profile edge-tts up -d
```

### Start with GPU Support
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile full up -d
```

---

## 📦 Available Providers

| Port | Provider | Type | Description |
|------|----------|------|-------------|
| 5000 | Edge TTS | Cloud | Microsoft's free cloud TTS (no API key needed) |
| 5001 | Piper | Local | Fast, lightweight ONNX-based TTS |
| 5002 | Coqui XTTS | Local | Zero-shot voice cloning |
| 5003 | Bark | Local | Expressive TTS with emotions/music |
| 5004 | ChatTTS | Local | Conversational speech synthesis |
| 5005 | Kokoro | Local | Lightweight 82M param model |
| 5006 | Mimic3 | Local | Privacy-focused offline TTS |
| 5007 | F5-TTS | Local | High-quality diffusion-based TTS |
| 5008 | StyleTTS2 | Local | Human-level expressive speech |
| 5009 | Orpheus | Local | LLM-based emotional speech |

---

## 🚀 Usage Examples

### Edge TTS (Port 5000)
```bash
# Generate speech
curl -X POST http://localhost:5000/api/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice_id": "en-US-AriaNeural"}'

# List available voices
curl http://localhost:5000/api/voices
```

### Piper TTS (Port 5001)
```bash
# Piper uses the Wyoming protocol
# Access via Home Assistant integration or:
echo "Hello from Piper" | nc localhost 5001
```

### Coqui XTTS (Port 5002)
```bash
# Generate with voice cloning (requires reference audio)
curl -X POST http://localhost:5002/tts_to_audio \
  -F "text=Hello world" \
  -F "speaker_wav=@reference.wav" \
  -F "language=en"
```

### Bark TTS (Port 5003)
```bash
# Generate expressive speech
curl -X POST http://localhost:5003/api/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "[laughs] Hello there!", "voice_id": "v2/en_speaker_0"}'
```

---

## 📁 Directory Structure

```
docker/
├── docker-compose.yml          # Main orchestration
├── docker-compose.gpu.yml      # GPU override
├── .env.example                # Environment template
├── README.md                   # This file
│
├── edge-tts/                   # Custom Edge TTS container
│   ├── Dockerfile
│   ├── server.py
│   └── requirements.txt
│
├── bark/                       # Custom Bark container
│   ├── Dockerfile
│   ├── server.py
│   └── requirements.txt
│
└── outputs/                    # Generated audio files
    ├── edge-tts/
    ├── piper/
    ├── coqui-xtts/
    ├── bark/
    ├── chattts/
    ├── kokoro/
    ├── mimic3/
    ├── f5-tts/
    ├── styletts2/
    └── orpheus/
```

---

## ⚙️ Configuration

1. Copy the environment template:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` to customize settings:
   - `PIPER_VOICE`: Default Piper voice
   - `SUNO_USE_SMALL_MODELS`: Use smaller Bark models
   - `MODEL_SIZE`: Orpheus model size (small/medium/large)

---

## 🎮 Profile Reference

| Profile | Containers Included |
|---------|---------------------|
| `full` | All 10 providers |
| `local` | All local (non-cloud) providers |
| `cloud` | Cloud-based providers only (Edge TTS) |
| `edge-tts` | Edge TTS only |
| `piper` | Piper only |
| `coqui-xtts` | Coqui XTTS only |
| `bark` | Bark only |
| `chattts` | ChatTTS only |
| `kokoro` | Kokoro only |
| `mimic3` | Mimic3 only |
| `f5-tts` | F5-TTS only |
| `styletts2` | StyleTTS2 only |
| `orpheus` | Orpheus only |

---

## 💻 GPU Requirements

| Provider | CPU | GPU (VRAM) |
|----------|-----|------------|
| Edge TTS | ✅ Fast | N/A (cloud) |
| Piper | ✅ Fast | Not needed |
| Coqui XTTS | ⚠️ Slow | 4GB+ recommended |
| Bark | ⚠️ Very slow | 6-12GB recommended |
| ChatTTS | ⚠️ Slow | 4GB+ recommended |
| Kokoro | ✅ Fast | 2GB (optional) |
| Mimic3 | ✅ Fast | Not needed |
| F5-TTS | ⚠️ Slow | 6GB+ recommended |
| StyleTTS2 | ⚠️ Slow | 4GB+ recommended |
| Orpheus | ⚠️ Slow | 4-12GB (depends on model) |

---

## 🔧 Troubleshooting

### Container won't start
```bash
# Check logs
docker compose logs edge-tts

# Rebuild container
docker compose build edge-tts
```

### Out of disk space
```bash
# Clean up unused images and volumes
docker system prune -a --volumes
```

### GPU not detected
```bash
# Verify NVIDIA runtime is available
docker run --gpus all nvidia/cuda:12.0-base nvidia-smi
```

---

## 📊 Health Checks

All containers expose health endpoints:

```bash
# Check all container health
docker compose ps

# Check specific service
curl http://localhost:5000/health
```

---

## 🛑 Stopping Services

```bash
# Stop all
docker compose --profile full down

# Stop and remove volumes (⚠️ deletes model cache!)
docker compose --profile full down -v
```

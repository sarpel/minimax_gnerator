# Installation Guide

This guide covers installing WakeGen on various platforms.

## Prerequisites

- **Python 3.10+** - Check with `python --version`
- **pip** - Python package installer
- **Git** - For cloning the repository

## Quick Install

```bash
# Clone and install
git clone https://github.com/yourusername/wakegen.git
cd wakegen
pip install -e .
```

## Detailed Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/wakegen.git
cd wakegen
```

### 2. Create Virtual Environment

**Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Core Package

```bash
pip install -e .
```

This installs WakeGen with basic dependencies (Edge TTS works out of the box).

### 4. Install Optional Providers

Different TTS providers have different dependencies:

```bash
# Lightweight providers (CPU-friendly)
pip install kokoro-onnx          # Kokoro TTS
pip install piper-tts            # Piper TTS
pip install mycroft-mimic3-tts   # Mimic 3

# High-quality providers (GPU recommended)
pip install TTS                  # Coqui XTTS
pip install styletts2            # StyleTTS 2
pip install f5-tts               # F5-TTS
pip install bark                 # Bark (Suno)
pip install chattts              # ChatTTS

# Quality verification
pip install openai-whisper       # ASR verification
```

## Platform-Specific Instructions

### Windows

```powershell
# Install Python 3.10+ from python.org
# Then in PowerShell:
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

For GPU support with PyTorch:
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Linux (Ubuntu/Debian)

```bash
# System dependencies
sudo apt-get update
sudo apt-get install -y \
    libsndfile1 \
    portaudio19-dev \
    python3-dev \
    ffmpeg

# Python setup
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### macOS

```bash
# Install Homebrew dependencies
brew install portaudio libsndfile ffmpeg

# Python setup
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Raspberry Pi

Optimized for Pi Zero 2 W, Pi 4, and Pi 5:

```bash
# System dependencies
sudo apt-get update
sudo apt-get install -y \
    libsndfile1 \
    portaudio19-dev \
    python3-dev \
    libopenblas-dev \
    libatlas-base-dev

# Install WakeGen
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Install CPU-friendly providers
pip install piper-tts kokoro-onnx mycroft-mimic3-tts
```

**Recommended providers for Raspberry Pi:**
- Piper TTS (fastest)
- Mimic 3 (good quality)
- Kokoro TTS (balanced)

### Docker

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libsndfile1 \
    portaudio19-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN pip install -e .

ENTRYPOINT ["wakegen"]
```

Build and run:
```bash
docker build -t wakegen .
docker run -v $(pwd)/output:/app/output wakegen generate --text "hey assistant" --count 10
```

## GPU Setup

### NVIDIA CUDA

For GPU-accelerated providers:

```bash
# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify GPU is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Apple Silicon (M1/M2/M3)

PyTorch MPS backend is automatically used:

```bash
pip install torch torchvision torchaudio

# Verify MPS is available
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

## Verify Installation

```bash
# Check CLI is working
wakegen --help

# List available providers
wakegen list-providers

# Test generation
wakegen generate --text "hello world" --count 1

# Check installed providers
wakegen list-providers --available
```

## Provider Installation Summary

| Provider | Install Command | GPU Required |
|----------|-----------------|--------------|
| Edge TTS | *(built-in)* | No |
| Kokoro | `pip install kokoro-onnx` | No |
| Piper | `pip install piper-tts` | No |
| Mimic 3 | `pip install mycroft-mimic3-tts` | No |
| Coqui XTTS | `pip install TTS` | Yes |
| F5-TTS | `pip install f5-tts` | Recommended |
| StyleTTS 2 | `pip install styletts2` | Recommended |
| Orpheus | `pip install orpheus-speech` | For large models |
| Bark | `pip install bark` | Yes |
| ChatTTS | `pip install chattts` | Recommended |
| MiniMax | *(API key required)* | No |

## Troubleshooting

### "Command not found: wakegen"

Ensure virtual environment is activated:
```bash
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

### Import errors

Reinstall with dependencies:
```bash
pip install -e . --force-reinstall
```

### Audio library errors

Install system audio dependencies:
```bash
# Ubuntu/Debian
sudo apt-get install libsndfile1 portaudio19-dev

# macOS
brew install portaudio libsndfile
```

### CUDA out of memory

Reduce batch size or use CPU:
```bash
wakegen generate --text "hello" --count 10 --device cpu
```

### Provider not available

Check specific provider dependencies:
```bash
wakegen list-providers --check-deps
```

## Development Installation

For contributing to WakeGen:

```bash
# Clone with development dependencies
git clone https://github.com/yourusername/wakegen.git
cd wakegen

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode with test dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run type checking
mypy wakegen

# Run linting
ruff check wakegen
```

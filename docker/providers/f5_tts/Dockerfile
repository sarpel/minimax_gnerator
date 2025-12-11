# Use a base image with PyTorch and CUDA support
# F5-TTS works best with GPU
FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install F5-TTS specific dependencies
# We install directly from git or PyPI as per official docs
RUN pip install --no-cache-dir \
    f5-tts \
    fastapi \
    uvicorn \
    python-multipart

# Copy the server wrapper
COPY server.py .

# Expose the API port
EXPOSE 5000

# Run the server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "5000"]

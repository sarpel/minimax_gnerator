# Use a base image with PyTorch and CUDA support
# Coqui XTTS requires significant resources
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Coqui TTS and server dependencies
# Note: Accepting the Coqui license is implied by usage
RUN pip install --no-cache-dir \
    TTS \
    fastapi \
    uvicorn \
    python-multipart

# Copy the server wrapper
COPY server.py .

# Accept the Coqui license automatically (env var)
ENV COQUI_TOS_AGREED=1

# Expose the API port
EXPOSE 5000

# Run the server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "5000"]

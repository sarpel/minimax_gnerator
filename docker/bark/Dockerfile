# =============================================================================
# Bark TTS Docker Container
# =============================================================================
# CONCEPT: Bark is Suno's expressive text-to-speech model that can generate
# emotions, laughter, music, and non-verbal sounds. This container provides
# a FastAPI REST API wrapper.
#
# REQUIREMENTS:
#   - ~10GB disk for model cache
#   - 6-12GB VRAM for GPU (or use small models on CPU)
#
# BUILD: docker build -t wakegen-bark .
# RUN:   docker run -p 5003:8000 wakegen-bark
# =============================================================================

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# - git: Required for installing bark from GitHub
# - ffmpeg: Required for audio processing
# - build-essential: Required for compiling some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
  git \
  ffmpeg \
  curl \
  build-essential \
  && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
# This can take a while as bark has many dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI server code
COPY server.py .

# Create output directory
RUN mkdir -p /app/outputs

# Set environment variable for Bark to use smaller models by default
# This can be overridden at runtime with -e SUNO_USE_SMALL_MODELS=0
ENV SUNO_USE_SMALL_MODELS=1

# Expose the port that uvicorn will listen on
EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

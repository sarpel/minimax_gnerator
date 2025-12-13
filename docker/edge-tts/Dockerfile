# =============================================================================
# Edge TTS Docker Container
# =============================================================================
# CONCEPT: This is a lightweight Python container that wraps the edge-tts
# library with a FastAPI REST API. Edge TTS is Microsoft's free cloud-based
# TTS service - this container just provides a convenient HTTP interface.
#
# BUILD: docker build -t wakegen-edge-tts .
# RUN:   docker run -p 5000:8000 wakegen-edge-tts
# =============================================================================

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
  curl \
  && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir saves disk space by not caching pip downloads
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI server code
COPY server.py .

# Create output directory
RUN mkdir -p /app/outputs

# Expose the port that uvicorn will listen on
EXPOSE 8000

# Run the FastAPI server
# --host 0.0.0.0 makes it accessible from outside the container
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

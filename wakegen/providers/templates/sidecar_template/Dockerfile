# Use a base image that fits your requirements.
# For legacy providers, you might use python:3.8-slim or specific cuda images.
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (e.g., for audio libraries)
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the server code
COPY server.py .

# Expose the port (must match what you configure in WakeGen)
EXPOSE 5000

# Run the server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "5000"]

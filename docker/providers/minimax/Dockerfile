# Use a lightweight python image as Minimax is just an API client
FROM python:3.10-slim

WORKDIR /app

# Install requests/httpx
RUN pip install --no-cache-dir \
    httpx \
    fastapi \
    uvicorn \
    python-multipart

# Copy the server wrapper
COPY server.py .

# Expose the API port
EXPOSE 5000

# Run the server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "5000"]

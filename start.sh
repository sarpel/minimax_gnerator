#!/bin/bash

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "[ERROR] Virtual environment not found."
    echo "Please run './install.sh' first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

echo "==============================================================================="
echo " Starting WakeGen Web UI..."
echo "==============================================================================="
echo "Access the UI at: http://127.0.0.1:8000"
echo "Press Ctrl+C to stop the server."
echo ""

# Start the server in reload mode for development
python -m uvicorn wakegen.web.app:create_app --factory --reload --host 127.0.0.1 --port 8000

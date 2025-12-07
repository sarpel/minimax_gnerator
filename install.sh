#!/bin/bash

echo "==============================================================================="
echo " WakeGen One-Click Installer"
echo "==============================================================================="

# Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed or not in your PATH."
    echo "Please install Python 3.10+ and try again."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "[INFO] Creating virtual environment..."
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to create virtual environment."
        exit 1
    fi
    echo "[OK] Virtual environment created."
else
    echo "[INFO] Virtual environment already exists."
fi

# Activate virtual environment and install dependencies
echo "[INFO] Installing dependencies..."
source .venv/bin/activate
pip install -e ".[web]"

if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install dependencies."
    exit 1
fi

# Make start script executable
chmod +x start.sh

echo ""
echo "==============================================================================="
echo " Installation Complete!"
echo "==============================================================================="
echo "You can now run './start.sh' to launch the application."
echo ""

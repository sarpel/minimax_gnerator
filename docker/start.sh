#!/bin/bash
# =============================================================================
# WakeGen TTS Docker Stack - Start Helper Script (Linux/macOS)
# =============================================================================
# USAGE:
#   ./start.sh              - Start all providers
#   ./start.sh piper        - Start only Piper
#   ./start.sh gpu          - Start all with GPU support
#   ./start.sh piper gpu    - Start Piper with GPU support
# =============================================================================

set -e

# Change to script directory
cd "$(dirname "$0")"

# Default values
PROFILE="full"
USE_GPU=0

# Parse arguments
for arg in "$@"; do
    case $arg in
        gpu)
            USE_GPU=1
            ;;
        *)
            PROFILE="$arg"
            ;;
    esac
done

echo ""
echo "==============================================="
echo " WakeGen TTS Docker Stack"
echo "==============================================="
echo " Profile: $PROFILE"
echo " GPU: $USE_GPU"
echo "==============================================="
echo ""

# Build the docker compose command
if [ $USE_GPU -eq 1 ]; then
    echo "Starting with GPU support..."
    docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile "$PROFILE" up -d
else
    echo "Starting in CPU mode..."
    docker compose --profile "$PROFILE" up -d
fi

echo ""
echo "==============================================="
echo " Containers started successfully!"
echo "==============================================="
echo ""
echo "Access the services at:"
echo "  Edge TTS:    http://localhost:5000"
echo "  Piper:       http://localhost:5001"
echo "  Coqui XTTS:  http://localhost:5002"
echo "  Bark:        http://localhost:5003"
echo "  ChatTTS:     http://localhost:5004"
echo "  Kokoro:      http://localhost:5005"
echo "  Mimic3:      http://localhost:5006"
echo "  F5-TTS:      http://localhost:5007"
echo "  StyleTTS2:   http://localhost:5008"
echo "  Orpheus:     http://localhost:5009"
echo ""
echo "Run 'docker compose ps' to see container status"

#!/bin/bash
# =============================================================================
# WakeGen TTS Docker Stack - Stop Helper Script (Linux/macOS)
# =============================================================================
# USAGE:
#   ./stop.sh              - Stop all providers
#   ./stop.sh piper        - Stop only Piper
#   ./stop.sh clean        - Stop all and remove volumes
# =============================================================================

set -e

# Change to script directory
cd "$(dirname "$0")"

# Default values
PROFILE="full"
CLEAN=0

# Parse arguments
for arg in "$@"; do
    case $arg in
        clean)
            CLEAN=1
            ;;
        *)
            PROFILE="$arg"
            ;;
    esac
done

echo ""
echo "==============================================="
echo " WakeGen TTS Docker Stack - Stopping"
echo "==============================================="
echo " Profile: $PROFILE"
echo " Clean: $CLEAN"
echo "==============================================="
echo ""

if [ $CLEAN -eq 1 ]; then
    echo "WARNING: This will delete all cached models!"
    read -p "Are you sure? (y/N): " confirm
    if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
        docker compose --profile "$PROFILE" down -v
    else
        echo "Aborted."
        exit 0
    fi
else
    docker compose --profile "$PROFILE" down
fi

echo ""
echo "Containers stopped successfully."

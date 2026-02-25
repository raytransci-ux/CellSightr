#!/usr/bin/env bash
# CellCount - Hemocytometer Cell Counter
# Launch script for Linux/macOS
# ==========================================

set -e

echo ""
echo " ============================================"
echo "  CellCount - Hemocytometer Cell Counter"
echo " ============================================"
echo ""

# Determine project root
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

# Check for Python 3
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install Python 3.10+"
    exit 1
fi

# Check for virtual environment
if [ -d ".venv" ]; then
    echo "Using virtual environment..."
    source .venv/bin/activate
else
    echo "No .venv found. Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    echo "Installing dependencies..."
    pip install -r webapp/backend/requirements.txt
    echo ""
fi

# Check model weights
if [ ! -f "checkpoints/yolo/nano/weights/best.pt" ]; then
    echo "WARNING: No model weights found in checkpoints/yolo/"
    echo "The app will fail to start without trained model weights."
    echo "Copy the checkpoints/ folder from the development machine."
    exit 1
fi

export PYTHONPATH="$PROJECT_ROOT/webapp/backend"

echo "Starting CellCount server on http://localhost:8000"
echo "Press Ctrl+C to stop."
echo ""

python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --app-dir "$PROJECT_ROOT/webapp/backend"

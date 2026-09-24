#!/usr/bin/env bash

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "=== LastZAlert environment setup ==="

if ! command -v python3 >/dev/null 2>&1; then
    echo "[ERROR] python3 is not installed."
    exit 1
fi

if ! command -v adb >/dev/null 2>&1; then
    echo "[WARNING] adb is not installed."
    echo "Install it with: sudo apt install android-tools-adb"
fi

if ! command -v tesseract >/dev/null 2>&1; then
    echo "[WARNING] tesseract is not installed."
    echo "Install it with: sudo apt install tesseract-ocr"
fi

if [ ! -d "venv" ]; then
    echo "[SETUP] Creating Python virtual environment..."
    python3 -m venv venv
fi

echo "[SETUP] Installing Python dependencies..."
venv/bin/python -m pip install --upgrade pip
venv/bin/python -m pip install -r requirements.txt

echo "[SETUP] Configuring Git security hook..."
git config core.hooksPath .githooks

if [ ! -f ".env" ]; then
    echo "[SETUP] Creating local .env from .env.example..."
    cp .env.example .env
fi

chmod +x lastz_treasure_watcher.py

echo
echo "=== Setup completed ==="
echo
echo "Activate the environment:"
echo "  source venv/bin/activate"
echo
echo "Run LastZAlert:"
echo "  ./lastz_treasure_watcher.py"
echo
echo "Optional Discord notifications:"
echo "  Configure DISCORD_WEBHOOK_URL in .env"

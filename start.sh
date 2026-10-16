#!/usr/bin/env bash

# Recover project dependencies after Python upgrades
if ! ./venv/bin/python -c "import requests" >/dev/null 2>&1; then
    echo "[SETUP] Python dependencies missing. Installing..."
    if [ -f requirements.txt ]; then
        ./venv/bin/python -m pip install -r requirements.txt || exit 1
    else
        ./venv/bin/python -m pip install requests || exit 1
    fi
fi


# Recover pip if missing from the virtual environment
if ! ./venv/bin/python -m pip --version >/dev/null 2>&1; then
    echo "[SETUP] pip missing. Recovering..."
    ./venv/bin/python -m ensurepip --upgrade || {
        echo "[ERROR] Cannot recover pip. Install python3-venv."
        exit 1
    }
fi

# Ensure OpenCV is available
if ! ./venv/bin/python -c "import cv2" >/dev/null 2>&1; then
    echo "[SETUP] OpenCV missing. Installing..."
    ./venv/bin/python -m pip install opencv-python-headless || exit 1
fi


set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

ADB_DEVICE="${ADB_DEVICE:-}"

if [ -z "$ADB_DEVICE" ] && [ -f ".env" ]; then
    ADB_DEVICE="$("$PROJECT_DIR/venv/bin/python" -c 'from config import ADB_DEVICE; print(ADB_DEVICE)' 2>/dev/null)"
fi

if [ -z "$ADB_DEVICE" ]; then
    echo "[ERROR] ADB_DEVICE is not configured."
    echo
    echo "Configure it in .env, for example:"
    echo "  ADB_DEVICE=192.168.0.95:5555"
    exit 1
fi
TOUCH_GRAB_REMOTE="/data/local/tmp/touch_grab"

MONITOR_PID=""
TOUCH_GRAB_PID=""
BOT_PID=""
CLEANED_UP=0

cleanup() {
    if [ "$CLEANED_UP" -eq 1 ]; then
        return
    fi
    CLEANED_UP=1

    echo
    echo "=== Stopping LastZAlert services ==="

    if [ -n "$MONITOR_PID" ] && kill -0 "$MONITOR_PID" 2>/dev/null; then
        echo "[STOP] Monitor"
        kill "$MONITOR_PID" 2>/dev/null || true
        wait "$MONITOR_PID" 2>/dev/null || true
    fi

    if [ -n "$TOUCH_GRAB_PID" ] && kill -0 "$TOUCH_GRAB_PID" 2>/dev/null; then
        echo "[STOP] AndroidTouchGrab"
        kill "$TOUCH_GRAB_PID" 2>/dev/null || true
        wait "$TOUCH_GRAB_PID" 2>/dev/null || true
    fi

    echo "[STOP] Releasing Android input device..."
    adb -s "$ADB_DEVICE" shell "pkill -INT touch_grab 2>/dev/null || pkill touch_grab 2>/dev/null || true" >/dev/null 2>&1 || true

    if [ -n "$BOT_PID" ] && kill -0 "$BOT_PID" 2>/dev/null; then
        echo "[STOP] Bot"
        kill "$BOT_PID" 2>/dev/null || true
        wait "$BOT_PID" 2>/dev/null || true
    fi

    echo "=== All services stopped ==="
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [ ! -x "venv/bin/python" ]; then
    echo "[ERROR] Python virtual environment not found."
    echo "Run:"
    echo "  ./setup_env.sh"
    exit 1
fi

if ! command -v adb >/dev/null 2>&1; then
    echo "[ERROR] adb not found."
    exit 1
fi

echo "=== LastZAlert launcher ==="
echo

echo "[ADB] Connecting to $ADB_DEVICE..."
adb connect "$ADB_DEVICE" >/dev/null 2>&1 || true

if ! adb -s "$ADB_DEVICE" get-state 2>/dev/null | grep -q "^device$"; then
    echo "[ERROR] Android device $ADB_DEVICE is not available."
    exit 1
fi

echo "[ADB] Device connected."

echo
echo "[MONITOR] Checking port 8000..."

if ss -ltn '( sport = :8000 )' | grep -q LISTEN; then
    echo "[ERROR] Port 8000 is already in use."
    exit 1
fi

echo "[TOUCH] Cleaning previous AndroidTouchGrab instance..."

adb -s "$ADB_DEVICE" shell     "pkill -INT touch_grab 2>/dev/null || pkill touch_grab 2>/dev/null || true"     >/dev/null 2>&1 || true

sleep 1

echo "[TOUCH] Starting AndroidTouchGrab..."

adb -s "$ADB_DEVICE" shell "$TOUCH_GRAB_REMOTE" &
TOUCH_GRAB_PID=$!

sleep 1

if ! kill -0 "$TOUCH_GRAB_PID" 2>/dev/null; then
    echo "[ERROR] AndroidTouchGrab failed to start."
    exit 1
fi

echo "[TOUCH] AndroidTouchGrab running."

echo
echo "[MONITOR] Starting web monitor..."

(
    cd "$PROJECT_DIR/monitor"
    exec "$PROJECT_DIR/venv/bin/python" monitor.py
) &
MONITOR_PID=$!

sleep 1

if ! kill -0 "$MONITOR_PID" 2>/dev/null; then
    echo "[ERROR] Monitor failed to start."
    exit 1
fi

echo "[MONITOR] Monitor running."

echo
echo "[BOT] Starting LastZAlert..."
echo "Press Ctrl+C to stop everything."
echo

"$PROJECT_DIR/venv/bin/python" "$PROJECT_DIR/lastz_treasure_watcher.py" &
BOT_PID=$!

echo "[BOT] Running PID=$BOT_PID"

# Keep the monitor available even if the bot stops.
while true; do
    if ! kill -0 "$MONITOR_PID" 2>/dev/null; then
        echo "[ERROR] Monitor stopped."
        exit 1
    fi

    # Detect a terminated child, including an unreaped zombie.
    if ! ps -p "$BOT_PID" -o stat= 2>/dev/null | grep -qv '^[[:space:]]*Z'; then
        BOT_EXIT_CODE=0
        wait "$BOT_PID" || BOT_EXIT_CODE=$?
        BOT_PID=""

        echo "[BOT] Stopped (exit code: $BOT_EXIT_CODE)"
        echo "[MONITOR] Still running. Press Ctrl+C to stop everything."
        break
    fi

    sleep 2
done

# Preserve monitor access after the bot exits.
while kill -0 "$MONITOR_PID" 2>/dev/null; do
    sleep 2
done

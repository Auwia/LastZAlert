from flask import Flask, request, abort
import os
import sys
import time
from pathlib import Path

from treasure_detector import detect_treasure
import requests


# Allow this script to import the shared configuration from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DISCORD_WEBHOOK_URL


print("[+] server.py loaded")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

LATEST_PATH = os.path.join(UPLOAD_DIR, "latest.png")
LAST_ALERT_TIME = 0
MIN_SECONDS_BETWEEN_ALERTS = 600

app = Flask(__name__)

@app.route("/")
def index():
    return "ok", 200

@app.post("/upload_raw")
def upload_raw():
    global LAST_ALERT_TIME

    data = request.get_data()
    if not data:
        abort(400, "empty body")

    print("[+] Upload ricevuto, bytes:", len(data))

    with open(LATEST_PATH, "wb") as f:
        f.write(data)

    try:
        name, score = detect_treasure(LATEST_PATH)
        print(f"[i] Detect: name={name} score={score:.3f}")
    except Exception as e:
        print("[!] ERRORE detector:", e)
        name, score = None, 0.0

    now = time.time()
    if name and (now - LAST_ALERT_TIME) > MIN_SECONDS_BETWEEN_ALERTS:
        if not DISCORD_WEBHOOK_URL:
            print("[!] Discord webhook is not configured; notification skipped")
        else:
            try:
                response = requests.post(
                    DISCORD_WEBHOOK_URL,
                    json={
                        "content": f"🎁 Tesoro rilevato ({name}) score={score:.3f}"
                    },
                    timeout=10,
                )
                response.raise_for_status()
                print("[+] Discord notification sent")
            except requests.RequestException as exc:
                print(f"[!] Discord notification failed: {exc}")

        LAST_ALERT_TIME = now

    return "ok", 200

if __name__ == "__main__":
    print("[+] Avvio Flask su 0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000)


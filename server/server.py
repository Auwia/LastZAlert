from flask import Flask, request, abort
import os, time
from treasure_detector import detect_treasure
import requests

print("[+] server.py caricato")

DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1446565181265154190/pL-0gcgP09RlQqnqHqQDIdQqm505tqa744is2R_1eGA3Had4OXmhPgQrTLYXYzaMld0S"

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
        print("[🎁] TESORO RILEVATO, invio Discord")
        requests.post(DISCORD_WEBHOOK_URL, json={
            "content": f"🎁 Tesoro rilevato ({name}) score={score:.3f}"
        })
        LAST_ALERT_TIME = now

    return "ok", 200

if __name__ == "__main__":
    print("[+] Avvio Flask su 0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000)


#!/usr/bin/env python3
import os
import time
import subprocess
import ssl
from email.message import EmailMessage
import requests

import cv2
import numpy as np

# =============================
# CONFIG DISCORD WEBHOOK
# =============================
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1446565181265154190/pL-0gcgP09RlQqnqHqQDIdQqm505tqa744is2R_1eGA3Had4OXmhPgQrTLYXYzaMld0S"

def send_notification(text):
    if not DISCORD_WEBHOOK_URL:
        print("[!] DISCORD_WEBHOOK_URL non configurata.")
        return False

    payload = {"content": text}

    try:
        r = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
        print("Discord status:", r.status_code, r.text)
        return r.ok
    except Exception as e:
        print("[!] Errore Discord:", e)
        return False


# =============================
# CONFIG GENERALE
# =============================

ADB_CMD = "adb"
PACKAGE_NAME = "com.readygo.barrel.gp"

TEMPLATES_DIR = "treasures"
SCREENSHOT_PATH = "current_screen.png"

CHECK_INTERVAL_SEC = 10
MATCH_THRESHOLD = 0.30                 # più realistico per icona elicottero
MIN_SECONDS_BETWEEN_ALERTS = 600
CONSECUTIVE_HITS_REQUIRED = 3         # deve vederlo per 3 cicli

# Template che identificano il TESORO-EVENTO
EVENT_TEMPLATE_NAMES = {
    "tesoro_elicottero.jpg",
    "tesoro_elicottero2.jpg",
}

# ROI PRECISA
ROI_X_START_FRAC = 0.38
ROI_X_END_FRAC   = 0.72
ROI_Y_START_FRAC = 0.67
ROI_Y_END_FRAC   = 0.83

# Debug ROI
DEBUG_SAVE_ROI = True


# =============================
# UTILITY
# =============================

def run_cmd(cmd, timeout=30):
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=timeout, check=False, text=True)
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        print(f"[!] Timeout eseguendo: {' '.join(cmd)}")
        return 1, "", "timeout"


def check_adb_device():
    code, out, err = run_cmd([ADB_CMD, "devices"])
    if code != 0:
        print("[!] Errore eseguendo 'adb devices':", err)
        return False

    lines = out.strip().splitlines()
    devices = [l for l in lines[1:] if l.strip()]
    if not devices:
        print("[!] Nessun dispositivo ADB trovato.")
        return False

    print("[+] Dispositivo ADB rilevato:", devices[0])
    return True


def take_screenshot(path):
    try:
        with open(path, "wb") as f:
            proc = subprocess.run([ADB_CMD, "exec-out", "screencap", "-p"],
                                  stdout=f, stderr=subprocess.PIPE,
                                  timeout=30, check=False)
        if proc.returncode != 0:
            print("[!] Errore screencap:", proc.stderr.decode("utf-8", errors="ignore"))
            return False
        return True
    except Exception as e:
        print("[!] Eccezione screencap:", e)
        return False


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"[!] Impossibile leggere immagine: {path}")
    return img


def crop_roi_bottom_right(img):
    H, W = img.shape[:2]

    x_start = int(W * ROI_X_START_FRAC)
    x_end   = int(W * ROI_X_END_FRAC)
    y_start = int(H * ROI_Y_START_FRAC)
    y_end   = int(H * ROI_Y_END_FRAC)

    x_start = max(0, min(x_start, W - 1))
    x_end   = max(x_start + 1, min(x_end, W))
    y_start = max(0, min(y_start, H - 1))
    y_end   = max(y_start + 1, min(y_end, H))

    roi = img[y_start:y_end, x_start:x_end]
    return roi, (x_start, y_start, x_end, y_end)


def compute_match_score(roi, template):
    h, w = template.shape[:2]
    H, W = roi.shape[:2]
    if H < h or W < w:
        return 0.0
    res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    return max_val


def kill_app():
    code, out, err = run_cmd([ADB_CMD, "shell", "am", "force-stop", PACKAGE_NAME])
    if code == 0:
        print(f"[+] App {PACKAGE_NAME} killata.")
    else:
        print(f"[!] Errore kill app: {err}")


def load_all_templates(directory):
    templates = []
    if not os.path.isdir(directory):
        print(f"[!] Cartella template '{directory}' non trovata.")
        return templates

    for name in os.listdir(directory):
        if not name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        full_path = os.path.join(directory, name)
        img = load_image(full_path)
        if img is not None:
            templates.append((name, img))
            print(f"[+] Template caricato: {full_path}")
    return templates


# =============================
# MAIN
# =============================

def main():
    print("=== Last Z Treasure Watcher (multi-template, ROI precisa) ===")

    if not check_adb_device():
        return

    templates = load_all_templates(TEMPLATES_DIR)
    if not templates:
        return

    last_alert_time = 0
    consecutive_hits = 0
    debug_done = False

    while True:
        print("\n[*] Catturo screenshot...")
        if not take_screenshot(SCREENSHOT_PATH):
            time.sleep(CHECK_INTERVAL_SEC)
            continue

        img = load_image(SCREENSHOT_PATH)
        if img is None:
            time.sleep(CHECK_INTERVAL_SEC)
            continue

        roi, (xs, ys, xe, ye) = crop_roi_bottom_right(img)

        if DEBUG_SAVE_ROI and not debug_done:
            cv2.imwrite("roi_debug.png", roi)
            print("[+] ROI salvata in roi_debug.png")
            debug_done = True

        event_best_score = 0.0
        event_best_name = None
        scores_debug = []

        for name, tmpl in templates:
            score = compute_match_score(roi, tmpl)
            scores_debug.append((name, score))

            # solo template evento
            if name in EVENT_TEMPLATE_NAMES and score > event_best_score:
                event_best_score = score
                event_best_name = name

        print("[i] Scores:", ", ".join(f"{n}={s:.3f}" for n, s in scores_debug))
        print(f"[i] EVENTO: {event_best_name} score={event_best_score:.3f} ROI=({xs},{ys})-({xe},{ye})")

        now = time.time()
        elapsed = now - last_alert_time

        if event_best_score >= MATCH_THRESHOLD:
            consecutive_hits += 1
            print(f"[i] EVENTO sopra soglia, hit {consecutive_hits}/{CONSECUTIVE_HITS_REQUIRED}")
        else:
            if consecutive_hits > 0:
                print("[i] EVENTO sotto soglia, azzero hit consecutivi")
            consecutive_hits = 0

        if (event_best_score >= MATCH_THRESHOLD and
            consecutive_hits >= CONSECUTIVE_HITS_REQUIRED and
            elapsed >= MIN_SECONDS_BETWEEN_ALERTS):

            print(f"[+] TESORO EVENTO rilevato! ({event_best_name}) score={event_best_score:.3f}")
            send_notification(f"🚁 Tesoro EVENTO rilevato! ({event_best_name})")
            kill_app()

            last_alert_time = now
            consecutive_hits = 0

        print(f"[*] Attendo {CHECK_INTERVAL_SEC} secondi...")
        time.sleep(CHECK_INTERVAL_SEC)


if __name__ == "__main__":
    main()


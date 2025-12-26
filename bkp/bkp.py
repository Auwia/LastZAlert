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
MATCH_THRESHOLD = 0.50          
MIN_SECONDS_BETWEEN_ALERTS = 600
CONSECUTIVE_HITS_REQUIRED = 1

# Template che consideriamo come "evento tesoro"
EVENT_TEMPLATE_NAMES = {
    "elicottero.jpg",
    "scavatore.png",
}

# ROI (puoi aggiustare se serve)
ROI_X_START_FRAC = 0.50
ROI_X_END_FRAC   = 0.82
ROI_Y_START_FRAC = 0.84
ROI_Y_END_FRAC   = 0.97

# Debug ROI
DEBUG_SAVE_ROI = True
DEBUG_ROI_PATH = "roi_debug.png"
DEBUG_SCREENSHOT_MARKED = "debug_screen_marked.png"

# =============================
# ANCHOR (ingresso nel gioco)
# =============================

ANCHORS_DIR = "anchors"
ANCHOR_MATCH_THRESHOLD = 0.65          # << con il tuo log 0.65 è troppo alto
ANCHOR_CONSECUTIVE_HITS = 3            # deve essere sopra soglia per 3 check di fila
WAIT_FOR_GAME_TIMEOUT = 50

# Cerco l'anchor SOLO in basso-destra (molto più stabile e veloce)
ANCHOR_ROI_X_START_FRAC = 0.70
ANCHOR_ROI_Y_START_FRAC = 0.70
ANCHOR_ROI_X_END_FRAC   = 1.00
ANCHOR_ROI_Y_END_FRAC   = 1.00

# Multi-scale: prova template leggermente più piccolo/più grande
ANCHOR_SCALES = [0.75, 0.85, 0.95, 1.0, 1.05, 1.15, 1.25]

# Debug anchor
DEBUG_ANCHOR = True
DEBUG_ANCHOR_MARKED = "debug_anchor_marked.png"

# =============================
# UTILITY
# =============================
def load_all_anchors(directory):
    anchors = []
    if not os.path.isdir(directory):
        print(f"[!] Cartella anchors '{directory}' non trovata.")
        return anchors

    for name in os.listdir(directory):
        if not name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        p = os.path.join(directory, name)
        img = load_image(p)
        if img is not None:
            anchors.append((name, img))
            print(f"[+] Anchor caricata: {p}")
    return anchors

def crop_frac(img, x0f, y0f, x1f, y1f):
    H, W = img.shape[:2]
    x0 = int(W * x0f); x1 = int(W * x1f)
    y0 = int(H * y0f); y1 = int(H * y1f)
    x0 = max(0, min(x0, W-1))
    y0 = max(0, min(y0, H-1))
    x1 = max(x0+1, min(x1, W))
    y1 = max(y0+1, min(y1, H))
    return img[y0:y1, x0:x1], (x0, y0, x1, y1)


def match_template_multiscale(search_img, template_img, scales):
    """
    Restituisce: best_score, best_loc, best_scale, best_size
    loc è relativo a search_img.
    """
    best_score = 0.0
    best_loc = (0, 0)
    best_scale = 1.0
    best_size = (template_img.shape[1], template_img.shape[0])  # (w,h)

    search_gray = cv2.cvtColor(search_img, cv2.COLOR_BGR2GRAY)
    tmpl_gray0 = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)

    for s in scales:
        tw = int(tmpl_gray0.shape[1] * s)
        th = int(tmpl_gray0.shape[0] * s)
        if tw < 8 or th < 8:
            continue

        tmpl_gray = cv2.resize(tmpl_gray0, (tw, th), interpolation=cv2.INTER_AREA)

        H, W = search_gray.shape[:2]
        if H < th or W < tw:
            continue

        res = cv2.matchTemplate(search_gray, tmpl_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > best_score:
            best_score = max_val
            best_loc = max_loc
            best_scale = s
            best_size = (tw, th)

    return best_score, best_loc, best_scale, best_size


def debug_anchor_mark(full_img, roi_coords, match_loc, match_size, out_path=DEBUG_ANCHOR_MARKED):
    """
    Disegna:
    - rettangolo verde: anchor search ROI
    - rettangolo giallo: box del match migliore
    """
    x0, y0, x1, y1 = roi_coords
    mx, my = match_loc
    mw, mh = match_size

    img = full_img.copy()
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 3)

    # match_loc è relativo alla ROI -> trasformo in coordinate assolute
    abs_x = x0 + mx
    abs_y = y0 + my
    cv2.rectangle(img, (abs_x, abs_y), (abs_x + mw, abs_y + mh), (0, 255, 255), 3)

    cv2.imwrite(out_path, img)
    print(f"[+] Debug anchor salvato in {out_path}")

def load_all_images_from_dir(directory):
    imgs = []
    if not os.path.isdir(directory):
        print(f"[!] Cartella '{directory}' non trovata.")
        return imgs
    for name in os.listdir(directory):
        if not name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        path = os.path.join(directory, name)
        img = load_image(path)
        if img is not None:
            imgs.append((name, img))
            print(f"[+] Anchor caricata: {path}")
    return imgs


def best_match_anywhere(full_img, template):
    """
    Match su schermo intero (non solo ROI).
    Ritorna (score, max_loc) dove max_loc è top-left del match.
    """
    h, w = template.shape[:2]
    H, W = full_img.shape[:2]
    if H < h or W < w:
        return 0.0, (0, 0)
    res = cv2.matchTemplate(full_img, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    return max_val, max_loc


def wait_until_in_game(timeout_sec=WAIT_FOR_GAME_TIMEOUT):
    """
    Aspetta finché non rileva un'anchor (segno che lo splash è finito e siamo in game).
    """
    anchors = load_all_images_from_dir(ANCHORS_DIR)
    if not anchors:
        print("[!] Nessuna anchor trovata. Metti almeno 1 immagine in anchors/")
        return False

    print(f"[*] Attendo ingresso nel gioco (timeout {timeout_sec}s)...")
    start = time.time()

    while time.time() - start < timeout_sec:
        if not take_screenshot(SCREENSHOT_PATH):
            time.sleep(1)
            continue

        img = load_image(SCREENSHOT_PATH)
        if img is None:
            time.sleep(1)
            continue

        best_score = 0.0
        best_name = None

        for name, a in anchors:
            score, _ = best_match_anywhere(img, a)
            if score > best_score:
                best_score = score
                best_name = name

        print(f"[i] Anchor best: {best_name} score={best_score:.3f} (soglia {ANCHOR_MATCH_THRESHOLD})")

        if best_score >= ANCHOR_MATCH_THRESHOLD:
            print("[+] Siamo dentro al gioco, avvio gestione popup.")
            return True

        time.sleep(1)

    print("[!] Timeout: non sono riuscito a capire se sei entrato nel gioco.")
    return False

def run_cmd(cmd, timeout=30):
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
            text=True
        )
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
            proc = subprocess.run(
                [ADB_CMD, "exec-out", "screencap", "-p"],
                stdout=f,
                stderr=subprocess.PIPE,
                timeout=30,
                check=False
            )
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
    """Ritaglia la ROI in base alle frazioni configurate."""
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


def debug_dump_roi(full_img, roi, coords):
    """Salva ROI e screenshot intero con un rettangolo verde, solo per debug."""
    xs, ys, xe, ye = coords

    try:
        cv2.imwrite(DEBUG_ROI_PATH, roi)
        print(f"[+] ROI di debug salvata in {DEBUG_ROI_PATH}")
    except Exception as e:
        print("[!] Errore salvando ROI di debug:", e)

    try:
        img_marked = full_img.copy()
        cv2.rectangle(img_marked, (xs, ys), (xe, ye), (0, 255, 0), 3)  # rettangolo verde
        cv2.imwrite(DEBUG_SCREENSHOT_MARKED, img_marked)
        print(f"[+] Screenshot marcato salvato in {DEBUG_SCREENSHOT_MARKED}")
    except Exception as e:
        print("[!] Errore salvando screenshot marcato:", e)

def start_game():
    """
    Avvia Last Z dalla home usando monkey.
    """
    print("[*] Avvio Last Z...")
    run_cmd([
        ADB_CMD, "shell", "monkey",
        "-p", PACKAGE_NAME,
        "-c", "android.intent.category.LAUNCHER",
        "1"
    ])

def adb_tap(x, y):
    """
    Tap in coordinate assolute (pixel).
    """
    run_cmd([ADB_CMD, "shell", "input", "tap", str(x), str(y)])


def adb_tap_frac(x_frac, y_frac, screen_w, screen_h):
    """
    Tap in coordinate relative (frazioni 0–1 dello schermo).
    Esempio: x_frac=0.5, y_frac=0.5 = centro schermo.
    """
    x = int(screen_w * x_frac)
    y = int(screen_h * y_frac)
    print(f"[*] Tap a ({x},{y}) frazioni=({x_frac:.2f},{y_frac:.2f})")
    adb_tap(x, y)

def handle_startup_popups(max_seconds=20):
    """
    Dopo l'avvio del gioco, cerca di chiudere i popup iniziali
    tappando:
      - sulla X in alto a destra (popup tipo Sophia)
      - sul tasto back rotondo in basso a sinistra (schermate tipo Helping Hands)
      - in basso al centro, fuori dal popup
    per alcuni secondi.
    """
    print("[*] Gestione popup di avvio...")

    # screenshot solo per conoscere le dimensioni dello schermo
    if not take_screenshot(SCREENSHOT_PATH):
        print("[!] Impossibile fare screenshot per le dimensioni schermo.")
        return

    img = load_image(SCREENSHOT_PATH)
    if img is None:
        print("[!] Impossibile leggere screenshot per le dimensioni schermo.")
        return

    H, W = img.shape[:2]
    print(f"[*] Dimensioni schermo: {W}x{H}")

    # --- coordinate RELATIVE (0..1) ricavate dai tuoi screenshot ---

    # 1) X in alto a destra per popup tipo Sophia
    X_BTN_X_FRAC = 0.95   # molto vicino al bordo destro
    X_BTN_Y_FRAC = 0.06   # in alto

    # 2) tasto back rotondo in basso a sinistra (Helping Hands, altre schermate)
    BACK_BTN_X_FRAC = 0.08   # verso il bordo sinistro
    BACK_BTN_Y_FRAC = 0.93   # in basso

    # 3) tap fuori dal popup, zona centrale bassa
    OUTSIDE_FRAC_X = 0.50
    OUTSIDE_FRAC_Y = 0.94

    start = time.time()
    attempt = 0
    while time.time() - start < max_seconds:
        attempt += 1
        print(f"[*] Tentativo chiusura popup #{attempt}")

        # tap sulla X (se esiste)
        adb_tap_frac(X_BTN_X_FRAC, X_BTN_Y_FRAC, W, H)
        time.sleep(0.7)

        # tap sul tasto back in basso a sinistra
        adb_tap_frac(BACK_BTN_X_FRAC, BACK_BTN_Y_FRAC, W, H)
        time.sleep(0.7)

        # tap in basso al centro, fuori dal popup
        adb_tap_frac(OUTSIDE_FRAC_X, OUTSIDE_FRAC_Y, W, H)
        time.sleep(0.7)

    print("[*] Fine gestione popup di avvio.")

# =============================
# MAIN
# =============================

def main():
    print("=== Last Z Treasure Watcher (multi-template, ROI precisa) ===")

    if not check_adb_device():
        return

    start_game()

    anchors = load_all_anchors(ANCHORS_DIR)
    if not anchors:
        return

    if not wait_until_in_game():
            return

    handle_startup_popups(max_seconds=15)

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

        # Debug solo al primo giro
        if DEBUG_SAVE_ROI and not debug_done:
            debug_dump_roi(img, roi, (xs, ys, xe, ye))
            debug_done = True

        event_best_score = 0.0
        event_best_name = None
        scores_debug = []

        for name, tmpl in templates:
            score = compute_match_score(roi, tmpl)
            scores_debug.append((name, score))

            if name in EVENT_TEMPLATE_NAMES and score > event_best_score:
                event_best_score = score
                event_best_name = name

        print("[i] Scores:", ", ".join(f"{n}={s:.3f}" for n, s in scores_debug))
        print(f"[i] EVENTO: {event_best_name} score={event_best_score:.3f} "
              f"ROI=({xs},{ys})-({xe},{ye}) soglia={MATCH_THRESHOLD}")

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


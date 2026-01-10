#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
from datetime import datetime
import subprocess
import threading
from typing import List, Tuple, Optional

import cv2
import numpy as np
import requests

# ============================================================
# CONFIG
# ============================================================

ADB_CMD = "adb"
PACKAGE_NAME = "com.readygo.barrel.gp"

# Discord webhook (ATTENZIONE: se pubblico, meglio metterlo in env var)
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1446565181265154190/pL-0gcgP09RlQqnqHqQDIdQqm505tqa744is2R_1eGA3Had4OXmhPgQrTLYXYzaMld0S"

# Loop timing
CHECK_INTERVAL_SEC = 3

# Template match threshold
MATCH_THRESHOLD_TREASURE = 0.55
MATCH_THRESHOLD_HEAL     = 0.65
MATCH_THRESHOLD_HELP     = 0.40
MATCH_THRESHOLD_HOSPITAL = 0.60

# Anti-spam alert
MIN_SECONDS_BETWEEN_TREASURE_ALERTS = 600  # 10 min
CONSECUTIVE_HITS_REQUIRED_TREASURE = 3

# Paths
TEMPLATES_TREASURES_DIR = "treasures"
TEMPLATES_HEAL_DIR      = "heal"
TEMPLATES_HELP_DIR      = "help"
HEAL_FINISHED_DIR       = "heal_finished"
DEBUG_DIR               = "debug"
os.makedirs(DEBUG_DIR, exist_ok=True)

# Heal config
HEAL_BATCH_DEFAULT = 50
HEAL_DELAY_MULTIPLIER = 0.7
HEAL_FINISHED = os.path.join(HEAL_FINISHED_DIR, "screen_heal_loop.png")

# Help colleague config
HELP_COLLEAGUE_ROI = (0.74, 0.86, 0.70, 0.86)
TEMPLATES_HELP_COLLEAGUE_DIR = "colleague"
MATCH_THRESHOLD_HELP_COLLEAGUE = 0.31
HELP_COLLEAGUE_COOLDOWN = 20  # secondi

# HQ upgrade
TEMPLATES_HQ_UPGRADE_DIR = "hq_upgrade"
MATCH_THRESHOLD_HQ = 0.55
HQ_COOLDOWN = 15
last_hq_action = 0
HQ_BUBBLE_ROI  = (0.05, 0.40, 0.15, 0.55)   # chat area
HQ_GIFT_ROI    = (0.15, 0.85, 0.20, 0.70)   # banner centrale
HQ_OPEN_ROI    = (0.30, 0.70, 0.45, 0.80)   # bottone Open
HQ_CONFIRM_ROI = (0.25, 0.75, 0.60, 0.90)   # bottone Confirm

# Screenshot condiviso (riuso screen_treasure.png come "shared frame")
SCREENSHOT_PATH = os.path.join(DEBUG_DIR, "screen_treasure.png")
SCREENSHOT_LOCK = threading.Lock()

# ============================================================
# ROI (FRAZIONI dello schermo: x1,x2,y1,y2)
# Modifica qui se serve, come hai già fatto.
# ============================================================

# Tesoro: area basso-destra dove appare l’icona tesoro
TREASURE_ROI = (0.50, 0.82, 0.84, 0.97)

# Heal icon: nuvoletta croce rossa di solito sopra ospedale (zona centrale)
HEAL_ICON_ROI = (0.30, 0.70, 0.40, 0.75)

# Help icon: dove appare l’icona “stretta di mano” (spesso vicino ospedale/icone varie)
HELP_ICON_ROI = (0.30, 0.80, 0.40, 0.85)

# Dentro ospedale: prima riga campo numero (label su cui tappare per aprire tastiera)
# Metti qui la zona della label numerica della PRIMA RIGA (Shock Cavalry)
HOSPITAL_FIRST_ROW_NUMBER_LABEL_ROI = (0.78, 0.93, 0.33, 0.42)

#HOSPITAL_BANNER_ROI = (0.25, 0.75, 0.02, 0.14)
HOSPITAL_BANNER_ROI = (0.0, 1.0, 0.0, 0.22)

# ============================================================
# Debug
# ============================================================

DEBUG_SAVE_SCREENSHOTS = False   # salva screen interi
DEBUG_SAVE_ROIS        = True    # salva le ROI ritagliate
DEBUG_EVENTS_ONLY      = True  # scrive solo quando riconosce un evento


# ============================================================
# UTIL
# ============================================================

def log_event(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def run_cmd(cmd: List[str], timeout: int = 30) -> Tuple[int, str, str]:
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
        return 1, "", "timeout"


def check_adb_device() -> bool:
    code, out, err = run_cmd([ADB_CMD, "devices"])
    if code != 0:
        print("[!] adb devices error:", err)
        return False
    lines = out.strip().splitlines()
    devices = [l for l in lines[1:] if l.strip() and "device" in l]
    if not devices:
        print("[!] Nessun device ADB trovato.")
        return False
    print("[+] Device:", devices[0])
    return True


def take_screenshot(path: str) -> bool:
    try:
        proc = subprocess.run(
            [ADB_CMD, "exec-out", "screencap", "-p"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False
        )
        if proc.returncode != 0:
            print("[!] screencap error:", proc.stderr.decode("utf-8", errors="ignore"))
            return False
        with open(path, "wb") as f:
            f.write(proc.stdout)
        return True
    except Exception as e:
        print("[!] screencap exception:", e)
        return False

def screenshot_producer(stop_evt: threading.Event):
    tmp_path = SCREENSHOT_PATH + ".tmp"

    while not stop_evt.is_set():
        with SCREENSHOT_LOCK:
            proc = subprocess.run(
                [ADB_CMD, "exec-out", "screencap", "-p"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
                check=False
            )
            if proc.returncode == 0:
                with open(tmp_path, "wb") as f:
                    f.write(proc.stdout)
                os.replace(tmp_path, SCREENSHOT_PATH)  # atomico
        time.sleep(CHECK_INTERVAL_SEC)

def load_image(path: str):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        print("[!] Impossibile leggere immagine:", path)
    return img


def crop_roi(img, roi_frac: Tuple[float, float, float, float]):
    """roi_frac = (x1, x2, y1, y2) come FRAZIONI sullo schermo."""
    h, w = img.shape[:2]
    x1, x2, y1, y2 = roi_frac
    xs = int(w * x1)
    xe = int(w * x2)
    ys = int(h * y1)
    ye = int(h * y2)
    xs = max(0, min(xs, w - 1))
    xe = max(xs + 1, min(xe, w))
    ys = max(0, min(ys, h - 1))
    ye = max(ys + 1, min(ye, h))
    return img[ys:ye, xs:xe], (xs, ys, xe, ye)


def adb_tap(x: int, y: int):
    subprocess.run([ADB_CMD, "shell", "input", "tap", str(x), str(y)])

def adb_keyevent(code: int):
    subprocess.run([ADB_CMD, "shell", "input", "keyevent", str(code)])

def heal_sleep(seconds: float):
    time.sleep(seconds * HEAL_DELAY_MULTIPLIER)

def adb_input_text(txt: str):
    # Android input text: spazi vanno escape
    safe = txt.replace(" ", "%s")
    subprocess.run([ADB_CMD, "shell", "input", "text", safe])

def send_notification(text: str):
    if not DISCORD_WEBHOOK_URL:
        print("[!] DISCORD_WEBHOOK_URL non configurata. Notifica:", text)
        return False

    try:
        r = requests.post(DISCORD_WEBHOOK_URL, json={"content": text}, timeout=10)
        print("[i] Discord:", r.status_code, r.text[:200])
        return r.ok
    except Exception as e:
        print("[!] Errore Discord:", e)
        return False

def load_templates_from_dir(directory: str) -> List[Tuple[str, np.ndarray]]:
    templates = []
    if not os.path.isdir(directory):
        log_event(f"[!] Directory '{directory}' non trovata.")
        return templates

    for name in sorted(os.listdir(directory)):
        if not name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            continue
        full = os.path.join(directory, name)
        img = cv2.imread(full, cv2.IMREAD_COLOR)
        if img is None:
            print("[!] Template non leggibile:", full)
            continue
        templates.append((name, img))
    log_event(f"[+] Caricati {len(templates)} template da '{directory}'")
    return templates

def match_any(roi_img: np.ndarray, templates: List[Tuple[str, np.ndarray]]):
    """
    Ritorna:
      best_name, best_score, best_loc(x,y), best_size(h,w)
    """
    best_name = None
    best_score = 0.0
    best_loc = (0, 0)
    best_hw = (0, 0)

    for name, tmpl in templates:
        th, tw = tmpl.shape[:2]
        rh, rw = roi_img.shape[:2]
        if rh < th or rw < tw:
            continue

        res = cv2.matchTemplate(roi_img, tmpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        if score > best_score:
            best_score = float(score)
            best_name = name
            best_loc = loc
            best_hw = (th, tw)

    return best_name, best_score, best_loc, best_hw

def tap_match_in_fullscreen(roi_coords, match_loc, tmpl_hw):
    """Converte loc dentro ROI -> coordinate assolute e tappa al centro."""
    xs, ys, xe, ye = roi_coords
    mx, my = match_loc
    th, tw = tmpl_hw
    cx = xs + mx + tw // 2
    cy = ys + my + th // 2
    adb_tap(cx, cy)
    return cx, cy


# ============================================================
# THREAD 1: TREASURE WATCHER
# ============================================================

def treasure_watcher(stop_evt: threading.Event):
    templates = load_templates_from_dir(TEMPLATES_TREASURES_DIR)
    if not templates:
        print("[!] Nessun template tesoro. Thread treasure si ferma.")
        return

    last_alert = 0.0
    hits = 0

    while not stop_evt.is_set():
        with SCREENSHOT_LOCK:
            img = load_image(SCREENSHOT_PATH)

        if img is None:
            time.sleep(CHECK_INTERVAL_SEC)
            continue

        roi, coords = crop_roi(img, TREASURE_ROI)

        if DEBUG_SAVE_ROIS:
            cv2.imwrite(os.path.join(DEBUG_DIR, "roi_treasure.png"), roi)

        name, score, loc, hw = match_any(roi, templates)
        if not DEBUG_EVENTS_ONLY:
            log_event(f"[TREASURE] best={name} score={score:.3f} ROI={coords} thr={MATCH_THRESHOLD_TREASURE}")

        if score >= MATCH_THRESHOLD_TREASURE:
            hits += 1
        else:
            hits = 0

        now = time.time()
        if hits >= CONSECUTIVE_HITS_REQUIRED_TREASURE and (now - last_alert) >= MIN_SECONDS_BETWEEN_TREASURE_ALERTS:
            log_event(f"[TREASURE] RILEVATO {name} score={score:.3f} thr={MATCH_THRESHOLD_TREASURE}")
            send_notification(f"🎁 Tesoro rilevato! ({name}) score={score:.2f}")
            last_alert = now
            hits = 0

        time.sleep(CHECK_INTERVAL_SEC)


# ============================================================
# THREAD 2: HEAL AUTOMATION
# ============================================================
def heal_icon_watcher(stop_evt):
    templates = load_templates_from_dir(TEMPLATES_HEAL_DIR)

    while not stop_evt.is_set():
        img = load_image(SCREENSHOT_PATH)
        if img is None:
            time.sleep(1)
            continue

        roi, coords = crop_roi(img, HEAL_ICON_ROI)
        name, score, loc, hw = match_any(roi, templates)

        if score >= MATCH_THRESHOLD_HEAL:
            cx, cy = tap_match_in_fullscreen(coords, loc, hw)
            log_event(f"[HEAL] score={score:.3f} ≥ thr={MATCH_THRESHOLD_HEAL:.2f} tap heal_icon @ {cx},{cy}")
            time.sleep(2)  # debounce minimo

        time.sleep(1)

def hospital_watcher(stop_evt):
    templates = load_templates_from_dir("hospital")

    while not stop_evt.is_set():
        img = load_image(SCREENSHOT_PATH)
        if img is None:
            time.sleep(1)
            continue

        roi, coords = crop_roi(img, HOSPITAL_BANNER_ROI)
        name, score, loc, hw = match_any(roi, templates)

        if not DEBUG_EVENTS_ONLY:
            log_event(f"[HOSPITAL] best={name} score={score:.3f} thr={MATCH_THRESHOLD_HOSPITAL:.2f}")

        if score >= MATCH_THRESHOLD_HOSPITAL:
            # 1) tap label numerica prima riga
            xs, ys, xe, ye = crop_roi(img, HOSPITAL_FIRST_ROW_NUMBER_LABEL_ROI)[1]
            adb_tap((xs + xe)//2, (ys + ye)//2)
            log_event(f"[HOSPITAL] score={score:.3f} ≥ thr={MATCH_THRESHOLD_HOSPITAL:.2f} tap number label")

            time.sleep(1)

            # 2) inserisci batch
            adb_input_text(str(HEAL_BATCH_DEFAULT))
            adb_keyevent(66)
            log_event(f"[HOSPITAL] input batch={HEAL_BATCH_DEFAULT}")

            time.sleep(1)

            # 3) tap bottone Heal (zona fissa)
            adb_tap(
                int(img.shape[1] * 0.83),
                int(img.shape[0] * 0.86)
            )
            log_event("[HOSPITAL] tap HEAL button")

            time.sleep(3)

        time.sleep(1)

def help_heal_watcher(stop_evt):
    templates = load_templates_from_dir(TEMPLATES_HELP_DIR)
    last_tap = 0

    while not stop_evt.is_set():
        img = load_image(SCREENSHOT_PATH)
        if img is None:
            time.sleep(1)
            continue

        roi, coords = crop_roi(img, HELP_ICON_ROI)
        name, score, loc, hw = match_any(roi, templates)

        now = time.time()
        if score >= MATCH_THRESHOLD_HELP and now - last_tap > 10:
            cx, cy = tap_match_in_fullscreen(coords, loc, hw)
            log_event(f"[HELP-HEAL] score={score:.3f} ≥ thr={MATCH_THRESHOLD_HEAL:.2f} tap heal help @ {cx},{cy}")
            last_tap = now

        time.sleep(1)

def heal_finished_watcher(stop_evt):
    templates = load_templates_from_dir(HEAL_FINISHED_DIR)

    while not stop_evt.is_set():
        img = load_image(SCREENSHOT_PATH)
        if img is None:
            time.sleep(1)
            continue

        roi, coords = crop_roi(img, HEAL_ICON_ROI)
        name, score, loc, hw = match_any(roi, templates)

        if score >= MATCH_THRESHOLD_HEAL:
            cx, cy = tap_match_in_fullscreen(coords, loc, hw)
            log_event(f"[HEAL] finished, tap @ {cx},{cy}")
            time.sleep(3)

        time.sleep(1)

# ============================================================
# THREAD 3: HELP-COLLEAGUE AUTOMATION
# ============================================================
def help_colleague_watcher(stop_evt: threading.Event):
    templates = load_templates_from_dir(TEMPLATES_HELP_COLLEAGUE_DIR)
    if not templates:
        print("[HELP-COLLEAGUE] Nessun template, thread fermo.")
        return

    last_click = 0.0

    while not stop_evt.is_set():
        with SCREENSHOT_LOCK:
            img = load_image(SCREENSHOT_PATH)

        if img is None:
            time.sleep(CHECK_INTERVAL_SEC)
            continue

        roi, coords = crop_roi(img, HELP_COLLEAGUE_ROI)

        if DEBUG_SAVE_ROIS:
            cv2.imwrite(os.path.join(DEBUG_DIR, "roi_help_colleague.png"), roi)

        name, score, loc, hw = match_any(roi, templates)
        if not DEBUG_EVENTS_ONLY:
            log_event(f"[HELP-COLLEAGUE] best={name} score={score:.3f} thr={MATCH_THRESHOLD_HEAL:.2f}")

        now = time.time()
        if score >= MATCH_THRESHOLD_HELP_COLLEAGUE and (now - last_click) >= HELP_COLLEAGUE_COOLDOWN:
            cx, cy = tap_match_in_fullscreen(coords, loc, hw)
            log_event(f"[HELP-COLLEAGUE] tap @ {cx},{cy}")
            last_click = now

        time.sleep(CHECK_INTERVAL_SEC)

# ============================================================
# THREAD 4: HQ Upgrade Gift automation
# ============================================================
def hq_upgrade_watcher(stop_evt: threading.Event):
    templates = load_templates_from_dir(TEMPLATES_HQ_UPGRADE_DIR)
    if not templates:
        print("[HQ] Nessun template, thread fermo.")
        return
    hq_state = "IDLE"

    while not stop_evt.is_set():
        with SCREENSHOT_LOCK:
            img = load_image(SCREENSHOT_PATH)
    
        if img is None:
            time.sleep(CHECK_INTERVAL_SEC)
            continue
    
        # ---------------------------
        # STATE: IDLE → cerca bubble
        # ---------------------------
        if hq_state == "IDLE":
            roi, coords = crop_roi(img, HQ_BUBBLE_ROI)
            name, score, loc, hw = match_any(roi, templates)
    
            if score >= MATCH_THRESHOLD_HQ and "bubble" in name.lower():
                if name is None:
                    continue
                now = time.time()
                if now - last_hq_action < HQ_COOLDOWN:
                    continue
                last_hq_action = now
                tap_match_in_fullscreen(coords, loc, hw)
                print("[HQ] bubble → chat")
                hq_state = "CHAT_OPENED"
                time.sleep(2)
            else:
                time.sleep(CHECK_INTERVAL_SEC)
            continue
    
        # ---------------------------
        # STATE: CHAT_OPENED → gift
        # ---------------------------
        if hq_state == "CHAT_OPENED":
            roi, coords = crop_roi(img, HQ_GIFT_ROI)
            name, score, loc, hw = match_any(roi, templates)
    
            if score >= MATCH_THRESHOLD_HQ and "gift" in name.lower():
                tap_match_in_fullscreen(coords, loc, hw)
                print("[HQ] gift banner")
                hq_state = "GIFT_OPENED"
                time.sleep(2)
            else:
                hq_state = "IDLE"
            continue
    
        # ---------------------------
        # STATE: GIFT_OPENED → open
        # ---------------------------
        if hq_state == "GIFT_OPENED":
            roi, coords = crop_roi(img, HQ_OPEN_ROI)
            name, score, loc, hw = match_any(roi, templates)
    
            if score >= MATCH_THRESHOLD_HQ and "open" in name.lower():
                tap_match_in_fullscreen(coords, loc, hw)
                print("[HQ] OPEN")
                hq_state = "WAIT_CONFIRM"
                time.sleep(2)
            else:
                hq_state = "IDLE"
            continue
    
        # ---------------------------
        # STATE: WAIT_CONFIRM
        # ---------------------------
        if hq_state == "WAIT_CONFIRM":
            roi, coords = crop_roi(img, HQ_CONFIRM_ROI)
            name, score, loc, hw = match_any(roi, templates)
    
            if score >= MATCH_THRESHOLD_HQ and "confirm" in name.lower():
                tap_match_in_fullscreen(coords, loc, hw)
                print("[HQ] CONFIRM → DONE")
            hq_state = "IDLE"
            time.sleep(3)
            continue


        time.sleep(CHECK_INTERVAL_SEC)

# ============================================================
# MAIN
# ============================================================

def main():
    print("=== Last Z Bot (Treasure + Heal, threaded) ===")

    if not check_adb_device():
        return

    stop_evt = threading.Event()

    t0 = threading.Thread(target=screenshot_producer, args=(stop_evt,), daemon=True )
    t1 = threading.Thread(target=treasure_watcher, args=(stop_evt,), daemon=True)
    t2 = threading.Thread( target=heal_icon_watcher, args=(stop_evt,), daemon=True ) 
    t3 = threading.Thread( target=hospital_watcher, args=(stop_evt,), daemon=True ) 
    t4 = threading.Thread( target=help_heal_watcher, args=(stop_evt,), daemon=True ) 
    t5 = threading.Thread( target=heal_finished_watcher, args=(stop_evt,), daemon=True ) 
    t6 = threading.Thread( target=help_colleague_watcher, args=(stop_evt,), daemon=True ) 
    t7 = threading.Thread( target=hq_upgrade_watcher, args=(stop_evt,), daemon=True )

    t0.start()
    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t5.start()
    t6.start()
    t7.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[!] Stop richiesto.")
        stop_evt.set()
        time.sleep(1)


if __name__ == "__main__":
    main()

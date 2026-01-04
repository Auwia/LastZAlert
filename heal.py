#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
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
CHECK_INTERVAL_SEC = 8

# Template match threshold
MATCH_THRESHOLD_TREASURE = 0.40
MATCH_THRESHOLD_HEAL     = 0.50
MATCH_THRESHOLD_HELP     = 0.40

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
MATCH_THRESHOLD_HELP_COLLEAGUE = 0.35
HELP_COLLEAGUE_COOLDOWN = 20  # secondi

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

# ============================================================
# Debug
# ============================================================

DEBUG_SAVE_SCREENSHOTS = True   # salva screen interi
DEBUG_SAVE_ROIS        = True   # salva le ROI ritagliate


# ============================================================
# UTIL
# ============================================================

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
        print(f"[!] Directory '{directory}' non trovata.")
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
    print(f"[+] Caricati {len(templates)} template da '{directory}'")
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
        snap_path = os.path.join(DEBUG_DIR, "screen_treasure.png")
        if not take_screenshot(snap_path):
            time.sleep(CHECK_INTERVAL_SEC)
            continue

        img = load_image(snap_path)
        if img is None:
            time.sleep(CHECK_INTERVAL_SEC)
            continue

        roi, coords = crop_roi(img, TREASURE_ROI)

        if DEBUG_SAVE_ROIS:
            cv2.imwrite(os.path.join(DEBUG_DIR, "roi_treasure.png"), roi)

        name, score, loc, hw = match_any(roi, templates)
        print(f"[TREASURE] best={name} score={score:.3f} ROI={coords} thr={MATCH_THRESHOLD_TREASURE}")

        if score >= MATCH_THRESHOLD_TREASURE:
            hits += 1
        else:
            hits = 0

        now = time.time()
        if hits >= CONSECUTIVE_HITS_REQUIRED_TREASURE and (now - last_alert) >= MIN_SECONDS_BETWEEN_TREASURE_ALERTS:
            print(f"[TREASURE] RILEVATO {name} score={score:.3f}")
            send_notification(f"🎁 Tesoro rilevato! ({name}) score={score:.2f}")
            last_alert = now
            hits = 0

        time.sleep(CHECK_INTERVAL_SEC)


# ============================================================
# THREAD 2: HEAL AUTOMATION
# ============================================================

def heal_watcher(stop_evt: threading.Event):
    heal_templates = load_templates_from_dir(TEMPLATES_HEAL_DIR)
    help_templates = load_templates_from_dir(TEMPLATES_HELP_DIR)
    heal_finished_templates = load_templates_from_dir(HEAL_FINISHED_DIR)

    if not heal_templates:
        print("[!] Nessun template heal. Thread heal si ferma.")
        return

    # help templates possono anche mancare: in quel caso facciamo solo heal
    while not stop_evt.is_set():
        snap_path = os.path.join(DEBUG_DIR, "screen_treasure.png")
        if not take_screenshot(snap_path):
            time.sleep(CHECK_INTERVAL_SEC)
            continue

        img = load_image(snap_path)
        if img is None:
            continue

        roi_f, coords_f = crop_roi(img, HEAL_ICON_ROI)
    
        name_f, score_f, loc_f, hw_f = match_any(roi_f, heal_finished_templates)
    
        if score_f >= MATCH_THRESHOLD_HEAL:
            print(f"[HEAL] heal finished detected ({name_f}) score={score_f:.3f}")
            tap_match_in_fullscreen(coords_f, loc_f, hw_f)
            heal_sleep(HEAL_DELAY_MULTIPLIER)
            continue

        snap_path = os.path.join(DEBUG_DIR, "screen_heal.png")
        if not take_screenshot(snap_path):
            heal_sleep(HEAL_DELAY_MULTIPLIER)
            continue

        img = load_image(snap_path)
        if img is None:
            heal_sleep(HEAL_DELAY_MULTIPLIER)
            continue

        # 1) cerca icona heal sulla mappa
        roi_map, coords_map = crop_roi(img, HEAL_ICON_ROI)
        name, score, loc, hw = match_any(roi_map, heal_templates)
        print(f"[HEAL] heal_icon best={name} score={score:.3f} thr={MATCH_THRESHOLD_HEAL}")

        if score < MATCH_THRESHOLD_HEAL:
            time.sleep(5)
            continue

        # click icona heal
        cx, cy = tap_match_in_fullscreen(coords_map, loc, hw)
        print(f"[HEAL] tap heal_icon @ {cx},{cy}")
        heal_sleep(HEAL_DELAY_MULTIPLIER)

        # 2) dentro ospedale: tap sulla label numerica (prima riga) per aprire tastiera
        snap_hosp = os.path.join(DEBUG_DIR, "screen_hospital.png")
        take_screenshot(snap_hosp)
        img_h = load_image(snap_hosp)
        if img_h is None:
            heal_sleep(HEAL_DELAY_MULTIPLIER) 
            continue

        roi_label, coords_label = crop_roi(img_h, HOSPITAL_FIRST_ROW_NUMBER_LABEL_ROI)

        # tap al centro della ROI label (non template: è “zona fissa” che vuoi)
        xs, ys, xe, ye = coords_label
        adb_tap((xs + xe) // 2, (ys + ye) // 2)
        print("[HEAL] tap number label (keyboard open)")
        heal_sleep(HEAL_DELAY_MULTIPLIER)

        # inserisci batch
        adb_input_text(str(HEAL_BATCH_DEFAULT))
        time.sleep(0.3)
        adb_keyevent(66)  # ENTER = OK (spesso funziona)
        print(f"[HEAL] input batch={HEAL_BATCH_DEFAULT} + ENTER")
        heal_sleep(HEAL_DELAY_MULTIPLIER)

        # torna indietro alla mappa
        # adb_keyevent(4)  # BACK
        adb_tap(int(img_h.shape[1] * 0.83), int(img_h.shape[0] * 0.86))  # tap bottone blu "Heal"
        heal_sleep(HEAL_DELAY_MULTIPLIER)

        # 3) cerca icona help e cliccala (se hai template)
        if help_templates:
            snap_help = os.path.join(DEBUG_DIR, "screen_help.png")
            take_screenshot(snap_help)
            img2 = load_image(snap_help)
            if img2 is not None:
                roi_help, coords_help = crop_roi(img2, HELP_ICON_ROI)
                if DEBUG_SAVE_ROIS:
                    cv2.imwrite(os.path.join(DEBUG_DIR, "roi_help_icon.png"), roi_help)

                n2, s2, loc2, hw2 = match_any(roi_help, help_templates)
                print(f"[HEAL] help_icon best={n2} score={s2:.3f} thr={MATCH_THRESHOLD_HELP}")

                if s2 >= MATCH_THRESHOLD_HELP:
                    cx2, cy2 = tap_match_in_fullscreen(coords_help, loc2, hw2)
                    print(f"[HEAL] tap help_icon @ {cx2},{cy2}")
                    heal_sleep(HEAL_DELAY_MULTIPLIER)

        # 4) aspetta un po’ prima del prossimo ciclo
        heal_sleep(HEAL_DELAY_MULTIPLIER)

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
        snap = os.path.join(DEBUG_DIR, "screen_help_colleague.png")
        if not take_screenshot(snap):
            time.sleep(CHECK_INTERVAL_SEC)
            continue

        img = load_image(snap)
        if img is None:
            time.sleep(CHECK_INTERVAL_SEC)
            continue

        roi, coords = crop_roi(img, HELP_COLLEAGUE_ROI)

        if DEBUG_SAVE_ROIS:
            cv2.imwrite(os.path.join(DEBUG_DIR, "roi_help_colleague.png"), roi)

        name, score, loc, hw = match_any(roi, templates)
        print(f"[HELP-COLLEAGUE] best={name} score={score:.3f}")

        now = time.time()
        if score >= MATCH_THRESHOLD_HELP_COLLEAGUE and (now - last_click) >= HELP_COLLEAGUE_COOLDOWN:
            cx, cy = tap_match_in_fullscreen(coords, loc, hw)
            print(f"[HELP-COLLEAGUE] tap @ {cx},{cy}")
            last_click = now

        time.sleep(CHECK_INTERVAL_SEC)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=== Last Z Bot (Treasure + Heal, threaded) ===")

    if not check_adb_device():
        return

    stop_evt = threading.Event()

    t1 = threading.Thread(target=treasure_watcher, args=(stop_evt,), daemon=True)
    t2 = threading.Thread(target=heal_watcher, args=(stop_evt,), daemon=True)
    t3 = threading.Thread(target=help_colleague_watcher, args=(stop_evt,), daemon=True)

    t1.start()
    t2.start()
    t3.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[!] Stop richiesto.")
        stop_evt.set()
        time.sleep(1)


if __name__ == "__main__":
    main()

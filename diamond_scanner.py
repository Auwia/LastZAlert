#!/usr/bin/env python3
# diamond_scanner.py

import cv2
import time
import json
import argparse
import os
import numpy as np
import easyocr
import subprocess
from pathlib import Path

# =========================
# CONFIG
# =========================

# ZOOM CONFIG
ZOOM_OUT_STEPS = 4       # quante volte zoommare out
ZOOM_OUT_DISTANCE = 350  # quanto "aperti" i pinch
ZOOM_SLEEP = 0.6

SCREENSHOT_PATH = "screen.png"

DIAMOND_TEMPLATES_DIR = "diamonds"
BUTTONS_DIR = "buttons"

STATE_DIR = "scan_state"
STATE_FILE = f"{STATE_DIR}/progress.json"

DEBUG_DIR = "ocr"
os.makedirs(DEBUG_DIR, exist_ok=True)
os.makedirs(STATE_DIR, exist_ok=True)

# OCR
OCR_READER = easyocr.Reader(['en'], gpu=False)

# =========================
# ADB UTILS
# =========================

def adb(cmd):
    subprocess.run(["adb"] + cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def adb_tap(x, y):
    adb(["shell", "input", "tap", str(x), str(y)])

def adb_swipe(x1, y1, x2, y2, duration=600):
    adb(["shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(duration)])

def adb_screenshot():
    adb(["exec-out", "screencap", "-p"],)

    subprocess.run(
        ["adb", "exec-out", "screencap", "-p"],
        stdout=open(SCREENSHOT_PATH, "wb")
    )

# =========================
# ZOOM UTILS
# =========================
def adb_zoom_out(steps=4, distance=350):
    """
    Zoom-out reale con pinch multitouch
    Funziona davvero su Android
    """
    cx, cy = 540, 960  # centro schermo (1080x1920)

    for _ in range(steps):
        cmd = [
            "adb", "shell", "input", "touchscreen", "swipe",
            str(cx), str(cy - 50),
            str(cx), str(cy - distance),
            "300",
            str(cx), str(cy + 50),
            str(cx), str(cy + distance),
        ]
        subprocess.run(cmd)
        time.sleep(ZOOM_SLEEP)

# =========================
# IMAGE UTILS
# =========================

def load_img(path):
    return cv2.imread(path)

def match_template(img, tmpl, threshold=0.85):
    res = cv2.matchTemplate(img, tmpl, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    return max_val, max_loc, tmpl.shape[:2]

def find_button(img, name, threshold=0.85):
    tmpl = load_img(f"{BUTTONS_DIR}/{name}")
    score, loc, (h, w) = match_template(img, tmpl, threshold)
    if score >= threshold:
        return (loc[0] + w // 2, loc[1] + h // 2), score
    return None, score

# =========================
# OCR
# =========================

def read_gathered_value(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)

    cv2.imwrite(f"{DEBUG_DIR}/gathered_roi.png", th)

    result = OCR_READER.readtext(th, detail=0)
    for txt in result:
        if "/" in txt:
            try:
                return int(txt.split("/")[0])
            except:
                pass
    return None

# =========================
# STATE
# =========================

def load_state():
    if os.path.exists(STATE_FILE):
        return json.load(open(STATE_FILE))
    return {
        "saved": 0,
        "moves": 0
    }

def save_state(state):
    json.dump(state, open(STATE_FILE, "w"), indent=2)

# =========================
# MAIN FSM
# =========================

def main(min_gathered, max_mines, max_time):

    start_time = time.time()
    state = load_state()

    print("[*] Diamond scanner started")

    # ---- OPEN VS ----
    adb_screenshot()
    img = load_img(SCREENSHOT_PATH)
    pos, _ = find_button(img, "vs.png", 0.8)
    if pos:
        adb_tap(*pos)
        time.sleep(2)

    # ---- ENEMY BUSTER ----
    adb_screenshot()
    img = load_img(SCREENSHOT_PATH)
    pos, _ = find_button(img, "enemy_booster.png", 0.8)
    if pos:
        adb_tap(*pos)
        time.sleep(2)

    # ---- TELEPORT ----
    adb_screenshot()
    img = load_img(SCREENSHOT_PATH)
    pos, _ = find_button(img, "teleport.png", 0.8)
    if pos:
        adb_tap(*pos)
        time.sleep(2)

    # ---- CANCEL ----
    adb_screenshot()
    img = load_img(SCREENSHOT_PATH)
    pos, _ = find_button(img, "x_cancel.png", 0.8)
    if pos:
        adb_tap(*pos)
        time.sleep(1)

    # ---- ZOOM OUT ----
    print("[*] Zooming out map")
    adb_zoom_out(
        steps=ZOOM_OUT_STEPS,
        distance=ZOOM_OUT_DISTANCE
    )
    time.sleep(1)

    # ---- SCAN LOOP ----
    while True:

        if time.time() - start_time > max_time:
            print("[!] Max time reached")
            break

        if state["saved"] >= max_mines:
            print("[!] Max mines saved")
            break

        adb_screenshot()
        img = load_img(SCREENSHOT_PATH)

        # ---- FIND DIAMONDS ----
        for fname in os.listdir(DIAMOND_TEMPLATES_DIR):
            tmpl = load_img(f"{DIAMOND_TEMPLATES_DIR}/{fname}")
            score, loc, (h, w) = match_template(img, tmpl, 0.75)
            if score < 0.75:
                continue

            cx = loc[0] + w // 2
            cy = loc[1] + h // 2

            adb_tap(cx, cy)
            time.sleep(1.2)

            adb_screenshot()
            popup = load_img(SCREENSHOT_PATH)

            # crop popup ROI (tuning needed)
            h_img, w_img = popup.shape[:2]
            roi = popup[int(h_img*0.45):int(h_img*0.6), int(w_img*0.3):int(w_img*0.7)]

            gathered = read_gathered_value(roi)

            if gathered is None or gathered < min_gathered:
                adb_tap(50, 50)  # tap grass
                time.sleep(0.5)
                continue

            # ---- SAVE BOOKMARK ----
            pos, _ = find_button(popup, "favourite.png", 0.8)
            if pos:
                adb_tap(*pos)
                time.sleep(1)

                adb_tap(w_img//2, h_img//2)
                adb(["shell", "input", "text", str(state["saved"] + 1)])
                time.sleep(0.5)

                adb_screenshot()
                conf = load_img(SCREENSHOT_PATH)
                pos, _ = find_button(conf, "confirm_blue.png", 0.8)
                if pos:
                    adb_tap(*pos)
                    time.sleep(1)

                state["saved"] += 1
                save_state(state)

            adb_tap(50, 50)
            time.sleep(0.5)

        # ---- MOVE MAP ----
        adb_swipe(600, 800, 200, 800)
        state["moves"] += 1
        save_state(state)
        time.sleep(1)

    print("[*] Scan finished")

# =========================
# ENTRY
# =========================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--min_gathered", type=int, default=60)
    parser.add_argument("--max_mines", type=int, default=50)
    parser.add_argument("--max_time", type=int, default=3600)

    args = parser.parse_args()

    main(args.min_gathered, args.max_mines, args.max_time)


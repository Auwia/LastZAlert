# bot_utils.py
import os
import cv2
import numpy as np
import subprocess

ADB_CMD = "adb"


def adb_tap(x: int, y: int):
    subprocess.run([ADB_CMD, "shell", "input", "tap", str(int(x)), str(int(y))],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def load_image(path: str):
    return cv2.imread(path, cv2.IMREAD_COLOR)


def crop_roi(img, roi_frac):
    """
    roi_frac = (x1, x2, y1, y2) in frazione [0..1]
    ritorna: (roi_img, (x_offset_px, y_offset_px))
    """
    h, w = img.shape[:2]
    x1, x2, y1, y2 = roi_frac
    x1p = int(w * x1)
    x2p = int(w * x2)
    y1p = int(h * y1)
    y2p = int(h * y2)

    roi = img[y1p:y2p, x1p:x2p]
    return roi, (x1p, y1p)


def load_templates(path: str):
    """
    Se path è una directory: carica tutti i .png/.jpg dentro (sorted).
    Se path è un file: carica solo quel file.
    Ritorna lista di tuple (name, img).
    """
    templates = []

    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            if not name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                continue
            full = os.path.join(path, name)
            img = cv2.imread(full, cv2.IMREAD_COLOR)
            if img is None:
                print(f"[!] Template non leggibile: {full}")
                continue
            templates.append((name, img))
        print(f"[+] Caricati {len(templates)} template da '{path}'")
        return templates

    # file singolo
    if os.path.isfile(path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[!] Template non leggibile: {path}")
            return []
        templates.append((os.path.basename(path), img))
        print(f"[+] Caricati 1 template da '{path}'")
        return templates

    print(f"[!] Path template non trovato: '{path}'")
    return []


def match_any(img, templates):
    """
    img: ROI (BGR)
    templates: [(name, tmpl_bgr), ...]
    ritorna: (best_name, best_score, best_loc, (th, tw))
    """
    best_name = None
    best_score = 0.0
    best_loc = (0, 0)
    best_hw = (0, 0)

    if img is None or img.size == 0:
        return None, 0.0, (0, 0), (0, 0)

    ih, iw = img.shape[:2]

    for name, tmpl in templates:
        th, tw = tmpl.shape[:2]
        if ih < th or iw < tw:
            continue

        res = cv2.matchTemplate(img, tmpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)

        if float(score) > best_score:
            best_score = float(score)
            best_name = name
            best_loc = loc
            best_hw = (th, tw)

    return best_name, best_score, best_loc, best_hw


#!/usr/bin/env python3
import re, os
import cv2
import numpy as np
import subprocess
import time
import pytesseract
from datetime import datetime

# =========================
# CONFIG
# =========================
ADB_TIMEOUT = 10

USE_ROI = (0.15, 0.42, 0.85, 0.68)
REWARD_ROI = (0.20, 0.35, 0.80, 0.60)

# posizione RELATIVA del numero rispetto al centro di USE
AMOUNT_FROM_USE = (-220, -90)   # x,y
AMOUNT_BOX_SIZE = (180, 70)     # w,h
AMOUNT_LABEL_POS = (878, 934)

THR_CHEST = 0.80
THR_UI    = 0.85

CONGR_ROI = (0.05, 0.02, 0.95, 0.22) 
THR_CONGR = 0.70 

DEBUG = False

# =========================
# LOG
# =========================
def log(msg):
    if not DEBUG and msg.startswith("[DEBUG]"):
        return
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def save_debug(name, img):
    if img is None or img.size == 0:
        return
    os.makedirs("debug", exist_ok=True)
    cv2.imwrite(os.path.join("debug", name), img)

def read_amount_at_label(img):
    # coordinate CENTRO numero (quelle che già funzionano)
    cx, cy = 878, 934

    # ROI stretta attorno al numero
    w, h = 80, 50
    x1 = max(0, cx - w//2)
    y1 = max(0, cy - h//2)
    x2 = min(img.shape[1], cx + w//2)
    y2 = min(img.shape[0], cy + h//2)

    roi = img[y1:y2, x1:x2]

    # DEBUG VISIVO
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(f"debug/amount_roi_{ts}.png", roi)

    # OCR
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)[1]

    text = pytesseract.image_to_string(
        gray,
        config="--psm 7 -c tessedit_char_whitelist=0123456789"
    ).strip()

    if text.isdigit():
        return int(text)

    return None

def adb_get_wm_size():
    p = adb(["shell", "wm", "size"])
    out = (p.stdout + p.stderr).decode("utf-8", errors="ignore")
    # Cerca prima Override, poi Physical
    m = re.search(r"Override size:\s*(\d+)x(\d+)", out)
    if not m:
        m = re.search(r"Physical size:\s*(\d+)x(\d+)", out)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))

# global cache
_DEVICE_SIZE = None

def adb_tap_scaled(x, y, shot_w, shot_h):
    global _DEVICE_SIZE
    if _DEVICE_SIZE is None:
        _DEVICE_SIZE = adb_get_wm_size()
        log(f"[DEBUG] wm size = {_DEVICE_SIZE}")

    if not _DEVICE_SIZE:
        # fallback: tap diretto
        adb_tap(int(x), int(y))
        return

    dev_w, dev_h = _DEVICE_SIZE
    sx = dev_w / float(shot_w)
    sy = dev_h / float(shot_h)

    tx = int(x * sx)
    ty = int(y * sy)
    log(f"[DEBUG] tap map ({x},{y}) -> ({tx},{ty}) scale=({sx:.4f},{sy:.4f})")
    adb_tap(tx, ty)

def debug_draw_tap(img, x, y, prefix="debug_tap"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{prefix}_{ts}.png"

    dbg = img.copy()

    # cerchio rosso
    cv2.circle(dbg, (int(x), int(y)), 25, (0, 0, 255), 3)

    # croce
    cv2.line(dbg, (int(x)-40, int(y)), (int(x)+40, int(y)), (0, 0, 255), 2)
    cv2.line(dbg, (int(x), int(y)-40), (int(x), int(y)+40), (0, 0, 255), 2)

    # coordinate stampate
    cv2.putText(
        dbg,
        f"({int(x)}, {int(y)})",
        (int(x)+10, int(y)-10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    os.makedirs("debug", exist_ok=True)
    path = os.path.join("debug", name)
    cv2.imwrite(path, dbg)

    log(f"[DEBUG] tap visualized -> {path}")


# =========================
# ADB HELPERS
# =========================
def adb(cmd):
    return subprocess.run(
        ["adb"] + cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=ADB_TIMEOUT
    )

def adb_screencap():
    p = adb(["exec-out", "screencap", "-p"])
    if p.returncode != 0:
        return None
    img = np.frombuffer(p.stdout, np.uint8)
    return cv2.imdecode(img, cv2.IMREAD_COLOR)

def adb_tap(x, y):
    adb(["shell", "input", "tap", str(x), str(y)])

def adb_input_text(txt):
    adb(["shell", "input", "text", txt])

def adb_enter():
    adb(["shell", "input", "keyevent", "66"])

# =========================
# IMAGE HELPERS
# =========================
def load_png(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)

def crop(img, roi):
    h, w = img.shape[:2]
    x1 = int(w * roi[0])
    y1 = int(h * roi[1])
    x2 = int(w * roi[2])
    y2 = int(h * roi[3])
    if x2 <= x1 or y2 <= y1:
        return None, (0, 0)
    return img[y1:y2, x1:x2], (x1, y1)

def match(template, img, thr):
    if img is None or template is None:
        return False, 0, None, None
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    return score >= thr, score, loc, template.shape[:2]

def tap_match(loc, size, offset):
    cx = offset[0] + loc[0] + size[1] // 2
    cy = offset[1] + loc[1] + size[0] // 2
    adb_tap(cx, cy)

# =========================
# OCR AMOUNT (MUST CHECK)
# =========================
def read_amount_near_use(img, ux, uy):
    ax = int(ux + AMOUNT_FROM_USE[0])
    ay = int(uy + AMOUNT_FROM_USE[1])
    w, h = AMOUNT_BOX_SIZE

    roi = img[ay:ay+h, ax:ax+w]
    if roi is None or roi.size == 0:
        return None

    save_debug("debug_amount_number.png", roi)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    txt = pytesseract.image_to_string(
        gray,
        config="--psm 7 -c tessedit_char_whitelist=0123456789"
    ).strip()

    return int(txt) if txt.isdigit() else None

# =========================
# OCR REWARD
# =========================
def ocr_reward_text(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # aumenta contrasto
    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)

    # binarizzazione soft
    g = cv2.threshold(g, 160, 255, cv2.THRESH_BINARY)[1]

    txt = pytesseract.image_to_string(
        g,
        config="--psm 6"
    )

    return txt

def clean_reward_text(txt):
    lines = []
    for l in txt.splitlines():
        l = l.strip()
        if len(l) < 4:
            continue
        if re.search(r"[A-Za-z]{3,}", l):
            lines.append(l)

    return " ".join(lines)

# =========================
# TEMPLATES
# =========================
CHESTS = [
    ("question_marked", load_png("chests/01.question_marked.png")),
    ("honor_crate",     load_png("chests/02.hero_battlefield_honor_crate.png")),
    ("treasure_hunt",   load_png("chests/03.treasure_hunt.png")),
    ("dual_master",     load_png("chests/04.dual_master_supply_box.png")),
]

TPL_USE   = load_png("use_button/use_blue.png")
TPL_CONGR = load_png("reward_popup/congratulations.png")

# =========================
# MAIN
# =========================
def main():
    log("=== Chest Opener Started ===")
    opened = 0

    while True:
        img = adb_screencap()
        if img is None:
            time.sleep(1)
            continue

        log(f"[DEBUG] frame size = {img.shape}")

        # 1️⃣ trova cassa
        found = False
        for name, tpl in CHESTS:
            ok, score, loc, size = match(tpl, img, THR_CHEST)
            if ok:
                tap_match(loc, size, (0, 0))
                log(f"[CHEST] {name} tapped (score={score:.3f})")
                found = True
                break

        # attesa minima + nuovo frame (POPUP)
        time.sleep(0.8)
        img = adb_screencap()
        if img is None:
            continue

        if not found:
            time.sleep(1)
            continue

        # === ORA IL POPUP È APERTO: CLICK SULLA LABEL ===
        ax, ay = AMOUNT_LABEL_POS
        log(f"[CHEST] clicking amount label at {ax},{ay}")
        shot_h, shot_w = img.shape[:2]
        debug_draw_tap(img, ax, ay, "click_amount")
        adb_tap_scaled(ax, ay, shot_w, shot_h)
        
        time.sleep(0.4)
        adb_input_text("1")
        adb_enter()
        time.sleep(0.6)

        time.sleep(1.2)
        img = adb_screencap()
        if img is None:
            continue

        # 2️⃣ trova USE
        img = adb_screencap()
        amount = read_amount_at_label(img)
        log(f"[CHEST] amount after input = {amount}")
        
        if amount != 1:
            log("[CHEST] amount != 1 → retry set")
            # NON tornare al loop principale
            time.sleep(0.5)
            continue 

        # ORA cerca USE
        # 3️⃣ aspetta USE SENZA TORNARE ALLA CHEST
        use_found = False
        for _ in range(10):  # ~5 secondi max
            img = adb_screencap()
            if img is None:
                continue
        
            roi_use, off_use = crop(img, USE_ROI)
            ok, score, uloc, usize = match(TPL_USE, roi_use, THR_UI)
        
            if ok:
                use_found = True
                break
        
            time.sleep(0.5)
        
        if not use_found:
            log("[CHEST] USE not matched but popup is open → retry USE only")
            time.sleep(0.5)
            continue  # continua NEL POPUP, NON AL LOOP PRINCIPALE

        # 4️⃣ premi USE
        tap_match(uloc, usize, off_use)
        log("[CHEST] USE tapped")

        # 5️⃣ reward
        time.sleep(1.0)
        img = adb_screencap()
        if img is None:
            log("[REWARD] screencap failed")
            continue
        
        # DEBUG: salva sempre cosa stiamo leggendo
        rimg, _ = crop(img, REWARD_ROI)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"debug/reward_roi_{ts}.png", rimg)
        
        raw = ocr_reward_text(rimg)
        clean = clean_reward_text(raw)
        
        opened += 1
        if clean:
            log(f"[REWARD #{opened}] {clean}")
        else:
            log(f"[REWARD #{opened}] (unreadable)")

        time.sleep(1)

# =========================
if __name__ == "__main__":
    main()


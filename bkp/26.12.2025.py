#!/usr/bin/env python3
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import time
import subprocess
import requests
import cv2

# =============================
# CONFIG DISCORD WEBHOOK
# =============================
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1446565181265154190/pL-0gcgP09RlQqnqHqQDIdQqm505tqa744is2R_1eGA3Had4OXmhPgQrTLYXYzaMld0S"

def send_notification(text: str) -> bool:
    if not DISCORD_WEBHOOK_URL:
        print("[!] DISCORD_WEBHOOK_URL non configurata.")
        return False
    try:
        r = requests.post(DISCORD_WEBHOOK_URL, json={"content": text}, timeout=10)
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
SCREENSHOT_FOLDER = "debug"
os.makedirs(SCREENSHOT_FOLDER, exist_ok=True)

CHECK_INTERVAL_SEC = 10
SCREENSHOTS_PER_CYCLE = 3
INTERVAL_BETWEEN_SHOTS = 1
MATCH_THRESHOLD = 0.40
MIN_SECONDS_BETWEEN_ALERTS = 600
CONSECUTIVE_HITS_REQUIRED = 1

ROI_X_START_FRAC = 0.50
ROI_X_END_FRAC   = 0.82
ROI_Y_START_FRAC = 0.84
ROI_Y_END_FRAC   = 0.97

DEBUG_SAVE_ROI = True
DEBUG_ONLY_ONCE = True
DEBUG_ROI_PATH = "roi_debug.png"
DEBUG_SCREENSHOT_MARKED = "debug_screen_marked.png"

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

def check_adb_device() -> bool:
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
            print("[!] Errore screencap:", proc.stderr.decode("utf-8", errors="ignore"))
            return False
        # Write the image to a temporary path
        tmp_path = path + ".tmp"
        with open(tmp_path, "wb") as f:
            f.write(proc.stdout)
        # Load and resize the image
        import cv2
        img = cv2.imread(tmp_path)
        if img is None:
            print("[!] Errore leggendo screenshot.")
            return False
        img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
        cv2.imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        os.remove(tmp_path)
        return True
    except Exception as e:
        print("[!] Eccezione screencap:", e)
        return False
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

def load_image(path: str):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"[!] Impossibile leggere immagine: {path}")
    return img

def crop_roi(img):
    H, W = img.shape[:2]
    xs = int(W * ROI_X_START_FRAC)
    xe = int(W * ROI_X_END_FRAC)
    ys = int(H * ROI_Y_START_FRAC)
    ye = int(H * ROI_Y_END_FRAC)
    xs = max(0, min(xs, W - 1))
    xe = max(xs + 1, min(xe, W))
    ys = max(0, min(ys, H - 1))
    ye = max(ys + 1, min(ye, H))
    roi = img[ys:ye, xs:xe]
    return roi, (xs, ys, xe, ye)

def compute_match_score(roi, template) -> float:
    h, w = template.shape[:2]
    H, W = roi.shape[:2]
    if H < h or W < w:
        return 0.0
    res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    return float(max_val)

def load_all_templates(directory: str):
    templates = []
    if not os.path.isdir(directory):
        print(f"[!] Cartella template '{directory}' non trovata.")
        return templates
    for name in sorted(os.listdir(directory)):
        if not name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        full_path = os.path.join(directory, name)
        img = load_image(full_path)
        if img is not None:
            templates.append((name, img))
            print(f"[+] Template caricato: {full_path}")
    if not templates:
        print(f"[!] Nessun template trovato in '{directory}'.")
    return templates

def debug_dump_roi(full_img, roi, coords):
    xs, ys, xe, ye = coords
    try:
        cv2.imwrite(DEBUG_ROI_PATH, roi)
        print(f"[+] ROI di debug salvata in {DEBUG_ROI_PATH}")
    except Exception as e:
        print("[!] Errore salvando ROI di debug:", e)
    try:
        img_marked = full_img.copy()
        cv2.rectangle(img_marked, (xs, ys), (xe, ye), (0, 255, 0), 3)
        cv2.imwrite(DEBUG_SCREENSHOT_MARKED, img_marked)
        print(f"[+] Screenshot marcato salvato in {DEBUG_SCREENSHOT_MARKED}")
    except Exception as e:
        print("[!] Errore salvando screenshot marcato:", e)


def take_multiple_screenshots_parallel(num_screenshots=3, delay_sec=1):
    screenshot_paths = []
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    with ThreadPoolExecutor(max_workers=num_screenshots) as executor:
        futures = []
        for i in range(num_screenshots):
            shot_path = os.path.join(SCREENSHOT_FOLDER, f"screen_{i}_{timestamp}.jpg")
            futures.append(executor.submit(take_screenshot_with_delay, shot_path, i * delay_sec))
            screenshot_paths.append(shot_path)
        for future in as_completed(futures):
            future.result()
    return screenshot_paths

def take_screenshot_with_delay(path: str, delay: int) -> bool:
    time.sleep(delay)
    return take_screenshot(path)


def main():
    # Pulizia della cartella debug all'avvio
    for fname in os.listdir(SCREENSHOT_FOLDER):
        try:
            os.remove(os.path.join(SCREENSHOT_FOLDER, fname))
        except Exception:
            pass

    print("=== Last Z Treasure Watcher (multi-screenshot mode) ===")
    if not check_adb_device():
        return
    templates = load_all_templates(TEMPLATES_DIR)
    if not templates:
        return
    last_alert_time = 0.0
    consecutive_hits = 0
    debug_done = False
    while True:
        screenshots = []
        for i in range(SCREENSHOTS_PER_CYCLE):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            shot_path = os.path.join(SCREENSHOT_FOLDER, f"screen_{i}_{timestamp}.jpg")
            if take_screenshot(shot_path):
                screenshots.append(shot_path)
            time.sleep(INTERVAL_BETWEEN_SHOTS)

        best_score = 0.0
        best_name = None
        scores_debug = []

        for path in screenshots:
            img = load_image(path)
            if img is None:
                continue
            roi, (xs, ys, xe, ye) = crop_roi(img)
            if DEBUG_SAVE_ROI and (not DEBUG_ONLY_ONCE or not debug_done):
                debug_dump_roi(img, roi, (xs, ys, xe, ye))
                debug_done = True
            for name, tmpl in templates:
                score = compute_match_score(roi, tmpl)
                scores_debug.append((f"{os.path.basename(path)}:{name}", score))
                if score > best_score:
                    best_score = score
                    best_name = f"{os.path.basename(path)}:{name}"

        print("[i] Scores:", ", ".join(f"{n}={s:.3f}" for n, s in scores_debug))
        print(f"[i] BEST MATCH: {best_name} score={best_score:.3f} soglia={MATCH_THRESHOLD}")
        now = time.time()
        elapsed = now - last_alert_time

        if best_score >= MATCH_THRESHOLD:
            consecutive_hits += 1
            print(f"[i] Sopra soglia, hit {consecutive_hits}/{CONSECUTIVE_HITS_REQUIRED}")
        else:
            if consecutive_hits > 0:
                print("[i] Sotto soglia, azzero hit consecutivi")
            consecutive_hits = 0

        if (best_score >= MATCH_THRESHOLD and
            consecutive_hits >= CONSECUTIVE_HITS_REQUIRED and
            elapsed >= MIN_SECONDS_BETWEEN_ALERTS):

            print(f"[+] RILEVATO! ({best_name}) score={best_score:.3f}")
            send_notification(f"🎁 Tesoro rilevato! (template: {best_name}, score={best_score:.3f})")
            last_alert_time = now
            consecutive_hits = 0


        print(f"[*] Attendo {CHECK_INTERVAL_SEC} secondi...")
        time.sleep(CHECK_INTERVAL_SEC)

if __name__ == "__main__":
    main()

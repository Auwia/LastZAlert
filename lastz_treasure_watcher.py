#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
from datetime import datetime
import subprocess
import threading
from typing import List, Tuple, Optional
from simple_events import SIMPLE_EVENTS
from treasure_flow import treasure_flow_watcher, TreasureFlow
from workflow_manager import WORKFLOW_MANAGER, Workflow
from donation_flow import DonationFlow
from ministry_flow import MinistryFlow
from forziere_flow import ForziereFlow
from heal_flow import HealFlow

import cv2
import numpy as np
import requests

# ============================================================
# CONFIG
# ============================================================

ADB_CMD = "adb"
PACKAGE_NAME = "com.readygo.barrel.gp"

USE_SEQUENTIAL = True

donation_flow_holder = {"flow": None}
ministry_flow_holder = {"flow": None}
forziere_flow_holder = {"flow": None}

# Discord webhook (ATTENZIONE: se pubblico, meglio metterlo in env var)
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1446565181265154190/pL-0gcgP09RlQqnqHqQDIdQqm505tqa744is2R_1eGA3Had4OXmhPgQrTLYXYzaMld0S"

# ============================================================
# VIDEO RECORDING (OPZIONALE)
# ============================================================

ENABLE_TREASURE_RECORDING = True      # ⬅ ON / OFF
TREASURE_RECORD_SECONDS  = 300         # durata max 5 minuti
TREASURE_RECORD_DIR      = "debug/recordings"

# Template match threshold
MATCH_THRESHOLD_TREASURE = 0.75
MATCH_THRESHOLD_HEAL     = 0.85
MATCH_THRESHOLD_HELP     = 0.80
MATCH_THRESHOLD_HOSPITAL = 0.85
last_hospital_action = 0
HOSPITAL_COOLDOWN = 5  # secondi

# Anti-spam alert
MIN_SECONDS_BETWEEN_TREASURE_ALERTS = 2  # 10 min
CONSECUTIVE_HITS_REQUIRED_TREASURE = 1

# Paths
TEMPLATES_TREASURES_DIR = "treasures"
TEMPLATES_HEAL_DIR      = "heal"
TEMPLATES_HELP_DIR      = "help"
HEAL_FINISHED_DIR       = "heal_finished"
DEBUG_DIR               = "debug"
os.makedirs(DEBUG_DIR, exist_ok=True)
os.makedirs(TREASURE_RECORD_DIR, exist_ok=True)

# Heal config
HEAL_BATCH_DEFAULT = 250
HEAL_BATCH_ALREADY_SET = False

# HQ upgrade
TEMPLATES_HQ_UPGRADE_DIR = "hq_upgrade"
MATCH_THRESHOLD_HQ = 0.55
HQ_COOLDOWN = 5
last_hq_action = 0
HQ_BUBBLE_ROI  = (0.50, 0.82, 0.84, 0.97)   # chat area
HQ_GIFT_ROI    = (0.15, 0.85, 0.20, 0.70)   # banner centrale
HQ_OPEN_ROI    = (0.30, 0.70, 0.45, 0.80)   # bottone Open
HQ_CONFIRM_ROI = (0.25, 0.75, 0.60, 0.90)   # bottone Confirm

# Screenshot condiviso (riuso screen_treasure.png come "shared frame")
CHECK_INTERVAL_SEC = 1
SCREENSHOT_PATH = os.path.join(DEBUG_DIR, "screen_treasure.png")
SCREENSHOT_ERROR_COUNT = 0
SCREENSHOT_ERROR_MAX = 3
SCREENSHOT_LOCK = threading.Lock()

REC_PROCESS = None
REC_REMOTE_PATH = None
REC_LOCAL_PATH = None
REC_ACTIVE = False
REC_LOCK = threading.Lock()

TREASURE_FLOW_EVENT = threading.Event()

# ============================================================
# ROI (FRAZIONI dello schermo: x1,x2,y1,y2)
# Modifica qui se serve, come hai già fatto.
# ============================================================

# Tesoro: area basso-destra dove appare l’icona tesoro
#TREASURE_ROI = (0.0, 1.0, 0.0, 1.0)
TREASURE_ROI = (0.50, 0.82, 0.84, 0.97)


# Heal icon: nuvoletta croce rossa di solito sopra ospedale (zona centrale)
HEAL_ICON_ROI = (0.0, 1.0, 0.0, 1.0)

# Dentro ospedale: prima riga campo numero (label su cui tappare per aprire tastiera)
# Metti qui la zona della label numerica della PRIMA RIGA (Shock Cavalry)
HOSPITAL_FIRST_ROW_NUMBER_LABEL_ROI = (0.78, 0.93, 0.33, 0.42)

#HOSPITAL_BANNER_ROI = (0.25, 0.75, 0.02, 0.14)
HOSPITAL_BANNER_ROI = (0.0, 1.0, 0.0, 0.22)

ALLIANCE_ICON_ROI   = (0.85, 0.98, 0.75, 0.95)
ALLIANCE_TECH_ROI = (0.05, 0.95, 0.25, 0.60)
RECOMMENDED_ROI    = (0.25, 0.75, 0.20, 0.55)
DONATE_BUTTON_ROI  = (0.20, 0.80, 0.70, 0.90)

# ============================================================
# Debug
# ============================================================

DEBUG_SAVE_SCREENSHOTS = False   # salva screen interi
DEBUG_SAVE_ROIS        = False    # salva le ROI ritagliate
DEBUG_EVENTS_ONLY      = True  # scrive solo quando riconosce un evento


# ============================================================
# UTIL
# ============================================================
ADB_DEVICE = "192.168.0.95:5555" 

def reset_adb():
    try:
        print("[ADB] killing server")
        subprocess.run([ADB_CMD, "kill-server"], timeout=5)
        time.sleep(2)

        print("[ADB] starting server")
        subprocess.run([ADB_CMD, "start-server"], timeout=5)
        time.sleep(2)

        print(f"[ADB] reconnecting to {ADB_DEVICE}")
        subprocess.run([ADB_CMD, "connect", ADB_DEVICE], timeout=10)
        time.sleep(2)

        print("[ADB] waiting for device")
        subprocess.run([ADB_CMD, "wait-for-device"], timeout=10)

        print("[ADB] device reconnected")

    except Exception as e:
        print("[ADB] reset failed:", e)

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
    global SCREENSHOT_ERROR_COUNT
    tmp_path = SCREENSHOT_PATH + ".tmp"

    while not stop_evt.is_set():
        #print("[SEQUENTIAL] Tick loop in esecuzione...")
        try:
            with SCREENSHOT_LOCK:
                proc = subprocess.run(
                    [ADB_CMD, "exec-out", "screencap", "-p"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=10,          # ⬅ riduci
                    check=False
                )

                if proc.returncode != 0 or not proc.stdout:
                    err = proc.stderr.decode(errors="ignore")
                    print("[SCREENSHOT] screencap failed:", err)
                    SCREENSHOT_ERROR_COUNT += 1

                    if "error: closed" in err.lower():
                        SCREENSHOT_ERROR_COUNT = SCREENSHOT_ERROR_MAX
                else:
                    SCREENSHOT_ERROR_COUNT = 0
                    with open(tmp_path, "wb") as f:
                        f.write(proc.stdout)
                    os.replace(tmp_path, SCREENSHOT_PATH)

        except subprocess.TimeoutExpired:
            print("[SCREENSHOT] adb screencap TIMEOUT – retry")
            SCREENSHOT_ERROR_COUNT += 1

        except Exception as e:
            print("[SCREENSHOT] exception:", e)
            SCREENSHOT_ERROR_COUNT += 1

        if SCREENSHOT_ERROR_COUNT >= SCREENSHOT_ERROR_MAX:
            print("[SCREENSHOT] troppi errori → reset adb")
            reset_adb()
            SCREENSHOT_ERROR_COUNT = 0

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

def tap_outside_popup(img):
    h, w = img.shape[:2]
    # angolo in alto a sinistra: quasi sempre sicuro
    x = int(w * 0.05)
    y = int(h * 0.05)
    adb_tap(x, y)
    return x, y

def tap_match_in_fullscreen(roi_coords, match_loc, tmpl_hw):
    """Converte loc dentro ROI -> coordinate assolute e tappa al centro."""
    xs, ys, xe, ye = roi_coords
    mx, my = match_loc
    th, tw = tmpl_hw
    cx = xs + mx + tw // 2
    cy = ys + my + th // 2
    adb_tap(cx, cy)
    return cx, cy

def forziere_flow_tick():
    flow = forziere_flow_holder.get("flow")
    if flow is None:
        return

    with SCREENSHOT_LOCK:
        img = load_image(SCREENSHOT_PATH)

    if img is not None:
        flow.step(img)

def stop_treasure_recording():
    global REC_PROCESS, REC_REMOTE_PATH, REC_LOCAL_PATH, REC_ACTIVE

    with REC_LOCK:
        if not REC_ACTIVE:
            return

        print("[REC] STOP requested")

        try:
            # 1. termina processo locale
            if REC_PROCESS:
                REC_PROCESS.terminate()
                REC_PROCESS.wait(timeout=5)

        except Exception:
            pass

        # 2. kill eventuale screenrecord sul device
        subprocess.run([ADB_CMD, "shell", "pkill", "screenrecord"])

        time.sleep(1)

        # 3. pull file se esiste
        if REC_REMOTE_PATH:
            print("[REC] pulling file...")
            subprocess.run([ADB_CMD, "pull", REC_REMOTE_PATH, REC_LOCAL_PATH])
            subprocess.run([ADB_CMD, "shell", "rm", REC_REMOTE_PATH])
            print(f"[REC] SAVED → {REC_LOCAL_PATH}")

        REC_PROCESS = None
        REC_REMOTE_PATH = None
        REC_LOCAL_PATH = None
        REC_ACTIVE = False

def start_treasure_recording(tag: str = "treasure"):
    global REC_PROCESS, REC_REMOTE_PATH, REC_LOCAL_PATH, REC_ACTIVE

    if not ENABLE_TREASURE_RECORDING:
        return

    with REC_LOCK:
        if REC_ACTIVE:
            print("[REC] già attivo, skip")
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        REC_REMOTE_PATH = f"/sdcard/{tag}_{ts}.mp4"
        REC_LOCAL_PATH  = os.path.join(TREASURE_RECORD_DIR, f"{tag}_{ts}.mp4")

        print(f"[REC] START → {REC_REMOTE_PATH}")

        REC_PROCESS = subprocess.Popen(
            [
                ADB_CMD, "shell", "screenrecord",
                f"--time-limit={TREASURE_RECORD_SECONDS}",
                "--bit-rate=4000000",
                REC_REMOTE_PATH
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        REC_ACTIVE = True

def start_treasure_recording_bkp(tag: str = "treasure"):

    if not ENABLE_TREASURE_RECORDING:
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    remote_path = f"/sdcard/{tag}_{ts}.mp4"
    local_path  = os.path.join(TREASURE_RECORD_DIR, f"{tag}_{ts}.mp4")

    def _record():
        try:
            print(f"[REC] start screenrecord → {remote_path}")
            subprocess.run(
                [
                    ADB_CMD, "shell", "screenrecord",
                    f"--time-limit={TREASURE_RECORD_SECONDS}",
                    "--bit-rate=4000000",
                    remote_path
                ],
                timeout=TREASURE_RECORD_SECONDS + 5
            )

            print("[REC] pull video")
            subprocess.run([ADB_CMD, "pull", remote_path, local_path], timeout=30)
            subprocess.run([ADB_CMD, "shell", "rm", remote_path], timeout=10)

            print(f"[REC] saved → {local_path}")

        except Exception as e:
            print("[REC] error:", e)

    threading.Thread(target=_record, daemon=True).start()

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

            start_treasure_recording("treasure")

            WORKFLOW_MANAGER.force(Workflow.TREASURE)

            if WORKFLOW_MANAGER.acquire(Workflow.TREASURE):
                # 0) TAP sul tesoro appena rilevato (apre la chat)
                cx, cy = tap_match_in_fullscreen(coords, loc, hw)
                log_event(f"[TREASURE] tap icon @ {cx},{cy} -> open chat")

            TREASURE_FLOW_EVENT.set()
            WORKFLOW_MANAGER.preempt_lower_priority(Workflow.TREASURE)
            treasure_flow_watcher.flow.trigger()

            last_alert = now
            hits = 0

        time.sleep(CHECK_INTERVAL_SEC)

def hospital_watcher(stop_evt):
    global last_hospital_action
    global HEAL_BATCH_ALREADY_SET

    templates = load_templates_from_dir("hospital")

    while not stop_evt.is_set():
        if not WORKFLOW_MANAGER.can_run(Workflow.HEAL):
            time.sleep(0.5)
            continue

        img = load_image(SCREENSHOT_PATH)
        if img is None:
            time.sleep(0.5)
            continue

        roi, coords = crop_roi(img, HOSPITAL_BANNER_ROI)
        name, score, loc, hw = match_any(roi, templates)

        if not DEBUG_EVENTS_ONLY:
            log_event(f"[HOSPITAL] best={name} score={score:.3f} thr={MATCH_THRESHOLD_HOSPITAL:.2f}")

        if score >= MATCH_THRESHOLD_HOSPITAL:
            now = time.time()
            if now - last_hospital_action < HOSPITAL_COOLDOWN:
                time.sleep(0.5)
                continue

            if not WORKFLOW_MANAGER.acquire(Workflow.HEAL):
                continue

            last_hospital_action = now

            # 1) tap label numerica prima riga
            if not HEAL_BATCH_ALREADY_SET:
                xs, ys, xe, ye = crop_roi(img, HOSPITAL_FIRST_ROW_NUMBER_LABEL_ROI)[1]
                adb_tap((xs + xe)//2, (ys + ye)//2)
                log_event(f"[HOSPITAL] first heal → tap number label")
            
                time.sleep(1)

                # 2) inserisci batch
                adb_input_text(str(HEAL_BATCH_DEFAULT))
                adb_keyevent(66)
                log_event(f"[HOSPITAL] input batch={HEAL_BATCH_DEFAULT}")
                time.sleep(0.5)

                HEAL_BATCH_ALREADY_SET = True
                log_event("[HOSPITAL] batch set complete, next heals will skip input")

            # 3) tap bottone Heal (zona fissa)
            heal_x = int(img.shape[1] * 0.83)
            heal_y = int(img.shape[0] * 0.86)
            log_event(f"[HOSPITAL] tap HEAL @ {heal_x},{heal_y}")
            adb_tap(heal_x, heal_y)

            time.sleep(0.5)

            WORKFLOW_MANAGER.release(Workflow.HEAL)

        time.sleep(0.5)

# ============================================================
# THREAD 4: HQ Upgrade Gift automation
# ============================================================
def hq_upgrade_watcher(stop_evt: threading.Event):
    global last_hq_action

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
                if not WORKFLOW_MANAGER.can_run(Workflow.HQ):
                    time.sleep(0.5)
                    continue
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
            if WORKFLOW_MANAGER.acquire(Workflow.HQ):
                print("[WF] HQ acquisito")

            roi, coords = crop_roi(img, HQ_GIFT_ROI)
            name, score, loc, hw = match_any(roi, templates)
    
            if score >= MATCH_THRESHOLD_HQ and "gift" in name.lower():
                tap_match_in_fullscreen(coords, loc, hw)
                print("[HQ] gift banner")
                hq_state = "GIFT_OPENED"
                time.sleep(2)
            else:
                hq_state = "IDLE"
                WORKFLOW_MANAGER.release(Workflow.HQ)
                print("[WF] HQ rilasciato")
            continue
    
        # ---------------------------
        # STATE: GIFT_OPENED → open
        # ---------------------------
        if hq_state == "GIFT_OPENED":
            if WORKFLOW_MANAGER.acquire(Workflow.HQ):
                print("[WF] HQ acquisito")

            roi, coords = crop_roi(img, HQ_OPEN_ROI)
            name, score, loc, hw = match_any(roi, templates)
    
            if score >= MATCH_THRESHOLD_HQ and "open" in name.lower():
                tap_match_in_fullscreen(coords, loc, hw)
                print("[HQ] OPEN")
                hq_state = "WAIT_CONFIRM"
                time.sleep(2)
            else:
                hq_state = "IDLE"
                WORKFLOW_MANAGER.release(Workflow.HQ)
                print("[WF] HQ rilasciato")
            continue
    
        # ---------------------------
        # STATE: WAIT_CONFIRM
        # ---------------------------
        if hq_state == "WAIT_CONFIRM":
            if WORKFLOW_MANAGER.acquire(Workflow.HQ):
                print("[WF] HQ acquisito")

            roi, coords = crop_roi(img, HQ_CONFIRM_ROI)
            name, score, loc, hw = match_any(roi, templates)
    
            if score >= MATCH_THRESHOLD_HQ and "confirm" in name.lower():
                tap_match_in_fullscreen(coords, loc, hw)
                print("[HQ] CONFIRM → DONE")
            hq_state = "IDLE"
            WORKFLOW_MANAGER.release(Workflow.HQ)
            print("[WF] HQ rilasciato")

            time.sleep(3)
            continue


        time.sleep(CHECK_INTERVAL_SEC)

# ============================================================
# THREAD 14: GENERIC 1-CLICK EVENTS AUTOMATION
# ============================================================
def simple_event_watcher(stop_evt):
    # carica template per ogni evento UNA SOLA VOLTA
    event_templates = {}
    for name, cfg in SIMPLE_EVENTS.items():
        templates = load_templates_from_dir(cfg["templates"])
        if not templates:
            log_event(f"[{name.upper()}] Nessun template, evento disabilitato")
            continue
        event_templates[name] = templates

    last_fire = {name: 0.0 for name in event_templates}

    GENERIC_GLOBAL_COOLDOWN = 10
    last_generic_fire = 0.0

    while not stop_evt.is_set():
        GENERIC_GLOBAL_COOLDOWN = 10
        last_generic_fire = 0.0

        now = time.time()
        if now - last_generic_fire < GENERIC_GLOBAL_COOLDOWN:
            time.sleep(0.5)
            continue

        if not WORKFLOW_MANAGER.acquire(Workflow.GENERIC):
            time.sleep(0.2)
            continue

        did_anything = False

        with SCREENSHOT_LOCK:
            img = load_image(SCREENSHOT_PATH)

        if img is None:
            WORKFLOW_MANAGER.release(Workflow.GENERIC)
            time.sleep(0.5)
            continue

        for name, templates in event_templates.items():
            cfg = SIMPLE_EVENTS[name]
            now = time.time()

            if now - last_fire[name] < cfg["cooldown"]:
                continue

            roi_img, roi_coords = crop_roi(img, cfg["roi"])
            name_t, score, loc, hw = match_any(roi_img, templates)

            if not DEBUG_EVENTS_ONLY:
                log_event(
                    f"[{name.upper()}] best={name_t} "
                    f"score={score:.3f} thr={cfg['threshold']:.2f}"
                )

            if score >= cfg["threshold"]:
                if cfg.get("tap") == "OUTSIDE":
                    #adb_keyevent(4)
                    tap_outside_popup(img)
                else:
                    cx, cy = tap_match_in_fullscreen(roi_coords, loc, hw)
                    log_event(f"[{name.upper()}] score: {score} threshold: {cfg['threshold']} tap @ {cx},{cy}")

                last_fire[name] = now
                did_anything = True
                time.sleep(1.0)
                last_generic_fire = now

        WORKFLOW_MANAGER.release(Workflow.GENERIC)

        # Se ha fatto qualcosa, dormi un po’ di più per evitare spam
        time.sleep(0.2 if not did_anything else 1)

# ============================================================
# TICK MODE WRAPPERS (per uso sequenziale)
# ============================================================

# 1. TREASURE WATCHER
def treasure_watcher_tick(stop_evt):
    if not hasattr(treasure_watcher_tick, "_gen"):
        treasure_watcher_tick._gen = treasure_watcher_loop(stop_evt)
    try:
        next(treasure_watcher_tick._gen)
    except StopIteration:
        del treasure_watcher_tick._gen

def treasure_watcher_loop(stop_evt):
    templates = load_templates_from_dir(TEMPLATES_TREASURES_DIR)
    if not templates:
        return

    last_alert = 0.0
    hits = 0

    while not stop_evt.is_set():
        #print("[DEBUG] treasure_watcher_loop attivo")
        with SCREENSHOT_LOCK:
            img = load_image(SCREENSHOT_PATH)

        if img is None:
            yield; continue

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
            log_event(f"[TREASURE] RILEVATO {name} score={score:.3f}")
            send_notification(f"🎁 Tesoro rilevato! ({name})")

            start_treasure_recording("treasure")

            WORKFLOW_MANAGER.force(Workflow.TREASURE)

            if WORKFLOW_MANAGER.acquire(Workflow.TREASURE):
                cx, cy = tap_match_in_fullscreen(coords, loc, hw)
                log_event(f"[TREASURE] tap icon @ {cx},{cy} -> open chat")
                time.sleep(1.5)

            TREASURE_FLOW_EVENT.set()
            WORKFLOW_MANAGER.preempt_lower_priority(Workflow.TREASURE)
            treasure_flow_watcher.flow.trigger()
            last_alert = now
            hits = 0

        yield
        time.sleep(CHECK_INTERVAL_SEC)

# 2. HOSPITAL WATCHER
def hospital_watcher_tick(stop_evt):
    if not hasattr(hospital_watcher_tick, "_gen"):
        hospital_watcher_tick._gen = hospital_watcher_loop(stop_evt)
    try:
        next(hospital_watcher_tick._gen)
    except StopIteration:
        del hospital_watcher_tick._gen

def hospital_watcher_loop(stop_evt):
    global last_hospital_action, HEAL_BATCH_ALREADY_SET
    templates = load_templates_from_dir("hospital")

    while not stop_evt.is_set():
        if not WORKFLOW_MANAGER.can_run(Workflow.HEAL):
            yield; time.sleep(0.5); continue

        img = load_image(SCREENSHOT_PATH)
        if img is None:
            yield; time.sleep(0.5); continue

        roi, coords = crop_roi(img, HOSPITAL_BANNER_ROI)
        name, score, loc, hw = match_any(roi, templates)

        if not DEBUG_EVENTS_ONLY:
            log_event(f"[HOSPITAL] best={name} score={score:.3f}")

        if score >= MATCH_THRESHOLD_HOSPITAL:
            now = time.time()
            if now - last_hospital_action < HOSPITAL_COOLDOWN:
                yield; time.sleep(0.5); continue

            if not WORKFLOW_MANAGER.acquire(Workflow.HEAL):
                yield; continue

            last_hospital_action = now

            if not HEAL_BATCH_ALREADY_SET:
                xs, ys, xe, ye = crop_roi(img, HOSPITAL_FIRST_ROW_NUMBER_LABEL_ROI)[1]
                adb_tap((xs + xe) // 2, (ys + ye) // 2)
                log_event("[HOSPITAL] tap number label")
                time.sleep(1)

                adb_input_text(str(HEAL_BATCH_DEFAULT))
                adb_keyevent(66)
                log_event(f"[HOSPITAL] input batch={HEAL_BATCH_DEFAULT}")
                time.sleep(0.5)

                HEAL_BATCH_ALREADY_SET = True

            heal_x = int(img.shape[1] * 0.83)
            heal_y = int(img.shape[0] * 0.86)
            adb_tap(heal_x, heal_y)
            log_event("[HOSPITAL] tap HEAL button")
            time.sleep(0.5)
            WORKFLOW_MANAGER.release(Workflow.HEAL)

        yield
        time.sleep(0.5)

# 3. DONATION WATCHER
def donation_watcher_tick(stop_evt):
    if not hasattr(donation_watcher_tick, "_gen"):
        donation_watcher_tick._gen = donation_watcher_loop(stop_evt)
    try:
        next(donation_watcher_tick._gen)
    except StopIteration:
        del donation_watcher_tick._gen

def donation_watcher_loop(stop_evt):
    from donation_flow import donation_watcher

    while not stop_evt.is_set():
        # Chiamiamo la funzione già esistente (blocca il flow finché termina)
        donation_watcher(stop_evt, SCREENSHOT_PATH, SCREENSHOT_LOCK, log_event)
        yield

_hq_upgrade_state = {"state": "IDLE"}
_hq_templates = None

def hq_upgrade_watcher_tick(stop_evt):
    global last_hq_action, _hq_upgrade_state, _hq_templates

    if _hq_templates is None:
        _hq_templates = load_templates_from_dir(TEMPLATES_HQ_UPGRADE_DIR)
        if not _hq_templates:
            log_event("[HQ] Nessun template trovato. Tick disabilitato.")
            return

    if stop_evt.is_set():
        return

    with SCREENSHOT_LOCK:
        img = load_image(SCREENSHOT_PATH)

    if img is None:
        return

    state = _hq_upgrade_state["state"]

    if state == "IDLE":
        roi, coords = crop_roi(img, HQ_BUBBLE_ROI)
        name, score, loc, hw = match_any(roi, _hq_templates)

        if score >= MATCH_THRESHOLD_HQ and "bubble" in name.lower():
            if not WORKFLOW_MANAGER.can_run(Workflow.HQ):
                return
            now = time.time()
            if now - last_hq_action < HQ_COOLDOWN:
                return
            last_hq_action = now
            tap_match_in_fullscreen(coords, loc, hw)
            log_event("[HQ] bubble → chat")
            _hq_upgrade_state["state"] = "CHAT_OPENED"
            time.sleep(2)
        return

    elif state == "CHAT_OPENED":
        if WORKFLOW_MANAGER.acquire(Workflow.HQ):
            roi, coords = crop_roi(img, HQ_GIFT_ROI)
            name, score, loc, hw = match_any(roi, _hq_templates)

            if score >= MATCH_THRESHOLD_HQ and "gift" in name.lower():
                tap_match_in_fullscreen(coords, loc, hw)
                log_event("[HQ] gift banner")
                _hq_upgrade_state["state"] = "GIFT_OPENED"
                time.sleep(2)
            else:
                _hq_upgrade_state["state"] = "IDLE"
                WORKFLOW_MANAGER.release(Workflow.HQ)
        return

    elif state == "GIFT_OPENED":
        roi, coords = crop_roi(img, HQ_OPEN_ROI)
        name, score, loc, hw = match_any(roi, _hq_templates)

        if score >= MATCH_THRESHOLD_HQ and "open" in name.lower():
            tap_match_in_fullscreen(coords, loc, hw)
            log_event("[HQ] OPEN")
            _hq_upgrade_state["state"] = "WAIT_CONFIRM"
            time.sleep(2)
        else:
            _hq_upgrade_state["state"] = "IDLE"
            WORKFLOW_MANAGER.release(Workflow.HQ)
        return

    elif state == "WAIT_CONFIRM":
        roi, coords = crop_roi(img, HQ_CONFIRM_ROI)
        name, score, loc, hw = match_any(roi, _hq_templates)

        if score >= MATCH_THRESHOLD_HQ and "confirm" in name.lower():
            tap_match_in_fullscreen(coords, loc, hw)
            log_event("[HQ] CONFIRM → DONE")
        _hq_upgrade_state["state"] = "IDLE"
        WORKFLOW_MANAGER.release(Workflow.HQ)
        time.sleep(3)
        return
_simple_event_templates = {}
_last_fire_simple_event = {}
_last_generic_fire = 0.0

def simple_event_watcher_tick(stop_evt):
    global _simple_event_templates, _last_fire_simple_event, _last_generic_fire

    if not _simple_event_templates:
        for name, cfg in SIMPLE_EVENTS.items():
            templates = load_templates_from_dir(cfg["templates"])
            if templates:
                _simple_event_templates[name] = templates
        _last_fire_simple_event = {name: 0.0 for name in _simple_event_templates}

    now = time.time()
    if now - _last_generic_fire < 10:
        return

    if not WORKFLOW_MANAGER.acquire(Workflow.GENERIC):
        return

    with SCREENSHOT_LOCK:
        img = load_image(SCREENSHOT_PATH)

    if img is None:
        WORKFLOW_MANAGER.release(Workflow.GENERIC)
        return

    did_anything = False

    for name, templates in _simple_event_templates.items():
        cfg = SIMPLE_EVENTS[name]
        if now - _last_fire_simple_event[name] < cfg["cooldown"]:
            continue

        roi_img, roi_coords = crop_roi(img, cfg["roi"])
        name_t, score, loc, hw = match_any(roi_img, templates)

        if not DEBUG_EVENTS_ONLY:
            log_event(f"[{name.upper()}] best={name_t} score={score:.3f} thr={cfg['threshold']:.2f}")

        if score >= cfg["threshold"]:
            if cfg.get("tap") == "OUTSIDE":
                cx, cy = tap_outside_popup(img) 
            elif cfg.get("tap") == "center":
                cx, cy = (img.shape[1] // 2, img.shape[0] // 2)
                adb_tap(cx, cy)
            else:
                cx, cy = tap_match_in_fullscreen(roi_coords, loc, hw)
        
            log_event(f"[{name.upper()}] score: {score} threshold: {cfg['threshold']} tap @ {cx},{cy}")
            _last_fire_simple_event[name] = now
            _last_generic_fire = now
            did_anything = True
            time.sleep(1.0)

    WORKFLOW_MANAGER.release(Workflow.GENERIC)

    if did_anything:
        time.sleep(1)

# 3b. TREASURE FLOW (chat explorer)
def treasure_flow_watcher_tick():
    with SCREENSHOT_LOCK:
        img = load_image(SCREENSHOT_PATH)

    if img is not None:
        treasure_flow_watcher.flow.step(img)

def donation_flow_tick():
    flow = donation_flow_holder.get("flow")
    if flow is None:
        return

    with SCREENSHOT_LOCK:
        img = load_image(SCREENSHOT_PATH)

    if img is not None:
        flow.step(img)

def ministry_flow_tick():
    flow = ministry_flow_holder.get("flow")
    if flow is None:
        return

    with SCREENSHOT_LOCK:
        img = load_image(SCREENSHOT_PATH)

    if img is not None:
        flow.step(img)

# ============================================================
# MAIN
# ============================================================

def log(msg):
    print(msg)

def main():
    ministry_flow = MinistryFlow()

    print("=== Last Z Bot (Treasure + Heal, threaded) ===")

    if not check_adb_device():
        return

    stop_evt = threading.Event()

    if not USE_SEQUENTIAL:

        t0 = threading.Thread(target=screenshot_producer, args=(stop_evt,), daemon=True )
        t1 = threading.Thread(target=treasure_watcher, args=(stop_evt,), daemon=True)
        t3 = threading.Thread( target=hospital_watcher, args=(stop_evt,), daemon=True ) 
        t5 = threading.Thread( target=treasure_flow_watcher, args=(stop_evt, SCREENSHOT_PATH, SCREENSHOT_LOCK, log_event), daemon=True ) 
        t7 = threading.Thread( target=hq_upgrade_watcher, args=(stop_evt,), daemon=True )
        t14 = threading.Thread( target=simple_event_watcher, args=(stop_evt,), daemon=True )
        t15 = threading.Thread( target=donation_watcher, args=(stop_evt, SCREENSHOT_PATH, SCREENSHOT_LOCK, log_event), daemon=True ) 
           
        t0.start()
        t1.start()
        t3.start()
        t5.start()
        t7.start()
        t14.start()
        t15.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[!] Stop richiesto.")
            stop_evt.set()
            stop_treasure_recording()
            time.sleep(1)

    else:
        threading.Thread(target=screenshot_producer, args=(stop_evt,), daemon=True).start()
        treasure_flow_watcher.flow = TreasureFlow(log_fn=log_event)
        donation_flow_holder["flow"] = DonationFlow(log_event)
        ministry_flow_holder["flow"] = MinistryFlow(log_fn=log_event)  
        forziere_flow_holder["flow"] = ForziereFlow(log_event)
        dflow = donation_flow_holder.get("flow")
        mflow = ministry_flow_holder.get("flow")
        fflow = forziere_flow_holder.get("flow")
        heal_flow = HealFlow(log_event)
        
        #LOGICA SEQUENZIALE
        try:
            while not stop_evt.is_set():
               # 1. Tesori
               treasure_watcher_tick(stop_evt)
                
               # 2. Heal
               if (
                   heal_flow.state.name == "IDLE"
                   and WORKFLOW_MANAGER.can_run(Workflow.HEAL)
                   and not WORKFLOW_MANAGER.is_active(Workflow.GENERIC)
               ):
                   heal_flow.trigger()
               
               img = load_image(SCREENSHOT_PATH)
               if img is not None:
                   heal_flow.step(img)
                
               # 3. Treasure flow (quando triggerato)
               treasure_flow_watcher_tick()
                
               # 4. HQ gifts
               hq_upgrade_watcher_tick(stop_evt)
               
               # 5. Eventi semplici
               if not WORKFLOW_MANAGER.is_active(Workflow.HEAL):
                   simple_event_watcher_tick(stop_evt)

               # 6. Donazioni
               if (
                   dflow is not None
                   and dflow.state.name == "IDLE"
                   and not WORKFLOW_MANAGER.is_active(Workflow.GENERIC)
                   and WORKFLOW_MANAGER.can_run(Workflow.DONATION)
               ):
                   dflow.trigger()
               donation_flow_tick()

               # 7. Ministry
               if (mflow is not None and mflow.state.name == "IDLE" and not WORKFLOW_MANAGER.is_active(Workflow.GENERIC) and WORKFLOW_MANAGER.can_run(Workflow.MINISTRY)):
                   if time.time() >= ministry_flow.cooldown_until:
                       mflow.trigger()
               ministry_flow_tick()
#               print("\n[!] Ministry flow Disabled!")

               # 8. Forziere
               if (
                   fflow is not None
                   and fflow.state.name == "IDLE"
                   and not WORKFLOW_MANAGER.is_active(Workflow.GENERIC)
                   and WORKFLOW_MANAGER.can_run(Workflow.FORZIERE)
               ):
                   fflow.trigger()
               forziere_flow_tick()
   
               # Dopo ogni ciclo, puoi dormire un attimo
               time.sleep(1)
    
        except KeyboardInterrupt:
            print("\n[!] Stop richiesto.")
            stop_evt.set()
            time.sleep(1)

if __name__ == "__main__":
    main()

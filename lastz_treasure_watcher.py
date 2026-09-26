#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import threading
import time
from datetime import datetime
from typing import List, Tuple

import cv2
import numpy as np
import requests

from config import DISCORD_WEBHOOK_URL

from bounty_flow import BountyFlow
from donation_flow import DonationFlow
from forziere_flow import ForziereFlow
from heal_flow import HealFlow
from hero_flow import HeroFlow
from ministry_flow import MinistryFlow
from rally_flow import RallyFlow, RALLY_TRIGGER_ROI
from research_flow import ResearchFlow
from tank_flow import TankFlow
from simple_events import SIMPLE_EVENTS
from treasure_flow_simplified import TreasureFlowSimplified
from workflow_manager import WORKFLOW_MANAGER, Workflow
from flow_control import is_flow_enabled

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ADB_CMD = "adb"
ADB_DEVICE = "192.168.0.95:5555"

DEBUG = False
DEBUG_EVENTS_ONLY = True
DEBUG_SAVE_ROIS = False

# ============================================================
# EXECUTION ENGINE
# ============================================================

# False = comportamento attuale
# True  = nuovo motore sincronizzato agli screenshot
ENABLE_SCREENSHOT_DRIVEN_ENGINE = True

# Piccola pausa prima di acquisire il frame successivo.
# In questa prima fase NON sostituisce gli sleep interni ai workflow.
SCREENSHOT_DRIVEN_DELAY_SEC = 0.30
# ============================================================


# ============================================================
# HERO SCHEDULE
# lunedì / venerdì / domenica alle 03:50 localtime
# ============================================================

HERO_RUN_WEEKDAYS = {0, 4, 6}
# datetime.weekday():
# 0 = lunedì
# 4 = venerdì
# 6 = domenica

HERO_RUN_HOUR = 3
HERO_RUN_MINUTE = 50

# Se alle 03:50 un altro workflow è attivo,
# HERO ha fino alle 03:59 per partire appena il bot torna libero.
HERO_START_WINDOW_MINUTES = 10

# Memorizza l'ultimo giorno completato per evitare doppie esecuzioni.
HERO_LAST_RUN_PATH = os.path.join(BASE_DIR, "hero_last_run.txt")

ENABLE_MULTI_RESOURCE_COLLECTION = True

# TANK: domenica dalle 04:15 alle 04:59.
ENABLE_TANK_FLOW = True
TANK_LAST_RUN_PATH = os.path.join(BASE_DIR, "tank_last_run.txt")


def _tank_already_ran_today(now):
    try:
        with open(TANK_LAST_RUN_PATH, encoding="utf-8") as f:
            return f.read().strip() == now.strftime("%Y-%m-%d")
    except FileNotFoundError:
        return False


def _mark_tank_completed_today():
    today = datetime.now().strftime("%Y-%m-%d")
    with open(TANK_LAST_RUN_PATH, "w", encoding="utf-8") as f:
        f.write(today + "\n")
    log_event(f"[TANK] completed today={today}")




DEBUG_DIR = os.path.join(BASE_DIR, "debug")
DEBUG_RALLY_DIR = os.path.join(DEBUG_DIR, "rally")
os.makedirs(DEBUG_DIR, exist_ok=True)
os.makedirs(DEBUG_RALLY_DIR, exist_ok=True)

SCREENSHOT_PATH = os.path.join(DEBUG_DIR, "screen_treasure.png")
SCREENSHOT_LOCK = threading.Lock()
SCREENSHOT_ERROR_COUNT = 0
SCREENSHOT_ERROR_MAX = 3
SCREENSHOT_ACTIVE_INTERVAL_SEC = 0.30
SCREENSHOT_IDLE_INTERVAL_SEC = 1.20

MAIN_LOOP_ACTIVE_SLEEP_SEC = 0.08
MAIN_LOOP_IDLE_SLEEP_SEC = 0.80

TEMPLATES_TREASURES_DIR = os.path.join(BASE_DIR, "treasures")
TEMPLATES_HEAL_DIR = os.path.join(BASE_DIR, "heal")
TEMPLATES_HQ_UPGRADE_DIR = os.path.join(BASE_DIR, "hq_upgrade")

MATCH_THRESHOLD_TREASURE = 0.75
MATCH_THRESHOLD_HEAL = 0.85
MATCH_THRESHOLD_HOSPITAL = 0.85
MATCH_THRESHOLD_HQ = 0.55

MIN_SECONDS_BETWEEN_TREASURE_ALERTS = 2
CONSECUTIVE_HITS_REQUIRED_TREASURE = 1
TREASURE_SCAN_INTERVAL_SEC = 1.5

HEAL_ICON_ROI = (0.697, 0.937, 0.581, 0.693)
HOSPITAL_BANNER_ROI = (0.0, 1.0, 0.0, 0.22)
HOSPITAL_FIRST_ROW_NUMBER_LABEL_ROI = (0.78, 0.93, 0.33, 0.42)
HEAL_BATCH_DEFAULT = 100
HEAL_BATCH_ALREADY_SET = False

HQ_BUBBLE_ROI = (0.50, 0.82, 0.84, 0.97)
HQ_GIFT_ROI = (0.15, 0.85, 0.20, 0.70)
HQ_OPEN_ROI = (0.30, 0.70, 0.45, 0.80)
HQ_CONFIRM_ROI = (0.25, 0.75, 0.60, 0.90)
HQ_COOLDOWN_SEC = 5

TREASURE_ROI = (0.50, 0.82, 0.84, 0.97)

RESOURCE_EVENTS = {"wood", "meal", "electricity", "alloy", "zelt", "experience"}
MULTI_RESOURCE_BLOCK_SECONDS = 1

MINISTRY_BLOCKER_ICON_DIR = os.path.join(
    BASE_DIR,
    "ministry",
    "blocker_icons",
)
HQ_VIEW_DIR = os.path.join(BASE_DIR, "ministry", "hq_view")

MINISTRY_BLOCKER_THRESHOLD = 0.80

# Last blocker reported in the log.
# Used only to suppress duplicate detection messages.
_last_ministry_blocker_name = None
HQ_VIEW_THRESHOLD = 0.80

HQ_VIEW_ROI = (0.72, 1.00, 0.82, 1.00)
LEFT_ICON_ROI = (0.00, 0.16, 0.33, 0.82)
TOP_ICON_ROI = (0.14, 0.55, 0.07, 0.12)

DONATION_MAIN_COOLDOWN_SEC = 180
RESEARCH_MAIN_COOLDOWN_SEC = 120

# ============================================================
# RUNTIME STATE
# ============================================================

_last_multi_resource_time = 0.0
_last_treasure_scan_ts = 0.0
_last_treasure_alert_ts = 0.0
_treasure_hits = 0
_last_hq_action_ts = 0.0
_last_donation_main_trigger = 0.0
_last_research_main_trigger = 0.0
_frame_counter = 0

_perf_tick_stats = {}

_simple_event_templates = {}
_last_fire_simple_event = {}
_last_generic_fire = 0.0

_hq_upgrade_state = {"state": "IDLE"}
_hq_templates = None

HEAL_ICON_TEMPLATES = []
MINISTRY_BLOCKER_TEMPLATES = []
HQ_VIEW_TEMPLATES = []
TREASURE_TEMPLATES = []

flows = {
    "bounty": None,
    "donation": None,
    "ministry": None,
    "forziere": None,
    "hero": None,
    "research": None,
    "rally": None,
    "treasure": None,
}

# ============================================================
# LOG / ADB / IMAGE HELPERS
# ============================================================

def log_event(msg: str) -> None:
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
            text=True,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "timeout"


def check_adb_device() -> bool:
    code, out, err = run_cmd([ADB_CMD, "devices"])
    if code != 0:
        print("[!] adb devices error:", err)
        return False

    devices = [line for line in out.strip().splitlines()[1:] if line.strip() and "device" in line]
    if not devices:
        print("[!] Nessun device ADB trovato.")
        return False

    print("[+] Device:", devices[0])
    return True


def reset_adb() -> None:
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
    except Exception as exc:
        print("[ADB] reset failed:", exc)


def adb_tap(x: int, y: int) -> None:
    subprocess.run([ADB_CMD, "shell", "input", "tap", str(x), str(y)])


def adb_keyevent(code: int) -> None:
    subprocess.run([ADB_CMD, "shell", "input", "keyevent", str(code)])


def adb_input_text(txt: str) -> None:
    subprocess.run([ADB_CMD, "shell", "input", "text", txt.replace(" ", "%s")])


def load_image(path: str):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None and DEBUG:
        print("[!] Impossibile leggere immagine:", path)
    return img


def crop_roi(img, roi_frac: Tuple[float, float, float, float]):
    h, w = img.shape[:2]
    x1, x2, y1, y2 = roi_frac
    xs = max(0, min(int(w * x1), w - 1))
    xe = max(xs + 1, min(int(w * x2), w))
    ys = max(0, min(int(h * y1), h - 1))
    ye = max(ys + 1, min(int(h * y2), h))
    return img[ys:ye, xs:xe], (xs, ys, xe, ye)


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
    best_name = None
    best_score = 0.0
    best_loc = (0, 0)
    best_hw = (0, 0)

    if roi_img is None or roi_img.size == 0:
        return best_name, best_score, best_loc, best_hw

    rh, rw = roi_img.shape[:2]
    for name, tmpl in templates:
        th, tw = tmpl.shape[:2]
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


def match_any_multiscale(
    roi_img: np.ndarray,
    templates: List[Tuple[str, np.ndarray]],
    scales=(0.80, 0.90, 1.0, 1.10, 1.20),
):
    best_name = None
    best_score = 0.0
    best_loc = (0, 0)
    best_hw = (0, 0)
    best_scale = 1.0

    if roi_img is None or roi_img.size == 0:
        return best_name, best_score, best_loc, best_hw, best_scale

    rh, rw = roi_img.shape[:2]
    for name, tmpl in templates:
        th0, tw0 = tmpl.shape[:2]
        for scale in scales:
            tw = int(tw0 * scale)
            th = int(th0 * scale)
            if tw < 8 or th < 8 or rh < th or rw < tw:
                continue

            tmpl_s = cv2.resize(
                tmpl,
                (tw, th),
                interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
            )
            res = cv2.matchTemplate(roi_img, tmpl_s, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(res)
            if score > best_score:
                best_score = float(score)
                best_name = name
                best_loc = loc
                best_hw = (th, tw)
                best_scale = scale

    return best_name, best_score, best_loc, best_hw, best_scale


def match_any_fast_scaled(roi_img, templates, scale=0.5):
    if roi_img is None or roi_img.size == 0:
        return None, 0.0, (0, 0), (0, 0)

    perf_total_t0 = time.time() if DEBUG else None

    # 1) Resize ROI
    perf_t0 = time.time() if DEBUG else None

    small_roi = cv2.resize(
        roi_img,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA
    )

    if DEBUG:
        perf_resize_roi = time.time() - perf_t0

    # 2) Resize templates
    perf_t0 = time.time() if DEBUG else None

    scaled_templates = []

    for name, tmpl in templates:
        th, tw = tmpl.shape[:2]
        tmpl_s = cv2.resize(
            tmpl,
            (max(8, int(tw * scale)), max(8, int(th * scale))),
            interpolation=cv2.INTER_AREA,
        )
        scaled_templates.append((name, tmpl_s))

    if DEBUG:
        perf_resize_templates = time.time() - perf_t0

    # 3) Template matching
    perf_t0 = time.time() if DEBUG else None

    name, score, loc, hw = match_any(small_roi, scaled_templates)

    if DEBUG:
        perf_match = time.time() - perf_t0
        perf_total = time.time() - perf_total_t0

        log_event(
            f"[FAST-PERF] scale={scale:.2f} "
            f"templates={len(templates):2d} "
            f"roi_resize={perf_resize_roi:.4f}s "
            f"tmpl_resize={perf_resize_templates:.4f}s "
            f"match={perf_match:.4f}s "
            f"total={perf_total:.4f}s"
        )

    if name is None:
        return None, score, loc, hw

    return (
        name,
        score,
        (int(loc[0] / scale), int(loc[1] / scale)),
        (int(hw[0] / scale), int(hw[1] / scale))
    )


def tap_match_in_fullscreen(roi_coords, match_loc, tmpl_hw):
    xs, ys, _, _ = roi_coords
    mx, my = match_loc
    th, tw = tmpl_hw
    cx = xs + mx + tw // 2
    cy = ys + my + th // 2
    adb_tap(cx, cy)
    return cx, cy


def tap_outside_popup(img):
    h, w = img.shape[:2]
    x = int(w * 0.04)
    y = int(h * 0.58)
    adb_tap(x, y)
    return x, y


def send_notification(text: str) -> bool:
    if not DISCORD_WEBHOOK_URL:
        log_event(f"[DISCORD] webhook non configurato: {text}")
        return False

    try:
        log_event(f"[DISCORD] sending: {text}")
        resp = requests.post(DISCORD_WEBHOOK_URL, json={"content": text}, timeout=10)
        log_event(f"[DISCORD] status={resp.status_code} ok={resp.ok}")
        return resp.ok
    except Exception as exc:
        log_event(f"[DISCORD] error: {exc}")
        return False


def timed_tick(name, fn, *args):
    if not DEBUG:
        return fn(*args)

    t0 = time.time()
    try:
        return fn(*args)
    finally:
        dur = time.time() - t0
        stats = _perf_tick_stats.setdefault(
            name,
            {"count": 0, "total": 0.0, "max": 0.0}
        )
        stats["count"] += 1
        stats["total"] += dur
        stats["max"] = max(stats["max"], dur)

        if dur >= 0.30:
            log_event(f"[SLOW-TICK] {name} dur={dur:.2f}s")


def print_perf_stats():
    if not DEBUG or not _perf_tick_stats:
        return

    log_event("[PERF] ===== TICK STATS =====")

    for name, stats in _perf_tick_stats.items():
        count = stats["count"]
        total = stats["total"]
        max_dur = stats["max"]
        avg = total / count if count else 0.0

        log_event(
            f"[PERF] {name:<15} "
            f"count={count:6d} "
            f"avg={avg:.4f}s "
            f"max={max_dur:.4f}s "
            f"total={total:.2f}s"
        )

    log_event("[PERF] ======================")


# ============================================================
# SCREENSHOT PRODUCER
# ============================================================

def take_screenshot(path: str) -> bool:
    try:
        proc = subprocess.run(
            [ADB_CMD, "exec-out", "screencap", "-p"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout:
            err = proc.stderr.decode("utf-8", errors="ignore")
            print("[SCREENSHOT] screencap failed:", err)
            return False

        with open(path, "wb") as file:
            file.write(proc.stdout)
        return True
    except Exception as exc:
        print("[SCREENSHOT] exception:", exc)
        return False

def capture_fresh_frame():
    """
    Modalità screenshot-driven.

    Esegue SINCRONAMENTE:
        ADB screencap
        -> scrittura file
        -> cv2.imread
        -> ritorno immagine

    Quando questa funzione ritorna, img appartiene sicuramente
    allo screenshot appena acquisito.
    """
    global SCREENSHOT_ERROR_COUNT, _frame_counter

    tmp_path = SCREENSHOT_PATH + ".frame.tmp"

    try:
        perf_total_t0 = time.time() if DEBUG else None
        perf_adb_t0 = time.time() if DEBUG else None

        proc = subprocess.run(
            [ADB_CMD, "exec-out", "screencap", "-p"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )

        if DEBUG:
            perf_adb = time.time() - perf_adb_t0

        if proc.returncode != 0 or not proc.stdout:
            err = proc.stderr.decode("utf-8", errors="ignore")

            log_event(f"[FRAME] screencap failed: {err}")
            SCREENSHOT_ERROR_COUNT += 1

            if SCREENSHOT_ERROR_COUNT >= SCREENSHOT_ERROR_MAX:
                log_event("[FRAME] troppi errori -> reset adb")
                reset_adb()
                SCREENSHOT_ERROR_COUNT = 0

            return None

        SCREENSHOT_ERROR_COUNT = 0

        perf_file_t0 = time.time() if DEBUG else None

        with open(tmp_path, "wb") as f:
            f.write(proc.stdout)

        with SCREENSHOT_LOCK:
            os.replace(tmp_path, SCREENSHOT_PATH)

        if DEBUG:
            perf_file = time.time() - perf_file_t0
            perf_decode_t0 = time.time()

        with SCREENSHOT_LOCK:
            img = cv2.imread(SCREENSHOT_PATH, cv2.IMREAD_COLOR)

        if DEBUG:
            perf_decode = time.time() - perf_decode_t0

        if img is None:
            log_event("[FRAME] screenshot acquisito ma cv2.imread fallita")
            return None

        _frame_counter += 1

        if DEBUG:
            perf_total = time.time() - perf_total_t0
            perf_mb = len(proc.stdout) / (1024 * 1024)

            log_event(
                f"[FRAME {_frame_counter:06d}] "
                f"captured {img.shape[1]}x{img.shape[0]}"
            )

            log_event(
                f"[FRAME-PERF] "
                f"adb={perf_adb:.3f}s "
                f"file={perf_file:.3f}s "
                f"decode={perf_decode:.3f}s "
                f"total={perf_total:.3f}s "
                f"bytes={perf_mb:.2f}MB"
            )

        return img

    except subprocess.TimeoutExpired:
        log_event("[FRAME] adb screencap TIMEOUT")
        SCREENSHOT_ERROR_COUNT += 1
        return None

    except Exception as exc:
        log_event(f"[FRAME] exception: {exc}")
        SCREENSHOT_ERROR_COUNT += 1
        return None

def wait_new_frame(delay=0.6):
    time.sleep(delay)
    with SCREENSHOT_LOCK:
        take_screenshot(SCREENSHOT_PATH)


def screenshot_producer(stop_evt: threading.Event) -> None:
    global SCREENSHOT_ERROR_COUNT
    tmp_path = SCREENSHOT_PATH + ".tmp"

    while not stop_evt.is_set():
        try:
            proc = subprocess.run(
                [ADB_CMD, "exec-out", "screencap", "-p"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,
                check=False,
            )

            if proc.returncode != 0 or not proc.stdout:
                err = proc.stderr.decode(errors="ignore")
                print("[SCREENSHOT] screencap failed:", err)
                SCREENSHOT_ERROR_COUNT += 1
                if "error: closed" in err.lower():
                    SCREENSHOT_ERROR_COUNT = SCREENSHOT_ERROR_MAX
            else:
                SCREENSHOT_ERROR_COUNT = 0
                with open(tmp_path, "wb") as file:
                    file.write(proc.stdout)
                with SCREENSHOT_LOCK:
                    os.replace(tmp_path, SCREENSHOT_PATH)

        except subprocess.TimeoutExpired:
            print("[SCREENSHOT] adb screencap TIMEOUT – retry")
            SCREENSHOT_ERROR_COUNT += 1
        except Exception as exc:
            print("[SCREENSHOT] exception:", exc)
            SCREENSHOT_ERROR_COUNT += 1

        if SCREENSHOT_ERROR_COUNT >= SCREENSHOT_ERROR_MAX:
            print("[SCREENSHOT] troppi errori -> reset adb")
            reset_adb()
            SCREENSHOT_ERROR_COUNT = 0

        time.sleep(SCREENSHOT_ACTIVE_INTERVAL_SEC if any_workflow_active() else SCREENSHOT_IDLE_INTERVAL_SEC)

# ============================================================
# FLOW TICKS
# ============================================================

def treasure_detect_tick(stop_evt: threading.Event, img=None) -> None:
    global _last_treasure_scan_ts, _last_treasure_alert_ts, _treasure_hits

    if not is_flow_enabled("treasure"):
        return

    if stop_evt.is_set() or not TREASURE_TEMPLATES:
        return

    now_scan = time.time()
    if now_scan - _last_treasure_scan_ts < TREASURE_SCAN_INTERVAL_SEC:
        return
    _last_treasure_scan_ts = now_scan

    if img is None:
        with SCREENSHOT_LOCK:
            img = load_image(SCREENSHOT_PATH)

    if img is None:
        return

    roi, coords = crop_roi(img, TREASURE_ROI)
    if DEBUG_SAVE_ROIS:
        cv2.imwrite(os.path.join(DEBUG_DIR, "roi_treasure.png"), roi)

    name, score, loc, hw = match_any(roi, TREASURE_TEMPLATES)
    if score >= MATCH_THRESHOLD_TREASURE:
        _treasure_hits += 1
    else:
        _treasure_hits = 0

    now = time.time()
    if _treasure_hits < CONSECUTIVE_HITS_REQUIRED_TREASURE:
        return
    if now - _last_treasure_alert_ts < MIN_SECONDS_BETWEEN_TREASURE_ALERTS:
        return

    log_event(f"[TREASURE] rilevato {name} score={score:.3f}")
    send_notification(f"🎁 Tesoro rilevato! ({name}) score={score:.2f}")

    flow = flows.get("treasure")
    if flow is not None:
        flow.trigger(coords, loc, hw)
        log_event("[TREASURE] detected -> simplified flow")
    else:
        log_event("[TREASURE] detected but workflow busy -> waiting")

    _last_treasure_alert_ts = now
    _treasure_hits = 0

def treasure_flow_tick(img=None) -> None:
    flow = flows.get("treasure")
    if flow is None:
        return

    if img is None:
        with SCREENSHOT_LOCK:
            img = load_image(SCREENSHOT_PATH)

    if img is not None:
        flow.step(img)

def heal_tick(heal_flow: HealFlow, img=None) -> None:
    if img is None:
        with SCREENSHOT_LOCK:
            img = load_image(SCREENSHOT_PATH)
    if img is None:
        return

    if (
        is_flow_enabled("heal")
        and heal_flow.state.name == "IDLE"
        and WORKFLOW_MANAGER.can_run(Workflow.HEAL)
        and not WORKFLOW_MANAGER.is_active(Workflow.GENERIC)
        and not WORKFLOW_MANAGER.is_active(Workflow.TREASURE)
        and not WORKFLOW_MANAGER.is_active(Workflow.MINISTRY)
        and not WORKFLOW_MANAGER.is_active(Workflow.RALLY)
    ):
        roi, coords = crop_roi(img, HEAL_ICON_ROI)
        name, score, loc, hw = match_any(roi, HEAL_ICON_TEMPLATES)
        if score >= MATCH_THRESHOLD_HEAL:
            xs, ys, _, _ = coords
            mx, my = loc
            th, tw = hw
            cx = xs + mx + tw // 2
            cy = ys + my + th // 2
            heal_flow.trigger((cx, cy))
            log_event(f"[HEAL] cerotto rilevato {name} score={score:.3f} @ {cx},{cy}")

    heal_flow.step(img)

def bounty_tick(img=None) -> None:
    flow = flows.get("bounty")
    if flow is None:
        return

    if img is None:
        with SCREENSHOT_LOCK:
            img = load_image(SCREENSHOT_PATH)
    if img is None:
        return

    flow.step(img)

def donation_tick(img=None) -> None:
    flow = flows.get("donation")

    if flow is None:
        return

    if img is None:
        with SCREENSHOT_LOCK:
            img = load_image(SCREENSHOT_PATH)
    if img is None:
        return

    flow.step(img)

def ministry_tick(img=None) -> None:
    flow = flows.get("ministry")
    if flow is None:
        return

    if img is None:
        with SCREENSHOT_LOCK:
            img = load_image(SCREENSHOT_PATH)
    if img is None:
        return

    flow.step(img)

def forziere_tick(img=None) -> None:
    flow = flows.get("forziere")
    if flow is None:
        return

    if img is None:
        with SCREENSHOT_LOCK:
            img = load_image(SCREENSHOT_PATH)
    if img is None:
        return

    flow.step(img)

def tank_tick(img=None) -> None:
    flow = flows.get("tank")
    if flow is not None:
        flow.step(img)


def maybe_trigger_tank(img=None) -> None:
    if not ENABLE_TANK_FLOW:
        return

    flow = flows.get("tank")
    if flow is None or flow.state.name != "IDLE":
        return

    now = datetime.now()

    if now.weekday() != 6:
        return

    if now.hour != 4 or now.minute < 15:
        return

    if _tank_already_ran_today(now):
        return

    if not WORKFLOW_MANAGER.is_idle():
        return

    if img is None or not hq_view_visible(img):
        return

    if flow.trigger():
        log_event(
            f"[TANK] weekly trigger -> "
            f"{now.strftime('%Y-%m-%d %H:%M:%S')}"
        )


def hero_tick(img=None) -> None:
    flow = flows.get("hero")
    if flow is None:
        return

    if img is None:
        with SCREENSHOT_LOCK:
            img = load_image(SCREENSHOT_PATH)
    if img is None:
        return

    flow.step(img)

def research_tick(img=None) -> None:
    flow = flows.get("research")
    if flow is None:
        return

    if img is None:
        with SCREENSHOT_LOCK:
            img = load_image(SCREENSHOT_PATH)
    if img is None:
        return

    flow.step(img)

def rally_tick(img=None) -> None:
    flow = flows.get("rally")
    if flow is None:
        return

    if img is None:
        with SCREENSHOT_LOCK:
            img = load_image(SCREENSHOT_PATH)
    if img is None:
        return

    flow.step(img)
    if DEBUG_SAVE_ROIS:
        roi, _ = crop_roi(img, RALLY_TRIGGER_ROI)
        cv2.imwrite(os.path.join(DEBUG_RALLY_DIR, "trigger_roi.png"), roi)

# ============================================================
# SIMPLE EVENTS / HQ / MINISTRY HELPERS
# ============================================================

def simple_event_watcher_tick(stop_evt: threading.Event, img=None) -> bool:
    global _simple_event_templates, _last_fire_simple_event, _last_generic_fire, _last_multi_resource_time

    if not is_flow_enabled("generic"):
        return False

    if not _simple_event_templates:
        for ev_name, cfg in SIMPLE_EVENTS.items():
            templates = load_templates_from_dir(cfg["templates"])
            if templates:
                _simple_event_templates[ev_name] = templates

        _last_fire_simple_event = {ev_name: 0.0 for ev_name in _simple_event_templates}
        log_event(f"[SIMPLE EVENTS] templates loaded={len(_simple_event_templates)}")

    now = time.time()
    if now - _last_generic_fire < 1:
        return

    if not WORKFLOW_MANAGER.acquire(Workflow.GENERIC):
        return

    hit = None
    try:
        if img is None:
            with SCREENSHOT_LOCK:
                img = load_image(SCREENSHOT_PATH)
        
        if img is None:
            return

        for ev_name, templates in _simple_event_templates.items():
            cfg = SIMPLE_EVENTS[ev_name]
            now = time.time()
            if now - _last_fire_simple_event[ev_name] < cfg["cooldown"]:
                continue

            roi_img, roi_coords = crop_roi(img, cfg["roi"])

            perf_t0 = time.time() if DEBUG else None

            if ev_name in ("confirm_popup", "cancel_popup"):
                name_t, score, loc, hw = match_any_fast_scaled(
                    roi_img, templates, scale=0.35
                )
            else:
                name_t, score, loc, hw = match_any(roi_img, templates)

            if DEBUG:
                perf_dur = time.time() - perf_t0
                log_event(
                    f"[SIMPLE-PERF] {ev_name:<16} "
                    f"templates={len(templates):2d} "
                    f"dur={perf_dur:.4f}s "
                    f"score={score:.3f}"
                )

            if DEBUG and ev_name == "cancel_popup" and score >= cfg["threshold"]:
                log_event(
                    f"[CANCEL-POS] template={name_t} "
                    f"score={score:.3f} "
                    f"loc={loc} "
                    f"size={hw}"
                )

            if score < cfg["threshold"]:
                continue

            if ENABLE_MULTI_RESOURCE_COLLECTION and ev_name in RESOURCE_EVENTS:
                if now - _last_multi_resource_time < MULTI_RESOURCE_BLOCK_SECONDS:
                    continue
                _last_multi_resource_time = now

            tap_mode = cfg.get("tap")
            if tap_mode == "OUTSIDE":
                cx, cy = tap_outside_popup(img)
            elif tap_mode == "center":
                cx = img.shape[1] // 2
                cy = img.shape[0] // 2
                adb_tap(cx, cy)
            elif tap_mode == "bottom_center":
                cx = img.shape[1] // 2
                cy = int(img.shape[0] * 0.95)
                adb_tap(cx, cy)
            elif tap_mode == "bottom_right":
                cx = int(img.shape[1] * 0.97)
                cy = int(img.shape[0] * 0.97)
                adb_tap(cx, cy)
            else:
                cx, cy = tap_match_in_fullscreen(roi_coords, loc, hw)

            log_event(f"[SIMPLE EVENTS] TAP event={ev_name} @ {cx},{cy}")

            if ev_name == "confirm_popup":
                h, w = img.shape[:2]
            
                x = int(w * 0.92)
                y = int(h * 0.945)
            
                time.sleep(1.0)
                adb_tap(x, y)
                time.sleep(1.0)
                adb_tap(x, y)
            
                log_event(f"[SIMPLE EVENTS] CALIBRA dopo confirm_popup @ {x},{y}")

            _last_fire_simple_event[ev_name] = now
            _last_generic_fire = now
            hit = ev_name
            time.sleep(0.2)
            break
    finally:
        WORKFLOW_MANAGER.release(Workflow.GENERIC)
        if hit is not None:
            time.sleep(0.3)

    return hit is not None


def _ensure_hq_lock() -> bool:
    if WORKFLOW_MANAGER.is_active(Workflow.HQ):
        return True
    return WORKFLOW_MANAGER.acquire(Workflow.HQ)


def hq_upgrade_watcher_tick(stop_evt: threading.Event, img=None) -> None:
    global _hq_templates, _last_hq_action_ts

    if _hq_templates is None:
        _hq_templates = load_templates_from_dir(TEMPLATES_HQ_UPGRADE_DIR)
        if not _hq_templates:
            log_event("[HQ] Nessun template trovato. Tick disabilitato.")
            return

    if stop_evt.is_set():
        return

    if img is None:
        with SCREENSHOT_LOCK:
            img = load_image(SCREENSHOT_PATH)
    
    if img is None:
        return

    state = _hq_upgrade_state["state"]

    if state == "IDLE":
        if not is_flow_enabled("hq"):
            return
        roi, coords = crop_roi(img, HQ_BUBBLE_ROI)
        name, score, loc, hw = match_any(roi, _hq_templates)
        if score >= MATCH_THRESHOLD_HQ and name and "bubble" in name.lower():
            now = time.time()

            if now - _last_hq_action_ts < HQ_COOLDOWN_SEC:
                return

            if not WORKFLOW_MANAGER.acquire(Workflow.HQ):
                return

            _last_hq_action_ts = now
            tap_match_in_fullscreen(coords, loc, hw)
            log_event("[HQ] bubble -> chat")
            _hq_upgrade_state["state"] = "CHAT_OPENED"
            time.sleep(2)
        return

    if not _ensure_hq_lock():
        return

    if state == "CHAT_OPENED":
        roi, coords = crop_roi(img, HQ_GIFT_ROI)
        name, score, loc, hw = match_any(roi, _hq_templates)
        if score >= MATCH_THRESHOLD_HQ and name and "gift" in name.lower():
            tap_match_in_fullscreen(coords, loc, hw)
            log_event("[HQ] gift banner")
            _hq_upgrade_state["state"] = "GIFT_OPENED"
            time.sleep(2)
        else:
            _hq_upgrade_state["state"] = "IDLE"
            WORKFLOW_MANAGER.release(Workflow.HQ)
        return

    if state == "GIFT_OPENED":
        roi, coords = crop_roi(img, HQ_OPEN_ROI)
        name, score, loc, hw = match_any(roi, _hq_templates)
        if score >= MATCH_THRESHOLD_HQ and name and "open" in name.lower():
            tap_match_in_fullscreen(coords, loc, hw)
            log_event("[HQ] OPEN")
            _hq_upgrade_state["state"] = "WAIT_CONFIRM"
            time.sleep(2)
        else:
            _hq_upgrade_state["state"] = "IDLE"
            WORKFLOW_MANAGER.release(Workflow.HQ)
        return

    if state == "WAIT_CONFIRM":
        roi, coords = crop_roi(img, HQ_CONFIRM_ROI)
        name, score, loc, hw = match_any(roi, _hq_templates)
        if score >= MATCH_THRESHOLD_HQ and name and "confirm" in name.lower():
            tap_match_in_fullscreen(coords, loc, hw)
            log_event("[HQ] CONFIRM -> DONE")
        _hq_upgrade_state["state"] = "IDLE"
        WORKFLOW_MANAGER.release(Workflow.HQ)
        time.sleep(3)

def hq_view_visible(img) -> bool:
    if not HQ_VIEW_TEMPLATES:
        return False

    roi, _ = crop_roi(img, HQ_VIEW_ROI)
    name, score, _, _ = match_any(roi, HQ_VIEW_TEMPLATES)
    if not DEBUG_EVENTS_ONLY:
        log_event(f"[HQ-VIEW] match={name} score={score:.3f}")
    return name is not None and score >= HQ_VIEW_THRESHOLD


def officer_icon_visible(img) -> bool:
    """
    Return True when any Ministry blocker icon is visible.

    Every template stored in ministry/blocker_icons automatically
    becomes a Ministry blocker. No Python change is required when
    adding new blocker icons.
    """
    if not MINISTRY_BLOCKER_TEMPLATES:
        return False

    roi_left, _ = crop_roi(img, LEFT_ICON_ROI)
    roi_top, _ = crop_roi(img, TOP_ICON_ROI)

    if DEBUG_SAVE_ROIS:
        cv2.imwrite(
            os.path.join(DEBUG_DIR, "officer_left.png"),
            roi_left
        )
        cv2.imwrite(
            os.path.join(DEBUG_DIR, "officer_top.png"),
            roi_top
        )

    left_name, left_score, _, _ = match_any(
        roi_left,
        MINISTRY_BLOCKER_TEMPLATES
    )

    top_name, top_score, _, _, top_scale = match_any_multiscale(
        roi_top,
        MINISTRY_BLOCKER_TEMPLATES
    )

    if top_score > left_score:
        best_name = top_name
        best_score = top_score
        best_source = "TOP"
        best_scale = top_scale
    else:
        best_name = left_name
        best_score = left_score
        best_source = "LEFT"
        best_scale = 1.0

    visible = (
        best_name is not None
        and best_score >= MINISTRY_BLOCKER_THRESHOLD
    )

    # Log blocker only when its state changes.
    # Avoid repeating the same detection on every scan.
    global _last_ministry_blocker_name

    current_blocker = best_name if visible else None

    if current_blocker != _last_ministry_blocker_name:
        if current_blocker is not None:
            log_event(
                f"[MINISTRY BLOCKER] detected={best_name} "
                f"score={best_score:.3f} "
                f"source={best_source} "
                f"scale={best_scale:.2f}"
            )
        elif _last_ministry_blocker_name is not None:
            log_event(
                f"[MINISTRY BLOCKER] cleared="
                f"{_last_ministry_blocker_name}"
            )

        _last_ministry_blocker_name = current_blocker

    elif DEBUG and not DEBUG_EVENTS_ONLY:
        log_event(
            f"[MINISTRY BLOCKER] best={best_name} "
            f"score={best_score:.3f} "
            f"source={best_source}"
        )

    return visible


# ============================================================
# SCHEDULING HELPERS
# ============================================================

def any_workflow_active() -> bool:
    return any(
        WORKFLOW_MANAGER.is_active(flow)
        for flow in (
            Workflow.GENERIC,
            Workflow.DONATION,
            Workflow.RESEARCH,
            Workflow.RALLY,
            Workflow.MINISTRY,
            Workflow.TREASURE,
            Workflow.HEAL,
            Workflow.FORZIERE,
            Workflow.HQ,
            Workflow.HERO,
            Workflow.BOUNTY,
        )
    )

def no_workflow_active() -> bool:
    return not any_workflow_active()


def can_start_common(flow: Workflow) -> bool:
    return WORKFLOW_MANAGER.can_run(flow) and no_workflow_active()


def _hero_today_key(now=None) -> str:
    now = now or datetime.now()
    return now.strftime("%Y-%m-%d")


def _hero_already_ran_today(now=None) -> bool:
    today = _hero_today_key(now)

    try:
        with open(HERO_LAST_RUN_PATH, "r", encoding="utf-8") as f:
            return f.read().strip() == today

    except FileNotFoundError:
        return False

    except Exception as exc:
        log_event(f"[HERO] errore lettura last-run: {exc}")
        return False

def _mark_hero_completed_today() -> None:
    today = _hero_today_key()

    try:
        with open(HERO_LAST_RUN_PATH, "w", encoding="utf-8") as f:
            f.write(today)

        log_event(f"[HERO] run completato e marcato: {today}")

    except Exception as exc:
        log_event(f"[HERO] errore scrittura last-run: {exc}")


def _hero_in_start_window(now=None) -> bool:
    now = now or datetime.now()

    if now.weekday() not in HERO_RUN_WEEKDAYS:
        return False

    now_minutes = now.hour * 60 + now.minute
    start_minutes = HERO_RUN_HOUR * 60 + HERO_RUN_MINUTE

    return (
        start_minutes
        <= now_minutes
        < start_minutes + HERO_START_WINDOW_MINUTES
    )

def init_research_flow():
    try:
        return ResearchFlow(log_event, notify_fn=send_notification)
    except TypeError:
        flow = ResearchFlow(log_event)
        setattr(flow, "notify", send_notification)
        return flow

# ============================================================
# MAIN LOOP
# ============================================================

def load_runtime_templates() -> None:
    global HEAL_ICON_TEMPLATES, MINISTRY_BLOCKER_TEMPLATES
    global HQ_VIEW_TEMPLATES, TREASURE_TEMPLATES

    MINISTRY_BLOCKER_TEMPLATES = load_templates_from_dir(
        MINISTRY_BLOCKER_ICON_DIR
    )
    HEAL_ICON_TEMPLATES = load_templates_from_dir(TEMPLATES_HEAL_DIR)
    HQ_VIEW_TEMPLATES = load_templates_from_dir(HQ_VIEW_DIR)
    TREASURE_TEMPLATES = load_templates_from_dir(TEMPLATES_TREASURES_DIR)


def init_flows():
    flows["bounty"] = BountyFlow(log_event)
    flows["donation"] = DonationFlow(log_event)
    flows["ministry"] = MinistryFlow(
        log_fn=log_event,
        screenshot_ctx={"path": SCREENSHOT_PATH, "lock": SCREENSHOT_LOCK, "load_image": load_image},
    )
    flows["forziere"] = ForziereFlow(log_event)
    flows["hero"] = HeroFlow(
        log_event,
        on_complete=_mark_hero_completed_today,
    )
    flows["tank"] = TankFlow(
        log_event,
        on_complete=_mark_tank_completed_today,
    )
    flows["rally"] = RallyFlow(log_event)
    flows["treasure"] = TreasureFlowSimplified(log_event)
    flows["research"] = init_research_flow()
    return HealFlow(log_event)

def maybe_trigger_bounty(img=None) -> None:
    if not is_flow_enabled("bounty"):
        return

    flow = flows.get("bounty")

    if flow is None or flow.state.name != "IDLE":
        return

    if not can_start_common(Workflow.BOUNTY):
        return

    if img is None:
        with SCREENSHOT_LOCK:
            img = load_image(SCREENSHOT_PATH)
    if img is None:
        return

    visible, name, score, coords, match_data = flow.is_bounty_visible(img)

    if not visible:
        return

    log_event(
        f"[BOUNTY] trigger visible "
        f"{name} score={score:.3f}"
    )

    flow.trigger()

def maybe_trigger_donation() -> None:
    global _last_donation_main_trigger
    if not is_flow_enabled("donation"):
        return

    flow = flows.get("donation")
    now = time.time()
    if flow is None or flow.state.name != "IDLE":
        return
    if now - _last_donation_main_trigger < DONATION_MAIN_COOLDOWN_SEC:
        return
    if now < getattr(flow, "cooldown_until", 0.0):
        return
    if not can_start_common(Workflow.DONATION):
        return

    _last_donation_main_trigger = now
    flow.trigger()


def maybe_trigger_ministry(img=None) -> None:
    if not is_flow_enabled("ministry"):
        return
    flow = flows.get("ministry")
    if flow is None or flow.state.name != "IDLE":
        return
    if time.time() < getattr(flow, "cooldown_until", 0.0):
        return
    if not can_start_common(Workflow.MINISTRY):
        return

    if img is None:
        with SCREENSHOT_LOCK:
            img = load_image(SCREENSHOT_PATH)
    if img is None:
        return

    if not hq_view_visible(img):
        if not DEBUG_EVENTS_ONLY:
            log_event("[MINISTRY] skip trigger -> not HQ view")
        return

    if officer_icon_visible(img):
        if not DEBUG_EVENTS_ONLY:
            log_event("[MINISTRY] skip trigger -> officer/application already present")
        return

    flow.trigger()


def maybe_trigger_forziere(img=None) -> None:
    if not is_flow_enabled("forziere"):
        return
    flow = flows.get("forziere")
    if flow is None or flow.state.name != "IDLE":
        return
    if not can_start_common(Workflow.FORZIERE):
        return

    if img is None:
        with SCREENSHOT_LOCK:
            img = load_image(SCREENSHOT_PATH)
    if img is None:
        return

    visible, _, score, _, _ = flow.is_forziere_visible(img)
    if visible:
        log_event(f"[FORZIERE-FLOW] trigger visibile score={score:.3f}")
        flow.trigger()

def maybe_trigger_hero() -> None:
    if not is_flow_enabled("hero"):
        return
    flow = flows.get("hero")

    if flow is None or flow.state.name != "IDLE":
        return

    now = datetime.now()

    # Solo lunedì / venerdì / domenica,
    # nella finestra 03:50 -> 03:59.
    if not _hero_in_start_window(now):
        return

    # Già completato oggi.
    if _hero_already_ran_today(now):
        return

    # Deve iniziare solo quando nessun altro workflow è attivo.
    if not can_start_common(Workflow.HERO):
        return

    if flow.trigger():
        log_event(
            f"[HERO] scheduled trigger -> "
            f"{now.strftime('%Y-%m-%d %H:%M:%S')}"
        )

def maybe_trigger_research(img=None) -> None:
    global _last_research_main_trigger
    if not is_flow_enabled("research"):
        return

    flow = flows.get("research")

    if flow is None or flow.state.name != "IDLE":
        return

    if not can_start_common(Workflow.RESEARCH):
        return

    if img is None:
        with SCREENSHOT_LOCK:
            img = load_image(SCREENSHOT_PATH)

    if img is None:
        return

    # --------------------------------------------------------
    # Research parte SOLO se:
    # A) è presente la signorina
    # oppure
    # B) è presente direttamente la provetta
    # --------------------------------------------------------

    entry = flow.entry_status(img)

    if not entry["start_ok"] and not entry["lab_ok"]:
        return

    now = time.time()

    if now - _last_research_main_trigger < RESEARCH_MAIN_COOLDOWN_SEC:
        return

    _last_research_main_trigger = now

    if entry["start_ok"]:
        log_event(
            f"[RESEARCH] trigger: START/lady detected "
            f"score={entry['start_score']:.3f}"
        )
    else:
        log_event(
            f"[RESEARCH] trigger: LAB/test-tube detected "
            f"score={entry['lab_score']:.3f}"
        )

    flow.trigger()

def maybe_trigger_rally(img=None) -> None:
    if not is_flow_enabled("rally"):
        return

    flow = flows.get("rally")
    if flow is None or flow.state.name != "IDLE":
        return
    if not can_start_common(Workflow.RALLY):
        return

    if img is None:
        with SCREENSHOT_LOCK:
            img = load_image(SCREENSHOT_PATH)
    if img is None:
        return

    roi, _ = crop_roi(img, RALLY_TRIGGER_ROI)
    _, score, _, _ = match_any(roi, flow.trigger_templates)
    if score >= 0.80:
        flow.trigger()

def screenshot_driven_run_active_workflow(
    stop_evt: threading.Event,
    heal_flow: HealFlow,
    img
) -> bool:
    """
    Se esiste un workflow attivo, esegue SOLO quello
    usando il frame corrente.

    Return:
        True  -> esisteva un workflow attivo
        False -> nessun workflow attivo
    """

    active = WORKFLOW_MANAGER.current()

    if active is None:
        return False

    if DEBUG:
        log_event(
            f"[FRAME {_frame_counter:06d}] "
            f"active workflow={active.name}"
        )

    if active == Workflow.TANK:
        timed_tick("TANK", tank_tick, img)

    elif active == Workflow.TREASURE:
        timed_tick("TREASURE-FLOW", treasure_flow_tick, img)

    elif active == Workflow.HEAL:
        timed_tick("HEAL", heal_tick, heal_flow, img)

    elif active == Workflow.HQ:
        timed_tick("HQ-UPGRADE", hq_upgrade_watcher_tick, stop_evt, img)

    elif active == Workflow.HERO:
        timed_tick("HERO", hero_tick, img)

    elif active == Workflow.BOUNTY:
        timed_tick("BOUNTY", bounty_tick, img)

    elif active == Workflow.DONATION:
        timed_tick("DONATION", donation_tick, img)

    elif active == Workflow.MINISTRY:
        timed_tick("MINISTRY", ministry_tick, img)

    elif active == Workflow.FORZIERE:
        timed_tick("FORZIERE", forziere_tick, img)

    elif active == Workflow.RESEARCH:
        timed_tick("RESEARCH", research_tick, img)

    elif active == Workflow.RALLY:
        timed_tick("RALLY", rally_tick, img)

    return True

def run_screenshot_driven_engine(
    stop_evt: threading.Event,
    heal_flow: HealFlow
) -> None:

    log_event("[ENGINE] SCREENSHOT-DRIVEN enabled")

    while not stop_evt.is_set():

        # ====================================================
        # 1. NUOVO FRAME
        # ====================================================

        img = capture_fresh_frame()

        if img is None:
            time.sleep(0.5)
            continue

        # ====================================================
        # 2. SE ESISTE UN WF ATTIVO,
        #    SOLO QUEL WF PUÒ USARE QUESTO FRAME
        # ====================================================

        if screenshot_driven_run_active_workflow(
            stop_evt,
            heal_flow,
            img
        ):
            time.sleep(SCREENSHOT_DRIVEN_DELAY_SEC)
            continue

        # ====================================================
        # 3. NESSUN WF ATTIVO:
        #    SCANSIONE SEQUENZIALE
        # ====================================================

        maybe_trigger_tank(img)

        if any_workflow_active():
            time.sleep(SCREENSHOT_DRIVEN_DELAY_SEC)
            continue

        timed_tick(
            "TREASURE-DETECT",
            treasure_detect_tick,
            stop_evt,
            img
        )

        if any_workflow_active():
            time.sleep(SCREENSHOT_DRIVEN_DELAY_SEC)
            continue


        timed_tick(
            "HEAL",
            heal_tick,
            heal_flow,
            img
        )

        if any_workflow_active():
            time.sleep(SCREENSHOT_DRIVEN_DELAY_SEC)
            continue


        timed_tick(
            "HQ-UPGRADE",
            hq_upgrade_watcher_tick,
            stop_evt,
            img
        )

        if any_workflow_active():
            time.sleep(SCREENSHOT_DRIVEN_DELAY_SEC)
            continue


        if can_start_common(Workflow.GENERIC):
            simple_event_action = timed_tick(
                "SIMPLE-EVENTS",
                simple_event_watcher_tick,
                stop_evt,
                img
            )

            if simple_event_action:
                time.sleep(SCREENSHOT_DRIVEN_DELAY_SEC)
                continue

        #
        # ATTENZIONE:
        # Generic può acquisire e rilasciare il lock
        # tutto dentro lo stesso tick.
        #
        # In questa prima versione lasciamo quindi
        # Simple Events invariato.
        # Lo renderemo "action-aware" nella fase 2.
        #


        maybe_trigger_hero()

        if any_workflow_active():
            continue

        timed_tick("HERO", hero_tick, img)

        if any_workflow_active():
            continue


        maybe_trigger_bounty(img)

        if any_workflow_active():
            continue

        timed_tick("BOUNTY", bounty_tick, img)

        if any_workflow_active():
            continue


        maybe_trigger_donation()

        if any_workflow_active():
            continue

        timed_tick("DONATION", donation_tick, img)

        if any_workflow_active():
            continue


        maybe_trigger_ministry(img)

        if any_workflow_active():
            continue

        timed_tick("MINISTRY", ministry_tick, img)

        if any_workflow_active():
            continue


        maybe_trigger_forziere(img)

        if any_workflow_active():
            continue

        timed_tick("FORZIERE", forziere_tick, img)

        if any_workflow_active():
            continue

        maybe_trigger_research(img)

        if any_workflow_active():
            continue

        timed_tick("RESEARCH", research_tick, img)

        if any_workflow_active():
            continue


        maybe_trigger_rally(img)

        if any_workflow_active():
            continue


        time.sleep(SCREENSHOT_DRIVEN_DELAY_SEC)

def main() -> None:
    print("=== Last Z Bot (sequential clean) ===")
    load_runtime_templates()

    if not check_adb_device():
        return

    # --------------------------------------------------------
    # STARTUP: elimina screenshot del run precedente
    # --------------------------------------------------------
    if os.path.exists(SCREENSHOT_PATH):
        try:
            os.remove(SCREENSHOT_PATH)
            log_event("[STARTUP] vecchio screenshot eliminato")
        except Exception as exc:
            log_event(f"[STARTUP] errore eliminazione screenshot: {exc}")

    stop_evt = threading.Event()

    if not ENABLE_SCREENSHOT_DRIVEN_ENGINE:
    
        # ========================================================
        # LEGACY ENGINE
        # ========================================================
    
        threading.Thread(
            target=screenshot_producer,
            args=(stop_evt,),
            daemon=True
        ).start()
    
        log_event("[STARTUP] LEGACY engine -> attendo primo screenshot nuovo...")
    
        while not stop_evt.is_set():
            if os.path.exists(SCREENSHOT_PATH):
                with SCREENSHOT_LOCK:
                    img = cv2.imread(
                        SCREENSHOT_PATH,
                        cv2.IMREAD_COLOR
                    )
    
                if img is not None:
                    log_event("[STARTUP] primo screenshot pronto -> avvio bot")
                    break
    
            time.sleep(0.1)
    
    else:
    
        log_event(
            "[STARTUP] SCREENSHOT-DRIVEN engine -> "
            "nessun producer asincrono"
        )

    # SOLO ORA inizializziamo i flow
    heal_flow = init_flows()

    try:
        if ENABLE_SCREENSHOT_DRIVEN_ENGINE:
            run_screenshot_driven_engine(
                stop_evt,
                heal_flow
            )
        else:
            while not stop_evt.is_set():
                timed_tick(
                    "TREASURE-DETECT",
                    treasure_detect_tick,
                    stop_evt
                )
                timed_tick(
                    "TREASURE-FLOW",
                    treasure_flow_tick
                )
    
                timed_tick(
                    "HEAL",
                    heal_tick,
                    heal_flow
                )
    
                timed_tick(
                    "HQ-UPGRADE",
                    hq_upgrade_watcher_tick,
                    stop_evt
                )
    
                if can_start_common(Workflow.GENERIC):
                    timed_tick(
                        "SIMPLE-EVENTS",
                        simple_event_watcher_tick,
                        stop_evt
                    )
    
                maybe_trigger_hero()
                timed_tick("HERO", hero_tick)
    
                maybe_trigger_bounty()
                timed_tick("BOUNTY", bounty_tick)
    
                maybe_trigger_donation()
                timed_tick("DONATION", donation_tick)
    
                maybe_trigger_ministry()
                timed_tick("MINISTRY", ministry_tick)
    
                maybe_trigger_forziere()
                timed_tick("FORZIERE", forziere_tick)
    
                maybe_trigger_research(img)
                timed_tick("RESEARCH", research_tick)
    
                maybe_trigger_rally()
    
                if WORKFLOW_MANAGER.is_active(Workflow.RALLY):
                    timed_tick("RALLY", rally_tick)
                    time.sleep(0.05)
                    continue
    
                time.sleep(
                    MAIN_LOOP_ACTIVE_SLEEP_SEC
                    if any_workflow_active()
                    else MAIN_LOOP_IDLE_SLEEP_SEC
                )

    except KeyboardInterrupt:
        print("\n[!] Stop richiesto.")
        stop_evt.set()

    
        if not ENABLE_SCREENSHOT_DRIVEN_ENGINE:
            time.sleep(1)

if __name__ == "__main__":
    main()

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

from donation_flow import DonationFlow
from forziere_flow import ForziereFlow
from heal_flow import HealFlow
from ministry_flow import MinistryFlow
from rally_flow import RallyFlow, RALLY_TRIGGER_ROI
from research_flow import ResearchFlow
from simple_events import SIMPLE_EVENTS
from treasure_flow_simplified import TreasureFlowSimplified
from workflow_manager import WORKFLOW_MANAGER, Workflow

# ============================================================
# CONFIG
# ============================================================

ADB_CMD = "adb"
ADB_DEVICE = "192.168.0.95:5555"

DEBUG = False
DEBUG_EVENTS_ONLY = True
DEBUG_SAVE_ROIS = False

ENABLE_MINISTRY_FLOW = True
ENABLE_RALLY_FLOW = False
ENABLE_MULTI_RESOURCE_COLLECTION = True

DISCORD_WEBHOOK_URL = os.environ.get(
    "DISCORD_WEBHOOK_URL",
    "https://discord.com/api/webhooks/1446565181265154190/pL-0gcgP09RlQqnqHqQDIdQqm505tqa744is2R_1eGA3Had4OXmhPgQrTLYXYzaMld0S",
)

DEBUG_DIR = "debug"
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

TEMPLATES_TREASURES_DIR = "treasures"
TEMPLATES_HEAL_DIR = "heal"
TEMPLATES_HQ_UPGRADE_DIR = "hq_upgrade"

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

SCIENCE_ICON_DIR = "ministry/science_icon"
CONSTRUCTION_ICON_DIR = "ministry/construction_icon"
CAPITALCLASH_ICON_DIR = "ministry/capital_clash"
HQ_VIEW_DIR = "ministry/hq_view"

SCIENCE_ICON_THRESHOLD = 0.80
CONSTRUCTION_ICON_THRESHOLD = 0.80
CAPITALCLASH_ICON_THRESHOLD = 0.80
HQ_VIEW_THRESHOLD = 0.80

HQ_VIEW_ROI = (0.72, 1.00, 0.82, 1.00)
LEFT_ICON_ROI = (0.00, 0.16, 0.33, 0.82)
TOP_ICON_ROI = (0.14, 0.55, 0.07, 0.12)

DONATION_MAIN_COOLDOWN_SEC = 300
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

_perf_tick_stats = {}

_simple_event_templates = {}
_last_fire_simple_event = {}
_last_generic_fire = 0.0

_hq_upgrade_state = {"state": "IDLE"}
_hq_templates = None

HEAL_ICON_TEMPLATES = []
SCIENCE_ICON_TEMPLATES = []
CONSTRUCTION_ICON_TEMPLATES = []
CAPITALCLASH_ICON_TEMPLATES = []
HQ_VIEW_TEMPLATES = []
TREASURE_TEMPLATES = []

flows = {
    "donation": None,
    "ministry": None,
    "forziere": None,
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

    small_roi = cv2.resize(roi_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    scaled_templates = []

    for name, tmpl in templates:
        th, tw = tmpl.shape[:2]
        tmpl_s = cv2.resize(
            tmpl,
            (max(8, int(tw * scale)), max(8, int(th * scale))),
            interpolation=cv2.INTER_AREA,
        )
        scaled_templates.append((name, tmpl_s))

    name, score, loc, hw = match_any(small_roi, scaled_templates)
    if name is None:
        return None, score, loc, hw

    return name, score, (int(loc[0] / scale), int(loc[1] / scale)), (int(hw[0] / scale), int(hw[1] / scale))


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
    t0 = time.time()
    try:
        return fn(*args)
    finally:
        dur = time.time() - t0
        stats = _perf_tick_stats.setdefault(name, {"count": 0, "total": 0.0, "max": 0.0})
        stats["count"] += 1
        stats["total"] += dur
        stats["max"] = max(stats["max"], dur)
        if DEBUG and dur >= 0.30:
            log_event(f"[SLOW-TICK] {name} dur={dur:.2f}s")

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

def treasure_detect_tick(stop_evt: threading.Event) -> None:
    global _last_treasure_scan_ts, _last_treasure_alert_ts, _treasure_hits

    if stop_evt.is_set() or not TREASURE_TEMPLATES:
        return

    now_scan = time.time()
    if now_scan - _last_treasure_scan_ts < TREASURE_SCAN_INTERVAL_SEC:
        return
    _last_treasure_scan_ts = now_scan

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

    WORKFLOW_MANAGER.force(Workflow.TREASURE)
    flow = flows.get("treasure")
    if flow is not None:
        flow.trigger(coords, loc, hw)
        log_event("[TREASURE] detected -> simplified flow")

    _last_treasure_alert_ts = now
    _treasure_hits = 0


def treasure_flow_tick() -> None:
    flow = flows.get("treasure")
    if flow is None:
        return

    with SCREENSHOT_LOCK:
        img = load_image(SCREENSHOT_PATH)
    if img is not None:
        flow.step(img)


def heal_tick(heal_flow: HealFlow) -> None:
    with SCREENSHOT_LOCK:
        img = load_image(SCREENSHOT_PATH)
    if img is None:
        return

    if (
        heal_flow.state.name == "IDLE"
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


def donation_tick() -> None:
    flow = flows.get("donation")
    if flow is None:
        return

    with SCREENSHOT_LOCK:
        img = load_image(SCREENSHOT_PATH)
    if img is not None:
        flow.step(img)


def ministry_tick() -> None:
    flow = flows.get("ministry")
    if flow is None:
        return

    with SCREENSHOT_LOCK:
        img = load_image(SCREENSHOT_PATH)
    if img is not None:
        flow.step(img)


def forziere_tick() -> None:
    flow = flows.get("forziere")
    if flow is None:
        return

    with SCREENSHOT_LOCK:
        img = load_image(SCREENSHOT_PATH)
    if img is not None:
        flow.step(img)


def research_tick() -> None:
    flow = flows.get("research")
    if flow is None:
        return

    with SCREENSHOT_LOCK:
        img = load_image(SCREENSHOT_PATH)
    if img is not None:
        flow.step(img)


def rally_tick() -> None:
    flow = flows.get("rally")
    if flow is None:
        return

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

def simple_event_watcher_tick(stop_evt: threading.Event) -> None:
    global _simple_event_templates, _last_fire_simple_event, _last_generic_fire, _last_multi_resource_time

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
            if ev_name in ("confirm_popup", "cancel_popup"):
                name_t, score, loc, hw = match_any_fast_scaled(roi_img, templates, scale=0.35)
            else:
                name_t, score, loc, hw = match_any(roi_img, templates)

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
            elif tap_mode == "bottom_right":
                cx = int(img.shape[1] * 0.97)
                cy = int(img.shape[0] * 0.97)
                adb_tap(cx, cy)
            else:
                cx, cy = tap_match_in_fullscreen(roi_coords, loc, hw)

            log_event(f"[SIMPLE EVENTS] TAP event={ev_name} @ {cx},{cy}")
            _last_fire_simple_event[ev_name] = now
            _last_generic_fire = now
            hit = ev_name
            time.sleep(0.2)
            break
    finally:
        WORKFLOW_MANAGER.release(Workflow.GENERIC)
        if hit is not None:
            time.sleep(0.3)


def _ensure_hq_lock() -> bool:
    if WORKFLOW_MANAGER.is_active(Workflow.HQ):
        return True
    return WORKFLOW_MANAGER.acquire(Workflow.HQ)


def hq_upgrade_watcher_tick(stop_evt: threading.Event) -> None:
    global _hq_templates, _last_hq_action_ts

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
        if score >= MATCH_THRESHOLD_HQ and name and "bubble" in name.lower():
            if not WORKFLOW_MANAGER.can_run(Workflow.HQ):
                return
            now = time.time()
            if now - _last_hq_action_ts < HQ_COOLDOWN_SEC:
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
    roi_left, _ = crop_roi(img, LEFT_ICON_ROI)
    roi_top, _ = crop_roi(img, TOP_ICON_ROI)

    if DEBUG_SAVE_ROIS:
        cv2.imwrite(os.path.join(DEBUG_DIR, "officer_left.png"), roi_left)
        cv2.imwrite(os.path.join(DEBUG_DIR, "officer_top.png"), roi_top)

    _, s_score_left, _, _ = match_any(roi_left, SCIENCE_ICON_TEMPLATES)
    _, c_score_left, _, _ = match_any(roi_left, CONSTRUCTION_ICON_TEMPLATES)
    _, cc_score_left, _, _ = match_any(roi_left, CAPITALCLASH_ICON_TEMPLATES)

    _, s_score_top, _, _, _ = match_any_multiscale(roi_top, SCIENCE_ICON_TEMPLATES)
    _, c_score_top, _, _, _ = match_any_multiscale(roi_top, CONSTRUCTION_ICON_TEMPLATES)

    if not DEBUG_EVENTS_ONLY:
        log_event(
            "[MINISTRY BLOCKER] "
            f"science L={s_score_left:.3f} T={s_score_top:.3f} | "
            f"construction L={c_score_left:.3f} T={c_score_top:.3f} | "
            f"capitalclash L={cc_score_left:.3f}"
        )

    left_visible = s_score_left >= SCIENCE_ICON_THRESHOLD or c_score_left >= CONSTRUCTION_ICON_THRESHOLD
    top_visible = s_score_top >= SCIENCE_ICON_THRESHOLD or c_score_top >= CONSTRUCTION_ICON_THRESHOLD
    return left_visible or top_visible

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
        )
    )


def no_workflow_active() -> bool:
    return not any_workflow_active()


def can_start_common(flow: Workflow) -> bool:
    return WORKFLOW_MANAGER.can_run(flow) and no_workflow_active()


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
    global HEAL_ICON_TEMPLATES, SCIENCE_ICON_TEMPLATES, CONSTRUCTION_ICON_TEMPLATES
    global CAPITALCLASH_ICON_TEMPLATES, HQ_VIEW_TEMPLATES, TREASURE_TEMPLATES

    SCIENCE_ICON_TEMPLATES = load_templates_from_dir(SCIENCE_ICON_DIR)
    CONSTRUCTION_ICON_TEMPLATES = load_templates_from_dir(CONSTRUCTION_ICON_DIR)
    HEAL_ICON_TEMPLATES = load_templates_from_dir(TEMPLATES_HEAL_DIR)
    CAPITALCLASH_ICON_TEMPLATES = load_templates_from_dir(CAPITALCLASH_ICON_DIR)
    HQ_VIEW_TEMPLATES = load_templates_from_dir(HQ_VIEW_DIR)
    TREASURE_TEMPLATES = load_templates_from_dir(TEMPLATES_TREASURES_DIR)


def init_flows():
    flows["donation"] = DonationFlow(log_event)
    flows["ministry"] = MinistryFlow(
        log_fn=log_event,
        screenshot_ctx={"path": SCREENSHOT_PATH, "lock": SCREENSHOT_LOCK, "load_image": load_image},
    )
    flows["forziere"] = ForziereFlow(log_event)
    flows["rally"] = RallyFlow(log_event)
    flows["treasure"] = TreasureFlowSimplified(log_event)
    flows["research"] = init_research_flow()
    return HealFlow(log_event)


def maybe_trigger_donation() -> None:
    global _last_donation_main_trigger

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


def maybe_trigger_ministry() -> None:
    if not ENABLE_MINISTRY_FLOW:
        return

    flow = flows.get("ministry")
    if flow is None or flow.state.name != "IDLE":
        return
    if time.time() < getattr(flow, "cooldown_until", 0.0):
        return
    if not can_start_common(Workflow.MINISTRY):
        return

    wait_new_frame(0.5)
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


def maybe_trigger_forziere() -> None:
    flow = flows.get("forziere")
    if flow is None or flow.state.name != "IDLE":
        return
    if not can_start_common(Workflow.FORZIERE):
        return

    img = load_image(SCREENSHOT_PATH)
    if img is None:
        return

    visible, _, score, _, _ = flow.is_forziere_visible(img)
    if visible:
        log_event(f"[FORZIERE-FLOW] trigger visibile score={score:.3f}")
        flow.trigger()


def maybe_trigger_research() -> None:
    global _last_research_main_trigger

    flow = flows.get("research")
    now = time.time()
    if flow is None or flow.state.name != "IDLE":
        return
    if now - _last_research_main_trigger < RESEARCH_MAIN_COOLDOWN_SEC:
        return
    if not can_start_common(Workflow.RESEARCH):
        return

    _last_research_main_trigger = now
    flow.trigger()


def maybe_trigger_rally() -> None:
    if not ENABLE_RALLY_FLOW:
        return

    flow = flows.get("rally")
    if flow is None or flow.state.name != "IDLE":
        return
    if not WORKFLOW_MANAGER.can_run(Workflow.RALLY):
        return

    img = load_image(SCREENSHOT_PATH)
    if img is None:
        return

    roi, _ = crop_roi(img, RALLY_TRIGGER_ROI)
    _, score, _, _ = match_any(roi, flow.trigger_templates)
    if score >= 0.80:
        flow.trigger()


def main() -> None:
    print("=== Last Z Bot (sequential clean) ===")
    load_runtime_templates()

    if not check_adb_device():
        return

    stop_evt = threading.Event()
    threading.Thread(target=screenshot_producer, args=(stop_evt,), daemon=True).start()

    heal_flow = init_flows()

    try:
        while not stop_evt.is_set():
            timed_tick("TREASURE-DETECT", treasure_detect_tick, stop_evt)
            timed_tick("TREASURE-FLOW", treasure_flow_tick)

            timed_tick("HEAL", heal_tick, heal_flow)
            timed_tick("HQ-UPGRADE", hq_upgrade_watcher_tick, stop_evt)

            if can_start_common(Workflow.GENERIC):
                timed_tick("SIMPLE-EVENTS", simple_event_watcher_tick, stop_evt)

            maybe_trigger_donation()
            timed_tick("DONATION", donation_tick)

            maybe_trigger_ministry()
            timed_tick("MINISTRY", ministry_tick)

            maybe_trigger_forziere()
            timed_tick("FORZIERE", forziere_tick)

            maybe_trigger_research()
            timed_tick("RESEARCH", research_tick)

            maybe_trigger_rally()
            if WORKFLOW_MANAGER.is_active(Workflow.RALLY):
                timed_tick("RALLY", rally_tick)
                time.sleep(0.05)
                continue

            time.sleep(MAIN_LOOP_ACTIVE_SLEEP_SEC if any_workflow_active() else MAIN_LOOP_IDLE_SLEEP_SEC)

    except KeyboardInterrupt:
        print("\n[!] Stop richiesto.")
        stop_evt.set()
        time.sleep(1)


if __name__ == "__main__":
    main()

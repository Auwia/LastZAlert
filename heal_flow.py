#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from enum import Enum

from workflow_manager import Workflow, WORKFLOW_MANAGER
from bot_utils import crop_roi, load_templates, match_any, adb_tap
import subprocess

# ============================================================
# CONFIG
# ============================================================

ADB_CMD = "adb"

DEBUG = False

THR_HOSPITAL = 0.85
ACTION_COOLDOWN_SEC = 1.0
STALL_TIMEOUT_SEC = 90

# ROI (usa le stesse che hai nel main)
HOSPITAL_BANNER_ROI = (0.0, 1.0, 0.0, 0.22)
FIRST_ROW_LABEL_ROI = (0.78, 0.93, 0.33, 0.42)
FIRST_ROW_LABEL_XY = (1420, 820) 

# coordinate FISSE 
HOSPITAL_ICON_XY = (960, 1080) 
HEAL_BUTTON_XY = (900, 2120)
FIRST_ROW_LABEL_ROI = (0.78, 0.93, 0.33, 0.42)
HEAL_BATCH = 50

# ============================================================
# ADB HELPERS (STILE VECCHIO – GLOBALI)
# ============================================================
def crop_roi_local(img, roi_frac):
    """
    Replica ESATTA di crop_roi del vecchio heal.py
    roi_frac = (x1, x2, y1, y2) frazioni schermo
    Ritorna: roi_img, (xs, ys, xe, ye)
    """
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

def adb_input_text(txt: str):
    safe = txt.replace(" ", "%s")
    subprocess.run(
        [ADB_CMD, "shell", "input", "text", safe],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


def adb_keyevent(code: int):
    subprocess.run(
        [ADB_CMD, "shell", "input", "keyevent", str(code)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

# ============================================================
# STATE
# ============================================================
class HealState(Enum):
    IDLE = 0
    OPEN_HOSPITAL = 1
    WAIT_HOSPITAL_UI = 2
    SET_BATCH = 3
    TAP_HEAL = 4

# ============================================================
# HEAL FLOW (DONATION STYLE)
# ============================================================

class HealFlow:
    def __init__(self, log_fn):
        self.log = log_fn

        self.state = HealState.IDLE
        self.last_action_ts = 0.0
        self.last_progress_ts = 0.0
        self.batch_set = False

        self.templates = {
            "hospital": load_templates("hospital"),
            "heal_icon": load_templates("heal"),
        }

        self.log("[HEAL-FLOW] inizializzato")

    # --------------------------------------------------------

    def _cooldown_ok(self):
        return (time.time() - self.last_action_ts) >= ACTION_COOLDOWN_SEC

    def _mark_action(self):
        self.last_action_ts = time.time()

    # --------------------------------------------------------
    def trigger(self):
        if self.state != HealState.IDLE:
            return
        if not WORKFLOW_MANAGER.can_run(Workflow.HEAL):
            return
        if not WORKFLOW_MANAGER.acquire(Workflow.HEAL):
            return

        self.state = HealState.OPEN_HOSPITAL
        self.last_action_ts = time.time()
        self.last_progress_ts = time.time()
        if DEBUG:
            self.log("[HEAL-FLOW] trigger -> OPEN_HOSPITAL")

    # --------------------------------------------------------

    def step(self, img):
        if self.state == HealState.IDLE:
            return

        if not self._cooldown_ok():
            return

        # watchdog anti-stallo (copiato da Donation)
        if (time.time() - self.last_progress_ts) > STALL_TIMEOUT_SEC:
            self.log("[HEAL-FLOW] STALL → reset + release")
            self.state = HealState.IDLE
            WORKFLOW_MANAGER.release(Workflow.HEAL)
            self._mark_action()
            return

        # ----------------------------------------------------
        # 1) open hospital
        # ----------------------------------------------------
        if self.state == HealState.OPEN_HOSPITAL:
            name, score, loc, hw = match_any(img, self.templates["heal_icon"])
        
            if name and score >= 0.80:
                cx = loc[0] + hw[1] // 2
                cy = loc[1] + hw[0] // 2
                adb_tap(cx, cy)
                if DEBUG:
                    self.log(f"[HEAL-FLOW] heal icon tap @ {cx},{cy} score={score:.3f}")
                self.state = HealState.WAIT_HOSPITAL_UI
                self.last_progress_ts = time.time()
                self._mark_action()
                return
            else:
                if DEBUG:
                    self.log("[HEAL-FLOW] heal icon NOT found → abort")
                self.state = HealState.IDLE
                WORKFLOW_MANAGER.release(Workflow.HEAL)
                self._mark_action()
                return

        # ----------------------------------------------------
        # 2) wait hospital UI
        # ----------------------------------------------------
        if self.state == HealState.WAIT_HOSPITAL_UI:
            roi, _ = crop_roi(img, HOSPITAL_BANNER_ROI)
            name, score, *_ = match_any(roi, self.templates["hospital"])
            if name and score >= THR_HOSPITAL:
                if DEBUG:
                    self.log("[HEAL-FLOW] hospital UI detected")
                self.state = HealState.SET_BATCH if not self.batch_set else HealState.TAP_HEAL
                self.last_progress_ts = time.time()
                self._mark_action()
                time.sleep(0.8)
            return

        # ----------------------------------------------------
        # 3) set batch (una sola volta)
        # ----------------------------------------------------
        if self.state == HealState.SET_BATCH:
            roi_label, coords = crop_roi_local(img, FIRST_ROW_LABEL_ROI)
            if roi_label is None or roi_label.size == 0:
                return
        
            xs, ys, xe, ye = coords
        
            adb_tap((xs + xe) // 2, (ys + ye) // 2)
            time.sleep(0.4)
        
            adb_input_text(str(HEAL_BATCH))
            time.sleep(0.3)
            adb_keyevent(66)  # ENTER
            time.sleep(0.5)
            adb_tap(900, 2120)
        
            self.log(f"[HEAL-FLOW] batch set (ROI) = {HEAL_BATCH}")
            self.state = HealState.IDLE
            self.batch_set = False
            WORKFLOW_MANAGER.release(Workflow.HEAL)
            if DEBUG:
                self.log("[HEAL-FLOW] released.")
            self._mark_action()
            return

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import re
import os
import cv2
import pytesseract
from enum import Enum
from typing import Optional

from workflow_manager import Workflow, WORKFLOW_MANAGER
from bot_utils import load_image, crop_roi, load_templates, match_any, adb_tap

# ============================================================
# DEBUG
# ============================================================

DEBUG_DONATION = False
DEBUG_DIR = "debug/donation"
os.makedirs(DEBUG_DIR, exist_ok=True)

def dbg_log(log_fn, msg):
    if DEBUG_DONATION:
        log_fn(msg)

# ============================================================
# CONFIG (INVARIATA)
# ============================================================

THR = 0.85
THR_RECOMMENDED = 0.56

ROI_ALLIANCE_ICON = (0.85, 0.98, 0.75, 0.95)
ROI_ALLIANCE_TECH = (0.0, 1.0, 0.0, 1.0)
ROI_RECOMMENDED   = (0.0, 1.0, 0.0, 1.0)
ROI_DONATE_BUTTON = (0.20, 0.80, 0.70, 0.90)

ROI_ATTEMPTS = (0.45, 0.85, 0.62, 0.75)
ROI_COOLDOWN = (0.20, 0.95, 0.70, 0.92)

DEFAULT_COOLDOWN = 60
ACTION_COOLDOWN_SEC = 1.0

# ============================================================
# OCR
# ============================================================

def _ocr_text(img):
    if img is None or img.size == 0:
        return ""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)[1]
    return pytesseract.image_to_string(gray, config="--psm 6")

def _parse_attempts(txt: str) -> int:
    m = re.search(r"(\d+)\s*/\s*(\d+)", txt)
    return int(m.group(1)) if m else 0

def _parse_cooldown_seconds(txt: str) -> int:
    m = re.search(r"(\d+):(\d+):(\d+)", txt)
    if not m:
        return DEFAULT_COOLDOWN
    h, m_, s = map(int, m.groups())
    return h * 3600 + m_ * 60 + s

# ============================================================
# TEMPLATES (INVARIATI)
# ============================================================

def _load_donation_templates():
    def pick(path1, path2):
        t = load_templates(path1)
        if t:
            return t
        return load_templates(path2)

    return {
        "alliance_icon":   pick("donation/alliance_icon.png",   "alliance_icon.png"),
        "alliance_button": pick("donation/alliance_button.png", "alliance_button.png"),
        "recommended":     pick("donation/recommended.png",     "recommended.png"),
        "donate_button":   pick("donation/donate_button.png",   "donate_button.png"),
    }

# ============================================================
# FLOW STATE
# ============================================================

class DonationState(Enum):
    IDLE = 0
    FIND_ALLIANCE_ICON = 1
    WAIT_ALLIANCE_MENU = 2
    FIND_TECH_BUTTON = 3
    FIND_RECOMMENDED = 4
    READ_ATTEMPTS = 5
    DONATE_LOOP = 6
    READ_COOLDOWN = 7

# ============================================================
# DONATION FLOW
# ============================================================

class DonationFlow:
    def __init__(self, log_fn=print):
        self.log = log_fn
        self.state = DonationState.IDLE
        self.last_action_ts = 0.0
        self.attempts_left = 0
        self.next_allowed = 0.0
        self.started_ts = 0.0
        self.initial_attempts = 0
        self.last_progress_ts = 0.0

        self.templates = _load_donation_templates()
        self.log("[DONATION-FLOW] inizializzato")

    # --------------------------------------------------------

    def _cooldown_ok(self) -> bool:
        return (time.time() - self.last_action_ts) >= ACTION_COOLDOWN_SEC

    def _mark_action(self):
        self.last_action_ts = time.time()

    # --------------------------------------------------------

    def trigger(self):
        if WORKFLOW_MANAGER.is_active(Workflow.TREASURE):
            return
        if self.state != DonationState.IDLE:
            return  # Già in corso, ignora
        if time.time() < self.next_allowed:
            return
        if not WORKFLOW_MANAGER.can_run(Workflow.GENERIC):
            return
        if not WORKFLOW_MANAGER.acquire(Workflow.DONATION):
            return

        self.started_ts = time.time()
        self.last_progress_ts = time.time()
    
        self.state = DonationState.FIND_ALLIANCE_ICON
        self._mark_action()
        self.log("[DONATION-FLOW] trigger -> FIND_ALLIANCE_ICON")

    # --------------------------------------------------------

    def step(self, img):
        if self.state == DonationState.IDLE:
            return

        if not self._cooldown_ok():
            return

        # watchdog: reset SOLO se non c'è progresso per troppo tempo
        STALL_TIMEOUT_SEC = 120
        if self.state != DonationState.IDLE and (time.time() - self.last_progress_ts) > STALL_TIMEOUT_SEC:
            self.log("[DONATION-FLOW] STALL → reset + release")
            self.state = DonationState.IDLE
            WORKFLOW_MANAGER.release(Workflow.DONATION)
            self._mark_action()
            return

        # ----------------------------------------------------
        # 1) Alliance icon
        # ----------------------------------------------------
        if self.state == DonationState.FIND_ALLIANCE_ICON:
            name, score, loc, hw = match_any(img, self.templates["alliance_icon"])
            if name and score >= THR:
                cx = loc[0] + hw[1] // 2
                cy = loc[1] + hw[0] // 2
                adb_tap(cx, cy)
                self.log(f"[DONATION-FLOW] alliance icon tap score={score:.3f}")
                self.state = DonationState.WAIT_ALLIANCE_MENU
                self._mark_action()
            return

        # ----------------------------------------------------
        # 2) Alliance menu ready
        # ----------------------------------------------------
        if self.state == DonationState.WAIT_ALLIANCE_MENU:
            name, score, _, _ = match_any(img, self.templates["alliance_button"])
            if name and score >= 0.2:
                self.state = DonationState.FIND_TECH_BUTTON
                self._mark_action()
            return

        # ----------------------------------------------------
        # 3) Alliance tech button
        # ----------------------------------------------------
        if self.state == DonationState.FIND_TECH_BUTTON:
            roi, (ox, oy) = crop_roi(img, ROI_ALLIANCE_TECH)
            name, score, loc, hw = match_any(roi, self.templates["alliance_button"])
            if name and score >= THR:
                adb_tap(
                    ox + loc[0] + hw[1] // 2,
                    oy + loc[1] + hw[0] // 2
                )
                self.log(f"[DONATION-FLOW] alliance tech tap score={score:.3f}")
                self.state = DonationState.FIND_RECOMMENDED
                self._mark_action()
            return

        # ----------------------------------------------------
        # 4) Recommended (👍 esagono, NON MAX)
        # ----------------------------------------------------
        if self.state == DonationState.FIND_RECOMMENDED:
            roi, (ox, oy) = crop_roi(img, ROI_RECOMMENDED)
            name, score, loc, hw = match_any(roi, self.templates["recommended"])
            if name and score >= THR_RECOMMENDED:
                # 👍 è a sinistra → spostiamo il click a destra
                tap_offset_x = int(hw[1] * 0.75)
                cx = ox + loc[0] + tap_offset_x
                cy = oy + loc[1] + hw[0] // 2
                adb_tap(cx, cy)
                self.last_progress_ts = time.time()
                time.sleep(0.5)
                self.log(f"[DONATION-FLOW] recommended tap score={score:.3f}")
                self.state = DonationState.READ_ATTEMPTS
                self.last_progress_ts = time.time()
                self._mark_action()
            return

        # ----------------------------------------------------
        # 5) OCR attempts
        # ----------------------------------------------------
        if self.state == DonationState.READ_ATTEMPTS:
            roi, _ = crop_roi(img, ROI_ATTEMPTS)
            txt = _ocr_text(roi)

            attempts = _parse_attempts(txt)
            
            if attempts == 0:
                self.log("[DONATION-FLOW] no attempts left → read cooldown")
                self.state = DonationState.READ_COOLDOWN
                self._mark_action()
                return
            
            if attempts < 0:
                self.log(f"[DONATION-FLOW] OCR attempts FAILED '{txt.strip()}', retry")
                self._mark_action()
                return
            
            self.attempts_left = attempts
            self.initial_attempts = attempts
            self.last_progress_ts = time.time()
            self.state = DonationState.DONATE_LOOP
            self._mark_action()

            self._mark_action()
            return

        # ----------------------------------------------------
        # 6) Donate loop
        # ----------------------------------------------------
        if self.state == DonationState.DONATE_LOOP:
            if self.attempts_left <= 0:
                self.state = DonationState.READ_COOLDOWN
                self._mark_action()
                return

            name, score, loc, hw = match_any(img, self.templates["donate_button"])
            if name and score >= THR:
                cx = loc[0] + hw[1] // 2
                cy = loc[1] + hw[0] // 2
                adb_tap(cx, cy)
                self.attempts_left -= 1
                self.last_progress_ts = time.time()
                self.log(f"[DONATION-FLOW] donate, left={self.attempts_left}")
                self._mark_action()
            return

        # ----------------------------------------------------
        # 7) OCR cooldown + END
        # ----------------------------------------------------
        if self.state == DonationState.READ_COOLDOWN:
            roi, _ = crop_roi(img, ROI_COOLDOWN)
            txt = _ocr_text(roi)
            sec = _parse_cooldown_seconds(txt)
            self.next_allowed = time.time() + sec
            self.donation_done_ts = time.time()
            self.log(f"[DONATION-FLOW] cooldown OCR='{txt.strip()}' -> {sec}s")

            # Cleanup UI: 3 tap in basso a sinistra (x=100, y varia)
            cleanup_coords = [(100, 2400), (100, 2400), (100, 2400)]
            for i, (x, y) in enumerate(cleanup_coords):
                time.sleep(0.5)
                adb_tap(x, y)
                self.log(f"[DONATION-FLOW] cleanup tap {i+1} @ ({x},{y})")

            self.state = DonationState.IDLE
            WORKFLOW_MANAGER.release(Workflow.DONATION)
            self.log("[DONATION-FLOW] completed -> IDLE")
            self._mark_action()
            return
    

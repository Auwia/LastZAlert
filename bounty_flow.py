#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import subprocess
from enum import Enum

import cv2

from workflow_manager import Workflow, WORKFLOW_MANAGER


# ============================================================
# CONFIG
# ============================================================

ADB_CMD = "adb"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BOUNTY_DIR = os.path.join(BASE_DIR, "bounty")

THR_BOUNTY = 0.985
THR_CLAIM = 0.80

# icona bounty: per ora tutto schermo
BOUNTY_ROI = (0.0, 1.0, 0.0, 1.0)

# schermata Bounty Missions:
# evitiamo solo la parte estrema superiore/inferiore
CLAIM_ROI = (0.0, 1.0, 0.20, 0.90)

WAIT_AFTER_BOUNTY_TAP = 1.5
WAIT_AFTER_CLAIM_TAP = 2.0
WAIT_AFTER_REWARD_CLOSE = 0.7

STALL_TIMEOUT = 35.0


class State(Enum):
    IDLE = 0
    OPEN_BOUNTY = 1
    WAIT_CLAIM = 2
    WAIT_REWARD = 3
    DONE = 4


class BountyFlow:

    def __init__(self, log_fn):
        self.log = log_fn

        self.state = State.IDLE
        self.state_since = time.time()
        self.started_at = 0.0

        self.bounty_templates = []
        self.claim_templates = []

        self._load_templates()

    # ========================================================
    # TEMPLATE
    # ========================================================

    def _load_templates(self):
        if not os.path.isdir(BOUNTY_DIR):
            self.log(f"[BOUNTY] directory non trovata: {BOUNTY_DIR}")
            return

        for name in sorted(os.listdir(BOUNTY_DIR)):
            if not name.lower().endswith(
                (".png", ".jpg", ".jpeg", ".webp")
            ):
                continue

            path = os.path.join(BOUNTY_DIR, name)

            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                self.log(f"[BOUNTY] template non leggibile: {path}")
                continue

            low = name.lower()

            if "claim" in low:
                self.claim_templates.append((name, img))

            elif "bounty" in low or "wanted" in low:
                self.bounty_templates.append((name, img))

        self.log(
            f"[BOUNTY] templates loaded "
            f"bounty={len(self.bounty_templates)} "
            f"claim={len(self.claim_templates)}"
        )

    # ========================================================
    # HELPERS
    # ========================================================

    @staticmethod
    def _crop_roi(img, roi_frac):
        h, w = img.shape[:2]

        x1, x2, y1, y2 = roi_frac

        xs = max(0, min(int(w * x1), w - 1))
        xe = max(xs + 1, min(int(w * x2), w))

        ys = max(0, min(int(h * y1), h - 1))
        ye = max(ys + 1, min(int(h * y2), h))

        return img[ys:ye, xs:xe], (xs, ys, xe, ye)

    @staticmethod
    def _match_any(roi_img, templates):
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

            result = cv2.matchTemplate(
                roi_img,
                tmpl,
                cv2.TM_CCOEFF_NORMED,
            )

            _, score, _, loc = cv2.minMaxLoc(result)

            if score > best_score:
                best_name = name
                best_score = float(score)
                best_loc = loc
                best_hw = (th, tw)

        return best_name, best_score, best_loc, best_hw

    @staticmethod
    def _adb_tap(x, y):
        subprocess.run(
            [
                ADB_CMD,
                "shell",
                "input",
                "tap",
                str(int(x)),
                str(int(y)),
            ],
            check=False,
        )

    def _tap_match(self, roi_coords, loc, hw):
        xs, ys, _, _ = roi_coords
        mx, my = loc
        th, tw = hw

        x = xs + mx + tw // 2
        y = ys + my + th // 2

        self._adb_tap(x, y)

        return x, y

    def _tap_outside(self, img):
        h, w = img.shape[:2]

        # stessa zona usata dal tuo simply.py
        x = w // 2
        y = int(h * 0.95)

        self._adb_tap(x, y)

        return x, y

    def _change_state(self, state):
        self.state = state
        self.state_since = time.time()

    # ========================================================
    # DETECTION USATA DAL MAIN
    # ========================================================

    def is_bounty_visible(self, img):
        if img is None or not self.bounty_templates:
            return False, None, 0.0, None, None

        roi, coords = self._crop_roi(img, BOUNTY_ROI)

        name, score, loc, hw = self._match_any(
            roi,
            self.bounty_templates,
        )

        return (
            score >= THR_BOUNTY,
            name,
            score,
            coords,
            (loc, hw),
        )

    # ========================================================
    # TRIGGER
    # ========================================================

    def trigger(self):
        if self.state != State.IDLE:
            return False

        if not WORKFLOW_MANAGER.acquire(Workflow.BOUNTY):
            return False

        self.started_at = time.time()

        self._change_state(State.OPEN_BOUNTY)

        self.log("[BOUNTY] trigger -> OPEN_BOUNTY")

        return True

    # ========================================================
    # RESET
    # ========================================================

    def _finish(self):
        self.log("[BOUNTY] completed + release")

        self.state = State.IDLE
        self.started_at = 0.0
        self.state_since = time.time()

        WORKFLOW_MANAGER.release(Workflow.BOUNTY)

    def _abort(self, reason):
        self.log(f"[BOUNTY] {reason} -> reset + release")

        self.state = State.IDLE
        self.started_at = 0.0
        self.state_since = time.time()

        WORKFLOW_MANAGER.release(Workflow.BOUNTY)

    # ========================================================
    # STEP
    # ========================================================

    def step(self, img):

        if self.state == State.IDLE:
            return

        now = time.time()

        # watchdog globale
        if (
            self.started_at > 0
            and now - self.started_at > STALL_TIMEOUT
        ):
            self._abort("STALL")
            return

        # ----------------------------------------------------
        # OPEN_BOUNTY
        # ----------------------------------------------------

        if self.state == State.OPEN_BOUNTY:

            roi, coords = self._crop_roi(
                img,
                BOUNTY_ROI,
            )

            name, score, loc, hw = self._match_any(
                roi,
                self.bounty_templates,
            )

            if score < THR_BOUNTY:
                return

            x, y = self._tap_match(
                coords,
                loc,
                hw,
            )

            self.log(
                f"[BOUNTY] tap bounty "
                f"{name} score={score:.3f} "
                f"@ {x},{y}"
            )

            self._change_state(State.WAIT_CLAIM)
            return

        # ----------------------------------------------------
        # WAIT_CLAIM
        # ----------------------------------------------------

        if self.state == State.WAIT_CLAIM:

            if now - self.state_since < WAIT_AFTER_BOUNTY_TAP:
                return

            roi, coords = self._crop_roi(
                img,
                CLAIM_ROI,
            )

            name, score, loc, hw = self._match_any(
                roi,
                self.claim_templates,
            )

            if score < THR_CLAIM:
                return

            x, y = self._tap_match(
                coords,
                loc,
                hw,
            )

            self.log(
                f"[BOUNTY] CLAIM "
                f"{name} score={score:.3f} "
                f"@ {x},{y}"
            )

            self._change_state(State.WAIT_REWARD)
            return

        # ----------------------------------------------------
        # WAIT_REWARD
        # ----------------------------------------------------

        if self.state == State.WAIT_REWARD:

            if now - self.state_since < WAIT_AFTER_CLAIM_TAP:
                return

            time.sleep(2.0)

            x, y = self._tap_outside(img)

            self.log(
                f"[BOUNTY] reward -> tap outside "
                f"@ {x},{y}"
            )

            self._change_state(State.DONE)
            return

        # ----------------------------------------------------
        # DONE
        # ----------------------------------------------------

        if self.state == State.DONE:

            if now - self.state_since < WAIT_AFTER_REWARD_CLOSE:
                return

            self._finish()

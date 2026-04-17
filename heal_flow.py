#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from enum import Enum
import subprocess

from workflow_manager import Workflow, WORKFLOW_MANAGER
from bot_utils import crop_roi, load_templates, match_any, adb_tap

ADB_CMD = "adb"

DEBUG = False

THR_HOSPITAL = 0.85
ACTION_COOLDOWN_SEC = 1.0
STALL_TIMEOUT_SEC = 90

HOSPITAL_BANNER_ROI = (0.0, 1.0, 0.0, 0.22)
FIRST_ROW_LABEL_ROI = (0.78, 0.93, 0.33, 0.42)

HEAL_BUTTON_XY = (900, 2120)
HEAL_BATCH = 40


def crop_roi_local(img, roi_frac):
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


class HealState(Enum):
    IDLE = 0
    OPEN_HOSPITAL = 1
    WAIT_HOSPITAL_UI = 2
    SET_BATCH = 3
    TAP_HEAL = 4


class HealFlow:
    def __init__(self, log_fn):
        self.log = log_fn

        self.state = HealState.IDLE
        self.last_action_ts = 0.0
        self.last_progress_ts = 0.0

        self.batch_set = False
        self.heal_icon_xy = None

        self.templates = {
            "hospital": load_templates("hospital"),
        }

        self.log("[HEAL-FLOW] inizializzato")

    def _cooldown_ok(self):
        return (time.time() - self.last_action_ts) >= ACTION_COOLDOWN_SEC

    def _mark_action(self):
        self.last_action_ts = time.time()

    def _release_and_reset(self, reason=None):
        if reason:
            self.log(reason)

        self.state = HealState.IDLE
        self.heal_icon_xy = None
        WORKFLOW_MANAGER.release(Workflow.HEAL)
        self._mark_action()

    def trigger(self, heal_icon_xy):
        if self.state != HealState.IDLE:
            return False

        if heal_icon_xy is None:
            return False

        if not WORKFLOW_MANAGER.can_run(Workflow.HEAL):
            return False

        if not WORKFLOW_MANAGER.acquire(Workflow.HEAL):
            return False

        self.heal_icon_xy = heal_icon_xy
        self.state = HealState.OPEN_HOSPITAL
        self.last_action_ts = time.time()
        self.last_progress_ts = time.time()

        if DEBUG:
            self.log(f"[HEAL-FLOW] trigger -> OPEN_HOSPITAL @ {heal_icon_xy}")

        return True

    def step(self, img):
        if self.state == HealState.IDLE:
            return

        if not self._cooldown_ok():
            return

        if (time.time() - self.last_progress_ts) > STALL_TIMEOUT_SEC:
            self._release_and_reset("[HEAL-FLOW] STALL → reset + release")
            return

        # 1) tap su icona heal passata dal MAIN
        if self.state == HealState.OPEN_HOSPITAL:
            if self.heal_icon_xy is None:
                self._release_and_reset("[HEAL-FLOW] nessuna coordinata heal -> abort")
                return

            adb_tap(*self.heal_icon_xy)

            if DEBUG:
                self.log(f"[HEAL-FLOW] tap heal icon @ {self.heal_icon_xy}")

            self.state = HealState.WAIT_HOSPITAL_UI
            self.last_progress_ts = time.time()
            self._mark_action()
            return

        # 2) aspetta UI ospedale
        if self.state == HealState.WAIT_HOSPITAL_UI:
            roi, _ = crop_roi(img, HOSPITAL_BANNER_ROI)
            name, score, *_ = match_any(roi, self.templates["hospital"])

            if name and score >= THR_HOSPITAL:
                if DEBUG:
                    self.log(f"[HEAL-FLOW] hospital UI detected score={score:.3f}")

                self.state = HealState.SET_BATCH if not self.batch_set else HealState.TAP_HEAL
                self.last_progress_ts = time.time()
                self._mark_action()

            return

        # 3) set batch solo la prima volta
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
            time.sleep(0.4)

            self.log(f"[HEAL-FLOW] batch impostato = {HEAL_BATCH}")

            self.batch_set = True
            self.state = HealState.TAP_HEAL
            self.last_progress_ts = time.time()
            self._mark_action()

            return

        # 4) tap bottone Heal
        if self.state == HealState.TAP_HEAL:
            adb_tap(*HEAL_BUTTON_XY)
            self.log("[HEAL-FLOW] tap HEAL")

            time.sleep(1.0)  # aspetta che l'hospital si chiuda da solo

            if self.heal_icon_xy:
                adb_tap(*self.heal_icon_xy)
                self.log(f"[HEAL-FLOW] tap HELP/CEROTTO @ {self.heal_icon_xy}")
                time.sleep(0.2)

            self.last_progress_ts = time.time()
            self._release_and_reset("[HEAL-FLOW] completed + release")

            return

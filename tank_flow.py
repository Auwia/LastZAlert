#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import subprocess
from enum import Enum, auto

import cv2

from bot_utils import adb_tap, adb_swipe
from workflow_manager import Workflow, WORKFLOW_MANAGER


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "tank_flow")

THR_TANK_ICON = 0.80
THR_UPGRADE = 0.85

SWIPES = (
    (650, 900, 200, 1150, 700),
    (650, 900, 250, 1050, 700),
    (650, 950, 350, 1250, 600),
)
MAX_SWIPES = len(SWIPES)
MAX_UPGRADES = 4
STALL_TIMEOUT_SEC = 120


class TankState(Enum):
    IDLE = auto()
    FIND_TANK = auto()
    WAIT_TANK_SCREEN = auto()
    UPGRADE = auto()
    EXIT = auto()


class TankFlow:

    def __init__(self, log_fn, on_complete=None):
        self.log = log_fn
        self.on_complete = on_complete

        self.icon = cv2.imread(
            os.path.join(TEMPLATE_DIR, "tank_farm_icon.png")
        )
        self.upgrade = cv2.imread(
            os.path.join(TEMPLATE_DIR, "upgrade_button.png")
        )

        if self.icon is None or self.upgrade is None:
            raise RuntimeError("[TANK] template mancanti")

        self.state = TankState.IDLE
        self.swipes = 0
        self.upgrades = 0
        self.last_progress = time.monotonic()

        self.log("[TANK] initialized")

    def _match(self, img, template):
        if img is None:
            return 0.0, None

        ih, iw = img.shape[:2]
        th, tw = template.shape[:2]

        if ih < th or iw < tw:
            return 0.0, None

        result = cv2.matchTemplate(
            img,
            template,
            cv2.TM_CCOEFF_NORMED
        )

        _, score, _, loc = cv2.minMaxLoc(result)

        return float(score), (
            loc[0] + tw // 2,
            loc[1] + th // 2
        )

    def trigger(self):
        if self.state != TankState.IDLE:
            return False

        if not WORKFLOW_MANAGER.acquire(Workflow.TANK):
            return False

        self.swipes = 0
        self.upgrades = 0
        self.last_progress = time.monotonic()
        self.state = TankState.FIND_TANK

        self.log("[TANK] scheduled trigger")
        return True

    def _finish(self, completed=False):
        self.log(
            f"[TANK] exit completed={completed} "
            f"upgrades={self.upgrades}/{MAX_UPGRADES}"
        )

        self.state = TankState.IDLE
        WORKFLOW_MANAGER.release(Workflow.TANK)

        if completed and self.on_complete:
            self.on_complete()

    def step(self, img):
        if self.state == TankState.IDLE:
            return

        if time.monotonic() - self.last_progress > STALL_TIMEOUT_SEC:
            self.log("[TANK] STALL -> release")
            self._finish()
            return

        if img is None:
            return

        if self.state == TankState.FIND_TANK:

            # Riproduce esattamente la navigazione verificata.
            # Ogni step riceve un nuovo frame dal main.
            if self.swipes < MAX_SWIPES:
                coords = SWIPES[self.swipes]
                adb_swipe(*coords)
                self.swipes += 1
                self.log(
                    f"[TANK] swipe {self.swipes}/{MAX_SWIPES} {coords}"
                )
                self.last_progress = time.monotonic()
                return

            score, xy = self._match(img, self.icon)

            if score < THR_TANK_ICON:
                self.log(
                    f"[TANK] icon not found score={score:.3f} "
                    f"after {MAX_SWIPES} swipes"
                )
                self._finish()
                return

            adb_tap(*xy)
            self.log(
                f"[TANK] icon tapped score={score:.3f} @ {xy}"
            )
            self.state = TankState.WAIT_TANK_SCREEN
            self.last_progress = time.monotonic()
            return

        if self.state == TankState.WAIT_TANK_SCREEN:

            score, _ = self._match(img, self.upgrade)

            if score >= THR_UPGRADE:
                self.log(
                    f"[TANK] upgrade screen ready score={score:.3f}"
                )
                self.state = TankState.UPGRADE
                self.last_progress = time.monotonic()
            return

        if self.state == TankState.UPGRADE:

            if self.upgrades >= MAX_UPGRADES:
                self.state = TankState.EXIT
                return

            score, xy = self._match(img, self.upgrade)

            if score < THR_UPGRADE:
                self.log(
                    f"[TANK] upgrade button not found score={score:.3f}"
                )
                self._finish(completed=False)
                return

            adb_tap(*xy)
            self.upgrades += 1

            self.log(
                f"[TANK] upgrade {self.upgrades}/{MAX_UPGRADES} "
                f"score={score:.3f}"
            )

            self.last_progress = time.monotonic()
            time.sleep(1.0)

            if self.upgrades >= MAX_UPGRADES:
                self.state = TankState.EXIT

            return

        if self.state == TankState.EXIT:
            back_ok = False
            try:
                subprocess.run(
                    ["adb", "shell", "input", "keyevent", "KEYCODE_BACK"],
                    check=True,
                    timeout=10
                )
                back_ok = True
                self.log("[TANK] BACK executed")
            except (subprocess.SubprocessError, OSError) as exc:
                self.log(f"[TANK] BACK failed: {exc}")
            finally:
                self._finish(
                    completed=back_ok and self.upgrades == MAX_UPGRADES
                )

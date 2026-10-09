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

THR_LADY = 0.75
THR_CONGRATS = 0.75
THR_TOOLS = 0.80
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

# PRIMO TEST:
# True = si ferma quando trova Upgrade, senza cliccarlo.
# Dopo il test riuscito lo metteremo a False.
SAFE_TEST_MODE = False


class TankState(Enum):
    IDLE = auto()
    FIND_LADY = auto()
    CLOSE_CONGRATS = auto()
    OPEN_TOOLS = auto()
    FIND_TANK = auto()
    WAIT_TANK_SCREEN = auto()
    UPGRADE = auto()
    EXIT = auto()


class TankFlow:

    def __init__(self, log_fn, on_complete=None):
        self.log = log_fn
        self.on_complete = on_complete

        self.lady = self._load_template("tank_lady_icon.png")
        self.congrats = self._load_template("congratulations.png")
        self.tools = self._load_template("tools_icon.png")
        self.icon = self._load_template("tank_farm_icon.png")
        self.upgrade = self._load_template("upgrade_button.png")

        self.state = TankState.IDLE
        self.swipes = 0
        self.upgrades = 0
        self.last_progress = time.monotonic()

        self.log(
            f"[TANK] initialized SAFE_TEST_MODE={SAFE_TEST_MODE}"
        )

    def _load_template(self, name):
        path = os.path.join(TEMPLATE_DIR, name)
        img = cv2.imread(path)

        if img is None:
            raise RuntimeError(f"[TANK] template mancante: {path}")

        return img

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
        self.state = TankState.FIND_LADY

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
            self.log(
                f"[TANK] STALL state={self.state.name} -> release"
            )
            self._finish(completed=False)
            return

        if img is None:
            return

        # ==================================================
        # 1. TRE SWIPE IDENTICI AL TEST RIUSCITO
        #    POI CERCA LA SIGNORINA
        # ==================================================
        if self.state == TankState.FIND_LADY:

            if self.swipes < MAX_SWIPES:
                coords = SWIPES[self.swipes]

                adb_swipe(*coords)

                self.swipes += 1
                self.last_progress = time.monotonic()

                self.log(
                    f"[TANK] swipe "
                    f"{self.swipes}/{MAX_SWIPES} {coords}"
                )
                return

            score, xy = self._match(img, self.lady)

            self.log(
                f"[TANK] lady score={score:.3f}"
            )

            if score < THR_LADY:
                self.log("[TANK] lady not found -> release")
                self._finish(completed=False)
                return

            adb_tap(*xy)

            self.log(
                f"[TANK] lady tapped "
                f"score={score:.3f} @ {xy}"
            )

            self.state = TankState.CLOSE_CONGRATS
            self.last_progress = time.monotonic()
            return

        # ==================================================
        # 2. CONGRATULATIONS
        # ==================================================
        if self.state == TankState.CLOSE_CONGRATS:

            score, xy = self._match(img, self.congrats)

            self.log(
                f"[TANK] congratulations score={score:.3f}"
            )

            if score < THR_CONGRATS:
                return

            # Il popup si chiude cliccando sul contenuto
            # riconosciuto, senza coordinate hardcoded.
            adb_tap(*xy)

            self.log(
                f"[TANK] congratulations closed "
                f"score={score:.3f} @ {xy}"
            )

            self.state = TankState.OPEN_TOOLS
            self.last_progress = time.monotonic()
            return

        # ==================================================
        # 3. MARTELLO + CACCIAVITE
        # ==================================================
        if self.state == TankState.OPEN_TOOLS:

            score, xy = self._match(img, self.tools)

            self.log(
                f"[TANK] tools score={score:.3f}"
            )

            if score < THR_TOOLS:
                return

            adb_tap(*xy)

            self.log(
                f"[TANK] tools tapped "
                f"score={score:.3f} @ {xy}"
            )

            self.state = TankState.FIND_TANK
            self.last_progress = time.monotonic()
            return

        # ==================================================
        # 4. TANK FARM
        # ==================================================
        if self.state == TankState.FIND_TANK:

            score, xy = self._match(img, self.icon)

            self.log(
                f"[TANK] tank farm score={score:.3f}"
            )

            if score < THR_TANK_ICON:
                return

            adb_tap(*xy)

            self.log(
                f"[TANK] tank farm tapped "
                f"score={score:.3f} @ {xy}"
            )

            self.state = TankState.WAIT_TANK_SCREEN
            self.last_progress = time.monotonic()
            return

        # ==================================================
        # 5. ASPETTA SCHERMATA UPGRADE
        # ==================================================
        if self.state == TankState.WAIT_TANK_SCREEN:

            score, xy = self._match(img, self.upgrade)

            self.log(
                f"[TANK] upgrade button score={score:.3f}"
            )

            if score < THR_UPGRADE:
                return

            self.log(
                f"[TANK] upgrade screen ready "
                f"score={score:.3f} @ {xy}"
            )

            # Primo test: NON eseguire upgrade.
            if SAFE_TEST_MODE:
                self.log(
                    "[TANK] SAFE TEST SUCCESS -> "
                    "Upgrade trovato, nessun click eseguito"
                )
                self._finish(completed=False)
                return

            self.state = TankState.UPGRADE
            self.last_progress = time.monotonic()
            return

        # ==================================================
        # 6. QUATTRO UPGRADE
        # ==================================================
        if self.state == TankState.UPGRADE:

            if self.upgrades >= MAX_UPGRADES:
                self.state = TankState.EXIT
                self.last_progress = time.monotonic()
                return

            score, xy = self._match(img, self.upgrade)

            if score < THR_UPGRADE:
                self.log(
                    f"[TANK] upgrade button lost "
                    f"score={score:.3f}"
                )
                self._finish(completed=False)
                return

            adb_tap(*xy)
            self.upgrades += 1

            self.log(
                f"[TANK] upgrade "
                f"{self.upgrades}/{MAX_UPGRADES} "
                f"score={score:.3f} @ {xy}"
            )

            self.last_progress = time.monotonic()

            # Il main fornirà un nuovo screenshot
            # prima del prossimo upgrade.
            time.sleep(1.0)

            if self.upgrades >= MAX_UPGRADES:
                self.state = TankState.EXIT

            return

        # ==================================================
        # 7. USCITA
        # ==================================================
        if self.state == TankState.EXIT:

            back_ok = False

            try:
                subprocess.run(
                    [
                        "adb",
                        "shell",
                        "input",
                        "keyevent",
                        "KEYCODE_BACK",
                    ],
                    check=True,
                    timeout=10,
                )

                back_ok = True
                self.log("[TANK] BACK executed")

            except (subprocess.SubprocessError, OSError) as exc:
                self.log(
                    f"[TANK] BACK failed: {exc}"
                )

            finally:
                self._finish(
                    completed=(
                        back_ok
                        and self.upgrades == MAX_UPGRADES
                    )
                )

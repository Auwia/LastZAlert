#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from datetime import datetime
from enum import Enum

from workflow_manager import Workflow, WORKFLOW_MANAGER
from bot_utils import load_templates, match_any, adb_tap

# ============================================================
# CONFIG
# ============================================================

DEBUG = False
THR = 0.75
THR_RECOMMENDED = 0.56
ACTION_COOLDOWN = 1.0
STALL_TIMEOUT = 40

# nodi da testare (ordine visivo)
NODE_COORDS = [
    (400, 900),   # top-left
    (800, 900),   # top-right
    (400, 1200),  # mid-left
    (800, 1200),  # mid-right
    (400, 1500),  # bottom-left
    (800, 1500),  # bottom-right
]

BACK = (100, 2400)

# ============================================================
# STATE
# ============================================================

class ResearchState(Enum):
    IDLE = 0
    FIND_START = 1
    TAP_LAB = 2
    TAP_RAPID = 3
    SCAN_NODE = 4
    START_RESEARCH = 5
    REPLENISH = 6
    HELP = 7
    EXIT = 8

# ============================================================
# FLOW
# ============================================================

class ResearchFlow:

    def __init__(self, log_fn=print):
        self.log = log_fn
        self.state = ResearchState.IDLE

        self.last_action_ts = 0
        self.last_progress_ts = 0

        self.lab_opened = False
        self.is_wednesday_mode = False

        self.node_index = 0

        self.templates = {
            "start": load_templates("research/start.png"),
            "lab": load_templates("research/lab_icon.png"),
            "rapid": load_templates("research/rapid_grow.png"),
            "recommended": load_templates("research/recommended.png") or load_templates("donation/recommended.png") or load_templates("recommended.png"),
            "research": load_templates("research/research_button.png"),
            "help": load_templates("research/help_button.png"),
            "replenish": load_templates("research/replenish.png"),
        }

        self.log("[RESEARCH-FLOW] initialized")

    # ---------------------------------------------------------

    def _is_wednesday_window(self, now=None):
        now = now or datetime.now()
        wd = now.weekday()  # Monday=0 ... Sunday=6

        # mercoledì 04:00 -> giovedì 04:00
        if wd == 2 and now.hour >= 4:
            return True
        if wd == 3 and now.hour < 4:
            return True
        return False

    def _tap_recommended(self, img, log_msg):
        name, score, loc, hw = match_any(img, self.templates["recommended"])

        if name and score >= THR_RECOMMENDED:
            # il template matcha il pollice a sinistra:
            # tap spostato verso destra sulla tile
            tap_offset_x = int(hw[1] * 0.75)
            adb_tap(loc[0] + tap_offset_x, loc[1] + hw[0] // 2)
            self.log(log_msg)
            return True

        return False

    def _do_exit(self):

        if self.lab_opened:
            adb_tap(*BACK)
            time.sleep(0.5)
            adb_tap(*BACK)

        self.lab_opened = False
        self.is_wednesday_mode = False

        if DEBUG:
            self.log("[RESEARCH] exit")

        self.state = ResearchState.IDLE
        WORKFLOW_MANAGER.release(Workflow.RESEARCH)
        self._mark()

    def _cooldown_ok(self):
        return (time.time() - self.last_action_ts) >= ACTION_COOLDOWN

    def _mark(self):
        now = time.time()
        self.last_action_ts = now
        self.last_progress_ts = now

    # ---------------------------------------------------------

    def trigger(self):

        if self.state != ResearchState.IDLE:
            return

        if not WORKFLOW_MANAGER.acquire(Workflow.RESEARCH):
            return

        self.is_wednesday_mode = self._is_wednesday_window()
        self.state = ResearchState.FIND_START
        self.last_progress_ts = time.time()
        self._mark()

    # ---------------------------------------------------------

    def step(self, img):
        if DEBUG:
            self.log(f"[RESEARCH] step state={self.state}")

        if self.state == ResearchState.IDLE:
            return

        if not self._cooldown_ok():
            return

        if (time.time() - self.last_progress_ts) > STALL_TIMEOUT:
            self.state = ResearchState.IDLE
            self.is_wednesday_mode = False
            WORKFLOW_MANAGER.release(Workflow.RESEARCH)
            return

        # -----------------------------------------------------
        # START
        # -----------------------------------------------------

        if self.state == ResearchState.FIND_START:

            name, score, loc, hw = match_any(img, self.templates["start"])
            if DEBUG:
                self.log(f"[RESEARCH] start score={score:.3f}")

            if name and score >= THR:
                adb_tap(loc[0] + hw[1]//2, loc[1] + hw[0]//2)
                self.lab_opened = True
                self.log("[RESEARCH] start tapped")
                time.sleep(1)

            else:
                if DEBUG:
                    self.log("[RESEARCH] start not found -> continue")

            self.state = ResearchState.TAP_LAB
            self._mark()
            return

        # -----------------------------------------------------

        if self.state == ResearchState.TAP_LAB:

            name, score, loc, hw = match_any(img, self.templates["lab"])

            if name and score >= THR:
                adb_tap(loc[0] + hw[1]//2, loc[1] + hw[0]//2)
                self.log("[RESEARCH] lab tapped")
                time.sleep(2)
                self.state = ResearchState.TAP_RAPID
                self._mark()
            else:
                if DEBUG:
                    self.log("[RESEARCH] lab not found -> exit")
                self._do_exit()
                self._mark()
            return

        # -----------------------------------------------------

        if self.state == ResearchState.TAP_RAPID:

            if self.is_wednesday_mode:
                if self._tap_recommended(img, "[RESEARCH] recommended category tapped"):
                    self.state = ResearchState.SCAN_NODE
                    self._mark()
                else:
                    self.log("[RESEARCH] recommended not found -> retry")
                    self._mark()
                return

            name, score, loc, hw = match_any(img, self.templates["rapid"])

            if name and score >= THR:
                adb_tap(loc[0] + hw[1]//2, loc[1] + hw[0]//2)
                self.log("[RESEARCH] rapid growth tapped")
                self.node_index = 0
                self.state = ResearchState.SCAN_NODE
                self._mark()
            else:
                self.log("[RESEARCH] rapid not found -> retry")
                self._mark()

            return

        # -----------------------------------------------------
        # SCAN TECH
        # -----------------------------------------------------

        if self.state == ResearchState.SCAN_NODE:

            if self.is_wednesday_mode:
                if self._tap_recommended(img, "[RESEARCH] recommended node tapped"):
                    self.state = ResearchState.START_RESEARCH
                    self._mark()
                else:
                    self.log("[RESEARCH] recommended node not found -> retry")
                    self._mark()
                return

            if self.node_index >= len(NODE_COORDS):
                self.log("[RESEARCH] no research available")
                self.state = ResearchState.EXIT
                return

            x, y = NODE_COORDS[self.node_index]
            adb_tap(x, y)

            self.node_index += 1
            self.state = ResearchState.START_RESEARCH
            self._mark()
            return

        # -----------------------------------------------------

        if self.state == ResearchState.START_RESEARCH:

            name, score, loc, hw = match_any(img, self.templates["research"])

            if name and score >= THR:

                adb_tap(loc[0] + hw[1]//2, loc[1] + hw[0]//2)
                self.log("[RESEARCH] research started")

                time.sleep(2)

                self.state = ResearchState.REPLENISH
                self._mark()
                return

            # chiudi popup (max level ecc.)
            adb_tap(50, 50)  # tap fuori popup
            time.sleep(0.3)

            # no research -> try next node
            self.state = ResearchState.SCAN_NODE
            self._mark()
            return

        # -----------------------------------------------------

        if self.state == ResearchState.REPLENISH:

            name_r, score_r, loc_r, hw_r = match_any(img, self.templates["replenish"])

            if name_r and score_r >= THR:
                adb_tap(loc_r[0] + hw_r[1]//2, loc_r[1] + hw_r[0]//2)
                self.log("[RESEARCH] replenish all tapped")
                time.sleep(1)

            self.state = ResearchState.HELP
            self._mark()
            return

        # -----------------------------------------------------

        if self.state == ResearchState.HELP:

            name, score, loc, hw = match_any(img, self.templates["help"])

            if name and score >= THR:
                adb_tap(loc[0] + hw[1]//2, loc[1] + hw[0]//2)
                self.log("[RESEARCH] alliance help sent")
                self.state = ResearchState.EXIT
                self._mark()
            return

        # -----------------------------------------------------

        if self.state == ResearchState.EXIT:

            if self.lab_opened:
                adb_tap(*BACK)
                time.sleep(0.5)
                adb_tap(*BACK)

            self.lab_opened = False
            self.is_wednesday_mode = False

            self.log("[RESEARCH] exit")

            self.state = ResearchState.IDLE
            WORKFLOW_MANAGER.release(Workflow.RESEARCH)
            self._mark()

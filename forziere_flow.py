#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import bot_utils

from enum import Enum
from workflow_manager import Workflow, WORKFLOW_MANAGER
from ocr_utils import read_timer_seconds
from bot_utils import (
    crop_roi,
    match_any,
    adb_tap,
    load_templates,
    debug_save,
)

# ============================================================
# CONFIG
# ============================================================

THR_FORZIERE = 0.91
THR_COLLECT  = 0.85

ROI_FORZIERE = (0.40, 0.60, 0.55, 0.75)   # zona forziere dorato (ADATTA SE SERVE)
ROI_FORZIERE_TIMER = (0.60, 0.95, 0.45, 0.55)
ROI_COLLECT = (0.0, 1.0, 0.0, 1.0)

BACK_TAP = (100, 2300)

ACTION_COOLDOWN = 1.0
STALL_TIMEOUT   = 20

DEBUG=True
bot_utils.DEBUG = True
bot_utils.DEBUG_DIR = "debug/forziere"

# ============================================================
# STATE
# ============================================================

class ForziereState(Enum):
    IDLE = 0
    FIND_FORZIERE = 1
    TAP_CLAIM = 2
    TAP_COLLECT = 3
    EXIT = 4

# ============================================================
# FLOW
# ============================================================

class ForziereFlow:
    def __init__(self, log_fn=print):
        self.log = log_fn
        self.state = ForziereState.IDLE
        self.last_action_ts = 0.0
        self.last_progress_ts = 0.0
        self.next_available_ts = 0.0
        self.dynamic_cooldown = 60
        self.claim_wait_start = 0.0

        self.templates = {
            "forziere_full": load_templates("forziere"),
            "collect": load_templates("collect_button"),
            "claim": load_templates("claim_button"),      # pulsante CLAIM
        }

        self.log("[FORZIERE-FLOW] inizializzato")

    # --------------------------------------------------------

    def _cooldown_ok(self):
        return (time.time() - self.last_action_ts) >= ACTION_COOLDOWN

    def _mark(self):
        self.last_action_ts = time.time()

    # --------------------------------------------------------

    def trigger(self):
        # non triggerare se in cooldown
        if time.time() < self.next_available_ts:
            return
    
        if self.state != ForziereState.IDLE:
            return
    
        if not WORKFLOW_MANAGER.acquire(Workflow.FORZIERE):
            return
    
        self.state = ForziereState.FIND_FORZIERE
        self.last_progress_ts = time.time()
        self._mark()

    # --------------------------------------------------------

    def step(self, img):
        if time.time() < self.next_available_ts:
            if self.state != ForziereState.IDLE:
                self.log("[FORZIERE-FLOW] in cooldown → force release")
                self.state = ForziereState.IDLE
                WORKFLOW_MANAGER.release(Workflow.FORZIERE)
            return
        if self.state == ForziereState.IDLE:
            return
        if not self._cooldown_ok():
            return

        # watchdog
        if (time.time() - self.last_progress_ts) > STALL_TIMEOUT:
            self.log("[FORZIERE-FLOW] STALL → reset")
            self.state = ForziereState.IDLE
            WORKFLOW_MANAGER.release(Workflow.FORZIERE)
            adb_tap(*BACK_TAP)
            time.sleep(0.5)
            return

        # ----------------------------------------------------
        # 1) trova forziere
        # ----------------------------------------------------
        if self.state == ForziereState.FIND_FORZIERE:
            roi, (ox, oy) = crop_roi(img, ROI_FORZIERE)
        
            name, score, loc, hw = match_any(roi, self.templates["forziere_full"])
        
            if name and score >= THR_FORZIERE:
                adb_tap(
                    ox + loc[0] + hw[1] // 2,
                    oy + loc[1] + hw[0] // 2
                )
                time.sleep(0.8)
                self.log(f"[FORZIERE-FLOW] forziere PIENO score={score:.3f}")

                self.state = ForziereState.TAP_CLAIM
                self.claim_wait_start = time.time()

                self.last_progress_ts = time.time()
                self._mark()
                return
        
            self.state = ForziereState.IDLE
            WORKFLOW_MANAGER.release(Workflow.FORZIERE)
            return

        # ----------------------------------------------------
        # 2) tap claim (opzionale con timeout 5s)
        # ----------------------------------------------------
        if self.state == ForziereState.TAP_CLAIM:
        
            roi, (ox, oy) = crop_roi(img, ROI_COLLECT)
            name, score, loc, hw = match_any(roi, self.templates["claim"])
        
            self.log(f"[FORZIERE-FLOW][DEBUG] claim score={score:.3f}")
        
            # Se troviamo CLAIM → tap
            if name and score >= 0.75:
                adb_tap(
                    ox + loc[0] + hw[1] // 2,
                    oy + loc[1] + hw[0] // 2
                )
                time.sleep(0.8)
                self.log("[FORZIERE-FLOW] CLAIM tapped")
        
                self.state = ForziereState.TAP_COLLECT
                self.last_progress_ts = time.time()
                self._mark()
                return
        
            # Se non trovato ma siamo ancora dentro 5s → continua a cercare
            if (time.time() - self.claim_wait_start) < 5:
                self.log(f"[FORZIERE-FLOW] claim wait {time.time() - self.claim_wait_start:.1f}s")
                return
        
            # Timeout scaduto → vai avanti comunque
            self.log("[FORZIERE-FLOW] CLAIM non presente → skip")
            self.state = ForziereState.TAP_COLLECT
            self.last_progress_ts = time.time()
            self._mark()
            return

        # ----------------------------------------------------
        # 3) tap collect
        # ----------------------------------------------------
        if self.state == ForziereState.TAP_COLLECT:
            self.last_progress_ts = time.time()
            roi, (ox, oy) = crop_roi(img, ROI_COLLECT)
            name, score, loc, hw = match_any(roi, self.templates["collect"])
            self.log(f"[FORZIERE-FLOW][DEBUG] collect score={score:.3f}")

            if name and score >= THR_COLLECT:
                adb_tap(
                    ox + loc[0] + hw[1] // 2,
                    oy + loc[1] + hw[0] // 2
                )
                time.sleep(0.8)
                self.log("[FORZIERE-FLOW] COLLECT")
                self.state = ForziereState.EXIT
                self._mark()
                return

            self.log("[FORZIERE-FLOW] collect non trovato → abort")
            self.state = ForziereState.IDLE
            WORKFLOW_MANAGER.release(Workflow.FORZIERE)
            adb_tap(*BACK_TAP)
            self._mark()
            return

        # ----------------------------------------------------
        # 4) exit
        # ----------------------------------------------------
        if self.state == ForziereState.EXIT:
            adb_tap(*BACK_TAP)
            time.sleep(0.8)
            adb_tap(*BACK_TAP)
            self.log("[FORZIERE-FLOW] EXIT")
            self.state = ForziereState.IDLE
            WORKFLOW_MANAGER.release(Workflow.FORZIERE)
            self._mark()


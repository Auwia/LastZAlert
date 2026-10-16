#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Troops training workflow for LastZAlert."""

import os
import time
from datetime import datetime
from enum import Enum, auto

from bot_utils import adb_tap, load_templates, match_any
from workflow_manager import Workflow, WORKFLOW_MANAGER

DEBUG = True
ENABLE_TROOPS_FLOW = False  # Keep disabled until templates are ready

THR = 0.75
STALL_TIMEOUT_SEC = 120
MAX_TROOP_BATCHES = 3

# Monday=0, Sunday=6
TROOPS_SCHEDULE = {
    0: ((20, 15, 23, 55),),
    1: ((16, 15, 19, 55),),
    2: ((7, 15, 11, 55),),
    3: ((4, 15, 7, 55),),
    4: ((12, 15, 15, 55),),
    5: ((0, 15, 3, 55), (20, 15, 23, 55)),
    6: ((12, 15, 15, 55),),
}

TROOP_TYPES = ("shooter", "assaulter", "rider")


class TroopsState(Enum):
    IDLE = auto()
    FIND_COMPLETED_HELMET = auto()
    COLLECT_COMPLETED = auto()
    FIND_EMPTY_HELMET = auto()
    OPEN_TRAINING = auto()
    TAP_TRAIN = auto()
    CHECK_REPLENISH = auto()
    CONFIRM_REPLENISH = auto()
    VERIFY_TRAINING = auto()
    RETURN_HQ = auto()
    NEXT_BUILDING = auto()
    EXIT = auto()


def troops_schedule_active(now=None):
    if now is None:
        now = datetime.now()

    current = now.hour * 60 + now.minute

    for sh, sm, eh, em in TROOPS_SCHEDULE.get(now.weekday(), ()):
        if sh * 60 + sm <= current <= eh * 60 + em:
            return True

    return False


class TroopsFlow:
    def __init__(self, log_fn):
        self.log = log_fn
        self.state = TroopsState.IDLE
        self.completed = set()
        self.last_progress = time.monotonic()

        self.templates = {
            "replenish": load_templates("research/replenish.png"),
        }

        self.log("[TROOPS] initialized (disabled)")

    def trigger(self):
        if not ENABLE_TROOPS_FLOW:
            return False

        if not troops_schedule_active():
            return False

        self.log("[TROOPS] trigger requested")
        return False  # No ADB actions until helmet templates are ready

    def step(self, img):
        # Future screenshot-driven state machine.
        return

    def match_replenish(self, img):
        """Reuse the existing Research replenish template."""
        name, score, loc, hw = match_any(
            img, self.templates["replenish"]
        )

        if DEBUG:
            self.log(
                f"[TROOPS] replenish "
                f"name={name} score={score:.3f}"
            )

        if name and score >= THR:
            return loc[0] + hw[1] // 2, loc[1] + hw[0] // 2

        return None

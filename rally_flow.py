#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from enum import Enum

from workflow_manager import WORKFLOW_MANAGER, Workflow
from bot_utils import (
    crop_roi,
    match_any,
    tap_match_in_fullscreen,
    load_templates,
    adb_tap,
)


BOTTOM_LEFT = (100, 2400)
BOTTOM_RIGHT = (1000, 2400)

# RALLY_TRIGGER_ROI = (0.05, 0.45, 0.68, 0.88)
RALLY_TRIGGER_ROI = (0.05, 0.58, 0.78, 0.98)
RALLY_MONSTER_ROI = (0.00, 1.00, 0.10, 0.80)

TRIGGER_THRESHOLD = 0.80
MONSTER_THRESHOLD = 0.80
ADD_THRESHOLD = 0.90
READY_THRESHOLD = 0.80
MARCH_THRESHOLD = 0.80
CONFIRM_THRESHOLD = 0.80


class RallyState(Enum):
    IDLE = 0
    OPEN_RALLY = 1
    FIND_MONSTER = 2
    CLICK_ADD = 3
    CLICK_MARCH = 4
    CONFIRM = 5
    DONE = 6


STATE_TIMEOUTS = {
    RallyState.OPEN_RALLY: 10,
    RallyState.CLICK_ADD: 8,
    RallyState.CLICK_MARCH: 6,
    RallyState.CONFIRM: 5,
}


class RallyFlow:
    def __init__(self, log_fn):
        self.log = log_fn
        self.state = RallyState.IDLE
        self.cooldown_until = 0
        self.state_started_at = 0

        self.trigger_templates = load_templates("rally/trigger")
        self.monster_templates = load_templates("rally/monsters")
        self.add_templates = load_templates("rally/add")
        self.march_templates = load_templates("rally/march")
        self.confirm_templates = load_templates("rally/confirm")

    def set_state(self, new_state, cooldown=0):
        now = time.time()
        self.state = new_state
        self.state_started_at = now
        self.cooldown_until = now + cooldown

    def trigger(self):
        if self.state != RallyState.IDLE:
            return

        # Respect cooldown after finish(), otherwise the flow can restart immediately.
        if time.time() < self.cooldown_until:
            return

        if not WORKFLOW_MANAGER.acquire(Workflow.RALLY):
            self.log("[RALLY] acquire failed")
            return

        self.log("[RALLY] trigger start")
        self.set_state(RallyState.OPEN_RALLY)

    def step(self, img):
        if self.state == RallyState.IDLE:
            return

        now = time.time()

        # Generic state timeout. This prevents the workflow from getting stuck forever
        # when a template is not found.
        timeout = STATE_TIMEOUTS.get(self.state)
        if timeout is not None and now - self.state_started_at > timeout:
            self.log(f"[RALLY] timeout in state {self.state.name}")

            # CONFIRM is optional: if it does not appear, just finish.
            # For the other states, tap bottom-left to try to leave the current screen.
            if self.state != RallyState.CONFIRM:
                adb_tap(*BOTTOM_LEFT)

            self.finish()
            return

        if now < self.cooldown_until:
            return

        # -----------------------------------------
        # OPEN RALLY LIST
        # -----------------------------------------
        if self.state == RallyState.OPEN_RALLY:
            roi, coords = crop_roi(img, RALLY_TRIGGER_ROI)
            name, score, loc, hw = match_any(roi, self.trigger_templates)

            if score >= TRIGGER_THRESHOLD:
                tap_match_in_fullscreen(coords, loc, hw)
                self.log("[RALLY] rally list opened")
                self.set_state(RallyState.FIND_MONSTER, cooldown=2)
                return

            return

        # -----------------------------------------
        # FIND MONSTER
        # -----------------------------------------
        if self.state == RallyState.FIND_MONSTER:
            roi, coords = crop_roi(img, RALLY_MONSTER_ROI)
            name, score, loc, hw = match_any(roi, self.monster_templates)

            if score >= MONSTER_THRESHOLD:
                self.log(f"[RALLY] monster found {name} score={score:.3f}")
                self.set_state(RallyState.CLICK_ADD, cooldown=1)
                return

            adb_tap(*BOTTOM_LEFT)
            self.log(f"[RALLY] no monster in ROI (score={score:.3f})")
            self.finish()
            return

        # -----------------------------------------
        # CLICK ADD
        # -----------------------------------------
        if self.state == RallyState.CLICK_ADD:
            name, score, loc, hw = match_any(img, self.add_templates)

            if score >= ADD_THRESHOLD:
                tap_match_in_fullscreen((0, 0, img.shape[1], img.shape[0]), loc, hw)
                self.log("[RALLY] join rally")
                self.set_state(RallyState.CLICK_MARCH, cooldown=2)
                return

            self.log(f"[RALLY] add not clickable (score={score:.3f}) → exit")
            adb_tap(*BOTTOM_LEFT)
            self.finish()
            return

        # -----------------------------------------
        # CLICK MARCH
        # -----------------------------------------
        if self.state == RallyState.CLICK_MARCH:
            name, score, loc, hw = match_any(img, self.march_templates)

            if score >= MARCH_THRESHOLD:
                tap_match_in_fullscreen((0, 0, img.shape[1], img.shape[0]), loc, hw)
                self.log("[RALLY] march clicked")
                self.set_state(RallyState.CONFIRM, cooldown=2)
                return

            self.log(f"[RALLY] march not clickable or disabled (score={score:.3f}) → exit")
            adb_tap(*BOTTOM_LEFT)
            self.finish()
            return

        # -----------------------------------------
        # OPTIONAL CONFIRM POPUP
        # -----------------------------------------
        if self.state == RallyState.CONFIRM:
            name, score, loc, hw = match_any(img, self.confirm_templates)

            if score >= CONFIRM_THRESHOLD:
                tap_match_in_fullscreen((0, 0, img.shape[1], img.shape[0]), loc, hw)
                self.log("[RALLY] confirm popup")
                self.finish()
                return

            # Optional popup: wait until CONFIRM timeout before finishing.
            return

    def finish(self):
        self.log("[RALLY] finished")
        self.set_state(RallyState.IDLE, cooldown=5)
        WORKFLOW_MANAGER.release(Workflow.RALLY)

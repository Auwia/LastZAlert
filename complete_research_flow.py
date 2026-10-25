#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from enum import Enum

from workflow_manager import Workflow, WORKFLOW_MANAGER
from bot_utils import load_templates, match_any, adb_tap


THR_COMPLETE_ICON = 0.95
THR_COMPLETE_BUTTON = 0.80

ACTION_COOLDOWN = 1.0
STALL_TIMEOUT = 30.0

QUEUE_OPEN_WAIT = 1.5
COMPLETE_WAIT = 2.0

# Research Queue close button, based on 1080x2408 screenshots.
QUEUE_CLOSE = (955, 580)


class CompleteResearchState(Enum):
    IDLE = 0
    OPEN_QUEUE = 1
    TAP_COMPLETE = 2
    CLOSE_QUEUE = 3


class CompleteResearchFlow:

    def __init__(self, log_fn=print):
        self.log = log_fn

        self.state = CompleteResearchState.IDLE
        self.last_action_ts = 0.0
        self.last_progress_ts = 0.0

        self.templates = {
            "complete_icon": load_templates(
                "research/research_complete.png"
            ),
            "complete_button": load_templates(
                "research/button_complete.png"
            ),
            "free_button": load_templates(
                "research/free_research.png"
            ),
        }

        self.log("[COMPLETE-RESEARCH] initialized")

    def _mark(self):
        now = time.time()
        self.last_action_ts = now
        self.last_progress_ts = now

    def _release(self):
        self.state = CompleteResearchState.IDLE
        WORKFLOW_MANAGER.release(Workflow.COMPLETE_RESEARCH)
        self._mark()
        self.log("[COMPLETE-RESEARCH] released")

    def trigger(self):
        if self.state != CompleteResearchState.IDLE:
            return False

        if not WORKFLOW_MANAGER.acquire(Workflow.COMPLETE_RESEARCH):
            return False

        self.state = CompleteResearchState.OPEN_QUEUE
        self._mark()
        self.log("[COMPLETE-RESEARCH] triggered")
        return True

    def step(self, img):
        if self.state == CompleteResearchState.IDLE:
            return

        now = time.time()

        if now - self.last_action_ts < ACTION_COOLDOWN:
            return

        if now - self.last_progress_ts > STALL_TIMEOUT:
            self.log(
                f"[COMPLETE-RESEARCH] STALL TIMEOUT "
                f"state={self.state.name} -> release"
            )
            self._release()
            return

        if self.state == CompleteResearchState.OPEN_QUEUE:

            name, score, loc, hw = match_any(
                img,
                self.templates["complete_icon"]
            )

            if name and score >= THR_COMPLETE_ICON:
                x = loc[0] + hw[1] // 2
                y = loc[1] + hw[0] // 2

                adb_tap(x, y)

                self.log(
                    f"[COMPLETE-RESEARCH] queue opened "
                    f"score={score:.3f} @ ({x},{y})"
                )

                time.sleep(QUEUE_OPEN_WAIT)

                self.state = CompleteResearchState.TAP_COMPLETE
                self._mark()

            return

        if self.state == CompleteResearchState.TAP_COMPLETE:

            # Find the best match between Complete and Free.
            candidates = []

            for button_type in ("complete_button", "free_button"):
                name, score, loc, hw = match_any(
                    img,
                    self.templates[button_type]
                )
                if name:
                    candidates.append(
                        (score, button_type, name, loc, hw)
                    )

            if candidates:
                score, button_type, name, loc, hw = max(
                    candidates,
                    key=lambda item: item[0]
                )
            else:
                name, score, loc, hw = None, 0.0, None, None

            if not name or score < THR_COMPLETE_BUTTON:
                self.log(
                    f"[COMPLETE-RESEARCH] No Complete button "
                    f"(score={score:.3f}) -> release"
                )
                self._release()
                return

            if name and score >= THR_COMPLETE_BUTTON:
                x = loc[0] + hw[1] // 2
                y = loc[1] + hw[0] // 2

                adb_tap(x, y)

                self.log(
                    f"[COMPLETE-RESEARCH] Complete tapped "
                    f"score={score:.3f} @ ({x},{y})"
                )

                time.sleep(COMPLETE_WAIT)

                self.log("[COMPLETE-RESEARCH] completed -> release")
                self._release()

            return

        if self.state == CompleteResearchState.CLOSE_QUEUE:

            adb_tap(*QUEUE_CLOSE)

            self.log("[COMPLETE-RESEARCH] queue closed")
            self._release()
            return

    def detect_trigger(self, img):
        """Detect the completed research icon on the home screen."""
        if self.state != CompleteResearchState.IDLE:
            return False

        name, score, loc, hw = match_any(
            img,
            self.templates["complete_icon"]
        )

        if name and score >= THR_COMPLETE_ICON:
            self.log(
                f"[COMPLETE-RESEARCH] completion detected "
                f"score={score:.3f}"
            )
            return True

        return False

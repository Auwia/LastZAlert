#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import os
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
WAIT_LIST_ADD_THRESHOLD = 0.70

RALLY_DEBUG_DIR = "debug/rally"
os.makedirs(RALLY_DEBUG_DIR, exist_ok=True)

MONSTER_SCALES = (0.75, 0.85, 0.95)

RALLY_MONSTER_SLOTS = [
    {
        "name": "row1",
        "roi": (0.03, 0.31, 0.12, 0.31),
        "add_tap": (0.945, 0.154),
    },
    {
        "name": "row2",
        "roi": (0.03, 0.31, 0.37, 0.56),
        "add_tap": (0.945, 0.415),
    },
    {
        "name": "row3",
        "roi": (0.03, 0.31, 0.62, 0.79),
        "add_tap": (0.945, 0.670),
    },
]

RALLY_TEAM_DETAILS_ROI = (0.20, 0.80, 0.02, 0.11)
TEAM_DETAILS_THRESHOLD = 0.85
TEAM_DETAILS_SCALES = (0.75, 0.80, 0.85, 0.90, 0.95, 1.00)

class RallyState(Enum):
    IDLE = 0
    OPEN_RALLY = 1
    WAIT_LIST = 2
    FIND_MONSTER = 3
    CLICK_ADD = 4
    CLICK_MARCH = 5
    CONFIRM = 6
    DONE = 7


STATE_TIMEOUTS = {
    RallyState.OPEN_RALLY: 10,
    RallyState.WAIT_LIST: 12,
    RallyState.CLICK_ADD: 8,
    RallyState.CLICK_MARCH: 6,
    RallyState.CONFIRM: 5,
}

def match_any_multiscale(roi_img, templates, scales):
    best_name = None
    best_score = 0.0
    best_loc = (0, 0)
    best_hw = (0, 0)
    best_scale = 1.0

    if roi_img is None or roi_img.size == 0:
        return best_name, best_score, best_loc, best_hw, best_scale

    rh, rw = roi_img.shape[:2]

    for name, tmpl in templates:
        th0, tw0 = tmpl.shape[:2]

        for scale in scales:
            tw = int(tw0 * scale)
            th = int(th0 * scale)

            if tw < 8 or th < 8:
                continue

            if rh < th or rw < tw:
                continue

            tmpl_s = cv2.resize(
                tmpl,
                (tw, th),
                interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            )

            res = cv2.matchTemplate(roi_img, tmpl_s, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(res)

            if score > best_score:
                best_name = name
                best_score = float(score)
                best_loc = loc
                best_hw = (th, tw)
                best_scale = scale

    return best_name, best_score, best_loc, best_hw, best_scale

def crop_roi4(img, roi_frac):
    roi, coords = crop_roi(img, roi_frac)

    if coords is not None and len(coords) == 4:
        return roi, coords

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

    return roi, (xs, ys, xe, ye)

class RallyFlow:
    def __init__(self, log_fn):
        self.log = log_fn
        self.state = RallyState.IDLE
        self.cooldown_until = 0
        self.state_started_at = 0
        self.pending_add_tap = None

        self.trigger_templates = load_templates("rally/trigger")
        self.monster_templates = load_templates("rally/monsters")
        self.add_templates = load_templates("rally/add")
        self.march_templates = load_templates("rally/march")
        self.confirm_templates = load_templates("rally/confirm")
        self.team_details_templates = load_templates("rally/team_details")

    def _tap_frac(self, img, frac, label):
        h, w = img.shape[:2]
        x = int(w * frac[0])
        y = int(h * frac[1])
        adb_tap(x, y)
        self.log(f"[RALLY] tap fixed {label} @ {x},{y}")
        return x, y

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
                self.log(f"[RALLY] rally trigger tapped {name} score={score:.3f} -> WAIT_LIST")
                self.set_state(RallyState.WAIT_LIST, cooldown=1)
                return

            return

        # -----------------------------------------
        # WAIT RALLY LIST / TEAM DETAILS PAGE
        # -----------------------------------------
        if self.state == RallyState.WAIT_LIST:
            roi, coords = crop_roi4(img, RALLY_TEAM_DETAILS_ROI)

            cv2.imwrite(os.path.join(RALLY_DEBUG_DIR, "wait_list_title_roi.png"), roi)

            name, score, loc, hw, scale = match_any_multiscale(
                roi,
                self.team_details_templates,
                TEAM_DETAILS_SCALES
            )

            self.log(
                f"[RALLY] waiting Team Details "
                f"title={name} score={score:.3f} scale={scale:.2f}"
            )

            if score >= TEAM_DETAILS_THRESHOLD:
                self.log(
                    f"[RALLY] Team Details ready "
                    f"title={name} score={score:.3f} scale={scale:.2f}"
                )
                self.set_state(RallyState.FIND_MONSTER, cooldown=0.5)
                return

            time.sleep(0.5)
            return

        # -----------------------------------------
        # FIND MONSTER IN FIXED ROW SLOTS
        # -----------------------------------------
        if self.state == RallyState.FIND_MONSTER:
            best = {
                "slot": None,
                "name": None,
                "score": 0.0,
                "scale": 1.0,
                "loc": (0, 0),
                "hw": (0, 0),
                "coords": (0, 0, 0, 0),
                "add_tap": None,
            }

            full_debug = img.copy()

            for idx, slot in enumerate(RALLY_MONSTER_SLOTS, start=1):
                roi, coords = crop_roi4(img, slot["roi"])
                cv2.imwrite(
                    os.path.join(RALLY_DEBUG_DIR, f"monster_slot_{idx}_{slot['name']}.png"),
                    roi
                )

                name, score, loc, hw, scale = match_any_multiscale(
                    roi,
                    self.monster_templates,
                    MONSTER_SCALES
                )

                xs, ys, xe, ye = coords
                cv2.rectangle(full_debug, (xs, ys), (xe, ye), (0, 255, 255), 2)
                cv2.putText(
                    full_debug,
                    f"{slot['name']} {score:.2f}",
                    (xs, max(20, ys - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

                if score > best["score"]:
                    best.update({
                        "slot": slot["name"],
                        "name": name,
                        "score": float(score),
                        "scale": scale,
                        "loc": loc,
                        "hw": hw,
                        "coords": coords,
                        "add_tap": slot["add_tap"],
                    })

            if best["name"] is not None:
                xs, ys, xe, ye = best["coords"]
                x, y = best["loc"]
                th, tw = best["hw"]

                cv2.rectangle(
                    full_debug,
                    (xs + x, ys + y),
                    (xs + x + tw, ys + y + th),
                    (0, 255, 0),
                    3
                )

            cv2.imwrite(os.path.join(RALLY_DEBUG_DIR, "monster_slots_full.png"), full_debug)

            if best["score"] >= MONSTER_THRESHOLD:
                self.pending_add_tap = best["add_tap"]

                self.log(
                    f"[RALLY] monster found "
                    f"slot={best['slot']} "
                    f"name={best['name']} "
                    f"score={best['score']:.3f} "
                    f"scale={best['scale']:.2f}"
                )

                self.set_state(RallyState.CLICK_ADD, cooldown=0.3)
                return

            self.log(
                f"[RALLY] no monster in fixed slots "
                f"best={best['name']} "
                f"slot={best['slot']} "
                f"score={best['score']:.3f} "
                f"scale={best['scale']:.2f}"
            )

            self.finish()
            return

        # -----------------------------------------
        # CLICK ADD
        # -----------------------------------------
        if self.state == RallyState.CLICK_ADD:
            if self.pending_add_tap is not None:
                self._tap_frac(img, self.pending_add_tap, "add")
                self.pending_add_tap = None
                self.log("[RALLY] join rally")
                self.set_state(RallyState.CLICK_MARCH, cooldown=2)
                return

            name, score, loc, hw = match_any(img, self.add_templates)

            if score >= ADD_THRESHOLD:
                tap_match_in_fullscreen((0, 0, img.shape[1], img.shape[0]), loc, hw)
                self.log(f"[RALLY] join rally template={name} score={score:.3f}")
                self.set_state(RallyState.CLICK_MARCH, cooldown=2)
                return

            self.log(f"[RALLY] add not found score={score:.3f} -> exit")
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
            #adb_tap(*BOTTOM_LEFT)
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
        self.set_state(RallyState.IDLE, cooldown=15)
        WORKFLOW_MANAGER.release(Workflow.RALLY)

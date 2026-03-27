import os
import time
import cv2
import numpy as np
from enum import Enum
from typing import Optional
from datetime import datetime
import subprocess
import pytesseract

from workflow_manager import WORKFLOW_MANAGER, Workflow
from bot_utils import load_templates

ADB_CMD = "adb"

# =========================
# TEMPLATE DIR
# =========================
BASE_DIR = "treasure_flow"

CHAT_LINK_DIR = os.path.join(BASE_DIR, "chat_link")
CHAT_UI_DIR = os.path.join(BASE_DIR, "chat_ui")
TREASURE_ICONS_DIR = os.path.join(BASE_DIR, "helicopter")
TOKEN_DIR = os.path.join(BASE_DIR, "gold_token")
CONGR_DIR = os.path.join(BASE_DIR, "congratulations")

# =========================
# THRESHOLDS
# =========================
THR_CHAT = 0.65
THR_CHAT_UI = 0.60
THR_ICON = 0.55
THR_TOKEN = 0.70
THR_CONGR = 0.70

# =========================
# ROI
# =========================
ROI_CHAT = (0.02, 0.98, 0.18, 0.92)

# se vuoi, in futuro puoi restringere anche questa
ROI_CHAT_UI = (0.0, 1.0, 0.0, 1.0)

# =========================
# TIMINGS
# =========================
ACTION_COOLDOWN_SEC = 1.0
WAIT_CHAT_AFTER_TAP_SEC = 1.0
WAIT_AFTER_LINK_TAP_SEC = 1.2
WAIT_AFTER_ICON_TAP_SEC = 1.0
WAIT_AFTER_TOKEN_TAP_SEC = 1.0

TIMEOUT_CHAT_LINK_SEC = 8.0
TIMEOUT_MAP_ICONS_SEC = 8.0
TIMEOUT_TOKEN_SEC = 8.0
TIMEOUT_CONGR_SEC = 8.0

# =========================
# UTILS
# =========================

def adb_tap(x, y):
    subprocess.run(
        [ADB_CMD, "shell", "input", "tap", str(x), str(y)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

def crop(img, roi):
    h, w = img.shape[:2]
    x1, x2, y1, y2 = roi
    xs, xe = int(w * x1), int(w * x2)
    ys, ye = int(h * y1), int(h * y2)

    xs = max(0, min(xs, w - 1))
    xe = max(xs + 1, min(xe, w))
    ys = max(0, min(ys, h - 1))
    ye = max(ys + 1, min(ye, h))

    return img[ys:ye, xs:xe], (xs, ys, xe, ye)

def match_any(img, templates):
    if img is None or len(img.shape) != 3:
        return None, 0.0

    best_score = 0.0
    best = None

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for name, tpl in templates:
        if tpl is None:
            continue

        if len(tpl.shape) == 3:
            tpl_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
        else:
            tpl_gray = tpl

        th, tw = tpl_gray.shape[:2]
        ih, iw = img_gray.shape[:2]

        if ih < th or iw < tw:
            continue

        res = cv2.matchTemplate(img_gray, tpl_gray, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)

        if score > best_score:
            best_score = float(score)
            best = (name, loc, (th, tw))

    return best, best_score

def tap_match(offset, loc, size):
    xs, ys, _, _ = offset
    x = xs + loc[0] + size[1] // 2
    y = ys + loc[1] + size[0] // 2
    adb_tap(x, y)
    return x, y

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[SIMPLIFIED] {ts} {msg}", flush=True)

# =========================
# OCR REWARD
# =========================
def ocr_reward(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]
    txt = pytesseract.image_to_string(gray, config="--psm 6")
    return txt

# =========================
# STATE
# =========================
class State(Enum):
    IDLE = 0
    GO_CHAT = 1
    WAIT_CHAT_LINK = 2
    WAIT_MAP_ICONS = 3
    WAIT_TOKEN = 4
    WAIT_CONGR = 5
    DONE = 6

# =========================
# FLOW
# =========================
class TreasureFlowSimplified:

    def __init__(self, log_fn=print):
        self.log = log_fn
        self.state = State.IDLE

        self.t_chat = load_templates(CHAT_LINK_DIR)
        self.t_chat_ui = load_templates(CHAT_UI_DIR)
        self.t_icons = load_templates(TREASURE_ICONS_DIR)
        self.t_token = load_templates(TOKEN_DIR)
        self.t_congr = load_templates(CONGR_DIR)

        self.start_coords = None
        self.start_loc = None
        self.start_hw = None

        self.last_action_ts = 0.0
        self.state_enter_ts = 0.0
        self.flow_start_ts = 0.0

    def set_state(self, new_state: State):
        self.state = new_state
        self.state_enter_ts = time.time()
        self.log(f"state -> {new_state.name}")

    def _mark_action(self):
        self.last_action_ts = time.time()

    def _seconds_from_last_action(self):
        return time.time() - self.last_action_ts

    def _seconds_in_state(self):
        return time.time() - self.state_enter_ts

    def trigger(self, coords, loc, hw):
        if self.state != State.IDLE:
            return

        if not WORKFLOW_MANAGER.acquire(Workflow.TREASURE):
            return

        self.start_coords = coords
        self.start_loc = loc
        self.start_hw = hw

        self.flow_start_ts = time.time()
        self._mark_action()
        self.set_state(State.GO_CHAT)
        self.log("trigger")

    def step(self, img):
        if self.state == State.IDLE:
            return

        # ====================
        # GO CHAT
        # ====================
        if self.state == State.GO_CHAT:
            if self.start_coords is None or self.start_loc is None or self.start_hw is None:
                self.log("missing start tap data")
                self.set_state(State.DONE)
                return

            x, y = tap_match(self.start_coords, self.start_loc, self.start_hw)
            self._mark_action()
            self.log(f"tap treasure/chat @ ({x},{y})")
            self.set_state(State.WAIT_CHAT_LINK)
            return

        # ====================
        # WAIT CHAT LINK
        # Pattern preso dal full:
        # 1) aspetto minimo dopo tap iniziale
        # 2) controllo se la chat è ancora visibile
        # 3) se sì cerco il link
        # 4) se la chat non è più visibile, assumo che il link abbia già aperto la schermata dopo
        # ====================
        if self.state == State.WAIT_CHAT_LINK:
            if self._seconds_from_last_action() < WAIT_CHAT_AFTER_TAP_SEC:
                return

            # Controllo UI chat
            chat_ui_visible = True

            if self.t_chat_ui:
                roi_chat_ui, _ = crop(img, ROI_CHAT_UI)
                best_ui, score_ui = match_any(roi_chat_ui, self.t_chat_ui)

                if best_ui:
                    name_ui, loc_ui, size_ui = best_ui
                    self.log(f"chat_ui best={name_ui} score={score_ui:.3f}")
                else:
                    self.log("chat_ui no match")

                if not (best_ui and score_ui >= THR_CHAT_UI):
                    chat_ui_visible = False

            # Se la chat non è più visibile, passo avanti come nel full
            if not chat_ui_visible:
                self.log("chat non più visibile -> WAIT_MAP_ICONS")
                self._mark_action()
                self.set_state(State.WAIT_MAP_ICONS)
                return

            # Se siamo ancora in chat, cerchiamo il link
            if not self.t_chat:
                self.log("manca template chat_link/")
                self.set_state(State.DONE)
                return

            roi, coords = crop(img, ROI_CHAT)
            best, score = match_any(roi, self.t_chat)

            self.log(
                f"ROI_CHAT: x={coords[0]}, y={coords[1]}, w={roi.shape[1]}, h={roi.shape[0]}"
            )

            if best:
                name, loc, size = best
                self.log(f"chat link best={name} score={score:.3f} loc={loc} size={size}")
            else:
                self.log("chat link no match")

            if best and score >= THR_CHAT:
                name, loc, size = best
                x, y = tap_match(coords, loc, size)
                self._mark_action()
                self.log(f"chat hyperlink tapped: {name} @ ({x},{y}) score={score:.3f}")
                self.set_state(State.WAIT_MAP_ICONS)
                return

            if self._seconds_in_state() > TIMEOUT_CHAT_LINK_SEC:
                self.log("timeout WAIT_CHAT_LINK -> DONE")
                self.set_state(State.DONE)

            return

        # ====================
        # WAIT ICONS
        # ====================
        if self.state == State.WAIT_MAP_ICONS:
            if self._seconds_from_last_action() < WAIT_AFTER_LINK_TAP_SEC:
                return

            if not self.t_icons:
                self.log("manca template helicopter/")
                self.set_state(State.DONE)
                return

            best, score = match_any(img, self.t_icons)

            if best:
                name, loc, size = best
                self.log(f"icon best={name} score={score:.3f} loc={loc} size={size}")
            else:
                self.log("icon no match")

            if best and score >= THR_ICON:
                name, loc, size = best
                x, y = tap_match((0, 0, img.shape[1], img.shape[0]), loc, size)
                self._mark_action()
                self.log(f"icon tap {name} @ ({x},{y}) score={score:.3f}")
                self.set_state(State.WAIT_TOKEN)
                return

            if self._seconds_in_state() > TIMEOUT_MAP_ICONS_SEC:
                self.log("timeout WAIT_MAP_ICONS -> DONE")
                self.set_state(State.DONE)

            return

        # ====================
        # WAIT TOKEN
        # ====================
        if self.state == State.WAIT_TOKEN:
            if self._seconds_from_last_action() < WAIT_AFTER_ICON_TAP_SEC:
                return

            if not self.t_token:
                self.log("manca template gold_token/")
                self.set_state(State.DONE)
                return

            best, score = match_any(img, self.t_token)

            if best:
                name, loc, size = best
                self.log(f"token best={name} score={score:.3f} loc={loc} size={size}")
            else:
                self.log("token no match")

            if best and score >= THR_TOKEN:
                name, loc, size = best
                x, y = tap_match((0, 0, img.shape[1], img.shape[0]), loc, size)
                self._mark_action()
                self.log(f"token tapped @ ({x},{y}) score={score:.3f}")
                self.set_state(State.WAIT_CONGR)
                return

            if self._seconds_in_state() > TIMEOUT_TOKEN_SEC:
                self.log("timeout WAIT_TOKEN -> DONE")
                self.set_state(State.DONE)

            return

        # ====================
        # WAIT CONGRATS
        # ====================
        if self.state == State.WAIT_CONGR:
            if self._seconds_from_last_action() < WAIT_AFTER_TOKEN_TAP_SEC:
                return

            if not self.t_congr:
                self.log("manca template congratulations/")
                self.set_state(State.DONE)
                return

            best, score = match_any(img, self.t_congr)

            if best:
                name, loc, size = best
                self.log(f"congr best={name} score={score:.3f} loc={loc} size={size}")
            else:
                self.log("congr no match")

            if best and score >= THR_CONGR:
                self.log(f"CONGRATS detected score={score:.3f}")

                txt = ocr_reward(img)
                self.log(f"REWARD: {txt}")

                # chiudi popup
                adb_tap(50, 50)

                # HQ in basso a destra
                h, w = img.shape[:2]
                adb_tap(int(w * 0.95), int(h * 0.95))

                self._mark_action()
                self.set_state(State.DONE)
                return

            if self._seconds_in_state() > TIMEOUT_CONGR_SEC:
                self.log("timeout WAIT_CONGR -> DONE")
                self.set_state(State.DONE)

            return

        # ====================
        # DONE
        # ====================
        if self.state == State.DONE:
            WORKFLOW_MANAGER.release(Workflow.TREASURE)
            self.log("done -> release")
            self.state = State.IDLE

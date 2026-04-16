import os
import time
import cv2
import numpy as np
from enum import Enum
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
THR_CHAT = 0.73
THR_CHAT_UI = 0.73
THR_ICON = 0.73
THR_TOKEN = 0.73
THR_CONGR = 0.73

# =========================
# ROI
# =========================
ROI_CHAT = (0.02, 0.98, 0.18, 0.92)
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
TIMEOUT_TOKEN_SEC = 800.0
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

        tpl_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY) if len(tpl.shape) == 3 else tpl

        if img_gray.shape[0] < tpl_gray.shape[0] or img_gray.shape[1] < tpl_gray.shape[1]:
            continue

        res = cv2.matchTemplate(img_gray, tpl_gray, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)

        if score > best_score:
            best_score = float(score)
            best = (name, loc, tpl_gray.shape)

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

def ocr_reward(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]
    return pytesseract.image_to_string(gray, config="--psm 6")

# =========================
# STATE
# =========================
class State(Enum):
    IDLE = 0
    GO_CHAT = 1
    WAIT_CHAT_LINK = 2
    WAIT_MAP_ICONS = 3
    WAIT_HELI_DISAPPEAR = 4
    WAIT_TOKEN = 5
    WAIT_CONGR = 6
    DONE = 7

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

        self._mark_action()
        self.set_state(State.GO_CHAT)

    def step(self, img):

        if self.state == State.IDLE:
            return

        if self.state == State.GO_CHAT:
            x, y = tap_match(self.start_coords, self.start_loc, self.start_hw)
            self._mark_action()
            self.set_state(State.WAIT_CHAT_LINK)
            return

        if self.state == State.WAIT_CHAT_LINK:
            if self._seconds_from_last_action() < WAIT_CHAT_AFTER_TAP_SEC:
                return

            roi, coords = crop(img, ROI_CHAT)
            best, score = match_any(roi, self.t_chat)

            if best and score >= THR_CHAT:
                _, loc, size = best
                tap_match(coords, loc, size)
                self._mark_action()
                self.set_state(State.WAIT_MAP_ICONS)
                return

            if self._seconds_in_state() > TIMEOUT_CHAT_LINK_SEC:
                self.set_state(State.DONE)
            return

        if self.state == State.WAIT_MAP_ICONS:
            best, score = match_any(img, self.t_icons)

            if best and score >= THR_ICON:
                _, loc, size = best
                tap_match((0, 0, img.shape[1], img.shape[0]), loc, size)
                self._mark_action()
                self.set_state(State.WAIT_TOKEN)
                return

            if self._seconds_in_state() > TIMEOUT_MAP_ICONS_SEC:
                self.set_state(State.DONE)
            return

        if self.state == State.WAIT_TOKEN:
            best, score = match_any(img, self.t_token)
            heli_best, heli_score = match_any(img, self.t_icons)

            self.log(f"[WAIT_TOKEN] heli={heli_score:.3f} token_score={score:.3f} time={self._seconds_in_state():.2f}")

            # se elicottero visibile → reset timer 
            if heli_best and heli_score >= THR_ICON:
                self.state_enter_ts = time.time()

            if best and score >= THR_TOKEN:
                _, loc, size = best
                tap_match((0, 0, img.shape[1], img.shape[0]), loc, size)
                self._mark_action()
                self.set_state(State.WAIT_CONGR)
                return
            else:
                  # 🔍 debug
                  self.log(f"[WAIT_TOKEN] token_score={score:.3f} time={self._seconds_in_state():.2f}")
                  # 📸 salva screenshot (sempre stesso file)
                  import os
                  os.makedirs("debug/treasure", exist_ok=True)
                  cv2.imwrite("debug/treasure/wait_token.png", img)

            if self._seconds_in_state() > TIMEOUT_TOKEN_SEC:
                self.set_state(State.DONE)
            return

        if self.state == State.WAIT_CONGR:
            best, score = match_any(img, self.t_congr)

            if best and score >= THR_CONGR:
                txt = ocr_reward(img)
                self.log(f"REWARD: {txt}")
                adb_tap(50, 50)
                self._mark_action()
                self.set_state(State.DONE)
                return

            if self._seconds_in_state() > TIMEOUT_CONGR_SEC:
                self.set_state(State.DONE)
            return

        if self.state == State.DONE:
            WORKFLOW_MANAGER.release(Workflow.TREASURE)
            self.state = State.IDLE

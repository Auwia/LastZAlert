import os
import time
import cv2
from enum import Enum
from datetime import datetime
import subprocess
import pytesseract

from workflow_manager import WORKFLOW_MANAGER, Workflow
from bot_utils import load_templates, match_any

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
THR_CHAT = 0.70
THR_CHAT_UI = 0.70
THR_ICON = 0.70
THR_TOKEN = 0.81
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

TIMEOUT_CHAT_LINK_SEC = 80.0
TIMEOUT_MAP_ICONS_SEC = 80.0
TIMEOUT_TOKEN_SEC = 800.0
TIMEOUT_CONGR_SEC = 55.0

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
    VERIFY_LINK_TAP = 3
    WAIT_MAP_ICONS = 4
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

    def _chat_ui_visible(self, img):
        roi, _ = crop(img, ROI_CHAT_UI)
        best_name, score, _, _ = match_any(roi, self.t_chat_ui)
        return (best_name is not None and score >= THR_CHAT_UI), score

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
            self.log("[GO_CHAT] entering, about to tap treasure")
            x, y = tap_match(self.start_coords, self.start_loc, self.start_hw)
            self.log(f"[GO_CHAT] tapped @ {x},{y}")
            self._mark_action()
            self.set_state(State.WAIT_CHAT_LINK)
            return

        if self.state == State.WAIT_CHAT_LINK:
            dt_action = self._seconds_from_last_action()
            dt_state = self._seconds_in_state()
            self.log(f"[WAIT_CHAT_LINK] dt_action={dt_action:.2f} dt_state={dt_state:.2f}")
        
            if dt_action < WAIT_CHAT_AFTER_TAP_SEC:
                return
        
            roi, coords = crop(img, ROI_CHAT)
            best_name, score, loc, size = match_any(roi, self.t_chat)
    
            self.log(f"[WAIT_CHAT_LINK] best={best_name} score={score:.6f} thr={THR_CHAT:.6f}")
        
            os.makedirs("debug/treasure", exist_ok=True)
            cv2.imwrite("debug/treasure/wait_chat_link_roi.png", roi)
        
            dbg = img.copy()
            xs, ys, xe, ye = coords
            cv2.rectangle(dbg, (xs, ys), (xe, ye), (0, 255, 0), 3)
            cv2.imwrite("debug/treasure/wait_chat_link_full_with_roi.png", dbg)
        
            if best_name is not None:
                roi_dbg = roi.copy()
                mx, my = loc
                h, w = size
                cv2.rectangle(roi_dbg, (mx, my), (mx + w, my + h), (0, 0, 255), 3)
                cv2.imwrite("debug/treasure/wait_chat_link_best_match.png", roi_dbg)
        
            if best_name is not None and score >= THR_CHAT:
                tap_match(coords, loc, size)
                self._mark_action()
                self.set_state(State.VERIFY_LINK_TAP)
                return
        
            if self._seconds_in_state() > TIMEOUT_CHAT_LINK_SEC:
                self.set_state(State.DONE)
            return

        if self.state == State.VERIFY_LINK_TAP:
            if self._seconds_from_last_action() < WAIT_AFTER_LINK_TAP_SEC:
                return

            chat_visible, chat_score = self._chat_ui_visible(img)

            if chat_visible:
                self.log(f"[VERIFY_LINK_TAP] still in chat chat_ui={chat_score:.3f} -> retry")
                self._mark_action()
                self.set_state(State.WAIT_CHAT_LINK)
                return

            self.log(f"[VERIFY_LINK_TAP] left chat chat_ui={chat_score:.3f} -> map")
            self.set_state(State.WAIT_MAP_ICONS)
            return

        if self.state == State.WAIT_MAP_ICONS:
            best_name, score, loc, size = match_any(img, self.t_icons)

            if best_name is not None and score >= THR_ICON:
                tap_match((0, 0, img.shape[1], img.shape[0]), loc, size)
                self._mark_action()
                self.set_state(State.WAIT_TOKEN)
                return

            if self._seconds_in_state() > TIMEOUT_MAP_ICONS_SEC:
                self.log(f"[WAIT_MAP_ICONS] timeout score={score:.3f}")
                self.set_state(State.DONE)
            return

        if self.state == State.WAIT_TOKEN:
            token_name, score, loc, size = match_any(img, self.t_token)
            heli_name, heli_score, _, _ = match_any(img, self.t_icons)
        
            self.log(f"[WAIT_TOKEN] heli={heli_score:.3f} token_score={score:.3f} time={self._seconds_in_state():.2f}")
        
            # se elicottero visibile → reset timer
            if heli_name is not None and heli_score >= THR_ICON:
                self.state_enter_ts = time.time()
        
            if token_name is not None and score >= THR_TOKEN:
                tap_match((0, 0, img.shape[1], img.shape[0]), loc, size)
                self._mark_action()
                self.set_state(State.WAIT_CONGR)
                return
        
            os.makedirs("debug/treasure", exist_ok=True)
            cv2.imwrite("debug/treasure/wait_token.png", img)
        
            if self._seconds_in_state() > TIMEOUT_TOKEN_SEC:
                self.log("[WAIT_TOKEN] timeout")
                self.set_state(State.DONE)
            return

        if self.state == State.WAIT_CONGR:
            best_name, score, _, _ = match_any(img, self.t_congr)

            if best_name is not None and score >= THR_CONGR:
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

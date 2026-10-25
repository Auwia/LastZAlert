# treasure_flow.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import time
import random
import threading
import subprocess
from enum import Enum
from typing import Optional, Tuple, List
from workflow_manager import WORKFLOW_MANAGER, Workflow
from datetime import datetime

import cv2
import numpy as np

# ============================================================
# CONFIG (puoi modificarle da fuori se vuoi)
# ============================================================
DEBUG = True
DEBUG_MISS_SNAPSHOT_SEC = 10.0

ADB_CMD = "adb"

# Templates folder (creala nel progetto)
# Suggerito:
# treasure_flow/
#   chat_link/         (template della scritta/link "Explore Treasure" nella chat)
#   helicopter/        (template elicottero in fiamme / elicottero)
#   magnifier/         (template lente)
#   march/             (template bottone March)
TEMPL_DIR = "treasure_flow"
CHAT_LINK_DIR   = os.path.join(TEMPL_DIR, "chat_link")
HELI_DIR        = os.path.join(TEMPL_DIR, "helicopter")
MAGNIFIER_DIR   = os.path.join(TEMPL_DIR, "magnifier")
MARCH_DIR       = os.path.join(TEMPL_DIR, "march")

# Match thresholds (tarali)
THR_CHAT_LINK   = 0.67
THR_HELI        = 0.45
THR_MAGNIFIER   = 0.45
THR_MARCH       = 0.45

# Loop / timings
FLOW_TICK_SEC = 0.25
ACTION_COOLDOWN_SEC = 1.2

# Quando timer <= 5 sec, spam tap
SPAM_WHEN_REMAINING_SEC = 5
SPAM_TAPS_PER_TICK = 8          # quanti tap per "tick"
SPAM_TICK_SLEEP = 0.02          # sleep fra tap durante lo spam

# Se non leggiamo più il timer per N tick, consideriamo "finito"
# Per-state timeouts (seconds).
TIMEOUT_CHAT_LINK_SEC = 80.0
TIMEOUT_MAP_ICONS_SEC = 580.0
TIMEOUT_MAGNIFIER_SEC = 80.0
TIMEOUT_MARCH_SEC = 80.0
TIMEOUT_AFTER_MARCH_SEC = 80.0

STATE_TIMEOUTS = {
    "IN_CHAT_FIND_LINK": TIMEOUT_CHAT_LINK_SEC,
    "IN_MAP_FIND_HELI": TIMEOUT_MAP_ICONS_SEC,
    "IN_HELI_FIND_MAGNIFIER": TIMEOUT_MAGNIFIER_SEC,
    "IN_TEAM_FIND_MARCH": TIMEOUT_MARCH_SEC,
    "AFTER_MARCH_TAP_HELI": TIMEOUT_AFTER_MARCH_SEC,
}


# ============================================================
# ROI (fractions x1,x2,y1,y2) - tarale con i tuoi screenshot
# ============================================================

# Link "Explore Treasure" nella chat: zona chat centrale
#ROI_CHAT_LINK = (0.0, 1.0, 0.0, 1.0)
ROI_CHAT_LINK = (0.02, 0.98, 0.18, 0.92)

# Elicottero sulla mappa: spesso centro/basso
ROI_HELI = (0.0, 1.0, 0.0, 1.0) #backup: (0.15, 0.85, 0.25, 0.80)

# Lente "Explore" sotto elicottero: zona sotto elicottero
ROI_MAGNIFIER = (0.0, 1.0, 0.0, 1.0) #backup: (0.35, 0.65, 0.55, 0.85)

# Bottone March: pop-up centrale/basso
ROI_MARCH = (0.0, 1.0, 0.0, 1.0) #backup: (0.20, 0.80, 0.55, 0.90)

# Timer sopra (come nel tuo screenshot con lente e timer vicino/ sopra)
# Metti qui la fascia dove compare tipo "01:59:54" ecc
ROI_TIMER_TEXT = (0.33, 0.67, 0.39, 0.49)

# Zona “tra elicottero e timer” dove vuoi spammare tap (random) quando <= 5 sec
ROI_SPAM_TAP = (0.35, 0.65, 0.35, 0.70)

# Icona "Headquarters" in basso a destra nella mappa
# (zona fissa, tappo al centro)
ROI_HEADQUARTERS_BTN = (0.78, 0.98, 0.78, 0.98)

# ============================================================
# OCR (opzionale)
# ============================================================

_HAS_TESSERACT = False
try:
    import pytesseract  # type: ignore
    _HAS_TESSERACT = True
except Exception:
    _HAS_TESSERACT = False


# ============================================================
# UTIL BASE
# ============================================================

def adb_tap(x: int, y: int):
    subprocess.run([ADB_CMD, "shell", "input", "tap", str(x), str(y)],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def load_image(path: str):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img

def crop_roi(img, roi_frac: Tuple[float, float, float, float]):
    h, w = img.shape[:2]
    x1, x2, y1, y2 = roi_frac
    xs = int(w * x1); xe = int(w * x2)
    ys = int(h * y1); ye = int(h * y2)
    xs = max(0, min(xs, w - 1))
    xe = max(xs + 1, min(xe, w))
    ys = max(0, min(ys, h - 1))
    ye = max(ys + 1, min(ye, h))
    return img[ys:ye, xs:xe], (xs, ys, xe, ye)

def load_templates_from_dir(directory: str) -> List[Tuple[str, np.ndarray]]:
    templates = []
    if not os.path.isdir(directory):
        return templates
    for name in sorted(os.listdir(directory)):
        if not name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            continue
        full = os.path.join(directory, name)
        im = cv2.imread(full, cv2.IMREAD_COLOR)
        if im is None:
            continue
        templates.append((name, im))
    return templates

def match_any_exact_sqdiff(
    roi_img: np.ndarray,
    templates: List[Tuple[str, np.ndarray]]
):
    """
    Matching 1:1 pixel-exact usando TM_SQDIFF_NORMED.
    Ritorna il template più simile (diff più basso).
    """
    roi_g = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)

    best_name = None
    best_diff = 1.0
    best_loc = (0, 0)
    best_hw = (0, 0)

    for name, tmpl in templates:
        tmpl_g = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
        th, tw = tmpl_g.shape[:2]
        rh, rw = roi_g.shape[:2]

        if rh < th or rw < tw:
            continue

        res = cv2.matchTemplate(
            roi_g, tmpl_g, cv2.TM_SQDIFF_NORMED
        )
        min_val, _, min_loc, _ = cv2.minMaxLoc(res)

        if min_val < best_diff:
            best_diff = float(min_val)
            best_name = name
            best_loc = min_loc
            best_hw = (th, tw)

    return best_name, best_diff, best_loc, best_hw

def match_any(roi_img: np.ndarray, templates: List[Tuple[str, np.ndarray]]):
    best_name = None
    best_score = 0.0
    best_loc = (0, 0)
    best_hw = (0, 0)

    for name, tmpl in templates:
        th, tw = tmpl.shape[:2]
        rh, rw = roi_img.shape[:2]
        if rh < th or rw < tw:
            continue
        res = cv2.matchTemplate(roi_img, tmpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        if score > best_score:
            best_score = float(score)
            best_name = name
            best_loc = loc
            best_hw = (th, tw)
    return best_name, best_score, best_loc, best_hw

def tap_match_in_fullscreen(roi_coords, match_loc, tmpl_hw):
    xs, ys, xe, ye = roi_coords
    mx, my = match_loc
    th, tw = tmpl_hw
    cx = xs + mx + tw // 2
    cy = ys + my + th // 2
    adb_tap(cx, cy)
    time.sleep(0.8)
    return cx, cy

def tap_center_of_roi(img, roi_frac):
    _, (xs, ys, xe, ye) = crop_roi(img, roi_frac)
    cx = (xs + xe) // 2
    cy = (ys + ye) // 2
    adb_tap(cx, cy)
    time.sleep(0.8)
    return cx, cy

def parse_timer_text_to_seconds(text: str) -> Optional[int]:
    """
    Accetta formati tipo:
      01:59:54
      00:29:52
      5:03  (se mai)
    """
    text = text.strip()
    # Prendi prima occorrenza tipo H+:MM:SS o MM:SS
    m = re.search(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", text)
    if not m:
        return None
    a = int(m.group(1))
    b = int(m.group(2))
    c = int(m.group(3)) if m.group(3) else None
    if c is None:
        # MM:SS
        return a * 60 + b
    # HH:MM:SS
    return a * 3600 + b * 60 + c

def log_event(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def spam_tap_in_roi(img, roi_frac, taps: int):
    h, w = img.shape[:2]
    x1, x2, y1, y2 = roi_frac
    xs = int(w * x1); xe = int(w * x2)
    ys = int(h * y1); ye = int(h * y2)
    xs = max(0, min(xs, w - 1))
    xe = max(xs + 1, min(xe, w))
    ys = max(0, min(ys, h - 1))
    ye = max(ys + 1, min(ye, h))

    for _ in range(taps):
        x = random.randint(xs, xe - 1)
        y = random.randint(ys, ye - 1)
        adb_tap(x, y)
        time.sleep(SPAM_TICK_SLEEP)

# ============================================================
# FLOW STATE MACHINE
# ============================================================

class FlowState(Enum):
    IDLE = 0
    GO_CHAT = 1
    IN_CHAT_FIND_LINK = 2
    IN_MAP_FIND_HELI = 3
    IN_HELI_FIND_MAGNIFIER = 4
    IN_TEAM_FIND_MARCH = 5
    AFTER_MARCH_TAP_HELI = 6
    DIGGING_WAIT_TIMER = 7
    DIGGING_SPAM = 8
    RETURN_HQ = 9

class TreasureFlow:
    """
    Uso:
      flow = TreasureFlow(log_fn=...)
      flow.trigger()  # quando il treasure è stato rilevato
      flow.step(img)  # chiamata ad ogni tick con lo screenshot corrente
    """
    def __init__(self, log_fn=print, stop_record_fn=None):
        os.makedirs("debug/treasure", exist_ok=True)
        self.log = log_fn
        self.stop_record_fn = stop_record_fn
        self.state = FlowState.IDLE
        self.last_action_ts = 0.0
        self.timer_missing_ticks = 0
        self.timer_disappeared_ts = None
        self.state_enter_ts = time.time()
        self.flow_start_ts = None
        self.debug_dir = None
        self.debug_counter = 0
        self.debug_last_state = None
        self.debug_last_miss_ts = 0.0

        self.t_chat = load_templates_from_dir(CHAT_LINK_DIR)
        self.t_heli = load_templates_from_dir(HELI_DIR)
        self.t_mag  = load_templates_from_dir(MAGNIFIER_DIR)
        self.t_march= load_templates_from_dir(MARCH_DIR)
        self.t_chat_ui = load_templates_from_dir(os.path.join(TEMPL_DIR, "chat_ui"))

        if not _HAS_TESSERACT:
            self.log("[TREASURE-FLOW] OCR timer: pytesseract NON disponibile (spam <=5s disabilitato)")

    def debug_snapshot(self, img, label):
        if not DEBUG or img is None or not self.debug_dir:
            return

        self.debug_counter += 1
        filename = (
            f"{self.debug_counter:04d}_"
            f"{self.state.name}_{label}.png"
        )
        path = os.path.join(self.debug_dir, filename)

        if cv2.imwrite(path, img):
            self.log(f"[TREASURE-DEBUG][SCREENSHOT] {path}")
        else:
            self.log(f"[TREASURE-DEBUG][ERROR] cannot save {path}")

    def debug_match_miss(self, img, label, name, score, threshold):
        if not DEBUG or (name and score >= threshold):
            return

        now = time.monotonic()

        if now - self.debug_last_miss_ts < DEBUG_MISS_SNAPSHOT_SEC:
            return

        self.debug_last_miss_ts = now

        self.log(
            f"[TREASURE-DEBUG][MISS] "
            f"state={self.state.name} "
            f"target={label} "
            f"score={score:.3f} "
            f"threshold={threshold:.3f}"
        )

        self.debug_snapshot(img, f"miss_{label}")

    def debug_match(self, label, name, score, threshold, loc, hw, img=None):
        if not DEBUG:
            return

        self.log(
            f"[TREASURE-DEBUG][MATCH] "
            f"state={self.state.name} "
            f"elapsed={self._seconds_in_state():.2f}s "
            f"target={label} "
            f"template={name} "
            f"score={score:.3f} "
            f"threshold={threshold:.3f} "
            f"matched={bool(name and score >= threshold)} "
            f"loc={loc} hw={hw}"
        )

        if img is not None:
            self.debug_match_miss(img, label, name, score, threshold)

    def read_timer_seconds(self, img) -> Optional[int]:
        if not _HAS_TESSERACT:
            return None
        roi, _ = crop_roi(img, ROI_TIMER_TEXT)
        # preprocessing per OCR: grayscale + threshold
        g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        g = cv2.GaussianBlur(g,(3,3),0)
        _, th = cv2.threshold(g, 150, 255, cv2.THRESH_BINARY)
        # whitelist per ridurre rumore
        config = "--psm 7 -c tessedit_char_whitelist=0123456789:"
        txt = pytesseract.image_to_string(th, config=config)  # type: ignore
        self.log(f"[TREASURE-FLOW] OCR raw='{txt}'")
        sec = parse_timer_text_to_seconds(txt)

        if DEBUG:
            self.log(
                f"[TREASURE-DEBUG][OCR] "
                f"state={self.state.name} "
                f"raw={txt!r} seconds={sec}"
            )

        cv2.imwrite("debug/treasure/timer_roi.png", roi)
        cv2.imwrite("debug/treasure/timer_thresh.png", th)

        return sec

    def trigger(self, coords=None, loc=None, hw=None):
    
        if not WORKFLOW_MANAGER.acquire(Workflow.TREASURE):
            return
    
        self.start_coords = coords
        self.start_loc = loc
        self.start_hw = hw

        self.flow_start_ts = time.time()
        self._timer_wait_started = None

        if DEBUG:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self.debug_dir = os.path.join(
                "debug", "treasure", f"run_{run_id}"
            )
            os.makedirs(self.debug_dir, exist_ok=True)
            self.debug_counter = 0
            self.debug_last_state = None
            self.log(f"[TREASURE-DEBUG][RUN] {self.debug_dir}")
        self.set_state(FlowState.GO_CHAT)
        self.timer_missing_ticks = 0
        self.timer_disappeared_ts = None
        self.log("[TREASURE-FLOW] trigger -> GO_CHAT")

    def set_state(self, new_state: FlowState):
        previous = self.state
        elapsed = time.time() - self.state_enter_ts

        self.state = new_state
        self.state_enter_ts = time.time()

        self.log(
            f"[TREASURE-DEBUG][STATE] "
            f"{previous.name} -> {new_state.name} "
            f"previous_duration={elapsed:.2f}s"
        )

        self.debug_last_state = None
        self.debug_last_miss_ts = 0.0

    def _seconds_in_state(self):
        return time.time() - self.state_enter_ts

    def _finish(self, reason):
        self.log(f"[TREASURE-FLOW] finish: {reason}")

        if self.stop_record_fn:
            self.stop_record_fn()

        self.set_state(FlowState.IDLE)
        self.flow_start_ts = None
        self.timer_disappeared_ts = None
        WORKFLOW_MANAGER.release(Workflow.TREASURE)

    def _check_state_timeout(self):
        timeout = STATE_TIMEOUTS.get(self.state.name)

        if timeout is None:
            return False

        elapsed = self._seconds_in_state()

        if elapsed <= timeout:
            return False

        self._finish(
            f"state timeout {self.state.name} "
            f"elapsed={elapsed:.1f}s limit={timeout:.1f}s"
        )
        return True

    def _cooldown_ok(self) -> bool:
        return (time.time() - self.last_action_ts) >= ACTION_COOLDOWN_SEC

    def _mark_action(self):
        self.last_action_ts = time.time()

    def step(self, img):
        if self.state == FlowState.IDLE:
            return

        if DEBUG and self.debug_last_state != self.state:
            self.debug_snapshot(img, "state_entry")
            self.debug_last_state = self.state

        # --------------------------------------------------
        # PER-STATE TIMEOUT
        # --------------------------------------------------
        if self._check_state_timeout():
            return

        # sicurezza anti-ripetizione
        if not self._cooldown_ok() and self.state not in (FlowState.DIGGING_SPAM,):
            return

        # -------------------------
        # 1) GO_CHAT: a questo punto tu hai già tappato il tesoro altrove o lo farai qui
        # Se vuoi farlo qui: aggiungi template del "tesoro" e tap.
        # Per ora consideriamo che dopo trigger tu sia già finito in chat o stai per finirci.
        # -------------------------
        if self.state == FlowState.GO_CHAT:
            if self.start_coords is None:
                self.log("[TREASURE-FLOW] missing coords")
                self._finish("flow stopped")
                return

            cx, cy = tap_match_in_fullscreen(
                self.start_coords,
                self.start_loc,
                self.start_hw
            )
        
            self.log(f"[TREASURE-FLOW] tap treasure @ {cx},{cy}")
        
            time.sleep(1.2)

            # Piccolo delay e passa a cercare il link in chat
            self.set_state(FlowState.IN_CHAT_FIND_LINK)
            self._mark_action()
            self.log("[TREASURE-FLOW] -> IN_CHAT_FIND_LINK")
            return

        # -------------------------
        # 2) IN_CHAT_FIND_LINK: trova "Explore Treasure" e tappalo
        # -------------------------
        if self.state == FlowState.IN_CHAT_FIND_LINK:
            # --- CHECK: siamo ancora in chat? ---
            roi_chat_ui, _ = crop_roi(img, (0.0, 1.0, 0.0, 1.0))  # zona input chat
            name_ui, score_ui, loc_ui, hw_ui = match_any(
                roi_chat_ui, self.t_chat_ui
            )
            self.debug_match(
                "chat_ui", name_ui, score_ui, 0.6, loc_ui, hw_ui, img
            )

            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            debug_dir = "debug/chat_ui"
            os.makedirs(debug_dir, exist_ok=True)
        
            if not (name_ui and score_ui >= 0.6):
                # Do not infer navigation from a missing UI template.
                self.log(
                    f"[TREASURE-FLOW] chat UI uncertain "
                    f"score={score_ui:.3f}; checking treasure link"
                )

            if not self.t_chat:
                self.log("[TREASURE-FLOW] manca template chat_link/, stop flow")
                self._finish("flow stopped")
                return

            roi, coords = crop_roi(img, ROI_CHAT_LINK)

            self.log(f"[DEBUG] ROI_CHAT_LINK: x={coords[0]}, y={coords[1]}, w={roi.shape[1]}, h={roi.shape[0]}")

            name, score, loc, hw = match_any(roi, self.t_chat)
            self.debug_match("chat_link", name, score, THR_CHAT_LINK, loc, hw, img)
            if name:
                self.log(f"[DEBUG] Match trovato: template={name}, score={score:.3f}, pos={loc}, dim={hw}")
            else:
                self.log(f"[DEBUG] Nessun match nella ROI della chat (link non trovato)")

            if name and score >= THR_CHAT_LINK:
                xs, ys, _, _ = coords
                cx, cy = tap_match_in_fullscreen(coords, loc, hw)
                self._mark_action()
                self.log(f"[TREASURE-FLOW] chat link tap @ {cx},{cy} score={score:.3f} thr={THR_CHAT_LINK}")
                time.sleep(1.0)
                self.set_state(FlowState.IN_MAP_FIND_HELI)
            else:
                    self.log(f"[TREASURE-FLOW] link non ancora valido: score={score:.3f} < thr={THR_CHAT_LINK}")

            cv2.imwrite("debug/treasure/debug_roi_chat_link.png", roi)

            return

        # -------------------------
        # 3) IN_MAP_FIND_HELI: trova elicottero e tappalo
        # -------------------------
        if self.state == FlowState.IN_MAP_FIND_HELI:
            if not self.t_heli:
                self.log("[TREASURE-FLOW] manca template helicopter/, stop flow")
                self._finish("flow stopped")
                return

            roi, coords = crop_roi(img, ROI_HELI)
            name, score, loc, hw = match_any(roi, self.t_heli)
            self.debug_match("helicopter", name, score, THR_HELI, loc, hw, img)
            if name and score >= THR_HELI:
                cx, cy = tap_match_in_fullscreen(coords, loc, hw)
                self._mark_action()
                self.log(f"[TREASURE-FLOW] heli tap @ {cx},{cy} score={score:.3f} thr={THR_HELI}")
                self.set_state(FlowState.IN_HELI_FIND_MAGNIFIER)
            return

        # -------------------------
        # 4) IN_HELI_FIND_MAGNIFIER: trova la lente "Explore" e tappala
        # -------------------------
        if self.state == FlowState.IN_HELI_FIND_MAGNIFIER:
            if not self.t_mag:
                self.log("[TREASURE-FLOW] manca template magnifier/, stop flow")
                self._finish("flow stopped")
                return

            roi, coords = crop_roi(img, ROI_MAGNIFIER)
            name, score, loc, hw = match_any(roi, self.t_mag)
            self.debug_match("magnifier", name, score, THR_MAGNIFIER, loc, hw, img)
            if name and score >= THR_MAGNIFIER:
                cx, cy = tap_match_in_fullscreen(coords, loc, hw)
                self._mark_action()
                self.log(f"[TREASURE-FLOW] magnifier tap @ {cx},{cy} score={score:.3f} thr={THR_MAGNIFIER}")
                self.set_state(FlowState.IN_TEAM_FIND_MARCH)
            return

        # -------------------------
        # 5) IN_TEAM_FIND_MARCH: tap sul bottone March (team già selezionato di default)
        # -------------------------
        if self.state == FlowState.IN_TEAM_FIND_MARCH:
            if not self.t_march:
                self.log("[TREASURE-FLOW] manca template march/, stop flow")
                self._finish("flow stopped")
                return

            roi, coords = crop_roi(img, ROI_MARCH)
            name, score, loc, hw = match_any(roi, self.t_march)
            self.debug_match("march", name, score, THR_MARCH, loc, hw, img)
            if name and score >= THR_MARCH:
                cx, cy = tap_match_in_fullscreen(coords, loc, hw) 
                self._mark_action()
                self.log(f"[TREASURE-FLOW] MARCH tap @ {cx},{cy} score={score:.3f} thr={THR_MARCH}")
                self.set_state(FlowState.AFTER_MARCH_TAP_HELI)
                self.timer_missing_ticks = 0
            return

        # -------------------------
        # AFTER_MARCH_TAP_HELI
        # -------------------------
        if self.state == FlowState.AFTER_MARCH_TAP_HELI:
        
            roi, coords = crop_roi(img, ROI_HELI)
            name, score, loc, hw = match_any(roi, self.t_heli)
            self.debug_match("helicopter", name, score, THR_HELI, loc, hw, img)
        
            if name and score >= THR_HELI:
                cx, cy = tap_match_in_fullscreen(coords, loc, hw)
        
                self.log(f"[TREASURE-FLOW] second heli tap @ {cx},{cy}")
        
                self._mark_action()
        
                time.sleep(1.2)
        
                self.set_state(FlowState.DIGGING_WAIT_TIMER)
        
            return

        # -------------------------
        # 6) DIGGING_WAIT_TIMER: aspetta timer OCR; se <=5 -> SPAM
        # -------------------------
        if self.state == FlowState.DIGGING_WAIT_TIMER:
            sec = self.read_timer_seconds(img)

            if sec is None:
                self.timer_missing_ticks += 1

                if self._timer_wait_started is None:
                    self._timer_wait_started = time.monotonic()

                elapsed = time.monotonic() - self._timer_wait_started

                if elapsed >= 90.0:
                    self.log(
                        "[TREASURE-FLOW] timer OCR unavailable for "
                        f"{elapsed:.1f}s -> abort"
                    )
                    self._finish("timer OCR timeout")
                    return

                return

            self._timer_wait_started = time.monotonic()

            # Timer presente
            self.timer_missing_ticks = 0
            self.log(f"[TREASURE-FLOW] timer={sec}s")

            if sec <= SPAM_WHEN_REMAINING_SEC and _HAS_TESSERACT:
                self.set_state(FlowState.DIGGING_SPAM)
                self._mark_action()

                duration = float(sec) + 1.0
                spam_start = time.monotonic()
                deadline = spam_start + duration
                taps = 0

                self.log(
                    f"[TREASURE-FLOW] timer={sec}s "
                    f"-> continuous spam for {duration:.1f}s"
                )

                while (
                    time.monotonic() < deadline
                    and self.state == FlowState.DIGGING_SPAM
                ):
                    spam_tap_in_roi(img, ROI_SPAM_TAP, 1)
                    taps += 1

                self.log(
                    f"[TREASURE-FLOW] spam completed "
                    f"taps={taps} "
                    f"planned={duration:.1f}s "
                    f"actual={time.monotonic() - spam_start:.2f}s"
                )

                self.set_state(FlowState.RETURN_HQ)
                self._mark_action()

            return

        # -------------------------
        # 7) DIGGING_SPAM: tap come un demonio nella zona fra elicottero e timer
        #    Stop quando timer finisce (non leggibile per N tick)
        # -------------------------
        if self.state == FlowState.DIGGING_SPAM:
            # Tapping is executed immediately when entering this state.
            # The state is normally completed inside DIGGING_WAIT_TIMER.
            return

        # -------------------------
        # 8) RETURN_HQ: tappa Headquarters e reset
        # -------------------------
        if self.state == FlowState.RETURN_HQ:
            cx, cy = tap_center_of_roi(img, ROI_HEADQUARTERS_BTN)
            self._mark_action()
            self.log(f"[TREASURE-FLOW] tap Headquarters @ {cx},{cy} -> IDLE")
            self._finish("flow completed")

            return


# ============================================================
# THREAD READY-TO-USE (tu lo colleghi al tuo screenshot_producer)
# ============================================================

def treasure_flow_watcher(stop_evt: threading.Event,
                         screenshot_path: str,
                         screenshot_lock: Optional[threading.Lock] = None,
                         log_fn=print):
    """
    - screenshot_path: lo stesso file che aggiorna screenshot_producer
    - screenshot_lock: se nel tuo main hai un lock globale, passalo qui
    """
    flow = TreasureFlow(log_fn=log_fn)

    # espone una funzione comoda: chiamala dal tuo treasure_watcher quando rilevi
    # esempio:
    #   flow.trigger()
    treasure_flow_watcher.flow = flow  # type: ignore

    while not stop_evt.is_set():
        if screenshot_lock:
            with screenshot_lock:
                img = load_image(screenshot_path)
        else:
            img = load_image(screenshot_path)

        if img is not None:
            flow.step(img)

        time.sleep(FLOW_TICK_SEC)

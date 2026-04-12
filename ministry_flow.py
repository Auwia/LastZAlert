#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import re
import os
import cv2
import pytesseract
import subprocess

from enum import Enum
from typing import Optional

from workflow_manager import Workflow, WORKFLOW_MANAGER
from bot_utils import adb_tap, crop_roi, match_any, load_templates, load_image

# ============================================================
# DEBUG
# ============================================================

DEBUG = False
DEBUG_DIR = "debug/ministry"
os.makedirs(DEBUG_DIR, exist_ok=True)

ADB_CMD = "adb"

def dbg_log(log_fn, msg):
    if DEBUG:
        log_fn(msg)

# ============================================================
# CONFIG
# ============================================================

THR = 0.85
ACTION_COOLDOWN_SEC = 1.0
OFFICER_TOTAL_SEC = 10 * 60  # 10 minuti

ROI_APPOINTMENT_LIST = (0.10, 0.90, 0.52, 0.62)
ROI_APPLICATION_NOTE = (0.18, 0.82, 0.74, 0.86)
ROI_OFFICER_NICKNAME = (0.0, 1.0, 0.0, 1.0)
ROI_OFFICER_DURATION = (0.18, 0.78, 0.34, 0.46)

CENTER_SCREEN = (540, 960)
BOTTOM_LEFT = (100, 2400)
BOTTOM_LEFT_PIXEL = (80, 2200)
BOTTOM_RIGHT = (1000, 2400)

# ============================================================
# OCR
# ============================================================
def _normalize_time_text(txt: str) -> str:
    if not txt:
        return ""

    txt = txt.strip()

    # rimuove spazi attorno ai due punti
    txt = re.sub(r"\s*:\s*", ":", txt)

    # rimuove spazi tra numeri (es: "4 8" → "48")
    txt = re.sub(r"(?<=\d)\s+(?=\d)", "", txt)

    # rimuove caratteri strani tipo newline
    txt = txt.replace("\n", "").replace("\r", "")

    return txt

def _parse_hhmmss(txt: str) -> Optional[int]:
    txt = _normalize_time_text(txt)

    m = re.search(r"(\d{1,2}:\d{2}:\d{2})", txt)
    if not m:
        return None

    h, m_, s = map(int, m.group(1).split(":"))
    return h*3600 + m_*60 + s

def _ocr_duration(img):
    if img is None or img.size == 0:
        return ""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)[1]
    config = "--psm 7 -c tessedit_char_whitelist=0123456789:"
    return pytesseract.image_to_string(gray, config=config)

def _parse_application_note(txt: str) -> Optional[int]:
    txt = txt.strip()

    txt = _normalize_time_text(txt)

    # cerca QUALSIASI hh:mm:ss nel testo
    m = re.search(r"(\d{1,2}:\d{2}:\d{2})", txt)
    if not m:
        return None

    h, m_, s = map(int, m.group(1).split(":"))
    return int(time.time() + h * 3600 + m_ * 60 + s)

def _parse_application_note_bkp(txt: str) -> Optional[int]:
    """
    Ritorna timestamp UNIX del momento in cui l'incarico parte
    """
    txt = txt.strip()

    # Caso A: "will take office in 01:23:24"
    m = re.search(r"in\s+(\d+):(\d+):(\d+)", txt)
    if m:
        h, m_, s = map(int, m.groups())
        return int(time.time() + h*3600 + m_*60 + s)

    # Caso B: "will take office at 2026-01-29 19:23:40"
    m = re.search(r"at\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})", txt)
    if m:
        dt = time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        return int(time.mktime(dt))

    return None

def _ocr_application_note_yellow(img):
    """
    OCR specifico per:
    'Application approved will take office in 00:xx:xx'
    (testo giallo con glow)
    """
    if img is None or img.size == 0:
        return ""

    # Converti in HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Maschera giallo
    lower_yellow = (15, 80, 120)
    upper_yellow = (40, 255, 255)
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Chiudi i buchi nei numeri
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Nero su bianco
    mask = cv2.bitwise_not(mask)

    # Ingrandisci (FONDAMENTALE)
    mask = cv2.resize(mask, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    config = "--psm 7 -c tessedit_char_whitelist=0123456789:"
    return pytesseract.image_to_string(mask, config=config)

def _ocr_application_note(img):
    if img is None or img.size == 0:
        return ""

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # range giallo (testato su UI simili)
    lower_yellow = (15, 80, 120)
    upper_yellow = (40, 255, 255)

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # testo bianco su nero
    mask = cv2.bitwise_not(mask)

    # ingrandisci per OCR
    mask = cv2.resize(mask, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    config = "--psm 6 -c tessedit_char_whitelist=0123456789:"
    return pytesseract.image_to_string(mask, config=config)

def _ocr_time_only_bkp(img):
    if img is None or img.size == 0:
        return ""

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # leggero blur per togliere glow
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # adaptive threshold (molto meglio qui)
    gray = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        31, 5
    )

    config = "--psm 7 -c tessedit_char_whitelist=0123456789:"
    return pytesseract.image_to_string(gray, config=config)

def _ocr_text(img):
    if img is None or img.size == 0:
        return ""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)[1]
    return pytesseract.image_to_string(gray, config="--psm 6")

def _parse_scheduled(txt: str) -> int:
    m = re.search(r"\((\d+)\s*/\s*50\)", txt)
    return int(m.group(1)) if m else 0

def adb_keyevent(code: int):
    subprocess.run(
        [ADB_CMD, "shell", "input", "keyevent", str(code)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

# ============================================================
# TEMPLATES
# ============================================================

def _load_ministry_templates():
    return {
        "map":                 load_templates("ministry/map.png"),
        "search":              load_templates("ministry/search.png"),
        "special":             load_templates("ministry/special.png"),
        "go_capital":          load_templates("ministry/go_capital.png"),
        "pres_palace":         load_templates("ministry/pres_palace.png"),
        "position":            load_templates("ministry/position.png"),
        "sec_constr":          load_templates("ministry/sec_construction.png"),
        "sec_science":         load_templates("ministry/sec_science.png"),
        "apply":               load_templates("ministry/apply.png"),
        "confirm":             load_templates("ministry/confirm.png"),
        "sec_constr_title":    load_templates("ministry/sec_constr_title.png"),
        "sec_science_title":   load_templates("ministry/sec_science_title.png"),
        "go_hq":               load_templates("ministry/go_hq.png"),
        "nickname":            load_templates("ministry/nickname_auwia81.png"),
        "science_icon":        load_templates("ministry/science_icon/science_icon.png"),
        "construction_icon":   load_templates("ministry/construction_icon/construction_icon.png"),
    }

# ============================================================
# STATE MACHINE
# ============================================================

class MinistryState(Enum):
    IDLE = 0
    START = 1
    TAP_MAP = 2
    TAP_SEARCH = 3
    TAP_SPECIAL = 4
    TAP_GO = 5
    TAP_CENTER = 6
    TAP_PALACE = 7
    TAP_POSITION = 8
    TAP_CONSTRUCTION = 9
    READ_X = 10
    BACK_FROM_X = 11
    SCROLL_UP = 12
    TAP_SCIENCE = 13
    READ_Y = 14
    APPLY_SCIENCE = 15
    APPLY_CONSTRUCTION = 16
    APPLY_CONSTRUCTION_DONE = 17
    APPLY_SCIENCE_DONE = 18
    BACK_TO_HQ = 19
    DONE = 20
    CONFIRM = 21
    TAP_GO_HQ = 22
    READ_APPLICATION_NOTE = 23
    EXIT_MINISTRY = 24

# ============================================================
# MINISTRY FLOW
# ============================================================

class MinistryFlow:
    def _schedule_confirm_popup_cleanup(self, delay_sec: int):
        import threading
    
        # anticipo + ritardo (tolleranza)
        early = max(0, delay_sec - 10)
        late  = delay_sec + 20
    
        def worker():
            self.log(f"[MINISTRY] popup cleanup scheduled in {early}s")
    
            time.sleep(early)
    
            start = time.time()
            timeout = late - early
    
            while time.time() - start < timeout:
                try:
                    ctx = self.screenshot_ctx
                    SCREENSHOT_PATH = ctx["path"]
                    SCREENSHOT_LOCK = ctx["lock"]
                    load_image = ctx["load_image"]

                    from simple_events import SIMPLE_EVENTS
                    from bot_utils import match_any
    
                    with SCREENSHOT_LOCK:
                        img = load_image(SCREENSHOT_PATH)
    
                    if img is None:
                        time.sleep(0.5)
                        continue
    
                    cfg = SIMPLE_EVENTS["confirm_popup"]
                    roi, coords = crop_roi(img, cfg["roi"])
                    templates = load_templates(cfg["templates"])
    
                    name, score, loc, hw = match_any(roi, templates)
    
                    if score >= cfg["threshold"]:
                        self.log("[MINISTRY] confirm popup detected → closing")
                        adb_tap(50, 50)  # OUTSIDE
                        return
    
                except Exception as e:
                    self.log(f"[MINISTRY] popup watcher error: {e}")
    
                time.sleep(0.5)
    
            self.log("[MINISTRY] popup cleanup timeout")
    
        threading.Thread(target=worker, daemon=True).start()

    def _tap_template(self, img, key, next_state=None, offset=(0,0), score_thr=THR):
        name, score, loc, hw = match_any(img, self.templates[key])
        if name and score >= score_thr:
            cx = loc[0] + hw[1] // 2 + offset[0]
            cy = loc[1] + hw[0] // 2 + offset[1]
            adb_tap(cx, cy)
            time.sleep(0.4)
            self.log(f"[MINISTRY] tap {key} score={score:.3f}")
            if next_state:
                self.state = next_state
            self._mark_action()
            return True
        return False

    def _precheck_exit(self, img) -> bool:
        if self._construction_icon_present(img): 
            self.log("[MINISTRY] precheck: already construction officer")
            return True
            
        if self._science_icon_present(img): 
            self.log("[MINISTRY] precheck: already science officer")
            return True
            
        if self._is_current_officer(img):
            self.log("[MINISTRY] precheck: already officer")
            return True

        if self._handle_current_officer(img):
            self.log("[MINISTRY] precheck: already officer")
            return True
    
        #if self._application_note_visible(img):
        #    self.log("[MINISTRY] precheck: application note visible")
        #    return True
    
        if self._already_applied(img):
            self.log("[MINISTRY] precheck: already applied (green row)")
            return True
    
        return False

    def _construction_icon_present(self, img) -> bool:
        name, score, _, _ = match_any(img, self.templates["construction_icon"])
        return name is not None and score >= 0.85
        
    def _science_icon_present(self, img) -> bool:
        name, score, _, _ = match_any(img, self.templates["science_icon"])
        return name is not None and score >= 0.85

    def _handle_current_officer(self, img) -> bool:
        if not self._is_current_officer(img):
            return False
    
        roi, _ = crop_roi(img, ROI_OFFICER_DURATION)
        txt = _ocr_duration(roi)
        elapsed = _parse_hhmmss(txt)
    
        if elapsed is None:
            # fallback: se non leggo, metto comunque un cooldown "breve"
            self.log("[MINISTRY] officer duration OCR failed → cooldown 10 min")
            self.cooldown_until = int(time.time()) + OFFICER_TOTAL_SEC
            return True
    
        remaining = max(0, OFFICER_TOTAL_SEC - elapsed)
        # buffer piccolo per sicurezza
        self.cooldown_until = int(time.time()) + remaining + 10
        self.log(f"[MINISTRY] already officer → remaining {remaining}s, cooldown set")
    
        return True

    def _is_current_officer(self, img) -> bool:
        roi, coords = crop_roi(img, ROI_OFFICER_NICKNAME)
    
        if roi is None or roi.size == 0:
            return False
    
        name, score, _, _ = match_any(roi, self.templates["nickname"])
    
        if DEBUG:
            ts = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(
                f"debug/ministry/nickname_roi_{name}_{score:.3f}_{ts}.png",
                roi
            )
            self.log(f"[MINISTRY][DEBUG] nickname match={name} score={score:.3f}")
    
        return name is not None and score >= 0.85

    def _application_note_visible(self, img) -> bool:
        roi, _ = crop_roi(img, ROI_APPLICATION_NOTE)
        if roi is None or roi.size == 0:
            return False
        
        cv2.imwrite(f"debug/ministry/application_note_visibile.png", roi)
        
        txt = _ocr_application_note_yellow(roi)
        self.log(f"[MINISTRY] NOTE raw OCR: {repr(txt)}")
        #return "will take office" in txt.lower()
        return _parse_hhmmss(txt) is not None

    def _already_applied(self, img) -> bool:
        """
        Rileva se l'utente è già nella lista (riga verde con cestino).
        NO OCR. Basato su colore.
        """
        roi, _ = crop_roi(img, (0.10, 0.90, 0.62, 0.82))  # zona lista
        if roi is None or roi.size == 0:
            return False
    
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
        # verde selezione riga
        lower = (35, 40, 40)
        upper = (85, 255, 255)
    
        mask = cv2.inRange(hsv, lower, upper)
        ratio = cv2.countNonZero(mask) / mask.size
    
        return ratio > 0.03

    def __init__(self, log_fn=print, screenshot_ctx=None):
        self.log = log_fn
        self.screenshot_ctx = screenshot_ctx
        self.state = MinistryState.IDLE
        self.last_action_ts = 0.0
        self.templates = _load_ministry_templates()
        self.x = None
        self.y = None
        self.xy_read = False
        self.cooldown_until = 0
        self.returning_to_construction = False
        self.log("[MINISTRY-FLOW] inizializzato")

    def _cooldown_ok(self) -> bool:
        return (time.time() - self.last_action_ts) >= ACTION_COOLDOWN_SEC

    def _mark_action(self):
        self.last_action_ts = time.time()

    def trigger(self):
        if self.state != MinistryState.IDLE:
            return
    
        now = time.time()
        if now < self.cooldown_until:
            return
    
        if not WORKFLOW_MANAGER.acquire(Workflow.MINISTRY):
            return
    
        time.sleep(0.3)
        self.state = MinistryState.TAP_MAP
        self.started_ts = time.time()
        self._mark_action()
        self.log("[MINISTRY-FLOW] trigger -> TAP_MAP")

    def step(self, img):
        if (
            self.state not in (MinistryState.IDLE, MinistryState.DONE)
            and hasattr(self, "started_ts")
            and time.time() - self.started_ts > 220
        ):
            self.log(f"[MINISTRY][WATCHDOG] timeout in state={self.state.name}")
            self.xy_read = False
            self.state = MinistryState.IDLE
            WORKFLOW_MANAGER.release(Workflow.MINISTRY)
            if hasattr(self, "started_ts"):
                del self.started_ts
            adb_keyevent(4)
            time.sleep(0.3)
            adb_keyevent(4)
            time.sleep(0.3)
            adb_keyevent(4)
            time.sleep(0.3)
            self._tap_template(img, "go_hq")
            return

        if self.state == MinistryState.DONE:
            return
        if self.state == MinistryState.IDLE or not self._cooldown_ok():
            return

        def tap_and_next(x, y, next_state):
            adb_tap(x, y)
            time.sleep(0.1)
            self.state = next_state
            self._mark_action()

        # --- Step machine below ---
        if self.state == MinistryState.TAP_MAP:
            name, score, loc, hw = match_any(img, self.templates["map"])
            self.log(f"[MINISTRY] TAP_MAP match={name} score={score:.3f}")
        
            if name and score >= THR:
                adb_tap(loc[0] + hw[1] // 2, loc[1] + hw[0] // 2)
                time.sleep(0.4)
                self.log(f"[MINISTRY] tap map score={score:.3f}")
                self.state = MinistryState.TAP_SEARCH
                self._mark_action()
            return

        if self.state == MinistryState.TAP_SEARCH:
            if self._tap_template(img, "search", MinistryState.TAP_SPECIAL):
                return

        if self.state == MinistryState.TAP_SPECIAL:
            if self._tap_template(img, "special", MinistryState.TAP_GO):
                return

        if self.state == MinistryState.TAP_GO:
            if self._tap_template(img, "go_capital", MinistryState.TAP_CENTER):
                return

        if self.state == MinistryState.TAP_CENTER:
            tap_and_next(*CENTER_SCREEN, MinistryState.TAP_PALACE)

        if self.state == MinistryState.TAP_PALACE:
            if self._tap_template(img, "pres_palace", MinistryState.TAP_POSITION):
                return

        if self.state == MinistryState.TAP_POSITION:
            if self._tap_template(img, "position", MinistryState.TAP_CONSTRUCTION):
                return

        if self.state == MinistryState.TAP_CONSTRUCTION:
            next_state = (
                MinistryState.APPLY_CONSTRUCTION_DONE
                if self.returning_to_construction
                else MinistryState.READ_X
            )
            if self._tap_template(img, "sec_constr", next_state):
                return

        if self.state == MinistryState.READ_X:
            time.sleep(0.3)
            if self._precheck_exit(img):
                self.log("[MINISTRY] precheck exit → EXIT_MINISTRY")
                self.state = MinistryState.EXIT_MINISTRY
                self._mark_action()
                return

            if self.xy_read:
                return

            name, score, _, _ = match_any(img, self.templates["sec_constr_title"])
            if not name or score < 0.8:
                return  # popup non pronto, aspetta frame successivo
        
            roi, _ = crop_roi(img, ROI_APPOINTMENT_LIST)
            if roi is None or roi.size == 0:
                self.log("[MINISTRY] ROI vuota, attendo frame successivo")
                return

            if DEBUG:
                cv2.imwrite("debug/ministry/roi_ministry_timer.png", roi)
                cv2.imwrite("debug/ministry/ministry_full.png", img)

            if (
                self._is_current_officer(img)
                or self._application_note_visible(img)
                or self._already_applied(img)
            ):
                self.log("[MINISTRY] già ufficiale / application presente → EXIT")
                self.state = MinistryState.EXIT_MINISTRY
                self._mark_action()
                return

            txt = _ocr_text(roi)
            self.x = _parse_scheduled(txt)
            self.log(f"[MINISTRY] READ_X = {self.x}")

            if self._already_applied(img):
                self.log("[MINISTRY] già applicato dopo READ_X → vai a NOTE")
                self.state = MinistryState.READ_APPLICATION_NOTE
                self._mark_action()
                return

            self.state = MinistryState.BACK_FROM_X
            self._mark_action()
            return

        if self.state == MinistryState.BACK_FROM_X:
            adb_tap(*BOTTOM_LEFT)
            time.sleep(0.2)
            self.state = MinistryState.SCROLL_UP
            self._mark_action()
            return

        if self.state == MinistryState.SCROLL_UP:
            # Simula uno swipe: bottom-right verso top-right (scroll lista verso su)
            from bot_utils import adb_swipe
            adb_swipe(1000, 2000, 1000, 1000, 300)
            self.state = MinistryState.TAP_SCIENCE
            self._mark_action()
            return

        if self.state == MinistryState.TAP_SCIENCE:
            if self._tap_template(img, "sec_science", MinistryState.READ_Y):
                return

        if self.state == MinistryState.READ_Y:
            time.sleep(0.3)
            if self._precheck_exit(img):
                self.log("[MINISTRY] precheck exit → EXIT_MINISTRY")
                self.state = MinistryState.EXIT_MINISTRY
                self._mark_action()
                return

            # aspetta che il popup sia davvero aperto
            name, score, _, _ = match_any(img, self.templates["sec_science_title"])
            if not name or score < 0.8:
                return  # popup non pronto, aspetta frame successivo
        
            roi, _ = crop_roi(img, ROI_APPOINTMENT_LIST)
            if roi is None or roi.size == 0:
                self.log("[MINISTRY] ROI vuota, attendo frame successivo")
                return
    
            if DEBUG:
                cv2.imwrite("debug/ministry/roi_ministry_science_timer.png", roi)
                cv2.imwrite("debug/ministry/ministry_science_full.png", img)

            if (
                self._is_current_officer(img)
                or self._application_note_visible(img)
                or self._already_applied(img)
            ):
                self.log("[MINISTRY] già ufficiale / application presente → EXIT")
                self.state = MinistryState.EXIT_MINISTRY
                self._mark_action()
                return
        
            txt = _ocr_text(roi)
            self.y = _parse_scheduled(txt)
            self.log(f"[MINISTRY] READ_Y = {self.y}")
            self.xy_read = True

            if self._already_applied(img):
                self.log("[MINISTRY] già applicato (riga verde rilevata)")
                self.state = MinistryState.READ_APPLICATION_NOTE
                self._mark_action()
            else:
                if self.y <= self.x:
                    self.state = MinistryState.APPLY_SCIENCE
                    self.log("[MINISTRY] decisione: APPLY SCIENCE")
                else:
                    self.state = MinistryState.APPLY_CONSTRUCTION
                    self.state = MinistryState.APPLY_CONSTRUCTION

            self._mark_action()
            return

        # -------------------------
        # APPLY SCIENCE
        # -------------------------
        if self.state == MinistryState.APPLY_SCIENCE:
            if self._tap_template(img, "apply", score_thr=0.6):
                self.state = MinistryState.CONFIRM
                self._mark_action()
            return
        
        # -------------------------
        # APPLY CONSTRUCTION
        # -------------------------
        if self.state == MinistryState.APPLY_CONSTRUCTION:
            self.log("[MINISTRY] construction chosen → switching back to construction")
            self.returning_to_construction = True
            adb_tap(*BOTTOM_LEFT)  # chiude popup science
            time.sleep(0.3)
            self.state = MinistryState.TAP_CONSTRUCTION
            self._mark_action()
            return

        if self.state == MinistryState.APPLY_CONSTRUCTION_DONE:
            self.returning_to_construction = False
            time.sleep(0.3)
        
            # 1. già ufficiale
            if self._is_current_officer(img):
                self.log("[MINISTRY] già ufficiale (construction) → note")
                self.state = MinistryState.READ_APPLICATION_NOTE
                self._mark_action()
                return
        
            # 2. application già approvata / in corso
            if self._application_note_visible(img):
                self.log("[MINISTRY] application già in corso → note")
                self.state = MinistryState.READ_APPLICATION_NOTE
                self._mark_action()
                return
        
            # 3. già applicato (riga verde)
            if self._already_applied(img):
                self.log("[MINISTRY] già applicato (construction) → note")
                self.state = MinistryState.READ_APPLICATION_NOTE
                self._mark_action()
                return
        
            # 4. solo ora cerco Apply
            if self._tap_template(img, "apply", score_thr=0.6):
                self.state = MinistryState.CONFIRM
                self._mark_action()
            return

        if self.state == MinistryState.CONFIRM:
        
            # caso 1: application note già visibile
            if self._application_note_visible(img):
                self.log("[MINISTRY] confirm skipped → application note visible")
                self.state = MinistryState.READ_APPLICATION_NOTE
                self._mark_action()
                return
        
            # caso 2: già ufficiale (edge case raro ma reale)
            if self._is_current_officer(img):
                self.log("[MINISTRY] confirm skipped → already officer")
                self.state = MinistryState.EXIT_MINISTRY
                self._mark_action()
                return
        
            # caso 3: compare davvero il bottone Confirm
            if self._tap_template(img, "confirm", MinistryState.READ_APPLICATION_NOTE):
                time.sleep(0.4)
                return
        
            # altrimenti: aspetta frame successivo (NO timeout qui)
            return

        if self.state == MinistryState.READ_APPLICATION_NOTE:
            roi, _ = crop_roi(img, ROI_APPLICATION_NOTE)
            if roi is None or roi.size == 0:
                return
        
            if DEBUG:
                cv2.imwrite("debug/ministry/roi_application_note.png", roi)
        
            txt = _ocr_application_note_yellow(roi)
            self.log(f"[MINISTRY][OCR NOTE] {repr(txt)}")
        
            start_ts = _parse_application_note(txt)
        
            # --- FALLBACK OBBLIGATORIO ---
            delay = 0
            if start_ts is None:
                self.log("[MINISTRY] application note non leggibile → possibile coda=0")
            
                # --- NUOVA LOGICA ---
                if self.x == 0 or self.y == 0:
                    self.log("[MINISTRY] coda=0 + OCR vuoto → tap extra per uscire")
                    adb_tap(*BOTTOM_LEFT)
                    time.sleep(0.4)
            
                # fallback cooldown
                self.log("[MINISTRY] cooldown forzato 10 min")
                cooldown = int(time.time()) + 600
                self.cooldown_until = cooldown
            else:
                delay = start_ts - int(time.time())
                cooldown = start_ts + 600
                self.cooldown_until = cooldown 
                self.log(f"[MINISTRY] cooldown until {time.ctime(cooldown)}")
            if delay > 0:
                self._schedule_confirm_popup_cleanup(delay)
        
            self.log(f"[MINISTRY] ministry finished, cooldown ignored")
            self.state = MinistryState.EXIT_MINISTRY
            return

        if self.state == MinistryState.EXIT_MINISTRY:
            adb_tap(*BOTTOM_LEFT)
            time.sleep(0.4)
            adb_tap(*BOTTOM_LEFT)
            time.sleep(0.4)
            adb_tap(*BOTTOM_LEFT)
            self._mark_action()
            self.state = MinistryState.BACK_TO_HQ
            return

        if self.state == MinistryState.BACK_TO_HQ:
            if self._tap_template(img, "go_hq"):
                self.xy_read = False
                self.x = 0
                self.y = 0
                self.state = MinistryState.IDLE
                WORKFLOW_MANAGER.release(Workflow.MINISTRY)
                if hasattr(self, "started_ts"):
                    del self.started_ts
                self.log("[MINISTRY-FLOW] completed -> IDLE")
            return


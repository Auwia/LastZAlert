#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import pytesseract
import re
import time
from datetime import datetime
from enum import Enum

from workflow_manager import Workflow, WORKFLOW_MANAGER
from bot_utils import load_templates, match_any, adb_tap

# ============================================================
# CONFIG
# ============================================================

DEBUG = False
THR = 0.75
THR_RECOMMENDED = 0.56
ACTION_COOLDOWN = 1.0
STALL_TIMEOUT = 40

# ordine = priorità
RESEARCH_PRIORITIES = [
    ("hero", "research/hero_training.png"),
    ("military", "research/military_strategies.png"),
    ("rapid", "research/rapid_grow.png"),
]

# rilevamento nodi
NODE_MIN_W = 120
NODE_MAX_W = 260
NODE_MIN_H = 100
NODE_MAX_H = 260

# quanto può essere "irregolare" il poligono
NODE_POLY_EPSILON = 0.045

# per considerare due nodi sulla stessa riga
NODE_ROW_TOLERANCE = 80

# evita header/footer
NODE_SCAN_Y_MIN = 180
NODE_SCAN_Y_MAX = 1820

BACK = (100, 2400)

# ============================================================
# STATE
# ============================================================

class ResearchState(Enum):
    IDLE = 0
    FIND_START = 1
    TAP_LAB = 2
    TAP_CATEGORY = 3
    SCAN_NODE = 4
    START_RESEARCH = 5
    REPLENISH = 6
    HELP = 7
    EXIT = 8


def find_next_research_node(img, log_fn=print):

    nodes = detect_research_nodes(img)
    nodes = sort_nodes_visual(nodes)

    if DEBUG:
        log_fn(f"[RESEARCH] geometric nodes={len(nodes)}")

    for node in nodes:

        progress = read_node_progress(img, node)

        if DEBUG:
            log_fn(
                f"[RESEARCH] node "
                f"@ {node['x']},{node['y']} "
                f"status={progress}"
            )

        if progress is None:
            continue

        if progress["max"]:
            continue

        node["progress"] = progress
        return node

    return None

def sort_nodes_visual(nodes):
    """
    Ordina:
        alto -> basso
        e nella stessa riga:
        sinistra -> destra
    """

    if not nodes:
        return []

    nodes = sorted(nodes, key=lambda n: n["y"])

    rows = []

    for node in nodes:

        inserted = False

        for row in rows:
            avg_y = sum(n["y"] for n in row) / len(row)

            if abs(node["y"] - avg_y) <= NODE_ROW_TOLERANCE:
                row.append(node)
                inserted = True
                break

        if not inserted:
            rows.append([node])

    result = []

    for row in rows:
        row.sort(key=lambda n: n["x"])
        result.extend(row)

    return result

def read_node_progress(img, node):

    cx = node["x"]
    cy = node["y"]

    h_img, w_img = img.shape[:2]

    # scala rispetto alla risoluzione reale 1080x2408
    sx = w_img / 1080.0
    sy = h_img / 2408.0

    # zona stretta dove appare MAX / 6/10 / 7/10
    half_w = int(80 * sx)

    y1 = int(cy + 65 * sy)
    y2 = int(cy + 130 * sy)

    x1 = int(cx - half_w)
    x2 = int(cx + half_w)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w_img, x2)
    y2 = min(h_img, y2)

    roi = img[y1:y2, x1:x2]

    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(
        gray,
        None,
        fx=5,
        fy=5,
        interpolation=cv2.INTER_CUBIC
    )

    variants = [gray]

    _, otsu = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    variants.append(otsu)

    _, inv = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    variants.append(inv)

    texts = []

    for variant in variants:

        txt = pytesseract.image_to_string(
            variant,
            config=(
                "--psm 7 "
                "-c tessedit_char_whitelist=MAX0123456789/"
            )
        )

        txt = txt.upper()
        txt = txt.replace(" ", "")
        txt = txt.replace("\n", "")
        txt = txt.replace("\x0c", "")

        if txt:
            texts.append(txt)

    if DEBUG:
        print(
            f"[RESEARCH][OCR] "
            f"node={cx},{cy} "
            f"roi=({x1},{y1})-({x2},{y2}) "
            f"texts={texts}"
        )

    for txt in texts:

        if "MAX" in txt:
            return {
                "text": "MAX",
                "current": None,
                "total": None,
                "max": True
            }

        m = re.search(r"(\d{1,2})/(\d{1,2})", txt)

        if m:
            current = int(m.group(1))
            total = int(m.group(2))

            # evita OCR assurdi
            if total <= 0:
                continue

            if current > total:
                continue

            return {
                "text": f"{current}/{total}",
                "current": current,
                "total": total,
                "max": current >= total
            }

    return None

def detect_research_nodes(img):
    """
    Cerca forme compatibili con i nodi della research tree.

    ritorna:
        [
            {
                "x": center_x,
                "y": center_y,
                "bbox": (x, y, w, h),
                "vertices": n
            },
            ...
        ]
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # evidenzia bordi
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 140)

    # chiude piccoli buchi nei contorni
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    nodes = []

    for cnt in contours:

        peri = cv2.arcLength(cnt, True)

        approx = cv2.approxPolyDP(
            cnt,
            NODE_POLY_EPSILON * peri,
            True
        )

        vertices = len(approx)

        # il nodo è fondamentalmente esagonale,
        # ma lasciamo tolleranza ai bordi grafici
        if vertices < 5 or vertices > 9:
            continue

        x, y, w, h = cv2.boundingRect(approx)

        if w < NODE_MIN_W or w > NODE_MAX_W:
            continue

        if h < NODE_MIN_H or h > NODE_MAX_H:
            continue

        if y < NODE_SCAN_Y_MIN or y > NODE_SCAN_Y_MAX:
            continue

        ratio = w / float(h)

        if ratio < 0.65 or ratio > 1.45:
            continue

        cx = x + w // 2
        cy = y + h // 2

        nodes.append({
            "x": cx,
            "y": cy,
            "bbox": (x, y, w, h),
            "vertices": vertices,
        })

    return nodes
        
# ============================================================
# FLOW
# ============================================================

class ResearchFlow:

    def __init__(self, log_fn=print, notify_fn=None):
        self.log = log_fn
        self.notify = notify_fn
        self.state = ResearchState.IDLE

        self.last_action_ts = 0
        self.last_progress_ts = 0

        self.lab_opened = False
        self.is_wednesday_mode = False

        self.node_index = 0

        self.templates = {
            "start": load_templates("research/start.png"),
            "lab": load_templates("research/lab_icon.png"),
            "rapid": load_templates("research/rapid_grow.png"),
            "recommended": load_templates("research/recommended.png") or load_templates("donation/recommended.png") or load_templates("recommended.png"),
            "research": load_templates("research/research_button.png"),
            "help": load_templates("research/help_button.png"),
            "replenish": load_templates("research/replenish.png"),
        }

        self.research_priorities = [
            ("hero", load_templates("research/hero_training.png")),
            ("military", load_templates("research/military_strategies.png")),
            ("rapid", load_templates("research/rapid_grow.png")),
        ]
        
        self.research_index = 0
        self.current_research = None

        self.log("[RESEARCH-FLOW] initialized")

    # ---------------------------------------------------------

    def _is_wednesday_window(self, now=None):
        now = now or datetime.now()
        wd = now.weekday()  # Monday=0 ... Sunday=6

        # mercoledì 04:00 -> giovedì 04:00
        if wd == 2 and now.hour >= 4:
            return True
        if wd == 3 and now.hour < 4:
            return True
        return False

    def _tap_recommended(self, img, log_msg):
        name, score, loc, hw = match_any(img, self.templates["recommended"])

        if name and score >= THR_RECOMMENDED:
            # il template matcha il pollice a sinistra:
            # tap spostato verso destra sulla tile
            tap_offset_x = int(hw[1] * 0.75)
            adb_tap(loc[0] + tap_offset_x, loc[1] + hw[0] // 2)
            self.log(log_msg)
            return True

        return False

    def _do_exit(self):

        if self.lab_opened:
            adb_tap(*BACK)
            time.sleep(0.5)
            adb_tap(*BACK)

        self.lab_opened = False
        self.is_wednesday_mode = False

        if DEBUG:
            self.log("[RESEARCH] exit")

        self.state = ResearchState.IDLE
        WORKFLOW_MANAGER.release(Workflow.RESEARCH)
        self._mark()

    def _cooldown_ok(self):
        return (time.time() - self.last_action_ts) >= ACTION_COOLDOWN

    def _mark(self):
        now = time.time()
        self.last_action_ts = now
        self.last_progress_ts = now

    # ---------------------------------------------------------

    def trigger(self):

        if self.state != ResearchState.IDLE:
            return

        if not WORKFLOW_MANAGER.acquire(Workflow.RESEARCH):
            return

        self.is_wednesday_mode = self._is_wednesday_window()
        self.state = ResearchState.FIND_START
        self.last_progress_ts = time.time()
        self._mark()

    # ---------------------------------------------------------

    def step(self, img):
        if DEBUG:
            self.log(f"[RESEARCH] step state={self.state}")

        if self.state == ResearchState.IDLE:
            return

        if not self._cooldown_ok():
            return

        if (time.time() - self.last_progress_ts) > STALL_TIMEOUT:
            self.state = ResearchState.IDLE
            self.is_wednesday_mode = False
            WORKFLOW_MANAGER.release(Workflow.RESEARCH)
            return

        # -----------------------------------------------------
        # START
        # -----------------------------------------------------

        if self.state == ResearchState.FIND_START:

            name, score, loc, hw = match_any(img, self.templates["start"])
            if DEBUG:
                self.log(f"[RESEARCH] start score={score:.3f}")

            if name and score >= THR:
                adb_tap(loc[0] + hw[1]//2, loc[1] + hw[0]//2)
                self.lab_opened = True
                self.log("[RESEARCH] start tapped")

                if self.notify:
                    self.notify("🔬 Ricerca completata / laboratorio libero rilevato!")

                time.sleep(1)

            else:
                if DEBUG:
                    self.log("[RESEARCH] start not found -> continue")

            self.state = ResearchState.TAP_LAB
            self._mark()
            return

        # -----------------------------------------------------

        if self.state == ResearchState.TAP_LAB:

            name, score, loc, hw = match_any(img, self.templates["lab"])

            if name and score >= THR:
                adb_tap(loc[0] + hw[1]//2, loc[1] + hw[0]//2)
                self.log("[RESEARCH] lab tapped")
                time.sleep(2)
                self.research_index = 0
                self.state = ResearchState.TAP_CATEGORY
                self._mark()
            else:
                if DEBUG:
                    self.log("[RESEARCH] lab not found -> exit")
                self._do_exit()
                self._mark()
            return

        # -----------------------------------------------------

        if self.state == ResearchState.TAP_CATEGORY:

            time.sleep(5.0)
        
            if self.research_index >= len(self.research_priorities):
                self.log("[RESEARCH] no research available in priorities")
                self.state = ResearchState.EXIT
                self._mark()
                return
        
            research_name, templates = \
                self.research_priorities[self.research_index]
        
            name, score, loc, hw = match_any(img, templates)

            if DEBUG:
                self.log(
                    f"[RESEARCH] category check "
                    f"{research_name} score={score:.3f}"
                )

            if name and score >= THR:
        
                adb_tap(
                    loc[0] + hw[1] // 2,
                    loc[1] + hw[0] // 2
                )
        
                self.current_research = research_name
        
                self.log(
                    f"[RESEARCH] category opened: {research_name}"
                )
        
                time.sleep(4)
        
                self.state = ResearchState.SCAN_NODE
                self._mark()
                return
        
            self.log(
                f"[RESEARCH] category {research_name} not found"
            )
        
            self.research_index += 1
            self._mark()
            return

        # -----------------------------------------------------
        # SCAN TECH
        # -----------------------------------------------------

        if self.state == ResearchState.SCAN_NODE:

            if self.is_wednesday_mode:
                if self._tap_recommended(img, "[RESEARCH] recommended node tapped"):
                    self.state = ResearchState.START_RESEARCH
                    self._mark()
                else:
                    self.log("[RESEARCH] recommended node not found -> retry")
                    self._mark()
                return

            if self.state == ResearchState.SCAN_NODE:
            
                node = find_next_research_node(
                    img,
                    self.log
                )
            
                if node:
            
                    p = node["progress"]
            
                    self.log(
                        f"[RESEARCH] next node "
                        f"{self.current_research}: "
                        f"{p['text']} "
                        f"@ {node['x']},{node['y']}"
                    )
            
                    adb_tap(
                        node["x"],
                        node["y"]
                    )

                    self.log("[RESEARCH] node tapped -> waiting popup")
                    time.sleep(3.0)

                    self.state = ResearchState.START_RESEARCH
                    self._mark()
                    return
            
                # questa categoria non ha niente disponibile
                self.log(
                    f"[RESEARCH] "
                    f"{self.current_research}: no available node"
                )
            
                # torna alla schermata Techs
                adb_tap(*BACK)
                time.sleep(1)
            
                self.research_index += 1
                self.current_research = None
            
                self.state = ResearchState.TAP_CATEGORY
                self._mark()
                return

        # -----------------------------------------------------

        if self.state == ResearchState.START_RESEARCH:

            name, score, loc, hw = match_any(img, self.templates["research"])

            if DEBUG:
                self.log(
                    f"[RESEARCH] research button "
                    f"name={name} score={score:.3f}"
                )

            if name and score >= THR:

                adb_tap(loc[0] + hw[1]//2, loc[1] + hw[0]//2)
                self.log("[RESEARCH] research started")

                if self.notify:
                    self.notify("🔬 Laboratorio libero: nuova ricerca avviata!")

                time.sleep(2)

                self.state = ResearchState.REPLENISH
                self._mark()
                return

            self._mark()
            return

        # -----------------------------------------------------

        if self.state == ResearchState.REPLENISH:

            name_r, score_r, loc_r, hw_r = match_any(img, self.templates["replenish"])

            if name_r and score_r >= THR:
                adb_tap(loc_r[0] + hw_r[1]//2, loc_r[1] + hw_r[0]//2)
                self.log("[RESEARCH] replenish all tapped")
                time.sleep(1)

            self.state = ResearchState.HELP
            self._mark()
            return

        # -----------------------------------------------------

        if self.state == ResearchState.HELP:

            name, score, loc, hw = match_any(img, self.templates["help"])

            if name and score >= THR:
                adb_tap(loc[0] + hw[1]//2, loc[1] + hw[0]//2)
                self.log("[RESEARCH] alliance help sent")
                self.state = ResearchState.EXIT
                self._mark()
            return

        # -----------------------------------------------------

        if self.state == ResearchState.EXIT:

            if self.lab_opened:
                adb_tap(*BACK)
                time.sleep(0.5)
                adb_tap(*BACK)

            self.lab_opened = False
            self.is_wednesday_mode = False

            self.log("[RESEARCH] exit")

            self.state = ResearchState.IDLE
            WORKFLOW_MANAGER.release(Workflow.RESEARCH)
            self._mark()

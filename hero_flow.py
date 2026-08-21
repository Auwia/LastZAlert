#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import subprocess
import time
from enum import Enum

import cv2
import numpy as np
import pytesseract

from workflow_manager import Workflow, WORKFLOW_MANAGER
from bot_utils import adb_tap


# ============================================================
# CONFIG
# ============================================================

DEBUG = False

# True = esegue tutto il flow ma NON preme realmente Upgrade.
# False = i 5 tap su Upgrade vengono eseguiti davvero.
DRY_RUN = True

# Hero icon nella vista HQ.
# Coordinate frazionarie x/y.
HQ_HERO_ICON_FRAC = (0.08, 0.90)

# ------------------------------------------------------------
# RICONOSCIMENTO GEOMETRICO CARD EROI
# ------------------------------------------------------------

# Area verticale utile della schermata Heroes.
HERO_CARD_Y_MIN_FRAC = 0.14
HERO_CARD_Y_MAX_FRAC = 0.91

# Dimensioni relative attese delle card.
# Nello screenshot 918x2048 le card sono circa 195x350 px.
HERO_CARD_MIN_W_FRAC = 0.16
HERO_CARD_MAX_W_FRAC = 0.27
HERO_CARD_MIN_H_FRAC = 0.13
HERO_CARD_MAX_H_FRAC = 0.21

HERO_CARD_RATIO_MIN = 0.45
HERO_CARD_RATIO_MAX = 0.70

# Quanto possono differire verticalmente due card della stessa riga.
HERO_ROW_TOLERANCE_FRAC = 0.035

# Area OCR del livello ALL'INTERNO di ogni card:
# parte bassa / sinistra dove compare "Lv.xxx".
LEVEL_ROI_X1_FRAC = 0.00
LEVEL_ROI_X2_FRAC = 0.68
LEVEL_ROI_Y1_FRAC = 0.66
LEVEL_ROI_Y2_FRAC = 0.90

OCR_SCALE = 3.0
LEVEL_MIN = 1
LEVEL_MAX = 200

# Scroll verso il basso della lista = swipe verso l'alto.
SWIPE_X_FRAC = 0.50
SWIPE_FROM_Y_FRAC = 0.78
SWIPE_TO_Y_FRAC = 0.43
SWIPE_DURATION_MS = 550
MAX_SCROLLS = 10

# Se dopo uno swipe la griglia cambia pochissimo, consideriamo
# raggiunta la fine della lista.
HERO_GRID_ROI = (0.02, 0.98, 0.14, 0.91)
SCROLL_DIFF_THRESHOLD = 2.5

# Pulsante Upgrade nella pagina dettaglio eroe.
UPGRADE_BUTTON_FRAC = (0.50, 0.87)
UPGRADE_TAPS = 5
UPGRADE_TAP_PAUSE = 0.40

OPEN_HEROES_WAIT_SEC = 2.0
OPEN_DETAIL_WAIT_SEC = 1.5
AFTER_SCROLL_WAIT_SEC = 1.2
STALL_TIMEOUT_SEC = 120

ADB_CMD = "adb"


# ============================================================
# STATE
# ============================================================

class HeroState(Enum):
    IDLE = 0
    OPEN_HEROES = 1
    SCAN_HEROES = 2
    UPGRADE = 3


# ============================================================
# HELPERS
# ============================================================

def _crop_frac(img, roi):
    h, w = img.shape[:2]
    x1f, x2f, y1f, y2f = roi

    x1 = max(0, min(int(w * x1f), w - 1))
    x2 = max(x1 + 1, min(int(w * x2f), w))
    y1 = max(0, min(int(h * y1f), h - 1))
    y2 = max(y1 + 1, min(int(h * y2f), h))

    return img[y1:y2, x1:x2], (x1, y1, x2, y2)


def _tap_frac(img, xy_frac):
    h, w = img.shape[:2]
    x = int(w * xy_frac[0])
    y = int(h * xy_frac[1])
    adb_tap(x, y)
    return x, y


def _adb_swipe(x1, y1, x2, y2, duration_ms):
    subprocess.run(
        [
            ADB_CMD, "shell", "input", "swipe",
            str(int(x1)), str(int(y1)),
            str(int(x2)), str(int(y2)),
            str(int(duration_ms)),
        ],
        check=False,
    )


def _grid_signature(img):
    roi, _ = _crop_frac(img, HERO_GRID_ROI)
    if roi is None or roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (120, 160), interpolation=cv2.INTER_AREA)


def _image_diff(sig_a, sig_b):
    if sig_a is None or sig_b is None:
        return 999.0
    diff = cv2.absdiff(sig_a, sig_b)
    return float(diff.mean())


def detect_hero_cards(img):
    """
    Riconosce geometricamente le card rettangolari degli eroi.

    Ritorna:
        [
            {
                "x": centro_x,
                "y": centro_y,
                "bbox": (x, y, w, h)
            },
            ...
        ]
    """
    h_img, w_img = img.shape[:2]

    min_w = int(w_img * HERO_CARD_MIN_W_FRAC)
    max_w = int(w_img * HERO_CARD_MAX_W_FRAC)
    min_h = int(h_img * HERO_CARD_MIN_H_FRAC)
    max_h = int(h_img * HERO_CARD_MAX_H_FRAC)

    y_min = int(h_img * HERO_CARD_Y_MIN_FRAC)
    y_max = int(h_img * HERO_CARD_Y_MAX_FRAC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 140)

    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates = []

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue

        approx = cv2.approxPolyDP(
            cnt,
            0.02 * peri,
            True,
        )

        # La card è rettangolare; lasciamo un po' di tolleranza
        # perché cornice/immagine possono produrre 4-8 vertici.
        vertices = len(approx)
        if vertices < 4 or vertices > 8:
            continue

        x, y, w, h = cv2.boundingRect(approx)

        if w < min_w or w > max_w:
            continue

        if h < min_h or h > max_h:
            continue

        if y < y_min or y > y_max:
            continue

        ratio = w / float(h)
        if ratio < HERO_CARD_RATIO_MIN or ratio > HERO_CARD_RATIO_MAX:
            continue

        candidates.append(
            {
                "x": x + w // 2,
                "y": y + h // 2,
                "bbox": (x, y, w, h),
                "vertices": vertices,
            }
        )

    # Elimina rettangoli duplicati/nidificati generati dalla cornice.
    candidates.sort(
        key=lambda c: c["bbox"][2] * c["bbox"][3],
        reverse=True,
    )

    cards = []

    for cand in candidates:
        cx = cand["x"]
        cy = cand["y"]

        duplicate = False

        for saved in cards:
            sw = saved["bbox"][2]
            sh = saved["bbox"][3]

            if (
                abs(cx - saved["x"]) <= sw * 0.18
                and abs(cy - saved["y"]) <= sh * 0.18
            ):
                duplicate = True
                break

        if not duplicate:
            cards.append(cand)

    return cards


def sort_hero_cards(cards, img_height):
    """
    Ordina le card:
        alto -> basso
        sinistra -> destra
    """
    if not cards:
        return []

    row_tolerance = max(
        25,
        int(img_height * HERO_ROW_TOLERANCE_FRAC),
    )

    cards = sorted(cards, key=lambda c: c["y"])
    rows = []

    for card in cards:
        inserted = False

        for row in rows:
            avg_y = sum(c["y"] for c in row) / len(row)

            if abs(card["y"] - avg_y) <= row_tolerance:
                row.append(card)
                inserted = True
                break

        if not inserted:
            rows.append([card])

    result = []

    for row in rows:
        row.sort(key=lambda c: c["x"])
        result.extend(row)

    return result


def _extract_level(text):
    if not text:
        return None

    cleaned = text.upper()
    cleaned = cleaned.replace(" ", "")
    cleaned = cleaned.replace("\n", "")

    # Prima prova con Lv.xxx
    m = re.search(r"L[VW][.\-:]?(\d{2,3})", cleaned)

    # Fallback: OCR può perdere "Lv."
    if not m:
        m = re.search(r"(\d{2,3})", cleaned)

    if not m:
        return None

    try:
        level = int(m.group(1))
    except ValueError:
        return None

    if LEVEL_MIN <= level <= LEVEL_MAX:
        return level

    return None


def read_hero_level(img, card):
    """
    Fa OCR soltanto nella zona del livello della singola card.
    """
    x, y, w, h = card["bbox"]

    x1 = x + int(w * LEVEL_ROI_X1_FRAC)
    x2 = x + int(w * LEVEL_ROI_X2_FRAC)

    y1 = y + int(h * LEVEL_ROI_Y1_FRAC)
    y2 = y + int(h * LEVEL_ROI_Y2_FRAC)

    x1 = max(0, min(x1, img.shape[1] - 1))
    x2 = max(x1 + 1, min(x2, img.shape[1]))
    y1 = max(0, min(y1, img.shape[0] - 1))
    y2 = max(y1 + 1, min(y2, img.shape[0]))

    roi = img[y1:y2, x1:x2]

    if roi is None or roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(
        gray,
        None,
        fx=OCR_SCALE,
        fy=OCR_SCALE,
        interpolation=cv2.INTER_CUBIC,
    )

    # Testo bianco con bordo nero: Otsu funziona meglio della soglia
    # fissa quando cambiano personaggio/sfondo.
    _, bw = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    txt = pytesseract.image_to_string(
        bw,
        config=(
            "--psm 7 "
            "-c tessedit_char_whitelist=Lv.0123456789"
        ),
    )

    level = _extract_level(txt)

    if level is None:
        # Secondo tentativo sull'immagine invertita.
        inv = cv2.bitwise_not(bw)

        txt2 = pytesseract.image_to_string(
            inv,
            config=(
                "--psm 7 "
                "-c tessedit_char_whitelist=Lv.0123456789"
            ),
        )

        level = _extract_level(txt2)

        if level is not None:
            txt = txt2

    if level is None:
        return None

    return {
        "level": level,
        "text": txt.strip(),
        "ocr_bbox": (x1, y1, x2, y2),
    }


def read_visible_hero_levels(img, log_fn=print):
    """
    1. riconosce geometricamente le card;
    2. le ordina visivamente;
    3. legge il livello dentro ogni card.

    Non viene fatto OCR casuale sull'intera schermata.
    """
    cards = detect_hero_cards(img)
    cards = sort_hero_cards(cards, img.shape[0])

    log_fn(f"[HERO-FLOW] card rettangolari rilevate: {len(cards)}")

    heroes = []

    for index, card in enumerate(cards, start=1):
        info = read_hero_level(img, card)

        if info is None:
            if DEBUG:
                x, y, w, h = card["bbox"]
                log_fn(
                    f"[HERO-FLOW] card {index} "
                    f"bbox={x},{y},{w},{h} -> OCR level FAILED"
                )
            continue

        hero = {
            **card,
            **info,
        }

        heroes.append(hero)

        log_fn(
            f"[HERO-FLOW] card {index}: "
            f"Lv.{hero['level']} "
            f"center={hero['x']},{hero['y']}"
        )

    return heroes


def find_first_drop_by_five(heroes, previous_level=None):
    """
    Cerca il primo eroe il cui livello è esattamente 5 sotto
    il livello dell'eroe precedente.

    previous_level consente di riconoscere il cambio anche quando
    avviene tra due schermate dopo uno scroll.
    """
    prev = previous_level

    for hero in heroes:
        current = hero["level"]

        if prev is not None and (prev - current) == 5:
            return hero

        prev = current

    return None


# ============================================================
# FLOW
# ============================================================

class HeroFlow:

    def __init__(self, log_fn=print, on_complete=None):
        self.log = log_fn
        self.on_complete = on_complete

        self.state = HeroState.IDLE
        self.last_progress_ts = 0.0

        self.previous_level = None
        self.scroll_count = 0

        self.waiting_after_scroll = False
        self.before_scroll_signature = None

        self.log("[HERO-FLOW] initialized")

    def _mark(self):
        self.last_progress_ts = time.time()

    def _release(self, completed=False, reason=""):
        if reason:
            self.log(f"[HERO-FLOW] {reason}")

        self.state = HeroState.IDLE
        self.previous_level = None
        self.scroll_count = 0
        self.waiting_after_scroll = False
        self.before_scroll_signature = None

        WORKFLOW_MANAGER.release(Workflow.HERO)

        if completed and self.on_complete:
            try:
                self.on_complete()
            except Exception as exc:
                self.log(f"[HERO-FLOW] on_complete error: {exc}")

        self._mark()

    def trigger(self):
        if self.state != HeroState.IDLE:
            return False

        # HERO usa il proprio lock/priorità nel workflow_manager.
        if not WORKFLOW_MANAGER.acquire(Workflow.HERO):
            return False

        self.state = HeroState.OPEN_HEROES
        self.previous_level = None
        self.scroll_count = 0
        self.waiting_after_scroll = False
        self.before_scroll_signature = None
        self._mark()

        self.log("[HERO-FLOW] trigger -> OPEN_HEROES")
        return True

    def step(self, img):
        if self.state == HeroState.IDLE:
            return

        if (time.time() - self.last_progress_ts) > STALL_TIMEOUT_SEC:
            self._release(completed=False, reason="STALL -> release")
            return

        # --------------------------------------------------------
        # 1. HQ -> schermata Heroes
        # --------------------------------------------------------
        if self.state == HeroState.OPEN_HEROES:
            x, y = _tap_frac(img, HQ_HERO_ICON_FRAC)
            self.log(f"[HERO-FLOW] tap hero icon @ {x},{y}")

            time.sleep(OPEN_HEROES_WAIT_SEC)

            self.state = HeroState.SCAN_HEROES
            self._mark()
            return

        # --------------------------------------------------------
        # 2. OCR livelli / eventuale scroll
        # --------------------------------------------------------
        if self.state == HeroState.SCAN_HEROES:

            current_signature = _grid_signature(img)

            if self.waiting_after_scroll:
                diff = _image_diff(
                    self.before_scroll_signature,
                    current_signature,
                )

                if DEBUG:
                    self.log(f"[HERO-FLOW] scroll image diff={diff:.2f}")

                self.waiting_after_scroll = False
                self.before_scroll_signature = None

                if diff < SCROLL_DIFF_THRESHOLD:
                    self._release(
                        completed=False,
                        reason="fine lista: scroll non muove più la griglia -> release",
                    )
                    return

            hits = read_visible_hero_levels(img, self.log)

            if not hits:
                self._release(
                    completed=False,
                    reason="nessun eroe/livello trovato -> release",
                )
                return

            levels = [h["level"] for h in hits]
            self.log(f"[HERO-FLOW] livelli visibili: {levels}")

            target = find_first_drop_by_five(
                hits,
                previous_level=self.previous_level,
            )

            if target is not None:
                self.log(
                    f"[HERO-FLOW] trovato livello inferiore di 5: "
                    f"Lv.{target['level']} @ {target['x']},{target['y']}"
                )

                # 3. La card è stata riconosciuta geometricamente:
                # tap al centro della card, non sul testo OCR.
                adb_tap(target["x"], target["y"])
                time.sleep(OPEN_DETAIL_WAIT_SEC)

                self.state = HeroState.UPGRADE
                self._mark()
                return

            # Manteniamo l'ultimo livello letto per riconoscere il cambio
            # anche se capita esattamente tra due schermate dopo lo scroll.
            self.previous_level = hits[-1]["level"]

            # Caso richiesto: tutti gli eroi visibili hanno lo stesso livello.
            if len(set(levels)) == 1:
                if self.scroll_count >= MAX_SCROLLS:
                    self._release(
                        completed=False,
                        reason="raggiunto MAX_SCROLLS senza trovare livello -5 -> release",
                    )
                    return

                h, w = img.shape[:2]

                self.before_scroll_signature = current_signature
                self.waiting_after_scroll = True
                self.scroll_count += 1

                x = int(w * SWIPE_X_FRAC)
                y1 = int(h * SWIPE_FROM_Y_FRAC)
                y2 = int(h * SWIPE_TO_Y_FRAC)

                self.log(
                    f"[HERO-FLOW] tutti stesso livello Lv.{levels[0]} "
                    f"-> scroll {self.scroll_count}/{MAX_SCROLLS}"
                )

                _adb_swipe(
                    x, y1,
                    x, y2,
                    SWIPE_DURATION_MS,
                )

                time.sleep(AFTER_SCROLL_WAIT_SEC)
                self._mark()
                return

            # Se l'OCR vede livelli diversi ma non esiste un salto esatto di 5,
            # non scegliamo un eroe a caso.
            self._release(
                completed=False,
                reason=(
                    "livelli diversi ma nessun salto esatto di 5 "
                    f"({levels}) -> release"
                ),
            )
            return

        # --------------------------------------------------------
        # 4. Upgrade x5
        # --------------------------------------------------------
        if self.state == HeroState.UPGRADE:
            h, w = img.shape[:2]
            x = int(w * UPGRADE_BUTTON_FRAC[0])
            y = int(h * UPGRADE_BUTTON_FRAC[1])

            for i in range(UPGRADE_TAPS):

                if DRY_RUN:
                    self.log(
                        f"[HERO-FLOW][DRY-RUN] "
                        f"Upgrade {i + 1}/{UPGRADE_TAPS} "
                        f"SIMULATO @ {x},{y}"
                    )
                else:
                    adb_tap(x, y)
                    self.log(
                        f"[HERO-FLOW] "
                        f"Upgrade {i + 1}/{UPGRADE_TAPS} @ {x},{y}"
                    )

                time.sleep(UPGRADE_TAP_PAUSE)

            # In DRY_RUN non chiamiamo on_complete:
            # così hero_last_run.txt NON viene marcato e puoi rifare il test.
            self._release(
                completed=not DRY_RUN,
                reason=(
                    "DRY-RUN completato -> release"
                    if DRY_RUN
                    else "Upgrade x5 completato -> release"
                ),
            )
            return

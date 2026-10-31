#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import cv2

THR_WORKERS = 0.99

TEMPLATE_WORKER_FREE = (
    Path(__file__).resolve().parent
    / "building"
    / "worker_free.png"
)


class WorkersFlow:
    def __init__(self, log_fn=print, notify_fn=None):
        self.log = log_fn
        self.notify = notify_fn
        self.was_free = False

        self.template = cv2.imread(
            str(TEMPLATE_WORKER_FREE),
            cv2.IMREAD_COLOR
        )

        if self.template is None:
            raise RuntimeError(
                f"[WORKERS] template missing: {TEMPLATE_WORKER_FREE}"
            )

    def step(self, img):
        if img is None:
            return

        th, tw = self.template.shape[:2]
        ih, iw = img.shape[:2]

        if ih < th or iw < tw:
            return

        # Cerca esclusivamente nell'area dell'indicatore operai.
        # Coordinate proporzionali alla risoluzione dello screenshot.
        x1, x2 = int(iw * 0.00), int(iw * 0.16)
        y1, y2 = int(ih * 0.17), int(ih * 0.25)
        roi = img[y1:y2, x1:x2]

        if roi.shape[0] < th or roi.shape[1] < tw:
            return

        result = cv2.matchTemplate(
            roi,
            self.template,
            cv2.TM_CCOEFF_NORMED
        )

        _, score, _, loc = cv2.minMaxLoc(result)

        free = score >= THR_WORKERS

        if free and not self.was_free:
            self.log(
                f"[WORKERS] worker free "
                f"score={score:.3f} loc={loc}"
            )

            if self.notify:
                self.notify("👷 Operaio libero!")

        elif not free and self.was_free:
            self.log("[WORKERS] worker busy -> rearmed")

        self.was_free = free

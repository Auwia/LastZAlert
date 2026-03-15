#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
from enum import Enum

from workflow_manager import WORKFLOW_MANAGER, Workflow
from bot_utils import crop_roi, match_any, tap_match_in_fullscreen, load_templates

import cv2


#RALLY_TRIGGER_ROI = (0.05, 0.45, 0.68, 0.88)
RALLY_TRIGGER_ROI = (0.05, 0.58, 0.78, 0.98)
RALLY_MONSTER_ROI = (0.00, 1.00, 0.10, 0.80)

TRIGGER_THRESHOLD = 0.80
MONSTER_THRESHOLD = 0.80
ADD_THRESHOLD = 0.80
READY_THRESHOLD = 0.80
MARCH_THRESHOLD = 0.80
CONFIRM_THRESHOLD = 0.80

class RallyState(Enum):

    IDLE = 0
    OPEN_RALLY = 1
    FIND_MONSTER = 2
    CLICK_ADD = 3
    CHECK_READY = 4
    CLICK_MARCH = 5
    CONFIRM = 6
    DONE = 7


class RallyFlow:

    def __init__(self, log_fn):

        self.log = log_fn
        self.state = RallyState.IDLE
        self.cooldown_until = 0

        self.trigger_templates = load_templates("rally/trigger")
        self.monster_templates = load_templates("rally/monsters")
        self.add_templates = load_templates("rally/add")
        self.ready_templates = load_templates("rally/ready")
        self.march_templates = load_templates("rally/march")
        self.confirm_templates = load_templates("rally/confirm")


    def trigger(self):

        if not WORKFLOW_MANAGER.acquire(Workflow.RALLY):
            return

        self.log("[RALLY] trigger start")
        self.state = RallyState.OPEN_RALLY


    def step(self, img):

        if self.state == RallyState.IDLE:
            return

        if time.time() < self.cooldown_until:
            return


        # -----------------------------------------
        # OPEN RALLY LIST
        # -----------------------------------------

        if self.state == RallyState.OPEN_RALLY:

            roi, coords = crop_roi(img, RALLY_TRIGGER_ROI)

            name, score, loc, hw = match_any(roi,self.trigger_templates)

            if score >= TRIGGER_THRESHOLD:

                tap_match_in_fullscreen(coords,loc,hw)

                self.log("[RALLY] rally list opened")

                self.state = RallyState.FIND_MONSTER
                self.cooldown_until = time.time()+2
                return


        # -----------------------------------------
        # FIND MONSTER
        # -----------------------------------------

        if self.state == RallyState.FIND_MONSTER:

            roi, coords = crop_roi(img,RALLY_MONSTER_ROI)

            name, score, loc, hw = match_any(roi,self.monster_templates)

            if score >= MONSTER_THRESHOLD:

                self.log(f"[RALLY] monster found {name}")

                self.state = RallyState.CLICK_ADD
                self.cooldown_until = time.time()+1
                return


        # -----------------------------------------
        # CLICK ADD
        # -----------------------------------------

        if self.state == RallyState.CLICK_ADD:

            name, score, loc, hw = match_any(img,self.add_templates)

            if score >= ADD_THRESHOLD:

                tap_match_in_fullscreen((0,0,img.shape[1],img.shape[0]),loc,hw)

                self.log("[RALLY] join rally")

                self.state = RallyState.CHECK_READY
                self.cooldown_until = time.time()+2
                return


        # -----------------------------------------
        # CHECK VEHICLE READY
        # -----------------------------------------

        if self.state == RallyState.CHECK_READY:

            name, score, loc, hw = match_any(img,self.ready_templates)

            if score >= READY_THRESHOLD:

                self.log("[RALLY] vehicle ready")

                self.state = RallyState.CLICK_MARCH
                return

            else:

                self.log("[RALLY] no vehicle available")
                self.finish()
                return


        # -----------------------------------------
        # CLICK MARCH
        # -----------------------------------------

        if self.state == RallyState.CLICK_MARCH:

            name, score, loc, hw = match_any(img,self.march_templates)

            if score >= MARCH_THRESHOLD:

                tap_match_in_fullscreen((0,0,img.shape[1],img.shape[0]),loc,hw)

                self.log("[RALLY] march clicked")

                self.state = RallyState.CONFIRM
                self.cooldown_until = time.time()+2
                return


        # -----------------------------------------
        # OPTIONAL CONFIRM POPUP
        # -----------------------------------------

        if self.state == RallyState.CONFIRM:

            name, score, loc, hw = match_any(img,self.confirm_templates)

            if score >= CONFIRM_THRESHOLD:

                tap_match_in_fullscreen((0,0,img.shape[1],img.shape[0]),loc,hw)

                self.log("[RALLY] confirm popup")

            self.finish()
            return


    def finish(self):

        self.log("[RALLY] finished")

        self.state = RallyState.IDLE
        self.cooldown_until = time.time()+5

        WORKFLOW_MANAGER.release(Workflow.RALLY)

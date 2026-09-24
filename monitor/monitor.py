#!/usr/bin/env python3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import os
import subprocess
import threading
import time

import cv2
import numpy as np

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from flow_control import (
    FLOW_NAMES,
    MINISTRY_NAMES,
    get_all_flow_states,
    get_all_ministry_states,
    is_flow_enabled,
    is_ministry_enabled,
    set_flow_enabled,
    set_ministry_enabled,
)

# =========================
# CONFIG
# =========================

HOST = "192.168.0.55"
PORT = 8000

BASE_DIR = Path(__file__).resolve().parent.parent

IMAGE_PATH = BASE_DIR / "debug" / "screen_treasure.png"
ICON_TEMPLATE_PATH = BASE_DIR / "boot" / "boot_icon.png"
HEAL_BATCH_PATH = BASE_DIR / "heal_batch.txt"

# Modalità controllo:
#   "adb"     -> Android / emulatore
#   "xdotool" -> Linux desktop X11
CONTROL_MODE = "adb"

# ---- ADB MODE ----
ADB_PATH = "adb"
ADB_SERIAL = None        # es: "emulator-5554" oppure None
ADB_BACK_KEYCODE = "4"   # KEYCODE_BACK
ADB_HOME_KEYCODE = "3"   # KEYCODE_HOME
ADB_BACK_COUNT = 3
ADB_BACK_DELAY = 0.35
ADB_CONFIRM_DELAY = 0.70

# coordinate dalla tua immagine
ADB_OK_X = 313
ADB_OK_Y = 1455

# match icona Last Z
ADB_TAP_MATCH_THRESHOLD = 0.80
ADB_SCREENSHOT_TIMEOUT = 15
ADB_ICON_SCALE_MIN = 4.45
ADB_ICON_SCALE_MAX = 4.75
ADB_ICON_SCALE_STEPS = 7

# Bottone CALIBRA: doppio tap basso-destra
ADB_CALIBRA_X_RATIO = 0.92
ADB_CALIBRA_Y_RATIO = 0.945
ADB_CALIBRA_SLEEP = 1.0

# ---- XDO MODE ----
GAME_WINDOW_NAME = ""    # opzionale; es: "BlueStacks" o nome finestra gioco
XDO_BACK_KEY = "Escape"  # cambia se nel tuo caso il "back" è un altro tasto
# XDO_OK_X = ...
# XDO_OK_Y = ...

ACTION_LOCK = threading.Lock()
CLIENT_GONE_ERRORS = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)

HTML = """<!doctype html>
<html lang="it">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LastZAlert Monitor</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #090c10;
      --panel: #11161d;
      --panel2: #171d25;
      --border: #29313c;
      --text: #e8edf2;
      --muted: #8995a3;
      --green: #38d67a;
      --red: #ef5350;
      --blue: #4ea1ff;
      --yellow: #ffd54f;
    }

    * {
      box-sizing: border-box;
    }

    html, body {
      margin: 0;
      min-height: 100%;
      background: var(--bg);
      color: var(--text);
      font-family: Arial, Helvetica, sans-serif;
    }

    body {
      min-height: 100vh;
    }

    .wrap {
      width: 100%;
      min-height: 100vh;
      display: grid;
      grid-template-columns: 290px minmax(0, 1fr);
      grid-template-rows: auto minmax(0, 1fr) auto;
    }

    .console-header {
      grid-column: 1 / -1;
      height: 58px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 20px;
      padding: 0 18px;
      background: #0d1117;
      border-bottom: 1px solid var(--border);
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      min-width: 0;
    }

    .brand-mark {
      width: 11px;
      height: 11px;
      border-radius: 50%;
      background: var(--green);
      box-shadow: 0 0 12px rgba(56, 214, 122, .75);
      flex: 0 0 auto;
    }

    .brand-title {
      font-size: 16px;
      font-weight: 700;
      letter-spacing: 1.2px;
      white-space: nowrap;
    }

    .brand-subtitle {
      margin-top: 2px;
      color: var(--muted);
      font-size: 10px;
      letter-spacing: 1.5px;
      text-transform: uppercase;
    }

    .status {
      max-width: 55vw;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      color: var(--muted);
      font-size: 13px;
      text-align: right;
    }

    .status.ok {
      color: var(--green);
    }

    .status.err {
      color: #ff8585;
    }

    .sidebar {
      grid-column: 1;
      grid-row: 2;
      min-height: 0;
      overflow-y: auto;
      padding: 14px;
      background: var(--panel);
      border-right: 1px solid var(--border);
    }

    .section {
      margin-bottom: 16px;
    }

    .section-title {
      margin: 0 0 8px;
      color: var(--muted);
      font-size: 10px;
      font-weight: 700;
      letter-spacing: 1.7px;
      text-transform: uppercase;
    }

    .toolbar {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 7px;
    }

    button,
    input {
      font: inherit;
    }

    button {
      min-height: 38px;
      padding: 8px 10px;
      border: 1px solid var(--border);
      border-radius: 7px;
      background: var(--panel2);
      color: var(--text);
      cursor: pointer;
      transition:
        background .15s ease,
        border-color .15s ease,
        transform .05s ease;
    }

    button:hover {
      background: #202833;
      border-color: #3b4654;
    }

    button:active {
      transform: translateY(1px);
    }

    button:disabled {
      opacity: .55;
      cursor: wait;
    }

    button.primary {
      border-color: #2864a8;
      background: #17375c;
    }

    button.primary:hover {
      background: #204a78;
    }

    button.danger {
      border-color: #743334;
      background: #451f20;
    }

    button.danger:hover {
      background: #5b2829;
    }

    #allWorkflowsBtn {
      min-height: 48px;
      font-size: 14px;
      font-weight: 800;
      letter-spacing: .7px;
      border-color: #29975b;
      background: #17613a;
    }

    #allWorkflowsBtn:hover {
      filter: brightness(1.15);
    }

    #allWorkflowsBtn.master-off {
      border-color: #743334;
      background: #451f20;
    }

    #allWorkflowsBtn.master-partial {
      border-color: #8a6d24;
      background: #554516;
    }

    #touchControlBtn {
      grid-column: 1 / -1;
    }

    .flow-list {
      display: flex;
      flex-direction: column;
      gap: 5px;
    }

    .flow-switch {
      min-height: 37px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 7px 10px;
      border: 1px solid var(--border);
      border-radius: 7px;
      background: var(--panel2);
      cursor: pointer;
      user-select: none;
    }

    .flow-switch:hover {
      background: #1d252e;
    }

    .flow-switch span {
      font-size: 12px;
      font-weight: 600;
      letter-spacing: .35px;
    }

    .flow-switch input {
      appearance: none;
      -webkit-appearance: none;
      width: 34px;
      height: 18px;
      margin: 0;
      padding: 0;
      position: relative;
      flex: 0 0 auto;
      border: 1px solid #4a5563;
      border-radius: 20px;
      background: #2a3038;
      cursor: pointer;
      transition: .18s ease;
    }

    .flow-switch input::after {
      content: "";
      position: absolute;
      top: 2px;
      left: 2px;
      width: 12px;
      height: 12px;
      border-radius: 50%;
      background: #9ba5b0;
      transition: .18s ease;
    }

    .flow-switch input:checked {
      border-color: #29975b;
      background: #17613a;
    }

    .flow-switch input:checked::after {
      left: 18px;
      background: #66ee9c;
    }

    .ministry-group {
      margin: 2px 0 3px 14px;
      padding: 5px 0 2px 9px;
      border-left: 2px solid #303b47;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }

    .ministry-group .flow-switch {
      min-height: 32px;
      background: #121820;
      border-color: #242d37;
    }

    .ministry-group .flow-switch span {
      color: #b8c2cc;
      font-size: 11px;
      font-weight: 500;
    }

    .heal-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 72px 48px;
      gap: 5px;
      margin-bottom: 5px;
    }

    .heal-row .flow-switch {
      min-width: 0;
    }

    #healBatchInput {
      width: 100%;
      min-width: 0;
      height: 37px;
      padding: 6px 7px;
      border: 1px solid var(--border);
      border-radius: 7px;
      background: #0e1319;
      color: var(--text);
      text-align: center;
    }

    #healBatchBtn {
      min-height: 37px;
      padding: 5px;
      font-size: 11px;
    }

    .main-panel {
      grid-column: 2;
      grid-row: 2;
      min-width: 0;
      min-height: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: auto;
      padding: 14px;
      background:
        radial-gradient(circle at center, #141a21 0, #0b0e12 70%);
    }

    .screen-wrap {
      position: relative;
      display: inline-block;
      max-width: 100%;
      max-height: calc(100vh - 112px);
      line-height: 0;
    }

    #screen {
      display: block;
      max-width: 100%;
      max-height: calc(100vh - 112px);
      width: auto;
      height: auto;
      object-fit: contain;
      border: 1px solid #303945;
      border-radius: 6px;
      background: #000;
      box-shadow: 0 12px 40px rgba(0, 0, 0, .45);
      -webkit-user-drag: none;
      user-select: none;
    }

    #screen.touch-enabled {
      cursor: crosshair;
      touch-action: none;
      outline: 2px solid var(--green);
      outline-offset: 2px;
    }

    .touch-marker {
      position: absolute;
      width: 22px;
      height: 22px;
      border: 3px solid var(--yellow);
      border-radius: 50%;
      transform: translate(-50%, -50%);
      pointer-events: none;
      opacity: 0;
      transition: opacity .15s ease;
      box-sizing: border-box;
      z-index: 10;
    }

    .touch-marker.show {
      opacity: 1;
    }

    .console-footer {
      grid-column: 1 / -1;
      grid-row: 3;
      min-height: 40px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 15px;
      padding: 7px 16px;
      background: #0d1117;
      border-top: 1px solid var(--border);
      color: var(--muted);
      font-size: 11px;
    }

    .footer-left,
    .footer-right {
      display: flex;
      align-items: center;
      gap: 14px;
      min-width: 0;
    }

    .footer-item {
      white-space: nowrap;
    }

    .touch-state {
      color: #c5ced8;
    }

    .small {
      color: var(--muted);
      font-size: 11px;
    }

    @media (max-width: 800px) {
      .wrap {
        display: flex;
        min-height: 100vh;
        flex-direction: column;
      }

      .console-header {
        width: 100%;
        height: auto;
        min-height: 52px;
        padding: 9px 12px;
      }

      .brand-subtitle {
        display: none;
      }

      .status {
        max-width: 48vw;
        font-size: 11px;
      }

      .sidebar {
        width: 100%;
        overflow: visible;
        padding: 10px;
        border-right: 0;
        border-bottom: 1px solid var(--border);
      }

      .toolbar {
        grid-template-columns: repeat(3, 1fr);
      }

      #touchControlBtn {
        grid-column: auto;
      }

      .flow-list {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }

      .ministry-group {
        grid-column: 1 / -1;
        margin-left: 8px;
      }

      .main-panel {
        width: 100%;
        overflow: visible;
        padding: 8px;
      }

      .screen-wrap,
      #screen {
        max-height: none;
        max-width: 100%;
      }

      .console-footer {
        width: 100%;
        flex-wrap: wrap;
      }

      .footer-right {
        display: none;
      }
    }

    @media (max-width: 480px) {
      .toolbar {
        grid-template-columns: 1fr 1fr;
      }

      .flow-list {
        grid-template-columns: 1fr;
      }

      .heal-row {
        grid-template-columns: minmax(0, 1fr) 70px 46px;
      }

      .brand-title {
        font-size: 13px;
      }
    }
  </style>
</head>
<body>
  <div class="wrap">

    <header class="console-header">
      <div class="brand">
        <div class="brand-mark"></div>
        <div>
          <div class="brand-title">LASTZALERT CONTROL CONSOLE</div>
          <div class="brand-subtitle">Android automation monitor</div>
        </div>
      </div>

      <div class="status" id="status">Connessione…</div>
    </header>

    <aside class="sidebar">

      <section class="section">
        <div class="section-title">Game control</div>

        <div class="toolbar">
          <button class="primary" id="tapIconBtn" onclick="tapLastZIcon()">▶ START</button>
          <button class="danger" id="closeBtn" onclick="closeGame()">■ CLOSE</button>

          <button id="homeBtn" onclick="androidHome()">⌂ HOME</button>
          <button id="backBtn" onclick="androidBack()">← BACK</button>

          <button id="calibraBtn" onclick="calibra()">◎ CALIBRATE</button>

          <button
            id="allWorkflowsBtn"
            class="master-workflows"
            onclick="toggleAllWorkflows()"
          >⚡ ALL WF: ON</button>

          <button id="touchControlBtn" onclick="toggleTouchControl()">📱 CONTROLLO TOUCH: OFF</button>
        </div>
      </section>

      <section class="section">
        <div class="section-title">Workflows</div>

        <div class="flow-list">

          <label class="flow-switch">
            <span>TREASURE</span>
            <input id="treasureSwitch" type="checkbox"
                   onchange="setFlowEnabled('treasure', this.checked)">
          </label>

          <label class="flow-switch">
            <span>HQ</span>
            <input id="hqSwitch" type="checkbox"
                   onchange="setFlowEnabled('hq', this.checked)">
          </label>

          <div class="heal-row">
            <label class="flow-switch">
              <span>HEAL</span>
              <input id="healSwitch" type="checkbox"
                     onchange="setFlowEnabled('heal', this.checked)">
            </label>

            <input id="healBatchInput"
                   type="number"
                   min="1"
                   step="1"
                   placeholder="Batch">

            <button id="healBatchBtn"
                    onclick="setHealBatch()">SET</button>
          </div>

          <label class="flow-switch">
            <span>DONATION</span>
            <input id="donationSwitch" type="checkbox"
                   onchange="setFlowEnabled('donation', this.checked)">
          </label>

          <label class="flow-switch">
            <span>MINISTRY</span>
            <input id="ministrySwitch" type="checkbox"
                   onchange="setFlowEnabled('ministry', this.checked)">
          </label>

          <div class="ministry-group">
            <label class="flow-switch">
              <span>CONSTRUCTION</span>
              <input id="ministryConstructionSwitch"
                     type="checkbox"
                     onchange="setMinistryEnabled('construction', this.checked)">
            </label>

            <label class="flow-switch">
              <span>AGRICULTURE</span>
              <input id="ministryAgricultureSwitch"
                     type="checkbox"
                     onchange="setMinistryEnabled('agriculture', this.checked)">
            </label>

            <label class="flow-switch">
              <span>SCIENCE</span>
              <input id="ministryScienceSwitch"
                     type="checkbox"
                     onchange="setMinistryEnabled('science', this.checked)">
            </label>
          </div>

          <label class="flow-switch">
            <span>FORZIERE</span>
            <input id="forziereSwitch" type="checkbox"
                   onchange="setFlowEnabled('forziere', this.checked)">
          </label>

          <label class="flow-switch">
            <span>RESEARCH</span>
            <input id="researchSwitch" type="checkbox"
                   onchange="setFlowEnabled('research', this.checked)">
          </label>

          <label class="flow-switch">
            <span>RALLY</span>
            <input id="rallySwitch" type="checkbox"
                   onchange="setFlowEnabled('rally', this.checked)">
          </label>

          <label class="flow-switch">
            <span>HERO</span>
            <input id="heroSwitch" type="checkbox"
                   onchange="setFlowEnabled('hero', this.checked)">
          </label>

          <label class="flow-switch">
            <span>BOUNTY</span>
            <input id="bountySwitch" type="checkbox"
                   onchange="setFlowEnabled('bounty', this.checked)">
          </label>

          <label class="flow-switch">
            <span>GENERIC</span>
            <input id="genericSwitch" type="checkbox"
                   onchange="setFlowEnabled('generic', this.checked)">
          </label>

        </div>
      </section>

    </aside>

    <main class="main-panel">
      <div class="screen-wrap" id="screenWrap">
        <img
          id="screen"
          src="/image?v=init"
          alt="screen_treasure.png"
          draggable="false"
        />
        <div id="touchMarker" class="touch-marker"></div>
      </div>
    </main>

    <footer class="console-footer">
      <div class="footer-left">
        <span class="touch-state" id="touchHelp">Controllo touch disattivato</span>
        <span class="footer-item">Mode: <strong id="modeLabel"></strong></span>
      </div>

      <div class="footer-right">
        <span class="footer-item">debug/screen_treasure.png</span>
        <span class="footer-item">boot/boot_icon.png</span>
        <span class="footer-item">heal_batch.txt</span>
      </div>
    </footer>

  </div>

  <script>
    const img = document.getElementById("screen");
    const statusEl = document.getElementById("status");
    const closeBtn = document.getElementById("closeBtn");
    const tapIconBtn = document.getElementById("tapIconBtn");
    const calibraBtn = document.getElementById("calibraBtn");
    const allWorkflowsBtn = document.getElementById("allWorkflowsBtn");
    const backBtn = document.getElementById("backBtn");
    const homeBtn = document.getElementById("homeBtn");
    const healBatchInput = document.getElementById("healBatchInput");
    const healBatchBtn = document.getElementById("healBatchBtn");
    const modeLabel = document.getElementById("modeLabel");
    const touchControlBtn = document.getElementById("touchControlBtn");
    const screenWrap = document.getElementById("screenWrap");
    const touchMarker = document.getElementById("touchMarker");
    const touchHelp = document.getElementById("touchHelp");
    const flowSwitches = {
      treasure: document.getElementById("treasureSwitch"),
      hq: document.getElementById("hqSwitch"),
      heal: document.getElementById("healSwitch"),
      donation: document.getElementById("donationSwitch"),
      ministry: document.getElementById("ministrySwitch"),
      forziere: document.getElementById("forziereSwitch"),
      generic: document.getElementById("genericSwitch"),
      research: document.getElementById("researchSwitch"),
      rally: document.getElementById("rallySwitch"),
      hero: document.getElementById("heroSwitch"),
      bounty: document.getElementById("bountySwitch")
    };

    const ministrySwitches = {
      construction: document.getElementById("ministryConstructionSwitch"),
      science: document.getElementById("ministryScienceSwitch"),
      agriculture: document.getElementById("ministryAgricultureSwitch")
    };

    function updateAllWorkflowsButton() {
      const switches = [
        ...Object.values(flowSwitches),
        ...Object.values(ministrySwitches)
      ].filter(Boolean);

      const enabledCount =
        switches.filter(sw => sw.checked).length;

      const allEnabled =
        switches.length > 0 &&
        enabledCount === switches.length;

      const allDisabled =
        enabledCount === 0;

      allWorkflowsBtn.classList.remove(
        "master-off",
        "master-partial"
      );

      if (allEnabled) {
        allWorkflowsBtn.textContent = "⚡ ALL WF: ON";
        allWorkflowsBtn.title =
          "Click to disable all workflows";

      } else if (allDisabled) {
        allWorkflowsBtn.textContent = "⚡ ALL WF: OFF";
        allWorkflowsBtn.classList.add("master-off");
        allWorkflowsBtn.title =
          "Click to enable all workflows";

      } else {
        allWorkflowsBtn.textContent =
          "⚡ ALL WF: " +
          enabledCount +
          "/" +
          switches.length;

        allWorkflowsBtn.classList.add(
          "master-partial"
        );

        allWorkflowsBtn.title =
          "Click to enable all workflows";
      }
    }


    async function toggleAllWorkflows() {
      const switches = [
        ...Object.values(flowSwitches),
        ...Object.values(ministrySwitches)
      ].filter(Boolean);

      const allEnabled =
        switches.length > 0 &&
        switches.every(sw => sw.checked);

      // If everything is ON -> switch everything OFF.
      // Otherwise -> switch everything ON.
      const newState = !allEnabled;

      allWorkflowsBtn.disabled = true;

      setStatus(
        newState
          ? "Attivazione di tutti i workflow…"
          : "Disattivazione di tutti i workflow…"
      );

      try {
        for (const [flow, sw] of Object.entries(flowSwitches)) {
          if (sw.checked !== newState) {
            await setFlowEnabled(flow, newState);
          }
        }

        for (const [ministry, sw] of Object.entries(ministrySwitches)) {
          if (sw.checked !== newState) {
            await setMinistryEnabled(
              ministry,
              newState
            );
          }
        }

        updateAllWorkflowsButton();

        const finalSwitches = [
          ...Object.values(flowSwitches),
          ...Object.values(ministrySwitches)
        ].filter(Boolean);

        const success =
          finalSwitches.every(
            sw => sw.checked === newState
          );

        if (success) {
          setStatus(
            newState
              ? "Tutti i workflow sono ON"
              : "Tutti i workflow sono OFF",
            "ok"
          );
        } else {
          setStatus(
            "Alcuni workflow non hanno cambiato stato",
            "err"
          );
        }

      } finally {
        allWorkflowsBtn.disabled = false;
        updateAllWorkflowsButton();
      }
    }

    let touchControlEnabled = false;
    let gestureStart = null;
    let pendingTap = null;
    let markerTimer = null;

    const SWIPE_MIN_DISTANCE_PX = 15;
    const LONG_PRESS_THRESHOLD_MS = 650;
    const DOUBLE_TAP_WINDOW_MS = 280;
    const DOUBLE_TAP_DISTANCE_PX = 30;


    function toggleTouchControl() {
      touchControlEnabled = !touchControlEnabled;

      img.classList.toggle(
        "touch-enabled",
        touchControlEnabled
      );

      touchControlBtn.textContent = touchControlEnabled
        ? "📱 CONTROLLO TOUCH: ON"
        : "📱 CONTROLLO TOUCH: OFF";

      touchHelp.textContent = touchControlEnabled
        ? "Tap · doppio tap · pressione lunga · trascina per swipe"
        : "Controllo touch disattivato";

      if (!touchControlEnabled) {
        gestureStart = null;
        cancelPendingTap();
      }

      setStatus(
        touchControlEnabled
          ? "Controllo touchscreen attivato"
          : "Controllo touchscreen disattivato",
        touchControlEnabled ? "ok" : ""
      );
    }


    function cancelPendingTap() {
      if (pendingTap && pendingTap.timer) {
        clearTimeout(pendingTap.timer);
      }

      pendingTap = null;
    }


    function pointDistance(a, b) {
      const dx = a.localX - b.localX;
      const dy = a.localY - b.localY;

      return Math.sqrt(
        dx * dx + dy * dy
      );
    }


    function eventToAndroidPoint(ev) {
      const rect = img.getBoundingClientRect();
      const wrapRect = screenWrap.getBoundingClientRect();

      if (
        rect.width <= 0 ||
        rect.height <= 0 ||
        img.naturalWidth <= 0 ||
        img.naturalHeight <= 0
      ) {
        throw new Error(
          "Dimensioni screenshot non disponibili"
        );
      }

      let localX = ev.clientX - rect.left;
      let localY = ev.clientY - rect.top;

      localX = Math.max(
        0,
        Math.min(rect.width - 1, localX)
      );

      localY = Math.max(
        0,
        Math.min(rect.height - 1, localY)
      );

      const androidX = Math.max(
        0,
        Math.min(
          img.naturalWidth - 1,
          Math.round(
            localX *
            img.naturalWidth /
            rect.width
          )
        )
      );

      const androidY = Math.max(
        0,
        Math.min(
          img.naturalHeight - 1,
          Math.round(
            localY *
            img.naturalHeight /
            rect.height
          )
        )
      );

      return {
        x: androidX,
        y: androidY,

        localX: localX,
        localY: localY,

        markerX: ev.clientX - wrapRect.left,
        markerY: ev.clientY - wrapRect.top
      };
    }


    function showTouchMarker(point) {
      touchMarker.style.left =
        point.markerX + "px";

      touchMarker.style.top =
        point.markerY + "px";

      touchMarker.classList.add("show");

      if (markerTimer) {
        clearTimeout(markerTimer);
      }

      markerTimer = setTimeout(() => {
        touchMarker.classList.remove("show");
      }, 450);
    }


    async function sendTouch(payload, description) {
      try {
        const r = await fetch(
          "/action/touch",
          {
            method: "POST",

            headers: {
              "Content-Type": "application/json"
            },

            body: JSON.stringify(payload)
          }
        );

        const data = await r.json();

        if (data.ok) {
          setStatus(
            description + " → " + data.detail,
            "ok"
          );
        } else {
          setStatus(
            "Errore touch: " +
            (data.error || "sconosciuto"),
            "err"
          );
        }

      } catch (e) {
        setStatus(
          "Errore comunicazione touch",
          "err"
        );
      }
    }


    function scheduleTap(point) {
      const now = performance.now();

      if (pendingTap) {
        const elapsed =
          now - pendingTap.time;

        const distance =
          pointDistance(
            point,
            pendingTap.point
          );

        if (
          elapsed <= DOUBLE_TAP_WINDOW_MS &&
          distance <= DOUBLE_TAP_DISTANCE_PX
        ) {
          clearTimeout(
            pendingTap.timer
          );

          const firstPoint =
            pendingTap.point;

          pendingTap = null;

          showTouchMarker(point);

          sendTouch(
            {
              action: "double_tap",
              x: firstPoint.x,
              y: firstPoint.y
            },
            "DOUBLE TAP " +
            firstPoint.x +
            "," +
            firstPoint.y
          );

          return;
        }

        clearTimeout(
          pendingTap.timer
        );

        const oldPoint =
          pendingTap.point;

        sendTouch(
          {
            action: "tap",
            x: oldPoint.x,
            y: oldPoint.y
          },
          "TAP " +
          oldPoint.x +
          "," +
          oldPoint.y
        );

        pendingTap = null;
      }

      const timer = setTimeout(() => {
        if (!pendingTap) {
          return;
        }

        const tapPoint =
          pendingTap.point;

        pendingTap = null;

        sendTouch(
          {
            action: "tap",
            x: tapPoint.x,
            y: tapPoint.y
          },
          "TAP " +
          tapPoint.x +
          "," +
          tapPoint.y
        );

      }, DOUBLE_TAP_WINDOW_MS);

      pendingTap = {
        point: point,
        time: now,
        timer: timer
      };
    }


    img.addEventListener(
      "pointerdown",
      (ev) => {
        if (!touchControlEnabled) {
          return;
        }

        if (
          ev.pointerType === "mouse" &&
          ev.button !== 0
        ) {
          return;
        }

        ev.preventDefault();

        try {
          img.setPointerCapture(
            ev.pointerId
          );
        } catch (e) {
        }

        let point;

        try {
          point =
            eventToAndroidPoint(ev);
        } catch (e) {
          setStatus(
            e.message,
            "err"
          );

          return;
        }

        gestureStart = {
          pointerId: ev.pointerId,
          point: point,
          clientX: ev.clientX,
          clientY: ev.clientY,
          time: performance.now()
        };

        showTouchMarker(point);
      }
    );


    img.addEventListener(
      "pointermove",
      (ev) => {
        if (
          !touchControlEnabled ||
          !gestureStart ||
          gestureStart.pointerId !== ev.pointerId
        ) {
          return;
        }

        ev.preventDefault();
      }
    );


    img.addEventListener(
      "pointerup",
      (ev) => {
        if (
          !touchControlEnabled ||
          !gestureStart ||
          gestureStart.pointerId !== ev.pointerId
        ) {
          return;
        }

        ev.preventDefault();

        let endPoint;

        try {
          endPoint =
            eventToAndroidPoint(ev);
        } catch (e) {
          gestureStart = null;

          setStatus(
            e.message,
            "err"
          );

          return;
        }

        const elapsed =
          performance.now() -
          gestureStart.time;

        const dx =
          ev.clientX -
          gestureStart.clientX;

        const dy =
          ev.clientY -
          gestureStart.clientY;

        const distance =
          Math.sqrt(
            dx * dx + dy * dy
          );

        const startPoint =
          gestureStart.point;

        gestureStart = null;

        try {
          if (
            img.hasPointerCapture(
              ev.pointerId
            )
          ) {
            img.releasePointerCapture(
              ev.pointerId
            );
          }
        } catch (e) {
        }

        showTouchMarker(endPoint);

        if (
          distance >=
          SWIPE_MIN_DISTANCE_PX
        ) {
          cancelPendingTap();

          const durationMs =
            Math.min(
              5000,
              Math.max(
                100,
                Math.round(elapsed)
              )
            );

          sendTouch(
            {
              action: "swipe",

              x1: startPoint.x,
              y1: startPoint.y,

              x2: endPoint.x,
              y2: endPoint.y,

              duration_ms: durationMs
            },
            "SWIPE " +
            startPoint.x +
            "," +
            startPoint.y +
            " → " +
            endPoint.x +
            "," +
            endPoint.y
          );

          return;
        }

        if (
          elapsed >=
          LONG_PRESS_THRESHOLD_MS
        ) {
          cancelPendingTap();

          const durationMs =
            Math.min(
              5000,
              Math.max(
                500,
                Math.round(elapsed)
              )
            );

          sendTouch(
            {
              action: "long_press",
              x: startPoint.x,
              y: startPoint.y,
              duration_ms: durationMs
            },
            "LONG PRESS " +
            startPoint.x +
            "," +
            startPoint.y +
            " " +
            durationMs +
            "ms"
          );

          return;
        }

        scheduleTap(
          endPoint
        );
      }
    );


    img.addEventListener(
      "pointercancel",
      (ev) => {
        if (
          gestureStart &&
          gestureStart.pointerId ===
          ev.pointerId
        ) {
          gestureStart = null;
        }
      }
    );


    img.addEventListener(
      "contextmenu",
      (ev) => {
        if (
          touchControlEnabled
        ) {
          ev.preventDefault();
        }
      }
    );


    img.addEventListener(
      "dragstart",
      (ev) => {
        ev.preventDefault();
      }
    );

    function setStatus(text, cls = "") {
      statusEl.textContent = text;
      statusEl.className = "status " + cls;
    }

    function refreshImage(version) {
      img.src = "/image?v=" + encodeURIComponent(version || Date.now());
    }

    async function fetchConfig() {
      try {
        const r = await fetch("/config");
        const data = await r.json();
        modeLabel.textContent = data.control_mode || "-";

        if (data.heal_batch !== undefined && data.heal_batch !== null) {
          healBatchInput.value = data.heal_batch;
        }
        
        if (data.flows) {
          Object.entries(flowSwitches).forEach(([flow, sw]) => {
            sw.checked = data.flows[flow] !== false;
          });
        }

        if (data.ministries) {
          Object.entries(ministrySwitches).forEach(([ministry, sw]) => {
            sw.checked = data.ministries[ministry] !== false;
          });
        }

        updateAllWorkflowsButton();
        
      } catch (e) {
        modeLabel.textContent = "errore";
      }
    }

    async function setMinistryEnabled(ministry, enabled) {
      const sw = ministrySwitches[ministry];

      if (!sw) {
        setStatus("Ministero non valido: " + ministry, "err");
        return;
      }

      sw.disabled = true;

      try {
        const r = await fetch("/action/set-ministry-enabled", {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            ministry: ministry,
            enabled: enabled
          })
        });

        const data = await r.json();

        if (data.ok) {
          sw.checked = data.enabled;

          setStatus(
            "MINISTRY " + ministry.toUpperCase() + " " +
            (data.enabled ? "ON" : "OFF"),
            "ok"
          );
        } else {
          sw.checked = !enabled;
          setStatus(
            "Errore: " + (data.error || "sconosciuto"),
            "err"
          );
        }

      } catch (e) {
        sw.checked = !enabled;
        setStatus("Errore chiamata comando", "err");

      } finally {
        sw.disabled = false;
        updateAllWorkflowsButton();
      }
    }


    async function closeGame() {
      closeBtn.disabled = true;
      setStatus("Invio comando chiusura gioco…");

      try {
        const r = await fetch("/action/close-game", { method: "POST" });
        const data = await r.json();

        if (data.ok) {
          setStatus("Comando eseguito: " + data.detail, "ok");
        } else {
          setStatus("Errore: " + (data.error || "sconosciuto"), "err");
        }
      } catch (e) {
        setStatus("Errore chiamata comando", "err");
      } finally {
        closeBtn.disabled = false;
      }
    }

    async function tapLastZIcon() {
      tapIconBtn.disabled = true;
      setStatus("Cerco l'icona Last Z sullo schermo…");

      try {
        const r = await fetch("/action/tap-lastz-icon", { method: "POST" });
        const data = await r.json();

        if (data.ok) {
          setStatus("Icona premuta: " + data.detail, "ok");
        } else {
          setStatus("Errore: " + (data.error || "sconosciuto"), "err");
        }
      } catch (e) {
        setStatus("Errore chiamata comando", "err");
      } finally {
        tapIconBtn.disabled = false;
      }
    }

    async function calibra() {
      calibraBtn.disabled = true;
      setStatus("Calibrazione: doppio tap basso-destra…");

      try {
        const r = await fetch("/action/calibra", { method: "POST" });
        const data = await r.json();

        if (data.ok) {
          setStatus("CALIBRA eseguito: " + data.detail, "ok");
        } else {
          setStatus("Errore: " + (data.error || "sconosciuto"), "err");
        }
      } catch (e) {
        setStatus("Errore chiamata comando", "err");
      } finally {
        calibraBtn.disabled = false;
      }
    }

    async function androidBack() {
      backBtn.disabled = true;
      setStatus("Invio BACK Android…");
    
      try {
        const r = await fetch("/action/back", { method: "POST" });
        const data = await r.json();
    
        if (data.ok) {
          setStatus("BACK eseguito: " + data.detail, "ok");
        } else {
          setStatus("Errore: " + (data.error || "sconosciuto"), "err");
        }
      } catch (e) {
        setStatus("Errore chiamata comando", "err");
      } finally {
        backBtn.disabled = false;
      }
    }

    async function androidHome() {
      homeBtn.disabled = true;
      setStatus("Invio HOME Android…");
    
      try {
        const r = await fetch("/action/home", { method: "POST" });
        const data = await r.json();
    
        if (data.ok) {
          setStatus("HOME eseguito: " + data.detail, "ok");
        } else {
          setStatus("Errore: " + (data.error || "sconosciuto"), "err");
        }
      } catch (e) {
        setStatus("Errore chiamata comando", "err");
      } finally {
        homeBtn.disabled = false;
      }
    }

    async function setHealBatch() {
      const value = parseInt(healBatchInput.value, 10);

      if (!Number.isInteger(value) || value <= 0) {
        setStatus("Heal batch non valido", "err");
        return;
      }

      healBatchBtn.disabled = true;
      setStatus("Salvo heal batch…");

      try {
        const r = await fetch("/action/set-heal-batch", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ value })
        });

        const data = await r.json();

        if (data.ok) {
          healBatchInput.value = data.value;
          setStatus("Heal batch salvato: " + data.value, "ok");
        } else {
          setStatus("Errore: " + (data.error || "sconosciuto"), "err");
        }
      } catch (e) {
        setStatus("Errore chiamata comando", "err");
      } finally {
        healBatchBtn.disabled = false;
      }
    }

    async function setFlowEnabled(flow, enabled) {
      const sw = flowSwitches[flow];

      if (!sw) {
        setStatus("Workflow non valido: " + flow, "err");
        return;
      }

      sw.disabled = true;

      try {
        const r = await fetch("/action/set-flow-enabled", {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            flow: flow,
            enabled: enabled
          })
        });

        const data = await r.json();

        if (data.ok) {
          sw.checked = data.enabled;

          setStatus(
            flow.toUpperCase() + " " + (data.enabled ? "ON" : "OFF"),
            "ok"
          );

        } else {
          sw.checked = !enabled;

          setStatus(
            "Errore: " + (data.error || "sconosciuto"),
            "err"
          );
        }

      } catch (e) {
        sw.checked = !enabled;

        setStatus(
          "Errore modifica " + flow,
          "err"
        );

      } finally {
        sw.disabled = false;
        updateAllWorkflowsButton();
      }
    }

    img.onerror = () => setStatus("Immagine non disponibile", "err");

    function startSSE() {
      const es = new EventSource("/events");

      es.onopen = () => {
        setStatus("In ascolto aggiornamenti…");
      };

      es.onmessage = (ev) => {
        try {
          const data = JSON.parse(ev.data);
          if (data.version) {
            refreshImage(data.version);
          }
        } catch (e) {
          refreshImage(Date.now());
        }
      };

      es.onerror = () => {
        setStatus("Connessione persa, ritento…", "err");
      };
    }

    fetchConfig();
    startSSE();
  </script>
</body>
</html>
"""


def is_client_gone_error(e):
    return isinstance(e, CLIENT_GONE_ERRORS) or getattr(e, "errno", None) in (32, 104)


def safe_send(handler, status_code, content_type, body, extra_headers=None):
    try:
        handler.send_response(status_code)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(len(body)))
        handler.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")

        if extra_headers:
            for k, v in extra_headers.items():
                handler.send_header(k, v)

        handler.end_headers()

        if body:
            handler.wfile.write(body)

        return True

    except Exception as e:
        if is_client_gone_error(e):
            return False
        raise


def get_image_version():
    try:
        st = IMAGE_PATH.stat()
        return f"{st.st_mtime_ns}-{st.st_size}"
    except FileNotFoundError:
        return "missing"


def json_response(handler, status_code, payload):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return safe_send(handler, status_code, "application/json; charset=utf-8", body)


def text_response(handler, status_code, text):
    body = text.encode("utf-8")
    return safe_send(handler, status_code, "text/plain; charset=utf-8", body)


def run_cmd(cmd, timeout=10):
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )


def adb_prefix():
    cmd = [ADB_PATH]
    if ADB_SERIAL:
        cmd += ["-s", ADB_SERIAL]
    return cmd


def adb_exec_bytes(args, timeout=ADB_SCREENSHOT_TIMEOUT):
    result = subprocess.run(
        adb_prefix() + args,
        capture_output=True,
        timeout=timeout,
        check=True,
    )
    return result.stdout


def adb_capture_screen_cv():
    data = adb_exec_bytes(["exec-out", "screencap", "-p"])
    if not data:
        raise RuntimeError("Screenshot adb vuoto")

    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Impossibile decodificare screenshot adb")

    return img


def find_template_on_screen(screen_bgr, template_path, threshold=ADB_TAP_MATCH_THRESHOLD):
    if not template_path.exists():
        raise RuntimeError(f"Template non trovato: {template_path}")

    template_bgr = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
    if template_bgr is None:
        raise RuntimeError(f"Impossibile leggere template: {template_path}")

    screen_gray = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)

    sh, sw = screen_gray.shape[:2]
    best = None

    for scale in np.linspace(ADB_ICON_SCALE_MIN, ADB_ICON_SCALE_MAX, ADB_ICON_SCALE_STEPS):
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        resized = cv2.resize(template_gray, None, fx=scale, fy=scale, interpolation=interp)
        th, tw = resized.shape[:2]

        if tw < 5 or th < 5 or tw > sw or th > sh:
            continue

        result = cv2.matchTemplate(screen_gray, resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if best is None or max_val > best["score"]:
            x, y = max_loc
            best = {
                "x": x,
                "y": y,
                "w": tw,
                "h": th,
                "center_x": x + tw // 2,
                "center_y": y + th // 2,
                "score": float(max_val),
                "scale": float(scale),
            }

    if best is None or best["score"] < threshold:
        score = 0.0 if best is None else best["score"]
        scale = 0.0 if best is None else best["scale"]
        raise RuntimeError(
            f"Icona non trovata con confidenza sufficiente "
            f"(score={score:.3f}, scale={scale:.2f}, threshold={threshold:.3f})"
        )

    return best


def activate_window_if_needed():
    if not GAME_WINDOW_NAME.strip():
        return

    out = run_cmd(["xdotool", "search", "--name", GAME_WINDOW_NAME])
    window_ids = [x.strip() for x in out.stdout.splitlines() if x.strip()]
    if not window_ids:
        raise RuntimeError(f"Nessuna finestra trovata con nome: {GAME_WINDOW_NAME}")

    run_cmd(["xdotool", "windowactivate", "--sync", window_ids[0]])


def read_heal_batch():
    try:
        value = HEAL_BATCH_PATH.read_text(encoding="utf-8").strip()
        return int(value)
    except Exception:
        return None


def set_heal_batch(value):
    try:
        value = int(value)
    except Exception:
        raise RuntimeError("heal batch deve essere un numero intero")

    if value <= 0:
        raise RuntimeError("heal batch deve essere maggiore di zero")

    HEAL_BATCH_PATH.parent.mkdir(parents=True, exist_ok=True)
    HEAL_BATCH_PATH.write_text(f"{value}\n", encoding="utf-8")
    return value

def android_back():
    if CONTROL_MODE != "adb":
        raise RuntimeError("back supporta solo CONTROL_MODE='adb'")

    run_cmd(adb_prefix() + [
        "shell", "input", "keyevent", ADB_BACK_KEYCODE
    ])

    return f"adb BACK keyevent {ADB_BACK_KEYCODE}"

def android_home():
    if CONTROL_MODE != "adb":
        raise RuntimeError("home supporta solo CONTROL_MODE='adb'")

    run_cmd(adb_prefix() + [
        "shell", "input", "keyevent", ADB_HOME_KEYCODE
    ])

    return f"adb HOME keyevent {ADB_HOME_KEYCODE}"

def touch_int(payload, key, minimum=0, maximum=10000):
    value = payload.get(key)

    if isinstance(value, bool):
        raise RuntimeError(
            f"{key} non valido"
        )

    try:
        value = int(value)
    except Exception:
        raise RuntimeError(
            f"{key} deve essere un numero intero"
        )

    if value < minimum or value > maximum:
        raise RuntimeError(
            f"{key} fuori range: {value}"
        )

    return value


def execute_touch_action(payload):
    if CONTROL_MODE != "adb":
        raise RuntimeError(
            "touch supporta solo CONTROL_MODE='adb'"
        )

    action = payload.get("action")

    if action == "tap":
        x = touch_int(
            payload,
            "x"
        )

        y = touch_int(
            payload,
            "y"
        )

        run_cmd(
            adb_prefix() + [
                "shell",
                "input",
                "tap",
                str(x),
                str(y),
            ]
        )

        detail = (
            f"tap ({x},{y})"
        )

        print(
            f"[TOUCH] {detail}",
            flush=True,
        )

        return detail


    if action == "double_tap":
        x = touch_int(
            payload,
            "x"
        )

        y = touch_int(
            payload,
            "y"
        )

        cmd = (
            adb_prefix() + [
                "shell",
                "input",
                "tap",
                str(x),
                str(y),
            ]
        )

        run_cmd(cmd)

        time.sleep(
            0.10
        )

        run_cmd(cmd)

        detail = (
            f"double tap ({x},{y})"
        )

        print(
            f"[TOUCH] {detail}",
            flush=True,
        )

        return detail


    if action == "long_press":
        x = touch_int(
            payload,
            "x"
        )

        y = touch_int(
            payload,
            "y"
        )

        duration_ms = touch_int(
            payload,
            "duration_ms",
            300,
            5000,
        )

        run_cmd(
            adb_prefix() + [
                "shell",
                "input",
                "swipe",
                str(x),
                str(y),
                str(x),
                str(y),
                str(duration_ms),
            ]
        )

        detail = (
            f"long press "
            f"({x},{y}) "
            f"{duration_ms}ms"
        )

        print(
            f"[TOUCH] {detail}",
            flush=True,
        )

        return detail


    if action == "swipe":
        x1 = touch_int(
            payload,
            "x1"
        )

        y1 = touch_int(
            payload,
            "y1"
        )

        x2 = touch_int(
            payload,
            "x2"
        )

        y2 = touch_int(
            payload,
            "y2"
        )

        duration_ms = touch_int(
            payload,
            "duration_ms",
            100,
            5000,
        )

        run_cmd(
            adb_prefix() + [
                "shell",
                "input",
                "swipe",
                str(x1),
                str(y1),
                str(x2),
                str(y2),
                str(duration_ms),
            ]
        )

        detail = (
            f"swipe "
            f"({x1},{y1}) "
            f"-> "
            f"({x2},{y2}) "
            f"{duration_ms}ms"
        )

        print(
            f"[TOUCH] {detail}",
            flush=True,
        )

        return detail


    raise RuntimeError(
        f"azione touch non valida: {action}"
    )

def close_game():
    if CONTROL_MODE == "adb":
        for _ in range(ADB_BACK_COUNT):
            run_cmd(adb_prefix() + ["shell", "input", "keyevent", ADB_BACK_KEYCODE])
            time.sleep(ADB_BACK_DELAY)

        time.sleep(ADB_CONFIRM_DELAY)
        run_cmd(adb_prefix() + ["shell", "input", "tap", str(ADB_OK_X), str(ADB_OK_Y)])
        return f"adb: {ADB_BACK_COUNT}x BACK + tap Confirm ({ADB_OK_X},{ADB_OK_Y})"

    if CONTROL_MODE == "xdotool":
        activate_window_if_needed()
        run_cmd(["xdotool", "key", XDO_BACK_KEY])
        time.sleep(0.7)
        run_cmd(["xdotool", "mousemove", str(XDO_OK_X), str(XDO_OK_Y), "click", "1"])
        return f"xdotool: {XDO_BACK_KEY} + click su OK ({XDO_OK_X},{XDO_OK_Y})"

    raise RuntimeError(f"CONTROL_MODE non valido: {CONTROL_MODE}")


def tap_lastz_icon():
    if CONTROL_MODE != "adb":
        raise RuntimeError("tap_lastz_icon supporta solo CONTROL_MODE='adb'")

    screen = adb_capture_screen_cv()
    match = find_template_on_screen(screen, ICON_TEMPLATE_PATH)

    print(
        f"[LASTZ ICON] score={match['score']:.3f} "
        f"scale={match['scale']:.2f} "
        f"center=({match['center_x']},{match['center_y']})"
    )

    run_cmd(adb_prefix() + [
        "shell", "input", "tap",
        str(match["center_x"]), str(match["center_y"]),
    ])

    return (
        f"tap su ({match['center_x']},{match['center_y']}) "
        f"score={match['score']:.3f} scale={match['scale']:.2f}"
    )


def calibra_bottom_right():
    if CONTROL_MODE != "adb":
        raise RuntimeError("calibra supporta solo CONTROL_MODE='adb'")

    screen = adb_capture_screen_cv()
    h, w = screen.shape[:2]

    x = int(w * ADB_CALIBRA_X_RATIO)
    y = int(h * ADB_CALIBRA_Y_RATIO)

    run_cmd(adb_prefix() + ["shell", "input", "tap", str(x), str(y)])
    time.sleep(ADB_CALIBRA_SLEEP)
    run_cmd(adb_prefix() + ["shell", "input", "tap", str(x), str(y)])

    return f"2x tap su ({x},{y}) con sleep {ADB_CALIBRA_SLEEP}s"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            return self.serve_index()

        if self.path.startswith("/image"):
            return self.serve_image()

        if self.path.startswith("/events"):
            return self.serve_events()

        if self.path.startswith("/config"):
            return json_response(self, 200, {
                "control_mode": CONTROL_MODE,
                "image_path": str(IMAGE_PATH),
                "icon_template_path": str(ICON_TEMPLATE_PATH),
                "heal_batch_path": str(HEAL_BATCH_PATH),
                "heal_batch": read_heal_batch(),
                "flows": get_all_flow_states(),
                "ministries": get_all_ministry_states(),
            })

        return text_response(self, 404, "404 Not Found\n")

    def do_POST(self):
        if self.path == "/action/close-game":
            return self.handle_locked_action(close_game)

        if self.path == "/action/tap-lastz-icon":
            return self.handle_locked_action(tap_lastz_icon)

        if self.path == "/action/calibra":
            return self.handle_locked_action(calibra_bottom_right)

        if self.path == "/action/back":
            return self.handle_locked_action(android_back)

        if self.path == "/action/touch":
            return self.handle_touch_action()

        if self.path == "/action/set-heal-batch":
            return self.handle_set_heal_batch()

        if self.path == "/action/set-flow-enabled":
            return self.handle_set_flow_enabled()

        if self.path == "/action/set-ministry-enabled":
            return self.handle_set_ministry_enabled()

        if self.path == "/action/home":
            return self.handle_locked_action(android_home)

        return json_response(self, 404, {"ok": False, "error": "not found"})

    def read_request_body(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length > 0:
            return self.rfile.read(length)
        return b""

    def handle_locked_action(self, action_func):
        self.read_request_body()

        if not ACTION_LOCK.acquire(blocking=False):
            return json_response(self, 409, {
                "ok": False,
                "error": "azione già in corso",
            })

        try:
            detail = action_func()
            return json_response(self, 200, {
                "ok": True,
                "detail": detail,
            })
        except subprocess.CalledProcessError as e:
            err = (e.stderr or e.stdout or str(e)).strip()
            return json_response(self, 500, {
                "ok": False,
                "error": err or "comando fallito",
            })
        except Exception as e:
            return json_response(self, 500, {
                "ok": False,
                "error": str(e),
            })
        finally:
            ACTION_LOCK.release()

    def handle_touch_action(self):
        try:
            raw = self.read_request_body()

            payload = json.loads(
                raw.decode("utf-8") or "{}"
            )

        except Exception as e:
            return json_response(
                self,
                400,
                {
                    "ok": False,
                    "error": (
                        "payload touch non valido: "
                        + str(e)
                    ),
                },
            )

        if not ACTION_LOCK.acquire(
            blocking=False
        ):
            return json_response(
                self,
                409,
                {
                    "ok": False,
                    "error": "azione già in corso",
                },
            )

        try:
            detail = execute_touch_action(
                payload
            )

            return json_response(
                self,
                200,
                {
                    "ok": True,
                    "detail": detail,
                },
            )

        except subprocess.CalledProcessError as e:
            err = (
                e.stderr
                or e.stdout
                or str(e)
            ).strip()

            return json_response(
                self,
                500,
                {
                    "ok": False,
                    "error": (
                        err
                        or "comando ADB fallito"
                    ),
                },
            )

        except Exception as e:
            return json_response(
                self,
                500,
                {
                    "ok": False,
                    "error": str(e),
                },
            )

        finally:
            ACTION_LOCK.release()

    def handle_set_heal_batch(self):
        try:
            raw = self.read_request_body()
            payload = json.loads(raw.decode("utf-8") or "{}")
            value = set_heal_batch(payload.get("value"))

            return json_response(self, 200, {
                "ok": True,
                "value": value,
                "file": str(HEAL_BATCH_PATH),
            })
        except Exception as e:
            return json_response(self, 500, {
                "ok": False,
                "error": str(e),
            })

    def handle_set_flow_enabled(self):
        try:
            raw = self.read_request_body()

            payload = json.loads(
                raw.decode("utf-8") or "{}"
            )

            flow = payload.get("flow")
            enabled = payload.get("enabled")

            if flow not in FLOW_NAMES:
                raise RuntimeError(
                    f"flow non valido: {flow}"
                )

            if not isinstance(enabled, bool):
                raise RuntimeError(
                    "enabled deve essere true/false"
                )

            value = set_flow_enabled(
                flow,
                enabled,
            )

            return json_response(
                self,
                200,
                {
                    "ok": True,
                    "flow": flow,
                    "enabled": value,
                },
            )

        except Exception as e:
            return json_response(
                self,
                500,
                {
                    "ok": False,
                    "error": str(e),
                },
            )

    def handle_set_ministry_enabled(self):
        try:
            raw = self.read_request_body()

            payload = json.loads(
                raw.decode("utf-8") or "{}"
            )

            ministry = payload.get("ministry")
            enabled = payload.get("enabled")

            if ministry not in MINISTRY_NAMES:
                raise RuntimeError(
                    f"ministero non valido: {ministry}"
                )

            if not isinstance(enabled, bool):
                raise RuntimeError(
                    "enabled deve essere true/false"
                )

            value = set_ministry_enabled(
                ministry,
                enabled,
            )

            return json_response(
                self,
                200,
                {
                    "ok": True,
                    "ministry": ministry,
                    "enabled": value,
                },
            )

        except Exception as e:
            return json_response(
                self,
                500,
                {
                    "ok": False,
                    "error": str(e),
                },
            )

    def serve_index(self):
        return safe_send(
            self,
            200,
            "text/html; charset=utf-8",
            HTML.encode("utf-8"),
        )

    def serve_image(self):
        if not IMAGE_PATH.exists():
            msg = f"File non trovato: {IMAGE_PATH}\n".encode("utf-8")
            return safe_send(self, 404, "text/plain; charset=utf-8", msg)

        try:
            data = IMAGE_PATH.read_bytes()
        except Exception as e:
            msg = f"Errore lettura immagine: {e}\n".encode("utf-8")
            return safe_send(self, 500, "text/plain; charset=utf-8", msg)

        return safe_send(
            self,
            200,
            "image/png",
            data,
            {
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    def serve_events(self):
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()

            last_version = None

            while True:
                version = get_image_version()

                if version != last_version:
                    payload = json.dumps({"version": version})
                    self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    last_version = version

                time.sleep(0.5)

        except Exception as e:
            if is_client_gone_error(e):
                return
            return


def main():
    ThreadingHTTPServer.daemon_threads = True
    server = ThreadingHTTPServer((HOST, PORT), Handler)

    print(f"Server attivo su http://{HOST}:{PORT}")
    print(f"Immagine monitorata: {IMAGE_PATH}")
    print(f"Template icona: {ICON_TEMPLATE_PATH}")
    print(f"Heal batch file: {HEAL_BATCH_PATH}")
    print(f"CONTROL_MODE = {CONTROL_MODE}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nChiusura server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

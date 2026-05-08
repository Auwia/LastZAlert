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

# =========================
# CONFIG
# =========================

HOST = "192.168.0.55"
PORT = 8000

IMAGE_PATH = Path("/home/auwia/LastZAlert/debug/screen_treasure.png")
ICON_TEMPLATE_PATH = Path("/home/auwia/project/LastZAlert/boot/boot_icon.png")

# Modalità controllo:
#   "adb"     -> Android / emulatore
#   "xdotool" -> Linux desktop X11
CONTROL_MODE = "adb"

# ---- ADB MODE ----
ADB_PATH = "adb"
ADB_SERIAL = None        # es: "emulator-5554" oppure None
ADB_BACK_KEYCODE = "4"   # KEYCODE_BACK
ADB_BACK_COUNT = 3
ADB_BACK_DELAY = 0.35
ADB_CONFIRM_DELAY = 0.70

# coordinate dalla tua immagine
ADB_OK_X = 313
ADB_OK_Y = 1455

# match icona
ADB_TAP_MATCH_THRESHOLD = 0.48
ADB_SCREENSHOT_TIMEOUT = 15

# ---- XDO MODE ----
GAME_WINDOW_NAME = ""    # opzionale; es: "BlueStacks" o nome finestra gioco
XDO_BACK_KEY = "Escape"  # cambia se nel tuo caso il "back" è un altro tasto
# XDO_OK_X = ...
# XDO_OK_Y = ...

ACTION_LOCK = threading.Lock()

HTML = """<!doctype html>
<html lang="it">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LastZAlert Monitor</title>
  <style>
    :root { color-scheme: dark; }
    html, body {
      margin: 0;
      padding: 0;
      background: #111;
      color: #eee;
      font-family: Arial, sans-serif;
      min-height: 100vh;
    }
    body {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 12px;
      padding: 12px;
      box-sizing: border-box;
    }
    .wrap {
      width: 100%;
      max-width: 96vw;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 10px;
    }
    img {
      max-width: 96vw;
      max-height: 80vh;
      object-fit: contain;
      border: 1px solid #333;
      background: #000;
      box-shadow: 0 0 20px rgba(0,0,0,.35);
    }
    .status {
      font-size: 14px;
      opacity: .9;
    }
    .small {
      font-size: 12px;
      opacity: .65;
    }
    .err { color: #ff8f8f; }
    .ok { color: #8fffaa; }
    .toolbar {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      justify-content: center;
    }
    button {
      background: #2a2a2a;
      color: #fff;
      border: 1px solid #444;
      border-radius: 8px;
      padding: 10px 14px;
      cursor: pointer;
      font-size: 14px;
    }
    button:hover { background: #333; }
    button.danger {
      background: #7a1f1f;
      border-color: #a33;
    }
    button.danger:hover {
      background: #912626;
    }
    button:disabled {
      opacity: .6;
      cursor: wait;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="status" id="status">Connessione…</div>

    <div class="toolbar">
      <button class="danger" id="closeBtn" onclick="closeGame()">Chiudi gioco</button>
      <button id="tapIconBtn" onclick="tapLastZIcon()">Premi icona Last Z</button>
      <button onclick="refreshNow()">Refresh immagine</button>
    </div>

    <img id="screen" src="/image?v=init" alt="screen_treasure.png" />
    <div class="small">File monitorato: /home/auwia/project/LastZAlert/debug/screen_treasure.png</div>
    <div class="small">Template icona: /home/auwia/project/LastZAlert/debug/lastz_icon.png</div>
    <div class="small">Controllo gioco: <span id="modeLabel"></span></div>
  </div>

  <script>
    const img = document.getElementById("screen");
    const statusEl = document.getElementById("status");
    const closeBtn = document.getElementById("closeBtn");
    const tapIconBtn = document.getElementById("tapIconBtn");
    const modeLabel = document.getElementById("modeLabel");

    function setStatus(text, cls = "") {
      statusEl.textContent = text;
      statusEl.className = "status " + cls;
    }

    function refreshImage(version) {
      img.src = "/image?v=" + encodeURIComponent(version || Date.now());
    }

    function refreshNow() {
      refreshImage(Date.now());
    }

    async function fetchConfig() {
      try {
        const r = await fetch("/config");
        const data = await r.json();
        modeLabel.textContent = data.control_mode || "-";
      } catch (e) {
        modeLabel.textContent = "errore";
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

    img.onload = () => {
      if (!statusEl.textContent.startsWith("Comando eseguito") &&
          !statusEl.textContent.startsWith("Icona premuta")) {
        setStatus("Immagine aggiornata: " + new Date().toLocaleTimeString());
      }
    };

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

def get_image_version():
    try:
        st = IMAGE_PATH.stat()
        return f"{st.st_mtime_ns}-{st.st_size}"
    except FileNotFoundError:
        return "missing"

def json_response(handler, status_code, payload):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status_code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
    handler.end_headers()
    handler.wfile.write(body)

def run_cmd(cmd):
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=10,
        check=True
    )

def adb_prefix():
    cmd = [ADB_PATH]
    if ADB_SERIAL:
        cmd += ["-s", ADB_SERIAL]
    return cmd

def adb_exec_bytes(args, timeout=ADB_SCREENSHOT_TIMEOUT):
    cmd = adb_prefix() + args
    result = subprocess.run(
        cmd,
        capture_output=True,
        timeout=timeout,
        check=True
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

    template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
    if template is None:
        raise RuntimeError(f"Impossibile leggere template: {template_path}")

    th, tw = template.shape[:2]
    sh, sw = screen_bgr.shape[:2]

    if tw > sw or th > sh:
        raise RuntimeError(
            f"Template più grande dello schermo: template={tw}x{th}, screen={sw}x{sh}"
        )

    result = cv2.matchTemplate(screen_bgr, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val < threshold:
        raise RuntimeError(
            f"Icona non trovata con confidenza sufficiente "
            f"(score={max_val:.3f}, threshold={threshold:.3f})"
        )

    x, y = max_loc
    center_x = x + tw // 2
    center_y = y + th // 2

    return {
        "x": x,
        "y": y,
        "w": tw,
        "h": th,
        "center_x": center_x,
        "center_y": center_y,
        "score": float(max_val),
    }

def activate_window_if_needed():
    if not GAME_WINDOW_NAME.strip():
        return

    out = run_cmd(["xdotool", "search", "--name", GAME_WINDOW_NAME])
    window_ids = [x.strip() for x in out.stdout.splitlines() if x.strip()]
    if not window_ids:
        raise RuntimeError(f"Nessuna finestra trovata con nome: {GAME_WINDOW_NAME}")
    run_cmd(["xdotool", "windowactivate", "--sync", window_ids[0]])

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

    run_cmd(adb_prefix() + [
        "shell", "input", "tap",
        str(match["center_x"]), str(match["center_y"])
    ])

    return (
        f"tap su ({match['center_x']},{match['center_y']}) "
        f"score={match['score']:.3f}"
    )

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
            return json_response(self, 200, {"control_mode": CONTROL_MODE})

        self.send_response(404)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"404 Not Found")

    def do_POST(self):
        if self.path == "/action/close-game":
            return self.handle_close_game()

        if self.path == "/action/tap-lastz-icon":
            return self.handle_tap_lastz_icon()

        self.send_response(404)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(b'{"ok":false,"error":"not found"}')

    def serve_index(self):
        body = HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.end_headers()
        self.wfile.write(body)

    def serve_image(self):
        if not IMAGE_PATH.exists():
            msg = f"File non trovato: {IMAGE_PATH}\n".encode("utf-8")
            self.send_response(404)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(msg)))
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.end_headers()
            self.wfile.write(msg)
            return

        try:
            data = IMAGE_PATH.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            msg = f"Errore lettura immagine: {e}\n".encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)

    def serve_events(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        last_version = None
        try:
            while True:
                version = get_image_version()
                if version != last_version:
                    payload = json.dumps({"version": version})
                    self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    last_version = version
                time.sleep(0.5)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            pass

    def handle_close_game(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length > 0:
            _ = self.rfile.read(length)

        if not ACTION_LOCK.acquire(blocking=False):
            return json_response(self, 409, {
                "ok": False,
                "error": "azione già in corso"
            })

        try:
            detail = close_game()
            return json_response(self, 200, {
                "ok": True,
                "detail": detail
            })
        except subprocess.CalledProcessError as e:
            err = (e.stderr or e.stdout or str(e)).strip()
            return json_response(self, 500, {
                "ok": False,
                "error": err or "comando fallito"
            })
        except Exception as e:
            return json_response(self, 500, {
                "ok": False,
                "error": str(e)
            })
        finally:
            ACTION_LOCK.release()

    def handle_tap_lastz_icon(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length > 0:
            _ = self.rfile.read(length)

        if not ACTION_LOCK.acquire(blocking=False):
            return json_response(self, 409, {
                "ok": False,
                "error": "azione già in corso"
            })

        try:
            detail = tap_lastz_icon()
            return json_response(self, 200, {
                "ok": True,
                "detail": detail
            })
        except subprocess.CalledProcessError as e:
            err = (e.stderr or e.stdout or str(e)).strip()
            return json_response(self, 500, {
                "ok": False,
                "error": err or "comando fallito"
            })
        except Exception as e:
            return json_response(self, 500, {
                "ok": False,
                "error": str(e)
            })
        finally:
            ACTION_LOCK.release()

def main():
    os.makedirs("/home/auwia/project/LastZAlert/monitor", exist_ok=True)
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Server attivo su http://{HOST}:{PORT}")
    print(f"Immagine monitorata: {IMAGE_PATH}")
    print(f"Template icona: {ICON_TEMPLATE_PATH}")
    print(f"CONTROL_MODE = {CONTROL_MODE}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\\nChiusura server...")
    finally:
        server.server_close()

if __name__ == "__main__":
    main()

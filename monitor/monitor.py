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
HEAL_BATCH_PATH = Path("/home/auwia/LastZAlert/heal_batch.txt")

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
ADB_TAP_MATCH_THRESHOLD = 0.48
ADB_SCREENSHOT_TIMEOUT = 15
ADB_ICON_SCALE_MIN = 0.5
ADB_ICON_SCALE_MAX = 6.0
ADB_ICON_SCALE_STEPS = 111

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
      text-align: center;
    }
    .small {
      font-size: 12px;
      opacity: .65;
      text-align: center;
    }
    .err { color: #ff8f8f; }
    .ok { color: #8fffaa; }
    .toolbar {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      justify-content: center;
      align-items: center;
    }
    button, input {
      border-radius: 8px;
      padding: 10px 14px;
      font-size: 14px;
    }
    button {
      background: #2a2a2a;
      color: #fff;
      border: 1px solid #444;
      cursor: pointer;
    }
    button:hover { background: #333; }
    button.danger {
      background: #7a1f1f;
      border-color: #a33;
    }
    button.danger:hover { background: #912626; }
    button:disabled {
      opacity: .6;
      cursor: wait;
    }
    input {
      width: 110px;
      background: #1b1b1b;
      color: #fff;
      border: 1px solid #444;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="status" id="status">Connessione…</div>

    <div class="toolbar">
      <button class="danger" id="closeBtn" onclick="closeGame()">Chiudi gioco</button>
      <button id="tapIconBtn" onclick="tapLastZIcon()">Premi icona Last Z</button>
      <button id="calibraBtn" onclick="calibra()">CALIBRA</button>
      <button id="backBtn" onclick="androidBack()">BACK</button>
      <button id="homeBtn" onclick="androidHome()">HOME</button>
      <button onclick="refreshNow()">Refresh immagine</button>
      <input id="healBatchInput" type="number" min="1" step="1" placeholder="Heal batch" />
      <button id="healBatchBtn" onclick="setHealBatch()">Set heal batch</button>
    </div>

    <img id="screen" src="/image?v=init" alt="screen_treasure.png" />
    <div class="small">File monitorato: /home/auwia/LastZAlert/debug/screen_treasure.png</div>
    <div class="small">Template icona: /home/auwia/project/LastZAlert/boot/boot_icon.png</div>
    <div class="small">Heal batch: /home/auwia/LastZAlert/heal_batch.txt</div>
    <div class="small">Controllo gioco: <span id="modeLabel"></span></div>
  </div>

  <script>
    const img = document.getElementById("screen");
    const statusEl = document.getElementById("status");
    const closeBtn = document.getElementById("closeBtn");
    const tapIconBtn = document.getElementById("tapIconBtn");
    const calibraBtn = document.getElementById("calibraBtn");
    const backBtn = document.getElementById("backBtn");
    const homeBtn = document.getElementById("homeBtn");
    const healBatchInput = document.getElementById("healBatchInput");
    const healBatchBtn = document.getElementById("healBatchBtn");
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

        if (data.heal_batch !== undefined && data.heal_batch !== null) {
          healBatchInput.value = data.heal_batch;
        }
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

    img.onload = () => {
      const t = statusEl.textContent;
      if (!t.startsWith("Comando eseguito") &&
          !t.startsWith("Icona premuta") &&
          !t.startsWith("CALIBRA eseguito") &&
          !t.startsWith("Heal batch salvato")) {
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

        if self.path == "/action/set-heal-batch":
            return self.handle_set_heal_batch()

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
    os.makedirs("/home/auwia/LastZAlert/monitor", exist_ok=True)

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

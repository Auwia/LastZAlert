#!/usr/bin/env python3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import os
import time

HOST = "0.0.0.0"
PORT = 8000

IMAGE_PATH = Path("/home/auwia/project/LastZAlert/debug/screen_treasure.png")

HTML = """<!doctype html>
<html lang="it">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LastZAlert Monitor</title>
  <style>
    :root {
      color-scheme: dark;
    }
    html, body {
      margin: 0;
      padding: 0;
      background: #111;
      color: #eee;
      font-family: Arial, sans-serif;
      height: 100%%;
    }
    body {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 12px;
      min-height: 100vh;
    }
    .wrap {
      width: 100%%;
      max-width: 96vw;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 10px;
    }
    img {
      max-width: 96vw;
      max-height: 88vh;
      object-fit: contain;
      border: 1px solid #333;
      background: #000;
      box-shadow: 0 0 20px rgba(0,0,0,.35);
    }
    .status {
      font-size: 14px;
      opacity: .85;
    }
    .small {
      font-size: 12px;
      opacity: .65;
    }
    .err {
      color: #ff8f8f;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="status" id="status">Connessione…</div>
    <img id="screen" src="/image?v=init" alt="screen_treasure.png" />
    <div class="small">File monitorato: /home/auwia/project/LastZAlert/debug/screen_treasure.png</div>
  </div>

  <script>
    const img = document.getElementById("screen");
    const statusEl = document.getElementById("status");

    function setStatus(text, isError = false) {
      statusEl.textContent = text;
      statusEl.className = isError ? "status err" : "status";
    }

    function refreshImage(version) {
      img.src = "/image?v=" + encodeURIComponent(version || Date.now());
    }

    img.onload = () => setStatus("Immagine aggiornata: " + new Date().toLocaleTimeString());
    img.onerror = () => setStatus("Immagine non disponibile", true);

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
        setStatus("Connessione persa, ritento…", true);
      };
    }

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

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        # silenzioso
        return

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            self.serve_index()
            return

        if self.path.startswith("/image"):
            self.serve_image()
            return

        if self.path.startswith("/events"):
            self.serve_events()
            return

        self.send_response(404)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"404 Not Found")

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

def main():
    os.makedirs("/home/auwia/project/LastZAlert/monitor", exist_ok=True)
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Server attivo su http://127.0.0.1:{PORT}")
    print(f"Immagine monitorata: {IMAGE_PATH}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\\nChiusura server...")
    finally:
        server.server_close()

if __name__ == "__main__":
    main()

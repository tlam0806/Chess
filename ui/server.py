#!/usr/bin/env python3

import json
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UI_DIR = ROOT / "ui"
ENGINE = ROOT / "build" / "chess_engine_api"


class EngineProcess:
    def __init__(self):
        self.lock = threading.Lock()
        self.proc = subprocess.Popen(
            [str(ENGINE)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            cwd=str(ROOT),
        )

    def command(self, line):
        with self.lock:
            assert self.proc.stdin is not None
            assert self.proc.stdout is not None
            self.proc.stdin.write(line + "\n")
            self.proc.stdin.flush()
            response = self.proc.stdout.readline()
            return json.loads(response)


ENGINE_PROCESS = EngineProcess()


class Handler(BaseHTTPRequestHandler):
    def send_json(self, payload, status=200):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.serve_file(UI_DIR / "index.html", "text/html")
        elif self.path == "/app.js":
            self.serve_file(UI_DIR / "app.js", "application/javascript")
        elif self.path == "/styles.css":
            self.serve_file(UI_DIR / "styles.css", "text/css")
        elif self.path == "/api/state":
            self.send_json(ENGINE_PROCESS.command("state"))
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        payload = json.loads(body) if body else {}

        if self.path == "/api/move":
            self.send_json(ENGINE_PROCESS.command(f"move {payload.get('move', '')}"))
        elif self.path == "/api/bot":
            depth = int(payload.get("depth", 4))
            self.send_json(ENGINE_PROCESS.command(f"bot {depth}"))
        elif self.path == "/api/undo":
            self.send_json(ENGINE_PROCESS.command("undo"))
        elif self.path == "/api/undo_turn":
            self.send_json(ENGINE_PROCESS.command("undo_turn"))
        elif self.path == "/api/reset":
            self.send_json(ENGINE_PROCESS.command("reset"))
        else:
            self.send_error(404)

    def serve_file(self, path, content_type):
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        return


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    print("Open http://127.0.0.1:8765")
    server.serve_forever()


if __name__ == "__main__":
    main()

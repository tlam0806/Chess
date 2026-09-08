#!/usr/bin/env python3

import json
import subprocess
import threading
from urllib.parse import parse_qs, unquote, urlparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
UI_DIR = Path(__file__).resolve().parent
ENGINE = ROOT / "build" / "chess_engine_api"
DATA_DIR = ROOT / "data"
DEFAULT_DATASET = DATA_DIR / "value_depth8_v6_r12_600games.jsonl"


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
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self.serve_file(UI_DIR / "index.html", "text/html")
        elif path == "/replay.html":
            self.serve_file(UI_DIR / "replay.html", "text/html")
        elif path == "/dataset.html":
            self.serve_file(UI_DIR / "dataset.html", "text/html")
        elif path == "/app.js":
            self.serve_file(UI_DIR / "app.js", "application/javascript")
        elif path == "/replay.js":
            self.serve_file(UI_DIR / "replay.js", "application/javascript")
        elif path == "/dataset.js":
            self.serve_file(UI_DIR / "dataset.js", "application/javascript")
        elif path == "/styles.css":
            self.serve_file(UI_DIR / "styles.css", "text/css")
        elif path == "/api/state":
            self.send_json(ENGINE_PROCESS.command("state"))
        elif path == "/api/replays":
            self.send_json(list_replays())
        elif path == "/api/replay":
            query = parse_qs(parsed.query)
            name = query.get("file", [""])[0]
            self.send_json(load_replay(name))
        elif path == "/api/dataset_game":
            query = parse_qs(parsed.query)
            limit = int(query.get("limit", ["160"])[0])
            name = query.get("file", [""])[0]
            self.send_json(load_dataset_game(limit, name))
        elif path == "/api/dataset_view":
            query = parse_qs(parsed.query)
            name = query.get("file", [""])[0]
            self.send_json(load_dataset_view(name))
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


def list_replays():
    if not DATA_DIR.exists():
        return {"ok": True, "replays": []}
    replays = []
    for match_dir in sorted(DATA_DIR.glob("matches*")):
        if not match_dir.is_dir():
            continue
        for path in sorted(match_dir.glob("*.txt")):
            replays.append(str(path.relative_to(DATA_DIR)))
    return {"ok": True, "replays": replays}


def load_replay(name):
    relative_path = Path(unquote(name))
    parts = relative_path.parts
    if (
        len(parts) != 2
        or parts[0].startswith(".")
        or not parts[0].startswith("matches")
        or Path(parts[1]).name != parts[1]
    ):
        return {"ok": False, "message": "invalid replay name"}

    path = DATA_DIR / relative_path
    if not path.exists() or path.suffix != ".txt":
        return {"ok": False, "message": "replay not found"}

    lines = path.read_text(encoding="utf-8").splitlines()
    header = {}
    moves = []
    in_moves = False

    for line in lines:
        if not line.strip():
            in_moves = True
            continue
        if not in_moves:
            key, _, value = line.partition(" ")
            header[key] = value
            continue

        parts = line.split()
        if len(parts) < 8:
            continue
        fields = {}
        for index in range(4, len(parts) - 1, 2):
            fields[parts[index]] = parts[index + 1]
        moves.append({
            "ply": int(parts[0]),
            "side": parts[1],
            "engine": parts[2],
            "move": parts[3],
            "score": int(fields.get("score", "0")),
            "nodes": int(fields.get("nodes", "0")),
            "depth": int(fields.get("depth", header.get("depth", "0"))),
            "stopped": fields.get("stopped", "0") == "1",
        })

    return {"ok": True, "name": str(relative_path), "header": header, "moves": moves}


def square_name(square):
    return "abcdefgh"[square & 7] + str((square >> 3) + 1)


def decode_feature(index):
    piece_square = index % 64
    index //= 64
    king_square = index % 64
    index //= 64
    king_context = index % 2
    index //= 2
    piece_side = index % 2
    index //= 2
    piece_type = index % 6
    return piece_type, piece_side, king_context, king_square, piece_square


def decode_sample(sample, index):
    pieces = ["P", "N", "B", "R", "Q", "K"]
    side_to_move = "white" if index % 2 == 0 else "black"
    board = {}
    seen = set()

    for feature in sample["features"]:
        piece_type, piece_side, king_context, _, relative_square = decode_feature(int(feature))
        if king_context != 0:
            continue
        key = (piece_type, piece_side, relative_square)
        if key in seen:
            continue
        seen.add(key)

        absolute_square = relative_square if side_to_move == "white" else relative_square ^ 56
        friendly = piece_side == 0
        white_piece = friendly if side_to_move == "white" else not friendly
        piece = pieces[piece_type]
        board[square_name(absolute_square)] = piece if white_piece else piece.lower()

    target = int(sample["target"])
    white_target = target if side_to_move == "white" else -target
    return {
        "index": index,
        "side_to_move": side_to_move,
        "target": target,
        "white_target": white_target,
        "board": board,
        "aux": sample["aux"],
    }


def dataset_path(name, suffix):
    if not name:
        return DEFAULT_DATASET if suffix == ".jsonl" else None
    relative_path = Path(unquote(name))
    if (
        relative_path.is_absolute()
        or any(part.startswith(".") for part in relative_path.parts)
        or relative_path.parts[:1] != ("data",)
        or relative_path.suffix != suffix
    ):
        return None
    return ROOT / relative_path


def load_dataset_view(name):
    path = dataset_path(name, ".json")
    if path is None or not path.exists():
        return {"ok": False, "message": "dataset view not found"}
    return json.loads(path.read_text(encoding="utf-8"))


def load_dataset_game(limit, name=""):
    path = dataset_path(name, ".jsonl")
    if path is None or not path.exists():
        return {"ok": False, "message": "dataset not found"}

    samples = []
    with path.open("r", encoding="utf-8") as file:
        for index, line in enumerate(file):
            if index >= limit:
                break
            line = line.strip()
            if not line:
                continue
            samples.append(decode_sample(json.loads(line), index))

    return {
        "ok": True,
        "dataset": str(path.relative_to(ROOT)),
        "samples": samples,
    }


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    print("Open http://127.0.0.1:8765")
    server.serve_forever()


if __name__ == "__main__":
    main()

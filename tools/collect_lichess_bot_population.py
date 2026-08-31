#!/usr/bin/env python3
"""Collect a resumable, account-capped Lichess BOT position population.

The collector starts from a seed account's PGN, discovers BOT opponents, then
downloads recent rated blitz/rapid games for those accounts.  A position is
eligible only when a BOT with a game-time rating inside the requested range is
to move.  At most one teacher-valid position is selected from each normalized
quarter of a game.

All durable state lives in SQLite.  The compact CBin and JSONL artifacts are
deterministic exports of that database, so an interrupted export is harmless.
The API token is read only from an environment variable and is never persisted.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import sqlite3
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import chess
    import chess.pgn
except ImportError as exc:  # pragma: no cover - exercised by the CLI environment
    raise SystemExit(
        "python-chess is required; install it with `python -m pip install python-chess`"
    ) from exc


CBIN_HEADER = b"CHSCBIN2" + struct.pack("<HHI", 2, 13, 1)
CBIN_RECORD_SIZE = 40
BOT_TITLE = "BOT"
USERNAME_RE = re.compile(r"[^a-zA-Z0-9_-]+")


@dataclass(frozen=True)
class Config:
    min_elo: int
    max_elo: int
    min_base: int
    max_base: int
    min_increment: int
    max_increment: int
    target_positions: int
    min_accounts: int
    account_position_cap: int
    games_per_account: int
    random_seed: int
    excluded_accounts: frozenset[str]


@dataclass(frozen=True)
class Candidate:
    quarter: int
    fen: str
    board_bytes: bytes
    aux_bits: int
    model_key: bytes
    ply: int
    phase: int
    player: str
    player_elo: int
    opponent: str
    opponent_elo: int | None
    result: int


def normalize_username(value: str | None) -> str:
    return (value or "").strip().lower()


def parse_int(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def parse_time_control(value: str | None) -> tuple[int, int] | None:
    if not value or "+" not in value:
        return None
    base_text, increment_text = value.split("+", 1)
    try:
        return int(base_text), int(increment_text)
    except ValueError:
        return None


def game_result_for_color(result: str | None, color: chess.Color) -> int:
    if result == "1/2-1/2" or result == "*" or result is None:
        return 0
    if result == "1-0":
        return 1 if color == chess.WHITE else -1
    if result == "0-1":
        return 1 if color == chess.BLACK else -1
    return 0


def pack_relative_board(board: chess.Board) -> tuple[bytes, int]:
    """Encode exactly the board+aux key consumed by the F2/F2M trainers."""
    us = board.turn
    nibbles = [0] * 64
    for square, piece in board.piece_map().items():
        relative_square = square if us == chess.WHITE else square ^ 56
        friendly = piece.color == us
        nibbles[relative_square] = piece.piece_type + (0 if friendly else 6)

    packed = bytearray(32)
    for square in range(0, 64, 2):
        packed[square // 2] = nibbles[square] | (nibbles[square + 1] << 4)

    them = not us
    aux_bits = 0
    aux_bits |= int(board.has_kingside_castling_rights(us)) << 0
    aux_bits |= int(board.has_queenside_castling_rights(us)) << 1
    aux_bits |= int(board.has_kingside_castling_rights(them)) << 2
    aux_bits |= int(board.has_queenside_castling_rights(them)) << 3
    if board.ep_square is not None:
        relative_ep = board.ep_square if us == chess.WHITE else board.ep_square ^ 56
        aux_bits |= 1 << 4
        aux_bits |= 1 << (5 + chess.square_file(relative_ep))
    return bytes(packed), aux_bits


def mirror_aux_bits(aux_bits: int) -> int:
    values = [(aux_bits >> index) & 1 for index in range(13)]
    mirrored = [values[1], values[0], values[3], values[2], values[4], *reversed(values[5:13])]
    result = 0
    for index, value in enumerate(mirrored):
        result |= value << index
    return result


def canonical_f2m_key(board_bytes: bytes, aux_bits: int) -> tuple[bytes, int, bytes]:
    nibbles: list[int] = []
    for value in board_bytes:
        nibbles.extend((value & 0x0F, value >> 4))
    king_squares = [square for square, code in enumerate(nibbles) if code == 6]
    if len(king_squares) != 1:
        raise ValueError("position must contain exactly one friendly king")
    if chess.square_file(king_squares[0]) >= 4:
        mirrored = [0] * 64
        for square, code in enumerate(nibbles):
            mirrored[square ^ 7] = code
        packed = bytearray(32)
        for square in range(0, 64, 2):
            packed[square // 2] = mirrored[square] | (mirrored[square + 1] << 4)
        board_bytes = bytes(packed)
        aux_bits = mirror_aux_bits(aux_bits)
    payload = board_bytes + struct.pack("<H", aux_bits)
    return board_bytes, aux_bits, hashlib.blake2b(payload, digest_size=16).digest()


def eligible_players(game: chess.pgn.Game, config: Config) -> dict[chess.Color, tuple[str, int, str, int | None]]:
    headers = game.headers
    result: dict[chess.Color, tuple[str, int, str, int | None]] = {}
    for color, prefix, other_prefix in (
        (chess.WHITE, "White", "Black"),
        (chess.BLACK, "Black", "White"),
    ):
        username = normalize_username(headers.get(prefix))
        rating = parse_int(headers.get(prefix + "Elo"))
        if (
            headers.get(prefix + "Title") != BOT_TITLE
            or rating is None
            or not config.min_elo <= rating <= config.max_elo
            or username in config.excluded_accounts
        ):
            continue
        result[color] = (
            username,
            rating,
            normalize_username(headers.get(other_prefix)),
            parse_int(headers.get(other_prefix + "Elo")),
        )
    return result


def game_is_eligible(game: chess.pgn.Game, config: Config) -> bool:
    headers = game.headers
    if headers.get("Variant", "Standard") != "Standard":
        return False
    if "rated" not in headers.get("Event", "").lower():
        return False
    tc = parse_time_control(headers.get("TimeControl"))
    if tc is None:
        return False
    base, increment = tc
    return (
        config.min_base <= base <= config.max_base
        and config.min_increment <= increment <= config.max_increment
    )


def deterministic_order(length: int, seed: int, game_id: str, quarter: int) -> Iterable[int]:
    if length == 0:
        return ()
    digest = hashlib.blake2b(
        f"{seed}:{game_id}:{quarter}".encode("utf-8"), digest_size=8
    ).digest()
    start = int.from_bytes(digest, "little") % length
    return ((start + offset) % length for offset in range(length))


def game_candidates(game: chess.pgn.Game, config: Config) -> list[list[Candidate]]:
    players = eligible_players(game, config)
    moves = list(game.mainline_moves())
    quarters: list[list[Candidate]] = [[], [], [], []]
    if not players or not moves:
        return quarters

    board = game.board()
    game_result = game.headers.get("Result")
    total_plies = len(moves)
    for zero_ply, move in enumerate(moves):
        player = players.get(board.turn)
        if player is not None and not board.is_check() and not board.is_game_over(claim_draw=False):
            username, rating, opponent, opponent_rating = player
            raw_board, raw_aux = pack_relative_board(board)
            board_bytes, aux_bits, model_key = canonical_f2m_key(raw_board, raw_aux)
            quarter = min(3, 4 * zero_ply // total_plies)
            quarters[quarter].append(
                Candidate(
                    quarter=quarter,
                    fen=board.fen(en_passant="fen"),
                    board_bytes=board_bytes,
                    aux_bits=aux_bits,
                    model_key=model_key,
                    ply=zero_ply + 1,
                    phase=min((len(board.piece_map()) - 1) // 4, 7),
                    player=username,
                    player_elo=rating,
                    opponent=opponent,
                    opponent_elo=opponent_rating,
                    result=game_result_for_color(game_result, board.turn),
                )
            )
        board.push(move)
    return quarters


def connect_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS accounts(
            username TEXT PRIMARY KEY,
            status TEXT NOT NULL DEFAULT 'pending',
            discovered_from TEXT,
            error TEXT
        );
        CREATE TABLE IF NOT EXISTS games(
            game_id TEXT PRIMARY KEY,
            source_account TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS positions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_key BLOB NOT NULL UNIQUE,
            board BLOB NOT NULL,
            aux_bits INTEGER NOT NULL,
            fen TEXT NOT NULL,
            game_id TEXT NOT NULL,
            quarter INTEGER NOT NULL,
            ply INTEGER NOT NULL,
            phase INTEGER NOT NULL,
            player TEXT NOT NULL,
            player_elo INTEGER NOT NULL,
            opponent TEXT NOT NULL,
            opponent_elo INTEGER,
            result INTEGER NOT NULL,
            time_control TEXT NOT NULL,
            source_account TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS positions_player_idx ON positions(player);
        """
    )
    return connection


def config_json(config: Config) -> str:
    return json.dumps(
        {
            **config.__dict__,
            "excluded_accounts": sorted(config.excluded_accounts),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def initialize_database(connection: sqlite3.Connection, config: Config, resume: bool) -> None:
    expected = config_json(config)
    row = connection.execute("SELECT value FROM metadata WHERE key='config'").fetchone()
    if row is None:
        connection.execute("INSERT INTO metadata(key,value) VALUES('config',?)", (expected,))
        connection.commit()
    elif row[0] != expected:
        raise RuntimeError("collector configuration differs from the existing database")
    elif not resume:
        raise RuntimeError("output already contains a collector database; pass --resume")


def cache_name(username: str) -> str:
    return USERNAME_RE.sub("_", username) + ".pgn.gz"


def download_user_pgn(
    username: str,
    cache_dir: Path,
    token: str,
    games_per_account: int,
    retries: int,
) -> str:
    cache_path = cache_dir / cache_name(username)
    if cache_path.exists():
        with gzip.open(cache_path, "rt", encoding="utf-8") as stream:
            return stream.read()

    query = urllib.parse.urlencode(
        {
            "max": games_per_account,
            "rated": "true",
            "perfType": "blitz,rapid",
            "clocks": "false",
            "evals": "false",
            "opening": "false",
        }
    )
    url = f"https://lichess.org/api/games/user/{urllib.parse.quote(username)}?{query}"
    headers = {
        "Accept": "application/x-chess-pgn",
        "Authorization": f"Bearer {token}",
        "User-Agent": "ChessNNUEPopulationCollector/1.0",
    }
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as response:
                payload = response.read().decode("utf-8")
            temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
            with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as stream:
                stream.write(payload)
            temporary.replace(cache_path)
            return payload
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < retries:
                wait_seconds = max(60, int(exc.headers.get("Retry-After", "60")))
                print(f"rate_limit username={username} wait={wait_seconds}s", flush=True)
                time.sleep(wait_seconds)
                continue
            if exc.code in (404, 410):
                raise RuntimeError(f"account unavailable: HTTP {exc.code}") from exc
            if 500 <= exc.code < 600 and attempt < retries:
                time.sleep(2**attempt)
                continue
            raise
        except (TimeoutError, urllib.error.URLError) as exc:
            if attempt >= retries:
                raise
            time.sleep(2**attempt)
    raise AssertionError("retry loop exhausted")


def parse_games(pgn_text: str) -> Iterable[chess.pgn.Game]:
    stream = io.StringIO(pgn_text)
    while True:
        game = chess.pgn.read_game(stream)
        if game is None:
            return
        if game.errors:
            continue
        yield game


def raw_pgn_blocks(stream: Iterable[str]) -> Iterable[str]:
    """Split a Lichess database stream without parsing non-BOT games."""
    block: list[str] = []
    for line in stream:
        if line.startswith("[Event ") and block:
            yield "".join(block)
            block = [line]
        else:
            block.append(line)
    if block:
        yield "".join(block)


def insert_database_game(
    connection: sqlite3.Connection,
    game: chess.pgn.Game,
    source_id: str,
    config: Config,
    counts: dict[str, int],
    total_positions: int,
) -> tuple[int, int]:
    """Insert one database game and return (new positions, total positions)."""
    if not game_is_eligible(game, config):
        return 0, total_positions
    quarters = game_candidates(game, config)
    if not any(quarters):
        return 0, total_positions
    game_id = game.headers.get("GameId") or game.headers.get("Site", "").rsplit("/", 1)[-1]
    if not game_id:
        return 0, total_positions
    inserted = connection.execute(
        "INSERT OR IGNORE INTO games(game_id,source_account) VALUES(?,?)",
        (game_id, source_id),
    ).rowcount
    if not inserted:
        return 0, total_positions

    added = 0
    time_control = game.headers.get("TimeControl", "")
    for quarter, candidates in enumerate(quarters):
        if total_positions >= config.target_positions:
            break
        for index in deterministic_order(len(candidates), config.random_seed, game_id, quarter):
            candidate = candidates[index]
            if counts.get(candidate.player, 0) >= config.account_position_cap:
                continue
            try:
                connection.execute(
                    """
                    INSERT INTO positions(
                        model_key,board,aux_bits,fen,game_id,quarter,ply,phase,
                        player,player_elo,opponent,opponent_elo,result,time_control,source_account
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        candidate.model_key,
                        candidate.board_bytes,
                        candidate.aux_bits,
                        candidate.fen,
                        game_id,
                        candidate.quarter,
                        candidate.ply,
                        candidate.phase,
                        candidate.player,
                        candidate.player_elo,
                        candidate.opponent,
                        candidate.opponent_elo,
                        candidate.result,
                        time_control,
                        source_id,
                    ),
                )
            except sqlite3.IntegrityError:
                continue
            counts[candidate.player] = counts.get(candidate.player, 0) + 1
            total_positions += 1
            added += 1
            break
    return added, total_positions


def process_database_stream(
    connection: sqlite3.Connection,
    stream: Iterable[str],
    source_id: str,
    config: Config,
    progress_interval: int,
    source_position_limit: int | None,
) -> dict[str, int]:
    stats = {
        "scanned_games": 0,
        "bot_games": 0,
        "parsed_bot_games": 0,
        "new_positions": 0,
    }
    counts = account_position_counts(connection)
    total_positions = connection.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
    pending_commits = 0
    try:
        for block in raw_pgn_blocks(stream):
            stats["scanned_games"] += 1
            if 'Title "BOT"' not in block:
                if progress_interval and stats["scanned_games"] % progress_interval == 0:
                    print(
                        "database_progress "
                        f"scanned_games={stats['scanned_games']} bot_games={stats['bot_games']} "
                        f"positions={total_positions} accounts={len(counts)}",
                        flush=True,
                    )
                continue
            stats["bot_games"] += 1
            game = chess.pgn.read_game(io.StringIO(block))
            if game is None or game.errors:
                continue
            stats["parsed_bot_games"] += 1
            added, total_positions = insert_database_game(
                connection, game, source_id, config, counts, total_positions
            )
            stats["new_positions"] += added
            pending_commits += 1
            if pending_commits >= 250:
                connection.commit()
                pending_commits = 0
            if total_positions >= config.target_positions and len(counts) >= config.min_accounts:
                break
            if (
                source_position_limit is not None
                and stats["new_positions"] >= source_position_limit
            ):
                break
            if progress_interval and stats["scanned_games"] % progress_interval == 0:
                print(
                    "database_progress "
                    f"scanned_games={stats['scanned_games']} bot_games={stats['bot_games']} "
                    f"positions={total_positions} accounts={len(counts)}",
                    flush=True,
                )
    finally:
        connection.commit()
    return stats


def discover_bot_accounts(connection: sqlite3.Connection, game: chess.pgn.Game, source: str) -> None:
    for prefix in ("White", "Black"):
        if game.headers.get(prefix + "Title") != BOT_TITLE:
            continue
        username = normalize_username(game.headers.get(prefix))
        if username:
            connection.execute(
                "INSERT OR IGNORE INTO accounts(username,discovered_from) VALUES(?,?)",
                (username, source),
            )


def bootstrap_seed(
    connection: sqlite3.Connection,
    seed_username: str,
    pgn_text: str,
) -> tuple[int, int]:
    games = 0
    accounts_before = connection.execute("SELECT COUNT(*) FROM accounts").fetchone()[0]
    with connection:
        for game in parse_games(pgn_text):
            game_id = game.headers.get("GameId") or game.headers.get("Site", "").rsplit("/", 1)[-1]
            if not game_id:
                continue
            games += 1
            connection.execute(
                "INSERT OR IGNORE INTO games(game_id,source_account) VALUES(?,?)",
                (game_id, seed_username),
            )
            discover_bot_accounts(connection, game, seed_username)
        connection.execute(
            "INSERT INTO accounts(username,status,discovered_from) VALUES(?, 'processed', 'seed') "
            "ON CONFLICT(username) DO UPDATE SET status='processed'",
            (seed_username,),
        )
    accounts_after = connection.execute("SELECT COUNT(*) FROM accounts").fetchone()[0]
    return games, accounts_after - accounts_before


def account_position_counts(connection: sqlite3.Connection) -> dict[str, int]:
    return dict(connection.execute("SELECT player,COUNT(*) FROM positions GROUP BY player"))


def process_account(
    connection: sqlite3.Connection,
    username: str,
    pgn_text: str,
    config: Config,
) -> dict[str, int]:
    stats = {"downloaded_games": 0, "new_games": 0, "eligible_games": 0, "positions": 0}
    counts = account_position_counts(connection)
    total_positions = connection.execute("SELECT COUNT(*) FROM positions").fetchone()[0]

    with connection:
        for game in parse_games(pgn_text):
            stats["downloaded_games"] += 1
            discover_bot_accounts(connection, game, username)
            game_id = game.headers.get("GameId") or game.headers.get("Site", "").rsplit("/", 1)[-1]
            if not game_id:
                continue
            inserted = connection.execute(
                "INSERT OR IGNORE INTO games(game_id,source_account) VALUES(?,?)",
                (game_id, username),
            ).rowcount
            if not inserted:
                continue
            stats["new_games"] += 1
            if not game_is_eligible(game, config):
                continue
            quarters = game_candidates(game, config)
            if not any(quarters):
                continue
            stats["eligible_games"] += 1
            time_control = game.headers.get("TimeControl", "")
            for quarter, candidates in enumerate(quarters):
                if total_positions >= config.target_positions:
                    break
                for index in deterministic_order(len(candidates), config.random_seed, game_id, quarter):
                    candidate = candidates[index]
                    if counts.get(candidate.player, 0) >= config.account_position_cap:
                        continue
                    try:
                        connection.execute(
                            """
                            INSERT INTO positions(
                                model_key,board,aux_bits,fen,game_id,quarter,ply,phase,
                                player,player_elo,opponent,opponent_elo,result,time_control,source_account
                            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                            """,
                            (
                                candidate.model_key,
                                candidate.board_bytes,
                                candidate.aux_bits,
                                candidate.fen,
                                game_id,
                                candidate.quarter,
                                candidate.ply,
                                candidate.phase,
                                candidate.player,
                                candidate.player_elo,
                                candidate.opponent,
                                candidate.opponent_elo,
                                candidate.result,
                                time_control,
                                username,
                            ),
                        )
                    except sqlite3.IntegrityError:
                        continue
                    counts[candidate.player] = counts.get(candidate.player, 0) + 1
                    total_positions += 1
                    stats["positions"] += 1
                    break
        connection.execute(
            "UPDATE accounts SET status='processed',error=NULL WHERE username=?", (username,)
        )
    return stats


def population_status(connection: sqlite3.Connection) -> dict[str, object]:
    return {
        "positions": connection.execute("SELECT COUNT(*) FROM positions").fetchone()[0],
        "position_accounts": connection.execute(
            "SELECT COUNT(DISTINCT player) FROM positions"
        ).fetchone()[0],
        "games": connection.execute("SELECT COUNT(*) FROM games").fetchone()[0],
        "processed_accounts": connection.execute(
            "SELECT COUNT(*) FROM accounts WHERE status='processed'"
        ).fetchone()[0],
        "pending_accounts": connection.execute(
            "SELECT COUNT(*) FROM accounts WHERE status='pending'"
        ).fetchone()[0],
        "error_accounts": connection.execute(
            "SELECT COUNT(*) FROM accounts WHERE status='error'"
        ).fetchone()[0],
    }


def export_artifacts(connection: sqlite3.Connection, output_dir: Path, config: Config) -> dict[str, object]:
    cbin_path = output_dir / "positions.cbin"
    jsonl_path = output_dir / "positions.jsonl"
    cbin_tmp = cbin_path.with_suffix(".cbin.tmp")
    jsonl_tmp = jsonl_path.with_suffix(".jsonl.tmp")
    phase_counts = [0] * 8
    quarter_counts = [0] * 4
    account_counts: dict[str, int] = {}
    rows = connection.execute(
        """
        SELECT board,aux_bits,fen,game_id,quarter,ply,phase,player,player_elo,
               opponent,opponent_elo,result,time_control,source_account,hex(model_key)
        FROM positions ORDER BY id
        """
    )
    records = 0
    with cbin_tmp.open("wb") as cbin, jsonl_tmp.open("w", encoding="utf-8") as metadata:
        cbin.write(CBIN_HEADER)
        for row in rows:
            (
                board_bytes, aux_bits, fen, game_id, quarter, ply, phase, player,
                player_elo, opponent, opponent_elo, result, time_control, source_id,
                model_key,
            ) = row
            cbin.write(board_bytes)
            cbin.write(struct.pack("<HhHh", aux_bits, 0, ply, result))
            metadata.write(
                json.dumps(
                    {
                        "record": records,
                        "fen": fen,
                        "game_id": game_id,
                        "quarter": quarter,
                        "ply": ply,
                        "phase": phase,
                        "player": player,
                        "player_elo": player_elo,
                        "opponent": opponent,
                        "opponent_elo": opponent_elo,
                        "result": result,
                        "time_control": time_control,
                        "source_id": source_id,
                        "model_key": model_key.lower(),
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            records += 1
            phase_counts[phase] += 1
            quarter_counts[quarter] += 1
            account_counts[player] = account_counts.get(player, 0) + 1
    cbin_tmp.replace(cbin_path)
    jsonl_tmp.replace(jsonl_path)
    manifest = {
        "format": "lichess-bot-population-v1",
        "config": json.loads(config_json(config)),
        "records": records,
        "accounts": len(account_counts),
        "max_account_records": max(account_counts.values(), default=0),
        "phase_counts": phase_counts,
        "quarter_counts": quarter_counts,
        "status": population_status(connection),
        "artifacts": {
            "cbin": cbin_path.name,
            "metadata": jsonl_path.name,
        },
    }
    manifest_tmp = output_dir / "manifest.json.tmp"
    manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_tmp.replace(output_dir / "manifest.json")
    return manifest


def next_pending_account(connection: sqlite3.Connection, excluded: frozenset[str]) -> str | None:
    placeholders = ",".join("?" for _ in excluded)
    query = "SELECT username FROM accounts WHERE status='pending'"
    parameters: tuple[str, ...] = ()
    if excluded:
        query += f" AND username NOT IN ({placeholders})"
        parameters = tuple(sorted(excluded))
    query += " ORDER BY rowid LIMIT 1"
    row = connection.execute(query, parameters).fetchone()
    return row[0] if row else None


def read_seed_pgn(args: argparse.Namespace, cache_dir: Path, token: str) -> str:
    if args.seed_pgn is not None:
        return args.seed_pgn.read_text(encoding="utf-8")
    return download_user_pgn(
        args.seed_user,
        cache_dir,
        token,
        args.games_per_account,
        args.api_retries,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed-user", default="TrumCoVuaa")
    parser.add_argument("--seed-pgn", type=Path)
    parser.add_argument(
        "--database-pgn",
        help="Read an official Lichess PGN stream from this path, or '-' for stdin",
    )
    parser.add_argument("--database-source-id", default="lichess-database")
    parser.add_argument("--database-progress-interval", type=int, default=1_000_000)
    parser.add_argument(
        "--database-source-position-limit",
        type=int,
        help="Stop this input stream after adding this many positions",
    )
    parser.add_argument("--token-env", default="LICHESS_BOT_TOKEN")
    parser.add_argument("--min-elo", type=int, default=2300)
    parser.add_argument("--max-elo", type=int, default=2600)
    parser.add_argument("--min-base", type=int, default=60)
    parser.add_argument("--max-base", type=int, default=600)
    parser.add_argument("--min-increment", type=int, default=0)
    parser.add_argument("--max-increment", type=int, default=10)
    parser.add_argument("--target-positions", type=int, default=200_000)
    parser.add_argument("--min-accounts", type=int, default=100)
    parser.add_argument("--account-position-cap", type=int, default=2_000)
    parser.add_argument("--games-per-account", type=int, default=1_000)
    parser.add_argument("--random-seed", type=int, default=20260826)
    parser.add_argument("--api-retries", type=int, default=4)
    parser.add_argument("--max-processed-accounts", type=int, default=2_000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()

    if args.min_elo > args.max_elo:
        raise ValueError("--min-elo must not exceed --max-elo")
    if (
        args.target_positions <= 0
        or args.games_per_account <= 0
        or args.min_accounts <= 0
        or args.account_position_cap <= 0
    ):
        raise ValueError("targets, account cap, and games per account must be positive")
    if (
        args.database_source_position_limit is not None
        and args.database_source_position_limit <= 0
    ):
        raise ValueError("--database-source-position-limit must be positive")

    seed_username = normalize_username(args.seed_user)
    config = Config(
        min_elo=args.min_elo,
        max_elo=args.max_elo,
        min_base=args.min_base,
        max_base=args.max_base,
        min_increment=args.min_increment,
        max_increment=args.max_increment,
        target_positions=args.target_positions,
        min_accounts=args.min_accounts,
        account_position_cap=args.account_position_cap,
        games_per_account=args.games_per_account,
        random_seed=args.random_seed,
        excluded_accounts=frozenset({seed_username}),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "pgn_cache"
    cache_dir.mkdir(exist_ok=True)
    connection = connect_database(args.output_dir / "collector.sqlite3")
    initialize_database(connection, config, args.resume or args.status or args.export_only)

    if args.status:
        print(json.dumps(population_status(connection), indent=2, sort_keys=True))
        return 0
    if args.export_only:
        print(json.dumps(export_artifacts(connection, args.output_dir, config), indent=2, sort_keys=True))
        return 0

    if args.database_pgn is not None:
        if args.database_pgn == "-":
            stream = sys.stdin
            owner = None
        else:
            owner = Path(args.database_pgn).open("r", encoding="utf-8")
            stream = owner
        try:
            stats = process_database_stream(
                connection,
                stream,
                args.database_source_id,
                config,
                args.database_progress_interval,
                args.database_source_position_limit,
            )
        finally:
            if owner is not None:
                owner.close()
        status = population_status(connection)
        print("database_stream_done " + json.dumps({**stats, **status}, sort_keys=True), flush=True)
        if (
            int(status["positions"]) >= config.target_positions
            and int(status["position_accounts"]) >= config.min_accounts
        ):
            manifest = export_artifacts(connection, args.output_dir, config)
            print("collection_complete " + json.dumps(manifest, sort_keys=True), flush=True)
            return 0
        return 2

    token = os.environ.get(args.token_env, "").strip()
    if not token:
        raise RuntimeError(f"missing API token in environment variable {args.token_env}")

    if connection.execute("SELECT COUNT(*) FROM accounts").fetchone()[0] == 0:
        seed_pgn = read_seed_pgn(args, cache_dir, token)
        games, accounts = bootstrap_seed(connection, seed_username, seed_pgn)
        print(f"bootstrap games={games} discovered_accounts={accounts}", flush=True)

    while True:
        status = population_status(connection)
        if (
            int(status["positions"]) >= config.target_positions
            and int(status["position_accounts"]) >= config.min_accounts
        ):
            break
        if int(status["processed_accounts"]) >= args.max_processed_accounts:
            raise RuntimeError("reached --max-processed-accounts before population target")
        username = next_pending_account(connection, config.excluded_accounts)
        if username is None:
            raise RuntimeError("account discovery queue exhausted before population target")
        try:
            pgn = download_user_pgn(
                username, cache_dir, token, config.games_per_account, args.api_retries
            )
            account_stats = process_account(connection, username, pgn, config)
            time.sleep(0.25)
        except Exception as exc:
            with connection:
                connection.execute(
                    "UPDATE accounts SET status='error',error=? WHERE username=?",
                    (str(exc)[:1000], username),
                )
            print(f"account_error username={username} error={exc}", file=sys.stderr, flush=True)
            continue
        status = population_status(connection)
        print(
            "account_done "
            f"username={username} downloaded_games={account_stats['downloaded_games']} "
            f"new_games={account_stats['new_games']} eligible_games={account_stats['eligible_games']} "
            f"new_positions={account_stats['positions']} total_positions={status['positions']} "
            f"position_accounts={status['position_accounts']} pending={status['pending_accounts']}",
            flush=True,
        )

    manifest = export_artifacts(connection, args.output_dir, config)
    print("collection_complete " + json.dumps(manifest, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

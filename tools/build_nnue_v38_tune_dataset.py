#!/usr/bin/env python3
"""Build the fixed, disjoint 4k/2k/2k V38 selective-search dataset."""

from __future__ import annotations

import argparse
import hashlib
import random
import struct
import subprocess
from pathlib import Path

HEADER_SIZE = 16
RECORD_SIZE = 40
MAGIC = b"CHSCBIN2"
COUNTS = {
    "tune": {"random": 2800, "balanced": 600, "tactical": 400, "endgame": 200},
    "selection": {"random": 1400, "balanced": 300, "tactical": 200, "endgame": 100},
    "holdout": {"random": 1400, "balanced": 300, "tactical": 200, "endgame": 100},
}


def pieces(board: bytes) -> list[int]:
    out: list[int] = []
    for byte in board:
        out.extend((byte & 15, byte >> 4))
    return out


def is_pawn_endgame(board: bytes) -> bool:
    codes = pieces(board)
    return any(code in (1, 7) for code in codes) and not any(
        code in (2, 3, 4, 5, 8, 9, 10, 11) for code in codes
    )


def pseudo_tactical(board: bytes) -> bool:
    """Cheap deterministic capture/promotion proxy; legality is checked by the tuner."""
    sq = pieces(board)
    for source, code in enumerate(sq):
        if not 1 <= code <= 6:
            continue
        rank, file = divmod(source, 8)
        if code == 1:
            if rank == 6:
                return True
            for df in (-1, 1):
                target = source + 8 + df
                if 0 <= file + df < 8 and target < 64 and 7 <= sq[target] <= 12:
                    return True
        elif code == 2:
            for dr, df in ((2, 1), (2, -1), (1, 2), (1, -2),
                           (-1, 2), (-1, -2), (-2, 1), (-2, -1)):
                rr, ff = rank + dr, file + df
                if 0 <= rr < 8 and 0 <= ff < 8 and 7 <= sq[rr * 8 + ff] <= 12:
                    return True
        elif code in (3, 4, 5):
            directions = []
            if code in (3, 5):
                directions += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
            if code in (4, 5):
                directions += [(1, 0), (-1, 0), (0, 1), (0, -1)]
            for dr, df in directions:
                rr, ff = rank + dr, file + df
                while 0 <= rr < 8 and 0 <= ff < 8:
                    target = sq[rr * 8 + ff]
                    if target:
                        if 7 <= target <= 12:
                            return True
                        break
                    rr, ff = rr + dr, ff + df
        elif code == 6:
            for dr in (-1, 0, 1):
                for df in (-1, 0, 1):
                    rr, ff = rank + dr, file + df
                    if (dr or df) and 0 <= rr < 8 and 0 <= ff < 8:
                        if 7 <= sq[rr * 8 + ff] <= 12:
                            return True
    return False


def board_fen(board: bytes, aux: int) -> str:
    codes = pieces(board)
    chars = ".PNBRQKpnbrqk"
    ranks: list[str] = []
    for rank in range(7, -1, -1):
        text, empty = "", 0
        for file in range(8):
            code = codes[rank * 8 + file]
            if code == 0:
                empty += 1
            else:
                if empty:
                    text += str(empty)
                    empty = 0
                text += chars[code]
        if empty:
            text += str(empty)
        ranks.append(text)
    castle = "".join(
        ch for bit, ch in enumerate("KQkq") if aux & (1 << bit)
    ) or "-"
    ep = "-"
    if aux & (1 << 4):
        for file in range(8):
            if aux & (1 << (5 + file)):
                ep = chr(ord("a") + file) + "6"
                break
    return "/".join(ranks) + f" w {castle} {ep} 0 1"


def key_for(board: bytes, aux: int) -> str:
    return hashlib.sha256(board + struct.pack("<H", aux)).hexdigest()[:24]


def reservoir_add(
    reservoir: list[tuple[str, str]], item: tuple[str, str], seen: int,
    limit: int, rng: random.Random,
) -> None:
    if len(reservoir) < limit:
        reservoir.append(item)
    else:
        index = rng.randrange(seen)
        if index < limit:
            reservoir[index] = item


def sample_corpus(
    root: Path, seed: int, excluded: set[str],
    counts: dict[str, dict[str, int]],
) -> dict[str, list[tuple[str, str]]]:
    need = {
        "random": sum(v["random"] for v in counts.values()),
        "tactical": sum(v["tactical"] for v in counts.values()),
        "endgame": sum(v["endgame"] for v in counts.values()),
    }
    rng = random.Random(seed)
    shards = sorted(root.glob("*.cbin.zst"))
    rng.shuffle(shards)
    pools = {name: [] for name in need}
    seen_count = {name: 0 for name in need}
    used: set[str] = set()
    for shard in shards:
        proc = subprocess.Popen(
            ["zstd", "-q", "-dc", str(shard)], stdout=subprocess.PIPE
        )
        assert proc.stdout is not None
        header = proc.stdout.read(HEADER_SIZE)
        if not header.startswith(MAGIC):
            raise RuntimeError(f"bad CBIN header: {shard}")
        while True:
            record = proc.stdout.read(RECORD_SIZE)
            if not record:
                break
            if len(record) != RECORD_SIZE:
                raise RuntimeError(f"truncated CBIN record: {shard}")
            board = record[:32]
            aux = struct.unpack_from("<H", record, 32)[0]
            key = key_for(board, aux)
            if key in excluded or key in used:
                continue
            category = (
                "endgame" if is_pawn_endgame(board)
                else "tactical" if pseudo_tactical(board)
                else "random"
            )
            seen_count[category] += 1
            reservoir_add(
                pools[category], (key, board_fen(board, aux)),
                seen_count[category], need[category], rng,
            )
        if proc.wait() != 0:
            raise RuntimeError(f"zstd failed: {shard}")
        if all(len(pools[name]) >= need[name] for name in need):
            break
    for name, count in need.items():
        if len(pools[name]) < count:
            raise RuntimeError(f"not enough {name}: {len(pools[name])} < {count}")
        used.update(key for key, _ in pools[name])
        rng.shuffle(pools[name])
    return pools


def balanced_prefixes(
    path: Path, seed: int, excluded: set[str],
    counts: dict[str, dict[str, int]],
) -> list[tuple[str, str]]:
    prefixes: dict[str, tuple[str, str]] = {}
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        moves = line.split()
        for length in range(4, len(moves) + 1):
            payload = "book:" + " ".join(moves[:length])
            key = hashlib.sha256(payload.encode()).hexdigest()[:24]
            if key not in excluded:
                prefixes[key] = (key, payload)
    out = list(prefixes.values())
    random.Random(seed ^ 0xB41A).shuffle(out)
    required = sum(v["balanced"] for v in counts.values())
    if len(out) < required:
        raise RuntimeError(f"not enough unique balanced prefixes: {len(out)}")
    return out[:required]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--balanced", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--balanced-total", type=int)
    parser.add_argument(
        "--exclude-dataset-dir", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.scale <= 0:
        raise SystemExit("--scale must be positive")
    counts = {
        split: {
            category: count * args.scale
            for category, count in split_counts.items()
        }
        for split, split_counts in COUNTS.items()
    }
    if args.balanced_total is not None:
        if args.balanced_total <= 0:
            raise SystemExit("--balanced-total must be positive")
        base_balanced_total = sum(
            split_counts["balanced"] for split_counts in COUNTS.values())
        assigned = 0
        for split_index, (split, split_counts) in enumerate(counts.items()):
            if split_index + 1 == len(counts):
                balanced_count = args.balanced_total - assigned
            else:
                balanced_count = (
                    args.balanced_total
                    * COUNTS[split]["balanced"]
                    // base_balanced_total
                )
                assigned += balanced_count
            deficit = split_counts["balanced"] - balanced_count
            if deficit < 0:
                raise SystemExit(
                    "--balanced-total exceeds scaled balanced allocation")
            split_counts["balanced"] = balanced_count
            split_counts["random"] += deficit
    excluded: set[str] = set()
    for excluded_dir in args.exclude_dataset_dir:
        for split in ("tune", "selection", "holdout"):
            path = excluded_dir / f"{split}.tsv"
            for row in path.read_text().splitlines():
                fields = row.split("\t", 2)
                if len(fields) >= 2:
                    excluded.add(fields[1])
    pools = sample_corpus(args.corpus, args.seed, excluded, counts)
    pools["balanced"] = balanced_prefixes(
        args.balanced, args.seed, excluded, counts)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    offsets = {name: 0 for name in pools}
    global_keys: set[str] = set()
    for split, split_counts in counts.items():
        rows: list[str] = []
        for category, count in split_counts.items():
            start = offsets[category]
            selected = pools[category][start:start + count]
            offsets[category] += count
            for key, payload in selected:
                if key in global_keys:
                    raise RuntimeError(f"duplicate position hash: {key}")
                global_keys.add(key)
                rows.append(f"{category}\t{key}\t{payload}")
        random.Random(args.seed ^ sum(map(ord, split))).shuffle(rows)
        (args.output_dir / f"{split}.tsv").write_text("\n".join(rows) + "\n")
    manifest = (
        f"seed={args.seed}\n"
        f"hash=sha256_96bit\n"
        f"excluded={len(excluded)}\n"
        f"total={len(global_keys)}\n"
        + "".join(
            f"{split}={sum(split_counts.values())}\n"
            for split, split_counts in counts.items()
        )
    )
    (args.output_dir / "manifest.txt").write_text(manifest)
    print(f"dataset_complete output={args.output_dir} total={len(global_keys)}")


if __name__ == "__main__":
    main()

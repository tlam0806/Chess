from __future__ import annotations

import io
import struct
from pathlib import Path

from tools.data import build_nnue_v38_tune_dataset as builder


def cbin_record(aux: int, ply: int) -> bytes:
    record = bytearray(builder.RECORD_SIZE)
    record[0] = 0x66  # Two kings; categorisation remains "random".
    struct.pack_into("<H", record, 32, aux)
    struct.pack_into("<H", record, 36, ply)
    return bytes(record)


def test_record_ply_reads_uint16_at_offset_36() -> None:
    record = bytearray(builder.RECORD_SIZE)
    struct.pack_into("<H", record, 36, 513)
    assert builder.record_ply(bytes(record)) == 513


def test_balanced_pool_uses_one_full_prefix_free_line_and_real_ply(
    tmp_path,
) -> None:
    source = tmp_path / "openings.txt"
    source.write_text(
        "e2e4 e7e5 g1f3 b8c6\n"
        "e2e4 e7e5 g1f3 b8c6 f1b5\n"  # Supersedes prior prefix.
        "d2d4 d7d5 c2c4 e7e6\n"
        "c2c4 e7e5 b1c3 g8f6\n"
        "c2c4 e7e5 b1c3 g8f6\n"  # Exact duplicate source line.
    )
    counts = {
        "tune": {"balanced": 1},
        "selection": {"balanced": 1},
        "holdout": {"balanced": 1},
    }
    rows = builder.balanced_prefixes([source], 7, set(), counts)
    assert len(rows) == 3
    moves = [tuple(payload[5:].split()) for _, payload, _ in rows]
    assert all(ply == len(line) for (_, _, ply), line in zip(rows, moves))
    assert tuple("e2e4 e7e5 g1f3 b8c6".split()) not in moves
    assert all(
        not (len(left) < len(right) and right[:len(left)] == left)
        for left in moves for right in moves
    )


def test_corpus_reservoir_scans_all_selected_shards_deterministically(
    tmp_path, monkeypatch,
) -> None:
    shard_a = tmp_path / "part_00001.cbin.zst"
    shard_b = tmp_path / "part_00002.cbin.zst"
    shard_a.touch()
    shard_b.touch()
    header = builder.MAGIC + bytes(builder.HEADER_SIZE - len(builder.MAGIC))
    streams = {
        shard_a: header + cbin_record(0, 11) + cbin_record(1, 12),
        shard_b: header + cbin_record(2, 81) + cbin_record(3, 82),
    }
    opened: list[Path] = []

    class FakeProcess:
        def __init__(self, command, **_kwargs):
            path = Path(command[-1])
            opened.append(path)
            self.stdout = io.BytesIO(streams[path])

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(builder.subprocess, "Popen", FakeProcess)
    counts = {
        "tune": {"random": 2, "tactical": 0, "endgame": 0},
    }
    first, selected_first = builder.sample_corpus(
        tmp_path, 123, set(), counts, shard_count=2)
    second, selected_second = builder.sample_corpus(
        tmp_path, 123, set(), counts, shard_count=2)
    assert len(first["random"]) == 2
    assert first == second
    assert selected_first == selected_second
    assert set(opened) == {shard_a, shard_b}
    assert len(opened) == 4

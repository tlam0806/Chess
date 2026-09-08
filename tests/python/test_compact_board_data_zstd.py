from __future__ import annotations

import _posixsubprocess
from pathlib import Path
from unittest.mock import patch

import torch

from chess_nnue.compact_board_data import (
    HEADER,
    RECORD_SIZE,
    open_reader,
    open_writer,
)


def test_zstd_streams_use_posix_spawn_instead_of_fork(tmp_path: Path) -> None:
    # Exercise libomp first, matching the long-running trainer. On macOS a
    # subsequent fork() from this state aborts in the child before exec.
    torch.mm(torch.ones((32, 32)), torch.ones((32, 32)))
    path = tmp_path / "sample.cbin.zst"
    with patch.object(
        _posixsubprocess,
        "fork_exec",
        side_effect=AssertionError("zstd launch fell back to fork"),
    ):
        stream, owner = open_writer(path)
        stream.write(HEADER + bytes(RECORD_SIZE))
        assert owner is not None
        owner.close()  # type: ignore[attr-defined]

        stream, owner = open_reader(path)
        assert stream.read() == HEADER + bytes(RECORD_SIZE)
        assert owner is not None
        owner.close()  # type: ignore[attr-defined]

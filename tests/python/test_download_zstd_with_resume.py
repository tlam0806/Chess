from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DOWNLOADER = REPO_ROOT / "tools" / "data" / "download_zstd_with_resume.sh"
ZSTD = Path("/opt/homebrew/bin/zstd")


def make_zstd(source: Path, destination: Path) -> None:
    source.write_bytes((b"stockfish-static-nnue\n" * 4096) + os.urandom(4096))
    subprocess.run(
        [str(ZSTD), "-q", "-f", str(source), "-o", str(destination)],
        check=True,
    )


def test_promotes_already_complete_partial_without_path_lookup(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    destination = tmp_path / "cached.zst"
    partial = Path(f"{destination}.part")
    make_zstd(source, partial)

    environment = os.environ.copy()
    environment["PATH"] = "/definitely/not/usable"
    completed = subprocess.run(
        ["/bin/zsh", str(DOWNLOADER), "file:///unused", str(destination)],
        check=True,
        env=environment,
        capture_output=True,
    )

    assert completed.stdout == b""
    assert b"download_complete=" in completed.stderr
    assert destination.exists()
    assert not partial.exists()
    subprocess.run([str(ZSTD), "-q", "-t", str(destination)], check=True)


def test_downloads_and_validates_fresh_file_url(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    remote = tmp_path / "remote.zst"
    destination = tmp_path / "cached.zst"
    make_zstd(source, remote)

    completed = subprocess.run(
        ["/bin/zsh", str(DOWNLOADER), remote.as_uri(), str(destination)],
        check=True,
        capture_output=True,
    )

    assert completed.stdout == b""
    assert b"download_complete=" in completed.stderr
    assert destination.read_bytes() == remote.read_bytes()
    subprocess.run([str(ZSTD), "-q", "-t", str(destination)], check=True)

from __future__ import annotations

import io
import shutil
import struct
import subprocess
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable

from chess_nnue.nnue_architectures import (
    is_dual_accumulator_transform,
    transform_features,
)
from chess_nnue.value_net import AUX_FEATURE_COUNT


MAGIC = b"CHSCBIN2"
FORMAT_VERSION = 2
TARGET_ENCODING_STOCKFISH_RAW = 1
HEADER = MAGIC + struct.pack(
    "<HHI",
    FORMAT_VERSION,
    AUX_FEATURE_COUNT,
    TARGET_ENCODING_STOCKFISH_RAW,
)
HEADER_SIZE = len(HEADER)
RECORD_SIZE = 40
SQUARES = 64
HORIZONTAL_MIRROR_TRANSFORM = "dual_full_king_square_concat_horizontal_mirror"


@dataclass(frozen=True)
class CompactSample:
    board: bytes
    aux_bits: int
    score: int
    ply: int
    result: int


def decode_feature(feature: int) -> tuple[int, int, int, int, int]:
    piece_square = feature % 64
    feature //= 64
    king_square = feature % 64
    feature //= 64
    king_context = feature % 2
    feature //= 2
    piece_side = feature % 2
    feature //= 2
    piece = feature
    return piece, piece_side, king_context, king_square, piece_square


def feature_index(
    piece: int,
    piece_side: int,
    king_context: int,
    king_square: int,
    piece_square: int,
) -> int:
    index = piece
    index = index * 2 + piece_side
    index = index * 2 + king_context
    index = index * 64 + king_square
    index = index * 64 + piece_square
    return index


def pack_aux(aux: list[int]) -> int:
    if len(aux) != AUX_FEATURE_COUNT:
        raise ValueError(f"expected {AUX_FEATURE_COUNT} aux entries, got {len(aux)}")
    bits = 0
    for index, value in enumerate(aux):
        if value:
            bits |= 1 << index
    return bits


def unpack_aux(aux_bits: int) -> list[int]:
    return [(aux_bits >> index) & 1 for index in range(AUX_FEATURE_COUNT)]


def uses_horizontal_mirror(transform: str) -> bool:
    return transform == HORIZONTAL_MIRROR_TRANSFORM


def horizontal_mirror_aux(aux: list[int]) -> list[int]:
    """Mirror castling sides and en-passant files without changing labels."""
    if len(aux) != AUX_FEATURE_COUNT:
        raise ValueError(f"expected {AUX_FEATURE_COUNT} aux entries, got {len(aux)}")
    return [
        aux[1],
        aux[0],
        aux[3],
        aux[2],
        aux[4],
        *reversed(aux[5:13]),
    ]


def horizontal_mirror_board(board: bytes) -> bytes:
    """Return an a<->h mirror of a compact board for offline canonical keys."""
    nibbles = unpack_board(board)
    mirrored = [0] * SQUARES
    for square, code in enumerate(nibbles):
        mirrored[square ^ 7] = code
    packed = bytearray(32)
    for square in range(0, SQUARES, 2):
        packed[square // 2] = mirrored[square] | (mirrored[square + 1] << 4)
    return bytes(packed)


def canonical_horizontal_mirror_input(
    board: bytes,
    aux_bits: int,
) -> tuple[bytes, int]:
    """Put the side-to-move king on files a-d for training and split keys."""
    nibbles = unpack_board(board)
    friendly_kings = [square for square, code in enumerate(nibbles) if code == 6]
    if len(friendly_kings) != 1:
        raise ValueError("compact board must contain exactly one friendly king")
    if (friendly_kings[0] & 7) < 4:
        return board, aux_bits
    mirrored_aux = horizontal_mirror_aux(unpack_aux(aux_bits))
    return horizontal_mirror_board(board), pack_aux(mirrored_aux)


def canonical_architecture_input(
    board: bytes,
    aux_bits: int,
    transform: str,
) -> tuple[bytes, int, list[int], list[int]]:
    """Return canonical board/key material plus sparse model inputs."""
    if uses_horizontal_mirror(transform):
        board, aux_bits = canonical_horizontal_mirror_input(board, aux_bits)
    elif not is_dual_accumulator_transform(transform) and transform not in {
        "base768",
        "king_bucket",
        "full_king_square",
    }:
        raise ValueError(f"unknown feature transform: {transform}")
    return (
        board,
        aux_bits,
        architecture_features_from_board(board, transform),
        unpack_aux(aux_bits),
    )


def pack_board_from_raw_features(raw_features: list[int]) -> bytes:
    nibbles = [0] * SQUARES
    seen: set[tuple[int, int, int]] = set()
    for raw in raw_features:
        piece, piece_side, _king_context, _king_square, piece_square = decode_feature(int(raw))
        key = (piece, piece_side, piece_square)
        if key in seen:
            continue
        seen.add(key)
        if not 0 <= piece < 6:
            raise ValueError(f"bad piece index: {piece}")
        if piece_side not in (0, 1):
            raise ValueError(f"bad piece side: {piece_side}")
        if not 0 <= piece_square < SQUARES:
            raise ValueError(f"bad square: {piece_square}")
        code = piece + 1 if piece_side == 0 else piece + 7
        if nibbles[piece_square] != 0 and nibbles[piece_square] != code:
            raise ValueError(f"conflicting pieces on square {piece_square}")
        nibbles[piece_square] = code

    packed = bytearray(32)
    for square in range(0, SQUARES, 2):
        packed[square // 2] = nibbles[square] | (nibbles[square + 1] << 4)
    return bytes(packed)


def unpack_board(board: bytes) -> list[int]:
    if len(board) != 32:
        raise ValueError(f"expected 32 board bytes, got {len(board)}")
    nibbles: list[int] = []
    for byte in board:
        nibbles.append(byte & 0x0F)
        nibbles.append((byte >> 4) & 0x0F)
    return nibbles


def raw_features_from_board(board: bytes) -> list[int]:
    nibbles = unpack_board(board)
    friendly_king = -1
    enemy_king = -1
    for square, code in enumerate(nibbles):
        if code == 6:
            friendly_king = square
        elif code == 12:
            enemy_king = square
    if friendly_king < 0 or enemy_king < 0:
        raise ValueError("compact board is missing one or both kings")

    raw_features: list[int] = []
    for square, code in enumerate(nibbles):
        if code == 0:
            continue
        if 1 <= code <= 6:
            piece = code - 1
            piece_side = 0
        elif 7 <= code <= 12:
            piece = code - 7
            piece_side = 1
        else:
            raise ValueError(f"bad packed piece code: {code}")
        raw_features.append(feature_index(piece, piece_side, 0, friendly_king, square))
        raw_features.append(feature_index(piece, piece_side, 1, enemy_king, square))
    raw_features.sort()
    return raw_features


def architecture_features_from_board(board: bytes, transform: str) -> list[int]:
    return transform_features(raw_features_from_board(board), transform)


def pack_record(board: bytes, aux_bits: int, score: int, ply: int, result: int) -> bytes:
    if len(board) != 32:
        raise ValueError(f"expected 32 board bytes, got {len(board)}")
    if not 0 <= aux_bits < (1 << AUX_FEATURE_COUNT):
        raise ValueError(f"aux bits out of range: {aux_bits}")
    if not -32768 <= score <= 32767:
        raise ValueError(f"score out of int16 range: {score}")
    if not 0 <= ply <= 0x3FFF:
        raise ValueError(f"ply out of binpack range: {ply}")
    if result not in (-1, 0, 1):
        raise ValueError(f"result must be -1, 0, or 1, got {result}")
    return board + struct.pack("<HhHh", aux_bits, score, ply, result)


def unpack_record(record: bytes) -> CompactSample:
    if len(record) != RECORD_SIZE:
        raise ValueError(f"expected {RECORD_SIZE} record bytes, got {len(record)}")
    aux_bits, score, ply, result = struct.unpack("<HhHh", record[32:])
    if ply > 0x3FFF:
        raise ValueError(f"ply out of binpack range: {ply}")
    if result not in (-1, 0, 1):
        raise ValueError(f"bad game result: {result}")
    return CompactSample(record[:32], aux_bits, score, ply, result)


def position_split_bucket(board: bytes, aux_bits: int, split_mod: int) -> int:
    if len(board) != 32:
        raise ValueError(f"expected 32 board bytes, got {len(board)}")
    if not 0 <= aux_bits < (1 << AUX_FEATURE_COUNT):
        raise ValueError(f"aux bits out of range: {aux_bits}")
    if split_mod <= 0:
        raise ValueError("split_mod must be positive")
    key = board + struct.pack("<H", aux_bits)
    return zlib.crc32(key) % split_mod


class ZstdReader:
    def __init__(self, path: Path) -> None:
        executable = shutil.which("zstd")
        if executable is None:
            raise FileNotFoundError("zstd executable is not available on PATH")
        self.process = subprocess.Popen(
            [executable, "-q", "-dc", "--", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            # Python 3.9 only selects posix_spawn when the executable is an
            # absolute path and close_fds is false. This is correctness-critical
            # after PyTorch initializes libomp: fork() runs libomp's atfork
            # handler in the child and macOS aborts before exec (SIGABRT 6).
            close_fds=False,
        )
        if self.process.stdout is None:
            raise RuntimeError("failed to open zstd stdout")
        self.stream: BinaryIO = self.process.stdout

    def close(self) -> None:
        self.stream.close()
        code = self.process.wait()
        # Consumers often stop after max_samples. Closing stdout early gives
        # zstd SIGPIPE, which is expected and does not indicate corrupt input.
        if code not in (0, -13, 70):
            raise RuntimeError(f"zstd failed with exit code {code}")


class ZstdWriter:
    def __init__(self, path: Path, level: int = 6) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        executable = shutil.which("zstd")
        if executable is None:
            raise FileNotFoundError("zstd executable is not available on PATH")
        self.process = subprocess.Popen(
            [executable, "-T0", f"-{level}", "-q", "-o", str(path), "-"],
            stdin=subprocess.PIPE,
            close_fds=False,
        )
        if self.process.stdin is None:
            raise RuntimeError("failed to open zstd stdin")
        self.stream: BinaryIO = self.process.stdin

    def close(self) -> None:
        self.stream.close()
        code = self.process.wait()
        if code != 0:
            raise RuntimeError(f"zstd failed with exit code {code}")


def open_reader(path: Path) -> tuple[BinaryIO, object | None]:
    if str(path).endswith(".zst"):
        reader = ZstdReader(path)
        return reader.stream, reader
    return path.open("rb"), None


def open_writer(path: Path, zstd_level: int = 6) -> tuple[BinaryIO, object | None]:
    if str(path).endswith(".zst"):
        writer = ZstdWriter(path, zstd_level)
        return writer.stream, writer
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("wb"), None


def validate_header(header: bytes, path: Path | str = "<stream>") -> None:
    if header == HEADER:
        return
    if header[:8] == b"CHSCBIN1":
        raise ValueError(
            f"legacy CHSCBIN1 data in {path} has no ply/result metadata; "
            "reconvert the original binpack with robotmoon_binpack_to_cbin"
        )
    raise ValueError(f"bad compact data header in {path}")


def compact_paths(path: Path) -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(f"compact data does not exist: {path}")
    if not path.is_dir():
        return [path]
    paths = sorted([*path.glob("*.cbin"), *path.glob("*.cbin.zst")])
    if not paths:
        raise RuntimeError(f"no compact shard files found in {path}")
    return paths


def validate_compact_dataset(path: Path) -> list[Path]:
    paths = compact_paths(path)
    # Validate every shard header before a long sweep. This is cheap compared
    # with training and prevents a mixed v1/v2 directory from failing late.
    for shard in paths:
        stream, owner = open_reader(shard)
        try:
            validate_header(stream.read(HEADER_SIZE), shard)
        finally:
            if owner is not None:
                owner.close()  # type: ignore[attr-defined]
            else:
                stream.close()
    return paths


def iter_compact_samples(path: Path) -> Iterable[CompactSample]:
    stream, owner = open_reader(path)
    try:
        header = stream.read(HEADER_SIZE)
        validate_header(header, path)
        while True:
            record = stream.read(RECORD_SIZE)
            if not record:
                break
            if len(record) != RECORD_SIZE:
                raise ValueError(f"truncated compact record in {path}")
            yield unpack_record(record)
    finally:
        if owner is not None:
            owner.close()  # type: ignore[attr-defined]
        else:
            stream.close()


def write_header(stream: BinaryIO) -> None:
    stream.write(HEADER)

from __future__ import annotations

import subprocess
import tempfile
import unittest
import json
from pathlib import Path
import sys

from chess_nnue.compact_board_data import iter_compact_samples, unpack_board


class RobotMoonConverterIntegrationTests(unittest.TestCase):
    def test_converter_can_exclude_positions_in_check(self) -> None:
        fixture = Path("build/robotmoon_binpack_fixture")
        converter = Path("build/robotmoon_binpack_to_cbin")
        if not fixture.exists() or not converter.exists():
            self.skipTest("build converter fixture targets before running this test")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binpack = root / "in_check_fixture.binpack"
            cbin = root / "filtered.cbin"
            subprocess.run([fixture, binpack, "--append-in-check"], check=True)
            process = subprocess.run(
                [
                    converter,
                    "--input",
                    binpack,
                    "--output",
                    cbin,
                    "--require-legal-move",
                    "--exclude-in-check",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            samples = list(iter_compact_samples(cbin))

        self.assertEqual(len(samples), 3)
        self.assertIn("in_check_filtered=1", process.stderr)

    def test_converter_preserves_metadata_and_engine_orientation(self) -> None:
        fixture = Path("build/robotmoon_binpack_fixture")
        converter = Path("build/robotmoon_binpack_to_cbin")
        if not fixture.exists() or not converter.exists():
            self.skipTest("build converter fixture targets before running this test")

        with tempfile.TemporaryDirectory() as directory:
            binpack = Path(directory) / "fixture.binpack"
            cbin = Path(directory) / "fixture.cbin"
            subprocess.run([fixture, binpack], check=True)
            subprocess.run(
                [converter, "--input", binpack, "--output", cbin, "--require-legal-move"],
                check=True,
                capture_output=True,
                text=True,
            )
            samples = list(iter_compact_samples(cbin))

        self.assertEqual(len(samples), 3)
        white, continuation, black = samples
        self.assertEqual((white.score, white.ply, white.result), (208, 10, 1))
        self.assertEqual((continuation.score, continuation.ply, continuation.result), (-123, 11, -1))
        self.assertEqual((black.score, black.ply, black.result), (-416, 33, -1))
        self.assertEqual(white.aux_bits, 0x11F)
        self.assertEqual(continuation.aux_bits, 0x0F)
        self.assertEqual(black.aux_bits, 0x11F)

        white_board = unpack_board(white.board)
        self.assertEqual(white_board[0], 4)   # friendly rook a1
        self.assertEqual(white_board[4], 6)   # friendly king e1
        self.assertEqual(white_board[35], 7)  # enemy pawn d5
        self.assertEqual(white_board[36], 1)  # friendly pawn e5
        self.assertEqual(white_board[56], 10) # enemy rook a8
        self.assertEqual(white_board[60], 12) # enemy king e8

        continuation_board = unpack_board(continuation.board)
        self.assertEqual(continuation_board[4], 6)   # black king e8 -> relative e1
        self.assertEqual(continuation_board[19], 7)  # white pawn d6 -> relative d3
        self.assertEqual(continuation_board[63], 10) # white rook h1 -> relative h8

        black_board = unpack_board(black.board)
        self.assertEqual(black_board[0], 4)   # black rook a8 -> relative a1
        self.assertEqual(black_board[4], 6)   # black king e8 -> relative e1
        self.assertEqual(black_board[35], 7)  # white pawn d4 -> relative d5
        self.assertEqual(black_board[36], 1)  # black pawn e4 -> relative e5
        self.assertEqual(black_board[56], 10) # white rook a1 -> relative a8
        self.assertEqual(black_board[60], 12) # white king e1 -> relative e8

    def test_streaming_wrapper_produces_readable_atomic_zstd_output(self) -> None:
        fixture = Path("build/robotmoon_binpack_fixture")
        converter = Path("build/robotmoon_binpack_to_cbin")
        if not fixture.exists() or not converter.exists():
            self.skipTest("build converter fixture targets before running this test")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binpack = root / "fixture.binpack"
            output = root / "fixture.cbin.zst"
            subprocess.run([fixture, binpack], check=True)
            process = subprocess.run(
                [
                    sys.executable,
                    "tools/data/robotmoon_binpack_zst_to_cbin.py",
                    "--input",
                    str(binpack),
                    "--output",
                    str(output),
                    "--converter",
                    str(converter),
                    "--require-legal-move",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(process.returncode, 0, msg=process.stdout + process.stderr)
            self.assertEqual(len(list(iter_compact_samples(output))), 3)
            self.assertFalse((root / "fixture.cbin.tmp.zst").exists())

    def test_position_deduplication_filters_repeated_binpack_chunks(self) -> None:
        fixture = Path("build/robotmoon_binpack_fixture")
        converter = Path("build/robotmoon_binpack_to_cbin")
        if not fixture.exists() or not converter.exists():
            self.skipTest("build converter fixture targets before running this test")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            single = root / "single.binpack"
            repeated = root / "repeated.binpack"
            cbin = root / "deduplicated.cbin"
            subprocess.run([fixture, single], check=True)
            payload = single.read_bytes()
            repeated.write_bytes(payload + payload)

            process = subprocess.run(
                [
                    converter,
                    "--input",
                    repeated,
                    "--output",
                    cbin,
                    "--deduplicate-positions",
                    "--dedup-expected-records",
                    "6",
                    "--require-legal-move",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            samples = list(iter_compact_samples(cbin))

        # The white and black fixtures are color/rank mirrors. The compact
        # representation normalizes both to the side-to-move perspective, so
        # they are one model input in addition to the continuation position.
        self.assertEqual(len(samples), 2)
        self.assertIn("duplicate_or_bloom_filtered=4", process.stderr)

    def test_atomic_wrapper_forwards_position_deduplication(self) -> None:
        fixture = Path("build/robotmoon_binpack_fixture")
        converter = Path("build/robotmoon_binpack_to_cbin")
        if not fixture.exists() or not converter.exists():
            self.skipTest("build converter fixture targets before running this test")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            single = root / "single.binpack"
            repeated = root / "repeated.binpack"
            output = root / "deduplicated.cbin.zst"
            subprocess.run([fixture, single], check=True)
            payload = single.read_bytes()
            repeated.write_bytes(payload + payload)

            process = subprocess.run(
                [
                    sys.executable,
                    "tools/data/robotmoon_binpack_zst_to_cbin.py",
                    "--input",
                    str(repeated),
                    "--output",
                    str(output),
                    "--converter",
                    str(converter),
                    "--deduplicate-positions",
                    "--dedup-expected-records",
                    "6",
                    "--require-legal-move",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            samples = list(iter_compact_samples(output))

        self.assertEqual(process.returncode, 0, msg=process.stdout + process.stderr)
        self.assertEqual(len(samples), 2)
        self.assertIn("duplicate_or_bloom_filtered=4", process.stderr)

    def test_atomic_wrapper_removes_partial_output_when_exact_limit_is_unreachable(self) -> None:
        fixture = Path("build/robotmoon_binpack_fixture")
        converter = Path("build/robotmoon_binpack_to_cbin")
        if not fixture.exists() or not converter.exists():
            self.skipTest("build converter fixture targets before running this test")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binpack = root / "fixture.binpack"
            output = root / "too_short.cbin.zst"
            subprocess.run([fixture, binpack], check=True)
            process = subprocess.run(
                [
                    sys.executable,
                    "tools/data/robotmoon_binpack_zst_to_cbin.py",
                    "--input",
                    str(binpack),
                    "--output",
                    str(output),
                    "--converter",
                    str(converter),
                    "--limit",
                    "10",
                    "--require-limit-reached",
                    "--require-legal-move",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            partial_output_exists = output.exists()

        self.assertNotEqual(process.returncode, 0)
        self.assertIn("required 10", process.stderr)
        self.assertFalse(partial_output_exists)

    def test_independent_validator_accepts_deduped_output_and_rejects_duplicates(self) -> None:
        fixture = Path("build/robotmoon_binpack_fixture")
        converter = Path("build/robotmoon_binpack_to_cbin")
        validator = Path("build/validate_robotmoon_cbin")
        if not fixture.exists() or not converter.exists() or not validator.exists():
            self.skipTest("build converter, validator, and fixture targets before running this test")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            single = root / "single.binpack"
            repeated = root / "repeated.binpack"
            deduped = root / "deduped.cbin"
            duplicated = root / "duplicated.cbin"
            subprocess.run([fixture, single], check=True)
            payload = single.read_bytes()
            repeated.write_bytes(payload + payload)

            subprocess.run(
                [
                    converter,
                    "--input",
                    repeated,
                    "--output",
                    deduped,
                    "--deduplicate-positions",
                    "--dedup-expected-records",
                    "6",
                    "--require-legal-move",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [converter, "--input", repeated, "--output", duplicated, "--require-legal-move"],
                check=True,
                capture_output=True,
                text=True,
            )

            valid = subprocess.run(
                [
                    validator,
                    "--input",
                    deduped,
                    "--spool-directory",
                    root / "valid_spool",
                    "--expected-records",
                    "2",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            invalid = subprocess.run(
                [
                    validator,
                    "--input",
                    duplicated,
                    "--spool-directory",
                    root / "invalid_spool",
                    "--expected-records",
                    "6",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(valid.returncode, 0, msg=valid.stdout + valid.stderr)
        self.assertEqual(json.loads(valid.stdout)["fingerprint_duplicates"], 0)
        self.assertEqual(invalid.returncode, 2, msg=invalid.stdout + invalid.stderr)
        self.assertGreater(json.loads(invalid.stdout)["fingerprint_duplicates"], 0)


if __name__ == "__main__":
    unittest.main()

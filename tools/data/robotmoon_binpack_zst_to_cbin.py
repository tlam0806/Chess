#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert RobotMoon/Stockfish .binpack(.zst) data to compact .cbin(.zst)."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--converter",
        type=Path,
        default=Path("build/robotmoon_binpack_to_cbin"),
        help="path to the compiled C++ converter",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress-interval", type=int, default=1_000_000)
    parser.add_argument("--zstd-level", type=int, default=6)
    parser.add_argument("--require-legal-move", action="store_true")
    parser.add_argument("--exclude-in-check", action="store_true")
    parser.add_argument("--require-limit-reached", action="store_true")
    parser.add_argument("--deduplicate-positions", action="store_true")
    parser.add_argument("--dedup-expected-records", type=int, default=0)
    parser.add_argument("--dedup-bits-per-record", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.converter.exists():
        raise SystemExit(f"converter not found: {args.converter}")

    output_is_zstd = str(args.output).endswith(".zst")
    temporary_output = (
        args.output.with_name(args.output.name.removesuffix(".zst") + ".tmp.zst")
        if output_is_zstd
        else args.output.with_name(args.output.name + ".tmp")
    )
    temporary_output.unlink(missing_ok=True)

    input_owner = None
    output_owner = None
    converter = None
    input_stream = None
    output_stream = None

    try:
        if str(args.input).endswith(".zst"):
            input_owner = subprocess.Popen(
                ["zstd", "-q", "-dc", str(args.input)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL if args.limit else None,
            )
            input_stream = input_owner.stdout
        else:
            input_stream = args.input.open("rb")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        if output_is_zstd:
            output_owner = subprocess.Popen(
                [
                    "zstd",
                    "-T0",
                    f"-{args.zstd_level}",
                    "-f",
                    "-q",
                    "-o",
                    str(temporary_output),
                    "-",
                ],
                stdin=subprocess.PIPE,
            )
            output_stream = output_owner.stdin
        else:
            output_stream = temporary_output.open("wb")

        if input_stream is None or output_stream is None:
            raise RuntimeError("failed to create converter streams")

        command = [
            str(args.converter),
            "--input",
            "-",
            "--output",
            "-",
            "--progress-interval",
            str(args.progress_interval),
        ]
        if args.limit:
            command.extend(["--limit", str(args.limit)])
        if args.deduplicate_positions:
            command.append("--deduplicate-positions")
            if args.dedup_expected_records:
                command.extend(
                    ["--dedup-expected-records", str(args.dedup_expected_records)]
                )
            command.extend(
                ["--dedup-bits-per-record", str(args.dedup_bits_per_record)]
            )
        if args.require_legal_move:
            command.append("--require-legal-move")
        if args.exclude_in_check:
            command.append("--exclude-in-check")
        if args.require_limit_reached:
            command.append("--require-limit-reached")

        converter = subprocess.Popen(
            command,
            stdin=input_stream,
            stdout=output_stream,
        )

        if input_owner is not None and input_owner.stdout is not None:
            input_owner.stdout.close()
        if output_owner is not None and output_owner.stdin is not None:
            output_owner.stdin.close()

        converter_code = converter.wait()
        input_code = input_owner.wait() if input_owner is not None else 0
        output_code = output_owner.wait() if output_owner is not None else 0

        if input_owner is None and input_stream is not None:
            input_stream.close()
        if output_owner is None and output_stream is not None:
            output_stream.close()

        if converter_code != 0:
            return converter_code
        if input_code not in (0, -13):
            print(f"zstd input failed with exit code {input_code}", file=sys.stderr)
            return 1
        if output_code != 0:
            print(f"zstd output failed with exit code {output_code}", file=sys.stderr)
            return 1
        temporary_output.replace(args.output)
        return 0
    finally:
        if converter is not None and converter.poll() is None:
            converter.terminate()
            converter.wait()
        if input_owner is not None and input_owner.poll() is None:
            input_owner.terminate()
            input_owner.wait()
        if output_owner is not None and output_owner.poll() is None:
            output_owner.terminate()
            output_owner.wait()
        if input_owner is None and input_stream is not None and not input_stream.closed:
            input_stream.close()
        if output_owner is None and output_stream is not None and not output_stream.closed:
            output_stream.close()
        temporary_output.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())

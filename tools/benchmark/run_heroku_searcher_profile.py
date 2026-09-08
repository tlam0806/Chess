#!/usr/bin/env python3
"""Run the V41 sampling profiler in an isolated Heroku one-off dyno."""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import statistics
import subprocess
import tarfile
from typing import Any


PROGRESS_SENTINEL = "CHESS_PROFILE_PROGRESS="
SUMMARY_SENTINEL = "CHESS_PROFILE_SUMMARY="
CHUNK_SENTINEL = "CHESS_PROFILE_CHUNK="
END_SENTINEL = "CHESS_PROFILE_ARTIFACT_END="
ERROR_SENTINEL = "CHESS_PROFILE_ERROR="


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def utc_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_command(
    args: argparse.Namespace,
    suite_base64: str,
    script_sha256: str,
) -> list[str]:
    command = [
        "heroku",
        "run",
        "--app",
        args.app,
        "--type",
        "worker",
        "--size",
        args.size,
        "--no-tty",
        "--no-notify",
        "--no-launcher",
        "--exit-code",
        "--",
        "python",
        "-u",
        "-",
        "--engine",
        args.remote_engine,
        "--model",
        args.remote_model,
        "--config",
        args.remote_config,
        "--engine-cwd",
        args.remote_engine_cwd,
        "--suite-base64",
        suite_base64,
        "--remote-script-sha256",
        script_sha256,
        "--depth",
        str(args.depth),
        "--blocks",
        str(args.blocks),
        "--frequency",
        str(args.frequency),
        "--timeout",
        str(args.timeout),
    ]
    if args.positions_limit is not None:
        command.extend(["--positions-limit", str(args.positions_limit)])
    return command


def extract_archive(archive_path: Path, output_dir: Path) -> None:
    root = output_dir.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            target = (output_dir / member.name).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"unsafe artifact path: {member.name}")
        archive.extractall(output_dir)


def parse_json_after(line: str, sentinel: str) -> dict[str, Any] | None:
    index = line.find(sentinel)
    if index < 0:
        return None
    return json.loads(line[index + len(sentinel) :])


def resolve_pprof(explicit: Path | None) -> Path:
    candidates = [
        explicit,
        Path(found) if (found := shutil.which("pprof")) else None,
        Path("/tmp/chess-pprof-bin/pprof"),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate.resolve()
    raise RuntimeError(
        "modern pprof is required; install with "
        "GOBIN=/tmp/chess-pprof-bin go install github.com/google/pprof@latest"
    )


def pprof_report(
    pprof: Path,
    engine: Path,
    profiles: list[Path],
    granularity: str,
    *,
    all_nodes: bool = False,
) -> tuple[str, str]:
    if not profiles:
        raise RuntimeError("no raw profiles available for symbolization")
    node_options = (
        ["-nodefraction=0", "-edgefraction=0", "-nodecount=10000"]
        if all_nodes
        else ["-nodecount=200"]
    )
    command = [
        str(pprof),
        "-top",
        "-flat",
        f"-{granularity}",
        *node_options,
        str(engine),
        *(str(path) for path in profiles),
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"pprof failed ({result.returncode}): {result.stderr[-4000:]}"
        )
    if "chess::" not in result.stdout:
        raise RuntimeError("pprof did not resolve engine symbols")
    return result.stdout, result.stderr


def write_merged_profile(
    pprof: Path,
    engine: Path,
    profiles: list[Path],
    output: Path,
) -> str:
    result = subprocess.run(
        [
            str(pprof),
            "-proto",
            f"-output={output}",
            str(engine),
            *(str(path) for path in profiles),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,
    )
    if result.returncode != 0 or not output.is_file():
        raise RuntimeError(f"pprof merge failed: {result.stderr[-4000:]}")
    return result.stderr


def duration_seconds(value: str, unit: str) -> float:
    scales = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}
    if unit not in scales:
        raise RuntimeError(f"unsupported pprof duration unit: {unit}")
    return float(value) * scales[unit]


FORWARD_FUNCTION_PATTERNS = (
    re.compile(r"PhaseQuantizedNnueModel::forward_positional_scalar$"),
    re.compile(r"::VnniNetwork::evaluate$"),
    re.compile(r"::Avx2Network::evaluate$"),
    re.compile(r"::PhaseCandidateKernel<.*>::evaluate$"),
)


def parse_forward_share(report: str) -> dict[str, Any]:
    total_match = re.search(
        r"\bof ([0-9.]+)(ns|us|ms|s) total$", report, re.MULTILINE
    )
    forward_matches: list[tuple[str, str, str]] = []
    for line in report.splitlines():
        row_match = re.match(
            r"^\s*([0-9.]+)(ns|us|ms|s)\s+[0-9.]+%\s+[0-9.]+%"
            r"\s+[0-9.]+(?:ns|us|ms|s)\s+[0-9.]+%\s{2,}(.+)$",
            line,
        )
        if row_match is None:
            continue
        symbol = row_match.group(3)
        if any(pattern.search(symbol) for pattern in FORWARD_FUNCTION_PATTERNS):
            forward_matches.append(row_match.groups())
    if total_match is None:
        raise RuntimeError("cannot parse total sampled time from pprof output")
    total = duration_seconds(*total_match.groups())
    forward = sum(
        duration_seconds(value, unit)
        for value, unit, _symbol in forward_matches
    )
    return {
        "total_seconds": total,
        "forward_seconds": forward,
        "forward_share": forward / total,
        "forward_symbols": sorted(
            {symbol for _value, _unit, symbol in forward_matches}
        ),
    }


def bootstrap_forward_share(
    rounds: list[dict[str, float]],
    *,
    replicates: int = 10000,
    seed: int = 20260824,
) -> dict[str, Any]:
    generator = random.Random(seed)
    count = len(rounds)
    samples = []
    for _ in range(replicates):
        selected = [rounds[generator.randrange(count)] for _ in range(count)]
        samples.append(
            sum(row["forward_seconds"] for row in selected)
            / sum(row["total_seconds"] for row in selected)
        )
    samples.sort()
    return {
        "method": "complete-round percentile bootstrap",
        "replicates": replicates,
        "seed": seed,
        "lower_95": samples[int(replicates * 0.025) - 1],
        "upper_95": samples[int(replicates * 0.975) - 1],
    }


def run(args: argparse.Namespace) -> Path:
    runner_bytes = Path(__file__).read_bytes()
    remote_script = args.remote_script.read_bytes()
    suite = args.suite.read_bytes()
    suite_base64 = base64.b64encode(suite).decode("ascii")
    script_sha256 = sha256_bytes(remote_script)
    output_dir = args.output_dir or Path(
        f"logs/nnue_v41_heroku_sampling_{utc_stamp()}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    console_path = output_dir / "heroku-console.log"
    command = build_command(args, suite_base64, script_sha256)

    chunks: dict[int, str] = {}
    summary = None
    end = None
    remote_error = None
    with console_path.open("w", encoding="utf-8") as console:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdin is not None
        process.stdin.write(remote_script.decode("utf-8"))
        process.stdin.close()
        assert process.stdout is not None
        for line in process.stdout:
            stripped = line.rstrip("\r\n")
            chunk_index = stripped.find(CHUNK_SENTINEL)
            if chunk_index >= 0:
                payload = stripped[chunk_index + len(CHUNK_SENTINEL) :]
                ordinal_text, encoded = payload.split(":", 1)
                chunks[int(ordinal_text)] = encoded
                continue
            console.write(line)
            console.flush()
            progress = parse_json_after(stripped, PROGRESS_SENTINEL)
            maybe_summary = parse_json_after(stripped, SUMMARY_SENTINEL)
            maybe_end = parse_json_after(stripped, END_SENTINEL)
            maybe_error = parse_json_after(stripped, ERROR_SENTINEL)
            if progress is not None:
                if progress["phase"] == "profiler_setup":
                    print(f"profiler setup: {progress['status']}", flush=True)
                elif progress["phase"] == "round":
                    print(
                        "round {round}: {mode}, nps={nps:,.0f}, cpu/wall={ratio:.3f}".format(
                            round=progress["round"] + 1,
                            mode=progress["mode"],
                            nps=progress["nps"],
                            ratio=progress["cpu_to_wall"],
                        ),
                        flush=True,
                    )
            elif maybe_summary is not None:
                summary = maybe_summary
            elif maybe_end is not None:
                end = maybe_end
            elif maybe_error is not None:
                remote_error = maybe_error
                print(f"remote profile error: {remote_error['error']}", flush=True)
            elif len(stripped) < 500:
                print(stripped, flush=True)
        return_code = process.wait()

    if return_code != 0 or summary is None or end is None:
        detail = remote_error["error"] if remote_error else "missing artifact sentinel"
        raise RuntimeError(
            f"Heroku profile failed (exit={return_code}): {detail}; see {console_path}"
        )
    expected_chunks = int(end["chunks"])
    if set(chunks) != set(range(expected_chunks)):
        raise RuntimeError(
            f"incomplete artifact: expected {expected_chunks} chunks, got {len(chunks)}"
        )
    archive = base64.b64decode("".join(chunks[index] for index in range(expected_chunks)))
    if len(archive) != end["archive_bytes"]:
        raise RuntimeError("artifact byte count mismatch")
    if sha256_bytes(archive) != end["archive_sha256"]:
        raise RuntimeError("artifact SHA-256 mismatch")

    archive_path = output_dir / "profile-artifact.tar.gz"
    archive_path.write_bytes(archive)
    extract_archive(archive_path, output_dir)
    (output_dir / "runner.snapshot.py").write_bytes(runner_bytes)
    (output_dir / "remote-script.snapshot.py").write_bytes(remote_script)
    (output_dir / "suite.snapshot.json").write_bytes(suite)
    pprof = resolve_pprof(args.pprof)
    engine = output_dir / "uci_nnue_v41.elf"
    profiles = sorted((output_dir / "raw").glob("*.prof*"))
    no_queenside = [
        profile for profile in profiles if "queenside_pressure" not in profile.name
    ]
    functions, functions_warning = pprof_report(
        pprof, engine, profiles, "functions"
    )
    functions_all, functions_all_warning = pprof_report(
        pprof, engine, profiles, "functions", all_nodes=True
    )
    lines, lines_warning = pprof_report(pprof, engine, profiles, "lines")
    functions_no_queenside, no_queenside_warning = pprof_report(
        pprof, engine, no_queenside, "functions", all_nodes=True
    )
    merged_warning = write_merged_profile(
        pprof, engine, profiles, output_dir / "pprof-merged.pb.gz"
    )
    (output_dir / "pprof-functions.txt").write_text(functions, encoding="utf-8")
    (output_dir / "pprof-functions-all.txt").write_text(
        functions_all, encoding="utf-8"
    )
    (output_dir / "pprof-lines.txt").write_text(lines, encoding="utf-8")
    (output_dir / "pprof-functions-no-queenside.txt").write_text(
        functions_no_queenside, encoding="utf-8"
    )
    round_profiles: dict[int, list[Path]] = {}
    for profile in profiles:
        match = re.match(r"r(\d+)-", profile.name)
        if match is None:
            raise RuntimeError(f"cannot identify profile round: {profile.name}")
        round_profiles.setdefault(int(match.group(1)), []).append(profile)
    round_analysis = []
    round_warnings = []
    round_report_dir = output_dir / "pprof-rounds"
    round_report_dir.mkdir(exist_ok=True)
    for round_index, current_profiles in sorted(round_profiles.items()):
        report, warning = pprof_report(
            pprof, engine, current_profiles, "functions", all_nodes=True
        )
        (round_report_dir / f"profile-r{round_index:02d}-functions.txt").write_text(
            report, encoding="utf-8"
        )
        round_analysis.append(
            {"round": round_index, **parse_forward_share(report)}
        )
        round_warnings.append(warning)
    analysis = {
        "overall": parse_forward_share(functions_all),
        "no_queenside_pressure": parse_forward_share(functions_no_queenside),
        "profiled_rounds": round_analysis,
        "round_forward_share_median": statistics.median(
            row["forward_share"] for row in round_analysis
        ),
        "round_forward_share_min": min(
            row["forward_share"] for row in round_analysis
        ),
        "round_forward_share_max": max(
            row["forward_share"] for row in round_analysis
        ),
        "bootstrap": bootstrap_forward_share(round_analysis),
    }
    (output_dir / "profile-analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "pprof-warnings.txt").write_text(
        "".join(
            [
                functions_warning,
                functions_all_warning,
                lines_warning,
                no_queenside_warning,
                merged_warning,
                *round_warnings,
            ]
        ),
        encoding="utf-8",
    )
    go = shutil.which("go")
    pprof_version = None
    if go:
        pprof_version = subprocess.run(
            [go, "version", "-m", str(pprof)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        ).stdout.strip()
    runner_metadata = {
        "app": args.app,
        "size": args.size,
        "remote_script": str(args.remote_script),
        "remote_script_sha256": script_sha256,
        "runner_sha256": sha256_bytes(runner_bytes),
        "suite": str(args.suite),
        "suite_sha256": sha256_bytes(suite),
        "archive_sha256": end["archive_sha256"],
        "pprof_path": str(pprof),
        "pprof_sha256": sha256_file(pprof),
        "pprof_version": pprof_version,
    }
    (output_dir / "runner.json").write_text(
        json.dumps(runner_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"saved profile: {output_dir}", flush=True)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", default="stormy-garden-92984")
    parser.add_argument("--size", default="basic")
    parser.add_argument(
        "--suite", type=Path, default=Path("benchmarks/uci_platform_v1.json")
    )
    parser.add_argument(
        "--remote-script",
        type=Path,
        default=Path("tools/benchmark/profile_uci_searcher_remote.py"),
    )
    parser.add_argument(
        "--remote-engine",
        default="/app/chess-engine/build-release/uci_nnue_v41",
    )
    parser.add_argument(
        "--remote-model",
        default=(
            "/app/chess-engine/models/quantized_scale_grid/"
            "old_score_huber200_lr_sweep_then_5ep_20260724_142758/"
            "best/phase_quantized_nnue.bin"
        ),
    )
    parser.add_argument(
        "--remote-config", default="/app/lichess-bot/config-nnue-v41.yml"
    )
    parser.add_argument("--remote-engine-cwd", default="/app/chess-engine")
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--frequency", type=int, default=100)
    parser.add_argument("--positions-limit", type=int)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--pprof", type=Path)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Profile two positions in one ABBA block before a full run.",
    )
    args = parser.parse_args()
    if args.smoke:
        args.blocks = 1
        args.positions_limit = 2
    return args


if __name__ == "__main__":
    run(parse_args())

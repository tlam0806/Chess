#!/usr/bin/env python3
"""Run paired non-LTO/LTO V41 benchmarks inside one Heroku dyno."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import os
import subprocess
import zlib
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence


PROGRESS_SENTINEL = "CHESS_LTO_AB_PROGRESS="
CHUNK_SENTINEL = "CHESS_LTO_AB_CHUNK="
END_SENTINEL = "CHESS_LTO_AB_END="
ERROR_SENTINEL = "CHESS_LTO_AB_ERROR="
VARIANTS = ("nonlto", "lto")
EXPECTED_KERNEL = "x86_avx512vnni_256"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def emit_progress(**payload: Any) -> None:
    print(
        PROGRESS_SENTINEL
        + json.dumps(payload, sort_keys=True, separators=(",", ":")),
        flush=True,
    )


def load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("chess_lto_benchmark", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import benchmark harness: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def kernel_handshake(engine: Path, model: Path) -> str | None:
    environment = os.environ.copy()
    environment["CHESS_NNUE_BACKEND"] = "vnni"
    completed = subprocess.run(
        [str(engine), str(model)],
        input="uci\nquit\n",
        text=True,
        capture_output=True,
        env=environment,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"UCI handshake failed with exit {completed.returncode}: "
            f"{completed.stderr[-1000:]}"
        )
    prefix = "info string nnue_kernel="
    for line in completed.stdout.splitlines():
        if line.startswith(prefix):
            return line.removeprefix(prefix)
    return None


def run_parity(test: Path, model: Path, parity: Path) -> dict[str, Any]:
    environment = os.environ.copy()
    environment["CHESS_NNUE_BACKEND"] = "vnni"
    completed = subprocess.run(
        [str(test), str(model), str(parity)],
        text=True,
        capture_output=True,
        env=environment,
        timeout=1800,
        check=False,
    )
    marker = "python_cpp_parity_positions=8192"
    if completed.returncode != 0 or marker not in completed.stdout.splitlines():
        raise RuntimeError(
            "phase parity failed: "
            f"returncode={completed.returncode} "
            f"stdout={completed.stdout[-2000:]!r} "
            f"stderr={completed.stderr[-2000:]!r}"
        )
    return {
        "status": "pass",
        "python_cpp_parity_positions": 8192,
        "stdout_tail": completed.stdout.splitlines()[-5:],
    }


def benchmark_arguments(
    args: argparse.Namespace,
    engine: Path,
    variant: str,
) -> list[str]:
    result = [
        "--engine",
        str(engine),
        "--model",
        str(args.model),
        "--config",
        str(args.config),
        "--engine-cwd",
        str(args.engine_cwd),
        "--spec",
        str(args.spec),
        "--host-label",
        args.host_label,
        "--source-label",
        f"{args.source_label}:{variant}",
        "--harness-sha256",
        args.harness_sha256,
        "--runner-sha256",
        args.runner_sha256,
    ]
    if args.profile == "smoke":
        result.extend(
            [
                "--fixed-depth",
                "7",
                "--fixed-depth",
                "8",
                "--fixed-rounds",
                "1",
                "--movetime-ms",
                "1000",
                "--movetime-rounds",
                "1",
                "--warmup-depth",
                "4",
                "--warmup-positions",
                "2",
                "--bootstrap-replicates",
                "1000",
            ]
        )
    return result


def fixed_depth_parity(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, dict[tuple[Any, ...], tuple[Any, ...]]] = {}
    for variant, result in results.items():
        by_variant[variant] = {
            (row["limit"], row["round"], row["position_id"]): (
                row["reported_depth"],
                row["score_type"],
                row["score_value"],
                row["nodes"],
                row["bestmove"],
            )
            for row in result["observations"]
            if row["mode"] == "fixed_depth"
        }
    keys = set(by_variant["nonlto"])
    mismatches = []
    if set(by_variant["lto"]) != keys:
        mismatches.append({"error": "fixed-depth case keys differ"})
    else:
        for key in sorted(keys):
            if by_variant["nonlto"][key] != by_variant["lto"][key]:
                mismatches.append(
                    {
                        "key": list(key),
                        "nonlto": list(by_variant["nonlto"][key]),
                        "lto": list(by_variant["lto"][key]),
                    }
                )
    return {
        "status": "pass" if not mismatches else "fail",
        "cases": len(keys),
        "mismatches": mismatches,
    }


def emit_result(result: dict[str, Any]) -> None:
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    compressed = zlib.compress(encoded, level=9)
    payload = base64.b64encode(compressed).decode("ascii")
    chunk_size = 6000
    chunks = [
        payload[index : index + chunk_size]
        for index in range(0, len(payload), chunk_size)
    ]
    for index, chunk in enumerate(chunks):
        print(
            f"{CHUNK_SENTINEL}{index + 1}/{len(chunks)}:{chunk}",
            flush=True,
        )
    print(
        END_SENTINEL
        + json.dumps(
            {
                "chunks": len(chunks),
                "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
                "json_sha256": hashlib.sha256(encoded).hexdigest(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nonlto-engine", type=Path, required=True)
    parser.add_argument("--lto-engine", type=Path, required=True)
    parser.add_argument("--nonlto-test", type=Path, required=True)
    parser.add_argument("--lto-test", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--parity", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--engine-cwd", type=Path, required=True)
    parser.add_argument("--order", required=True)
    parser.add_argument("--profile", choices=("smoke", "full"), required=True)
    parser.add_argument("--host-label", required=True)
    parser.add_argument("--source-label", required=True)
    parser.add_argument("--harness-sha256", required=True)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--runner-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        order = tuple(args.order.split(","))
        if len(order) != 2 or set(order) != set(VARIANTS):
            raise ValueError("--order must contain nonlto,lto exactly once")
        paths = {
            "nonlto_engine": args.nonlto_engine,
            "lto_engine": args.lto_engine,
            "nonlto_test": args.nonlto_test,
            "lto_test": args.lto_test,
            "model": args.model,
            "parity": args.parity,
            "config": args.config,
            "spec": args.spec,
            "harness": args.harness,
        }
        for label, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(f"{label} does not exist: {path}")
        if sha256_file(args.harness) != args.harness_sha256:
            raise RuntimeError("benchmark harness hash mismatch")
        if sha256_file(args.spec) != args.spec_sha256:
            raise RuntimeError("benchmark spec hash mismatch")

        engines = {"nonlto": args.nonlto_engine, "lto": args.lto_engine}
        tests = {"nonlto": args.nonlto_test, "lto": args.lto_test}
        identities = {}
        parity_results = {}
        for variant in VARIANTS:
            kernel = kernel_handshake(engines[variant], args.model)
            if kernel != EXPECTED_KERNEL:
                raise RuntimeError(
                    f"{variant} reported {kernel!r}, expected {EXPECTED_KERNEL!r}"
                )
            parity_results[variant] = run_parity(
                tests[variant], args.model, args.parity
            )
            identities[variant] = {
                "engine_sha256": sha256_file(engines[variant]),
                "test_sha256": sha256_file(tests[variant]),
                "kernel": kernel,
            }
            emit_progress(phase="preflight", variant=variant, status="pass")

        harness = load_module(args.harness)
        results = {}
        previous_backend = os.environ.get("CHESS_NNUE_BACKEND")
        try:
            os.environ["CHESS_NNUE_BACKEND"] = "vnni"
            for sequence, variant in enumerate(order):
                emit_progress(
                    phase="benchmark_start",
                    variant=variant,
                    sequence=sequence,
                    order=order,
                )
                benchmark_args = harness.build_parser().parse_args(
                    benchmark_arguments(args, engines[variant], variant)
                )
                result = harness.run_benchmark(benchmark_args)
                actual_kernel = (
                    result.get("uci", {}).get("handshake", {}).get("nnue_kernel")
                )
                if result.get("status") != "valid":
                    raise RuntimeError(f"{variant} benchmark returned invalid")
                if actual_kernel != EXPECTED_KERNEL:
                    raise RuntimeError(
                        f"{variant} benchmark reported {actual_kernel!r}"
                    )
                result["lto_ab"] = {
                    "variant": variant,
                    "order": list(order),
                    "sequence": sequence,
                }
                results[variant] = result
                emit_progress(
                    phase="benchmark_done",
                    variant=variant,
                    sequence=sequence,
                    status="pass",
                )
        finally:
            if previous_backend is None:
                os.environ.pop("CHESS_NNUE_BACKEND", None)
            else:
                os.environ["CHESS_NNUE_BACKEND"] = previous_backend

        fixed_parity = fixed_depth_parity(results)
        if fixed_parity["status"] != "pass":
            raise RuntimeError("fixed-depth signatures differ between binaries")
        emit_result(
            {
                "schema_version": 1,
                "status": "valid",
                "order": list(order),
                "identities": identities,
                "phase_parity": parity_results,
                "fixed_depth_parity": fixed_parity,
                "results": results,
            }
        )
        return 0
    except Exception as error:
        print(
            ERROR_SENTINEL
            + json.dumps(
                {"error_type": type(error).__name__, "error": str(error)},
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

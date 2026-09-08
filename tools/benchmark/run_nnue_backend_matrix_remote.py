#!/usr/bin/env python3
"""Run NNUE parity and a backend matrix inside one Linux/Heroku dyno."""

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


PROGRESS_SENTINEL = "CHESS_NNUE_MATRIX_PROGRESS="
CHUNK_SENTINEL = "CHESS_NNUE_MATRIX_CHUNK="
END_SENTINEL = "CHESS_NNUE_MATRIX_END="
ERROR_SENTINEL = "CHESS_NNUE_MATRIX_ERROR="
BACKENDS = ("scalar", "avx2", "vnni")
EXPECTED_KERNEL = {
    "scalar": "scalar",
    "avx2": "x86_avx2_exact",
    "vnni": "x86_avx512vnni_256",
    "auto": "x86_avx512vnni_256",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def emit_progress(**payload: Any) -> None:
    print(
        PROGRESS_SENTINEL + json.dumps(payload, sort_keys=True, separators=(",", ":")),
        flush=True,
    )


def load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("chess_uci_benchmark", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import benchmark harness: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def kernel_handshake(engine: Path, model: Path, backend: str) -> str | None:
    environment = os.environ.copy()
    environment["CHESS_NNUE_BACKEND"] = backend
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
            f"{backend} UCI handshake failed with exit {completed.returncode}: "
            f"{completed.stderr[-1000:]}"
        )
    prefix = "info string nnue_kernel="
    for line in completed.stdout.splitlines():
        if line.startswith(prefix):
            return line.removeprefix(prefix)
    return None


def run_parity(
    test: Path,
    model: Path,
    parity: Path,
    backend: str,
) -> dict[str, Any]:
    environment = os.environ.copy()
    environment["CHESS_NNUE_BACKEND"] = backend
    completed = subprocess.run(
        [str(test), str(model), str(parity)],
        text=True,
        capture_output=True,
        env=environment,
        timeout=1800,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{backend} phase parity failed with exit {completed.returncode}: "
            f"stdout={completed.stdout[-2000:]!r} stderr={completed.stderr[-2000:]!r}"
        )
    parity_marker = "python_cpp_parity_positions=8192"
    if parity_marker not in completed.stdout.splitlines():
        raise RuntimeError(
            f"{backend} parity did not confirm all 8192 fixture positions: "
            f"{completed.stdout[-2000:]!r}"
        )
    return {
        "status": "pass",
        "returncode": completed.returncode,
        "python_cpp_parity_positions": 8192,
        "stdout_tail": completed.stdout.splitlines()[-5:],
    }


def benchmark_arguments(
    args: argparse.Namespace,
    backend: str,
) -> list[str]:
    result = [
        "--engine",
        str(args.engine),
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
        args.source_label,
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
    grouped: dict[tuple[int, str], dict[str, set[tuple[Any, ...]]]] = {}
    for backend, result in results.items():
        for row in result["observations"]:
            if row["mode"] != "fixed_depth":
                continue
            key = (row["limit"], row["position_id"])
            grouped.setdefault(key, {}).setdefault(backend, set()).add(
                (
                    row["reported_depth"],
                    row["score_type"],
                    row["score_value"],
                    row["nodes"],
                    row["bestmove"],
                )
            )
    mismatches = []
    for (limit, position_id), by_backend in sorted(grouped.items()):
        combined = {
            signature for signatures in by_backend.values() for signature in signatures
        }
        if set(by_backend) != set(BACKENDS) or len(combined) != 1:
            mismatches.append(
                {
                    "limit": limit,
                    "position_id": position_id,
                    "signatures": {
                        backend: [list(value) for value in sorted(signatures)]
                        for backend, signatures in sorted(by_backend.items())
                    },
                }
            )
    return {
        "status": "pass" if not mismatches else "fail",
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
        print(f"{CHUNK_SENTINEL}{index + 1}/{len(chunks)}:{chunk}", flush=True)
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
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
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
        if len(order) != len(BACKENDS) or set(order) != set(BACKENDS):
            raise ValueError("--order must contain scalar,avx2,vnni exactly once")
        for label in ("engine", "test", "model", "parity", "config", "spec", "harness"):
            path = getattr(args, label)
            if not path.is_file():
                raise FileNotFoundError(f"{label} does not exist: {path}")
        actual_harness_sha256 = sha256_file(args.harness)
        if actual_harness_sha256 != args.harness_sha256:
            raise RuntimeError(
                "benchmark harness hash mismatch: "
                f"actual={actual_harness_sha256} expected={args.harness_sha256}"
            )
        actual_spec_sha256 = sha256_file(args.spec)
        if actual_spec_sha256 != args.spec_sha256:
            raise RuntimeError(
                "benchmark spec hash mismatch: "
                f"actual={actual_spec_sha256} expected={args.spec_sha256}"
            )

        auto_kernel = kernel_handshake(args.engine, args.model, "auto")
        if auto_kernel != EXPECTED_KERNEL["auto"]:
            raise RuntimeError(
                f"auto backend reported {auto_kernel!r}, expected {EXPECTED_KERNEL['auto']!r}"
            )
        emit_progress(phase="auto_gate", status="pass", kernel=auto_kernel)

        parity_results = {}
        for backend in BACKENDS:
            parity_results[backend] = run_parity(
                args.test, args.model, args.parity, backend
            )
            emit_progress(phase="parity", backend=backend, status="pass")

        harness = load_module(args.harness)
        benchmark_results = {}
        previous_backend = os.environ.get("CHESS_NNUE_BACKEND")
        try:
            for sequence, backend in enumerate(order):
                os.environ["CHESS_NNUE_BACKEND"] = backend
                emit_progress(
                    phase="benchmark_start",
                    backend=backend,
                    sequence=sequence,
                    order=order,
                )
                benchmark_args = harness.build_parser().parse_args(
                    benchmark_arguments(args, backend)
                )
                result = harness.run_benchmark(benchmark_args)
                actual_kernel = (
                    result.get("uci", {}).get("handshake", {}).get("nnue_kernel")
                )
                if result.get("status") != "valid":
                    raise RuntimeError(f"{backend} benchmark returned invalid status")
                if actual_kernel != EXPECTED_KERNEL[backend]:
                    raise RuntimeError(
                        f"{backend} reported {actual_kernel!r}, "
                        f"expected {EXPECTED_KERNEL[backend]!r}"
                    )
                result["matrix"] = {
                    "backend": backend,
                    "order": list(order),
                    "sequence": sequence,
                }
                benchmark_results[backend] = result
                emit_progress(
                    phase="benchmark_done",
                    backend=backend,
                    sequence=sequence,
                    status="pass",
                )
        finally:
            if previous_backend is None:
                os.environ.pop("CHESS_NNUE_BACKEND", None)
            else:
                os.environ["CHESS_NNUE_BACKEND"] = previous_backend

        parity = fixed_depth_parity(benchmark_results)
        if parity["status"] != "pass":
            raise RuntimeError("fixed-depth signatures differ across backends")
        emit_result(
            {
                "schema_version": 1,
                "status": "valid",
                "order": list(order),
                "auto_kernel": auto_kernel,
                "phase_parity": parity_results,
                "fixed_depth_parity": parity,
                "results": benchmark_results,
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

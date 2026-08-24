#!/usr/bin/env python3
"""Validate and compare scalar/AVX2/VNNI Heroku benchmark artifacts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


EXPECTED_KERNEL = {
    "scalar": "scalar",
    "avx2": "x86_avx2_exact",
    "vnni": "x86_avx512vnni_256",
}
PROVENANCE_KEYS = (
    "engine_sha256",
    "model_sha256",
    "spec_sha256",
    "effective_spec_sha256",
    "config_sha256",
    "harness_sha256",
    "runner_sha256",
)


def load_backend(
    directory: Path, backend: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary_path = directory / "summary.json"
    if not summary_path.is_file():
        raise ValueError(f"missing summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    runs = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("heroku-run-*.json"))
    ]
    if not runs:
        raise ValueError(f"no run artifacts in {directory}")
    for run in runs:
        if run.get("runner", {}).get("backend") != backend:
            raise ValueError(f"artifact backend mismatch in {directory}")
        actual = run.get("uci", {}).get("handshake", {}).get("nnue_kernel")
        if actual != EXPECTED_KERNEL[backend]:
            raise ValueError(
                f"{backend} reported {actual!r}, expected {EXPECTED_KERNEL[backend]!r}"
            )
        if run.get("status") != "valid":
            raise ValueError(f"invalid {backend} run in {directory}")
    if summary.get("status") != "valid":
        raise ValueError(f"invalid aggregate summary: {summary_path}")
    return summary, runs


def fixed_depth_signatures(
    all_runs: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], dict[str, set[tuple[Any, ...]]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for backend, runs in all_runs.items():
        for run in runs:
            for row in run["observations"]:
                if row["mode"] != "fixed_depth":
                    continue
                grouped[(row["limit"], row["position_id"])][backend].add(
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
        signatures = {
            signature
            for backend_signatures in by_backend.values()
            for signature in backend_signatures
        }
        if set(by_backend) != set(EXPECTED_KERNEL) or len(signatures) != 1:
            mismatches.append(
                {
                    "limit": limit,
                    "position_id": position_id,
                    "signatures": {
                        backend: [list(value) for value in sorted(values)]
                        for backend, values in sorted(by_backend.items())
                    },
                }
            )
    return mismatches


def provenance_check(all_runs: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    runs = [run for backend_runs in all_runs.values() for run in backend_runs]
    raw_values = {
        key: [run.get("provenance", {}).get(key) for run in runs]
        for key in PROVENANCE_KEYS
    }
    values = {
        key: sorted({str(value) for value in items if value is not None})
        for key, items in raw_values.items()
    }
    missing = sorted(
        key
        for key, items in raw_values.items()
        if any(value is None for value in items)
    )
    config_validation = [
        run.get("provenance", {}).get("config_validation", {}).get("status")
        for run in runs
    ]
    hashes_match = not missing and all(
        len(set(items)) == 1 for items in raw_values.values()
    )
    configs_match = bool(config_validation) and all(
        status == "match" for status in config_validation
    )
    return {
        "status": "pass" if hashes_match and configs_match else "fail",
        "values": values,
        "missing": missing,
        "hashes_match": hashes_match,
        "config_validation_match": configs_match,
    }


def performance_table(summaries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    by_backend = {
        backend: {
            (row["mode"], row["limit"]): float(row["aggregate_nps"])
            for row in summary["summaries"]
        }
        for backend, summary in summaries.items()
    }
    keys = set(by_backend["scalar"])
    if any(set(rows) != keys for rows in by_backend.values()):
        raise ValueError("backend summaries cover different benchmark modes/limits")
    result = []
    for mode, limit in sorted(keys):
        scalar = by_backend["scalar"][(mode, limit)]
        avx2 = by_backend["avx2"][(mode, limit)]
        vnni = by_backend["vnni"][(mode, limit)]
        result.append(
            {
                "mode": mode,
                "limit": limit,
                "scalar_nps": scalar,
                "avx2_nps": avx2,
                "vnni_nps": vnni,
                "avx2_over_scalar": avx2 / scalar,
                "vnni_over_scalar": vnni / scalar,
                "vnni_over_avx2": vnni / avx2,
            }
        )
    return result


def markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Heroku NNUE backend comparison",
        "",
        f"- Status: **{result['status']}**",
        f"- Cross-backend fixed-depth parity: `{result['fixed_depth_parity']['status']}`",
        f"- Provenance equality: `{result['provenance']['status']}`",
        f"- Promotion gate: `{result.get('promotion_gate', {}).get('status', 'not_evaluated')}`",
        "",
        "| Mode | Limit | Scalar NPS | AVX2 NPS | VNNI NPS | AVX2/scalar | VNNI/scalar | VNNI/AVX2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["performance"]:
        lines.append(
            "| {mode} | {limit} | {scalar_nps:,.0f} | {avx2_nps:,.0f} | "
            "{vnni_nps:,.0f} | {avx2_over_scalar:.3f}x | "
            "{vnni_over_scalar:.3f}x | {vnni_over_avx2:.3f}x |".format(**row)
        )
    paired = result.get("paired_speedups")
    if paired and paired.get("status") == "pass":
        lines.extend(
            [
                "",
                f"Paired bootstrap: {paired['replicates']:,} resamples ({paired['resampling']}).",
                "",
                "| Mode | Limit | AVX2/scalar 95% | VNNI/scalar 95% | VNNI/AVX2 95% |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for metric in paired["metrics"]:
            intervals = {
                name: metric["ratios"][name]["paired_bootstrap_interval95"]
                for name in (
                    "avx2_over_scalar",
                    "vnni_over_scalar",
                    "vnni_over_avx2",
                )
            }
            lines.append(
                "| {mode} | {limit} | [{a0:.3f}, {a1:.3f}] | "
                "[{v0:.3f}, {v1:.3f}] | [{x0:.3f}, {x1:.3f}] |".format(
                    mode=metric["mode"],
                    limit=metric["limit"],
                    a0=intervals["avx2_over_scalar"][0],
                    a1=intervals["avx2_over_scalar"][1],
                    v0=intervals["vnni_over_scalar"][0],
                    v1=intervals["vnni_over_scalar"][1],
                    x0=intervals["vnni_over_avx2"][0],
                    x1=intervals["vnni_over_avx2"][1],
                )
            )
    lines.extend(
        [
            "",
            "Ratios are descriptive for these sampled one-off dynos. A full run uses three dynos per backend; it is not a fleet-wide hardware guarantee.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scalar-dir", type=Path, required=True)
    parser.add_argument("--avx2-dir", type=Path, required=True)
    parser.add_argument("--vnni-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    directories = {
        "scalar": args.scalar_dir,
        "avx2": args.avx2_dir,
        "vnni": args.vnni_dir,
    }
    loaded = {
        backend: load_backend(directory, backend)
        for backend, directory in directories.items()
    }
    summaries = {backend: value[0] for backend, value in loaded.items()}
    all_runs = {backend: value[1] for backend, value in loaded.items()}
    mismatches = fixed_depth_signatures(all_runs)
    provenance = provenance_check(all_runs)
    result = {
        "schema_version": 1,
        "status": "valid"
        if not mismatches and provenance["status"] == "pass"
        else "invalid",
        "backends": {
            backend: {
                "directory": str(directories[backend]),
                "runs": len(runs),
                "kernel": EXPECTED_KERNEL[backend],
            }
            for backend, runs in all_runs.items()
        },
        "fixed_depth_parity": {
            "status": "pass" if not mismatches else "fail",
            "mismatches": mismatches,
        },
        "provenance": provenance,
        "performance": performance_table(summaries),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output_md.write_text(markdown(result), encoding="utf-8")
    print(f"comparison JSON: {args.output_json}")
    print(f"comparison report: {args.output_md}")
    return 0 if result["status"] == "valid" else 3


if __name__ == "__main__":
    raise SystemExit(main())

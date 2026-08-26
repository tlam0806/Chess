#!/usr/bin/env python3
"""Analyze the protocol-v2 NNUE forward-stage experiment across two hosts.

The experiment has three independent process/dyno clusters per host and twelve
rotating stage batches inside each cluster.  The primary estimator is the
median of the three cluster medians.  Confidence intervals use an independent
hierarchical bootstrap: clusters are sampled with replacement separately for
each host, then batches are sampled within each stage-order stratum.

This is a comparison of isolated kernel-core stage timings.  Stage timings are
not additive, and the cross-host ratios combine CPU, ISA, compiler, and hosting
differences rather than measuring a pure Heroku virtualization overhead.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import re
import statistics
from typing import Any, Mapping, Sequence


PROTOCOL = "phase_nnue_forward_stages_v2"
ANALYSIS_PROTOCOL = "phase_nnue_forward_stage_crosshost_analysis_v1"
STAGES = (
    "s1_input_to_hidden2",
    "s2_hidden2_to_hidden3",
    "s3_hidden3_to_output",
    "full",
)
METRICS = ("wall", "process_cpu")
CLUSTERS_PER_HOST = 3
REPEATS = 12
ORDERS = 4
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260825
OFFICIAL_CORPUS = {
    "samples": 256,
    "seed": 20260825,
    "checksum": "0xf5399956b2fa9ea8",
}
OFFICIAL_ITERATIONS = {
    "s1_input_to_hidden2": 4_000_000,
    "s2_hidden2_to_hidden3": 20_000_000,
    "s3_hidden3_to_output": 100_000_000,
    "full": 3_000_000,
}
OFFICIAL_STAGE_CHECKSUMS = {
    "s1_input_to_hidden2": "0xd264620b4b94ba37",
    "s2_hidden2_to_hidden3": "0x89dbdb9bf3a1ddde",
    "s3_hidden3_to_output": "0x61bbf9ec6191a24b",
    "full": "0x61bbf9ec6191a24b",
}
EXPECTED_TIMING = {
    "wall_clock": "std::chrono::steady_clock",
    "process_cpu_clock": "std::clock",
    "subtraction": "none",
    "stage_order": "rotating",
}
EXPECTED_HOST = {
    "local": {
        "backend_requested": "neon",
        "kernel": "arm_neon_dotprod_i8mm",
    },
    "heroku": {
        "backend_requested": "vnni",
        "kernel": "x86_avx512vnni_256",
    },
}
RUN_FILE_PATTERNS = {
    "local": re.compile(r"^local-forward-stages-run-(\d+)\.json$"),
    "heroku": re.compile(r"^heroku-forward-stages-run-(\d+)\.json$"),
}


class AnalysisError(ValueError):
    """Raised when an input violates the locked experiment protocol."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def _positive_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _finite_positive(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value)) and float(value) > 0.0


def _close(actual: Any, expected: float) -> bool:
    return _finite_positive(actual) and math.isclose(
        float(actual), expected, rel_tol=1e-9, abs_tol=1e-9
    )


def _load_documents(directory: Path, host: str) -> list[tuple[Path, dict[str, Any]]]:
    _require(directory.is_dir(), f"{host} input directory does not exist: {directory}")
    pattern = RUN_FILE_PATTERNS[host]
    selected: list[tuple[int, Path]] = []
    for path in directory.iterdir():
        match = pattern.fullmatch(path.name)
        if match is not None and path.is_file():
            selected.append((int(match.group(1)), path))
    selected.sort()
    _require(
        [ordinal for ordinal, _ in selected] == [1, 2, 3],
        f"{host} must contain exactly run ordinals 1, 2, 3",
    )
    documents: list[tuple[Path, dict[str, Any]]] = []
    for ordinal, path in selected:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise AnalysisError(f"cannot read {host} run {ordinal}: {path}") from error
        _require(isinstance(value, dict), f"{host} run {ordinal} is not an object")
        documents.append((path, value))
    return documents


def _validate_run(
    document: Mapping[str, Any],
    *,
    host: str,
    ordinal: int,
    path: Path | None,
) -> dict[str, Any]:
    label = f"{host} run {ordinal}" + (f" ({path})" if path is not None else "")
    _require(document.get("schema_version") == 2, f"{label}: schema_version is not 2")
    _require(document.get("protocol") == PROTOCOL, f"{label}: protocol is not v2")
    _require(document.get("status") == "valid", f"{label}: status is not valid")
    expected_host = EXPECTED_HOST[host]
    for field, expected in expected_host.items():
        _require(document.get(field) == expected, f"{label}: unexpected {field}")

    model_sha256 = document.get("model_sha256")
    _require(
        isinstance(model_sha256, str)
        and re.fullmatch(r"[0-9a-f]{64}", model_sha256) is not None,
        f"{label}: invalid model SHA-256",
    )

    corpus = document.get("corpus")
    _require(isinstance(corpus, dict), f"{label}: missing corpus identity")
    for field, expected in OFFICIAL_CORPUS.items():
        _require(corpus.get(field) == expected, f"{label}: corpus {field} mismatch")

    correctness = document.get("correctness")
    _require(isinstance(correctness, dict), f"{label}: missing correctness result")
    _require(correctness.get("status") == "pass", f"{label}: parity status is not pass")
    for field in ("scalar_parity", "stage_parity", "timed_sink_parity"):
        _require(correctness.get(field) is True, f"{label}: {field} did not pass")
    correctness_checksum = correctness.get("checksum")
    _require(
        isinstance(correctness_checksum, str) and correctness_checksum.startswith("0x"),
        f"{label}: invalid correctness checksum",
    )

    _require(document.get("timing") == EXPECTED_TIMING, f"{label}: timing contract mismatch")
    _require(document.get("warmup_evaluations") == 10_000, f"{label}: warmup mismatch")
    _require(document.get("repeats") == REPEATS, f"{label}: repeat count mismatch")

    run_identity = document.get("local_run" if host == "local" else "heroku_run")
    _require(isinstance(run_identity, dict), f"{label}: missing run identity")
    identity_ordinal = run_identity.get("ordinal" if host == "local" else "dyno_ordinal")
    _require(identity_ordinal == ordinal, f"{label}: run ordinal mismatch")
    if host == "local":
        _require(
            run_identity.get("independent_process") is True,
            f"{label}: process independence was not confirmed",
        )
    else:
        _require(run_identity.get("profile") == "full", f"{label}: not a full profile")
        _require(run_identity.get("size") == "basic", f"{label}: dyno is not Basic")

    stages = document.get("stages")
    _require(
        isinstance(stages, dict) and set(stages) == set(STAGES),
        f"{label}: stage set mismatch",
    )
    batches = document.get("batches")
    _require(
        isinstance(batches, list) and len(batches) == REPEATS * len(STAGES),
        f"{label}: batch count mismatch",
    )

    batch_by_stage_repeat: dict[tuple[str, int], Mapping[str, Any]] = {}
    stage_order_values: dict[str, dict[int, dict[str, list[float]]]] = {
        stage: {
            order: {metric: [] for metric in METRICS}
            for order in range(ORDERS)
        }
        for stage in STAGES
    }
    for repeat in range(REPEATS):
        repeat_rows = [
            row
            for row in batches
            if isinstance(row, dict) and row.get("repeat") == repeat
        ]
        _require(len(repeat_rows) == len(STAGES), f"{label}: repeat {repeat} is incomplete")
        _require(
            {row.get("order") for row in repeat_rows} == set(range(ORDERS)),
            f"{label}: repeat {repeat} has invalid order positions",
        )
        _require(
            {row.get("stage") for row in repeat_rows} == set(STAGES),
            f"{label}: repeat {repeat} has invalid stage set",
        )
        for row in repeat_rows:
            stage = row["stage"]
            order = row["order"]
            expected_stage = STAGES[(repeat + order) % len(STAGES)]
            _require(stage == expected_stage, f"{label}: rotating schedule mismatch")
            evaluations = row.get("evaluations")
            _require(
                evaluations == OFFICIAL_ITERATIONS[stage],
                f"{label}: {stage} evaluation count mismatch",
            )
            wall_ns = row.get("wall_ns")
            process_cpu_ns = row.get("process_cpu_ns")
            _require(_positive_int(wall_ns), f"{label}: invalid batch wall_ns")
            _require(_positive_int(process_cpu_ns), f"{label}: invalid batch process_cpu_ns")
            batch_by_stage_repeat[(stage, repeat)] = row
            stage_order_values[stage][order]["wall"].append(wall_ns / evaluations)
            stage_order_values[stage][order]["process_cpu"].append(
                process_cpu_ns / evaluations
            )

    cluster_medians: dict[str, dict[str, float]] = {}
    stage_checksums: dict[str, str] = {}
    timed_sinks: dict[str, str] = {}
    for stage in STAGES:
        row = stages[stage]
        _require(isinstance(row, dict), f"{label}: {stage} result is not an object")
        _require(
            row.get("evaluations_per_repeat") == OFFICIAL_ITERATIONS[stage],
            f"{label}: {stage} summary evaluation count mismatch",
        )
        _require(
            row.get("checksum") == OFFICIAL_STAGE_CHECKSUMS[stage],
            f"{label}: {stage} checksum mismatch",
        )
        timed_sink = row.get("timed_sink")
        _require(
            isinstance(timed_sink, str) and timed_sink.startswith("0x"),
            f"{label}: {stage} timed sink is invalid",
        )
        stage_checksums[stage] = row["checksum"]
        timed_sinks[stage] = timed_sink

        raw_wall = row.get("elapsed_ns")
        raw_cpu = row.get("process_cpu_ns")
        _require(
            isinstance(raw_wall, list)
            and isinstance(raw_cpu, list)
            and len(raw_wall) == REPEATS
            and len(raw_cpu) == REPEATS,
            f"{label}: {stage} raw timing arrays are invalid",
        )
        expected_wall = [
            batch_by_stage_repeat[(stage, repeat)]["wall_ns"]
            for repeat in range(REPEATS)
        ]
        expected_cpu = [
            batch_by_stage_repeat[(stage, repeat)]["process_cpu_ns"]
            for repeat in range(REPEATS)
        ]
        _require(raw_wall == expected_wall, f"{label}: {stage} wall array/batches differ")
        _require(raw_cpu == expected_cpu, f"{label}: {stage} CPU array/batches differ")

        wall_values = [value / OFFICIAL_ITERATIONS[stage] for value in expected_wall]
        cpu_values = [value / OFFICIAL_ITERATIONS[stage] for value in expected_cpu]
        wall_median = statistics.median(wall_values)
        cpu_median = statistics.median(cpu_values)
        _require(
            _close(row.get("median_ns_per_evaluation"), wall_median),
            f"{label}: {stage} reported wall median differs from raw batches",
        )
        _require(
            _close(row.get("median_process_cpu_ns_per_evaluation"), cpu_median),
            f"{label}: {stage} reported CPU median differs from raw batches",
        )
        for order in range(ORDERS):
            _require(
                len(stage_order_values[stage][order]["wall"]) == REPEATS // ORDERS,
                f"{label}: {stage} order {order} is not balanced",
            )
        cluster_medians[stage] = {"wall": wall_median, "process_cpu": cpu_median}

    return {
        "ordinal": ordinal,
        "path": str(path) if path is not None else None,
        "document": document,
        "model_sha256": model_sha256,
        "corpus": corpus,
        "correctness_checksum": correctness_checksum,
        "stage_checksums": stage_checksums,
        "timed_sinks": timed_sinks,
        "cluster_medians": cluster_medians,
        "stage_order_values": stage_order_values,
        "source_label": run_identity.get("source_label"),
    }


def _source_commit(source_label: Any) -> str | None:
    if not isinstance(source_label, str):
        return None
    match = re.search(r"(?:clean-stage-|source-commit:)([0-9a-f]{40})", source_label)
    return match.group(1) if match is not None else None


def _validate_experiment_identity(clusters: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    all_clusters = [cluster for values in clusters.values() for cluster in values]
    identity_fields = (
        "model_sha256",
        "corpus",
        "correctness_checksum",
        "stage_checksums",
        "timed_sinks",
    )
    for field in identity_fields:
        reference = all_clusters[0][field]
        _require(
            all(cluster[field] == reference for cluster in all_clusters[1:]),
            f"cross-host {field} identity mismatch",
        )
    commits = {
        commit
        for cluster in all_clusters
        if (commit := _source_commit(cluster.get("source_label"))) is not None
    }
    _require(len(commits) <= 1, "cross-host source commit identity mismatch")
    return {
        "model_sha256": all_clusters[0]["model_sha256"],
        "corpus": all_clusters[0]["corpus"],
        "correctness_checksum": all_clusters[0]["correctness_checksum"],
        "stage_checksums": all_clusters[0]["stage_checksums"],
        "timed_sinks": all_clusters[0]["timed_sinks"],
        "source_commit": next(iter(commits)) if commits else None,
    }


def _quantile(values: Sequence[float], probability: float) -> float:
    _require(bool(values), "cannot compute a quantile of an empty sample")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _distribution_summary(values: Sequence[float]) -> dict[str, Any]:
    return {
        "median": statistics.median(values),
        "ci95": [_quantile(values, 0.025), _quantile(values, 0.975)],
    }


def _bootstrap_host_estimate(
    clusters: Sequence[Mapping[str, Any]],
    *,
    stage: str,
    metric: str,
    rng: random.Random,
) -> float:
    cluster_estimates = []
    for _ in range(len(clusters)):
        cluster = clusters[rng.randrange(len(clusters))]
        values = []
        for order in range(ORDERS):
            stratum = cluster["stage_order_values"][stage][order][metric]
            values.extend(stratum[rng.randrange(len(stratum))] for _ in stratum)
        cluster_estimates.append(statistics.median(values))
    return statistics.median(cluster_estimates)


def _bootstrap(
    clusters: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    _require(replicates > 0, "bootstrap replicates must be positive")
    local_rng = random.Random(seed ^ 0x4D345F4C4F43414C)
    heroku_rng = random.Random(seed ^ 0x4845524F4B555F32)
    stage_results: dict[str, Any] = {}
    for stage in STAGES:
        metric_results: dict[str, Any] = {}
        for metric in METRICS:
            local_values: list[float] = []
            heroku_values: list[float] = []
            ratios: list[float] = []
            deltas: list[float] = []
            for _ in range(replicates):
                local = _bootstrap_host_estimate(
                    clusters["local"], stage=stage, metric=metric, rng=local_rng
                )
                heroku = _bootstrap_host_estimate(
                    clusters["heroku"], stage=stage, metric=metric, rng=heroku_rng
                )
                local_values.append(local)
                heroku_values.append(heroku)
                ratios.append(heroku / local)
                deltas.append(heroku - local)
            metric_results[metric] = {
                "local_ns_per_evaluation": _distribution_summary(local_values),
                "heroku_ns_per_evaluation": _distribution_summary(heroku_values),
                "ratio_heroku_over_local": _distribution_summary(ratios),
                "delta_heroku_minus_local_ns_per_evaluation": _distribution_summary(
                    deltas
                ),
            }
        stage_results[stage] = metric_results
    return {
        "replicates": replicates,
        "seed": seed,
        "confidence_level": 0.95,
        "design": {
            "host_sampling": "independent",
            "levels": ["cluster", "batch_within_cluster"],
            "within_cluster_strata": "stage_order_0_to_3",
            "estimator": "median_of_cluster_medians",
        },
        "stages": stage_results,
    }


def analyze_documents(
    local_documents: Sequence[Mapping[str, Any]],
    heroku_documents: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    local_paths: Sequence[Path | None] | None = None,
    heroku_paths: Sequence[Path | None] | None = None,
) -> dict[str, Any]:
    """Validate six run documents and return the cross-host analysis."""

    _require(
        len(local_documents) == CLUSTERS_PER_HOST,
        f"local must have exactly {CLUSTERS_PER_HOST} independent clusters",
    )
    _require(
        len(heroku_documents) == CLUSTERS_PER_HOST,
        f"heroku must have exactly {CLUSTERS_PER_HOST} independent clusters",
    )
    if local_paths is None:
        local_paths = [None] * CLUSTERS_PER_HOST
    if heroku_paths is None:
        heroku_paths = [None] * CLUSTERS_PER_HOST
    _require(len(local_paths) == CLUSTERS_PER_HOST, "local path count mismatch")
    _require(len(heroku_paths) == CLUSTERS_PER_HOST, "heroku path count mismatch")

    clusters = {
        "local": [
            _validate_run(document, host="local", ordinal=index, path=path)
            for index, (document, path) in enumerate(
                zip(local_documents, local_paths), start=1
            )
        ],
        "heroku": [
            _validate_run(document, host="heroku", ordinal=index, path=path)
            for index, (document, path) in enumerate(
                zip(heroku_documents, heroku_paths), start=1
            )
        ],
    }
    identity = _validate_experiment_identity(clusters)

    primary_stages: dict[str, Any] = {}
    for stage in STAGES:
        metric_results: dict[str, Any] = {}
        for metric in METRICS:
            local_cluster_medians = [
                cluster["cluster_medians"][stage][metric]
                for cluster in clusters["local"]
            ]
            heroku_cluster_medians = [
                cluster["cluster_medians"][stage][metric]
                for cluster in clusters["heroku"]
            ]
            local = statistics.median(local_cluster_medians)
            heroku = statistics.median(heroku_cluster_medians)
            metric_results[metric] = {
                "local_cluster_medians_ns_per_evaluation": local_cluster_medians,
                "heroku_cluster_medians_ns_per_evaluation": heroku_cluster_medians,
                "local_median_of_cluster_medians_ns_per_evaluation": local,
                "heroku_median_of_cluster_medians_ns_per_evaluation": heroku,
                "ratio_heroku_over_local": heroku / local,
                "delta_heroku_minus_local_ns_per_evaluation": heroku - local,
            }
        primary_stages[stage] = metric_results

    return {
        "schema_version": 1,
        "protocol": ANALYSIS_PROTOCOL,
        "status": "valid",
        "input_protocol": PROTOCOL,
        "identity": identity,
        "clusters": {
            host: [
                {
                    "ordinal": cluster["ordinal"],
                    "path": cluster["path"],
                    "source_label": cluster["source_label"],
                }
                for cluster in host_clusters
            ]
            for host, host_clusters in clusters.items()
        },
        "primary": {
            "estimator": "median_of_cluster_medians",
            "stages": primary_stages,
        },
        "bootstrap": _bootstrap(
            clusters, replicates=bootstrap_replicates, seed=bootstrap_seed
        ),
        "interpretation": {
            "scope": "isolated_kernel_core_stage_timing",
            "stages_are_additive": False,
            "ratio_is_pure_heroku_overhead": False,
        },
    }


def analyze_directories(
    local_directory: Path,
    heroku_directory: Path,
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    local_loaded = _load_documents(local_directory, "local")
    heroku_loaded = _load_documents(heroku_directory, "heroku")
    return analyze_documents(
        [document for _, document in local_loaded],
        [document for _, document in heroku_loaded],
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        local_paths=[path for path, _ in local_loaded],
        heroku_paths=[path for path, _ in heroku_loaded],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--heroku-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = analyze_directories(
            args.local_dir,
            args.heroku_dir,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
        )
    except AnalysisError as error:
        raise SystemExit(f"analysis failed: {error}") from error
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Compare complete-round CPU profiles from a local host and Heroku.

The input is deliberately profiler-neutral.  A host document contains search
round metadata plus either raw flat symbol counts or already-normalized flat
bucket counts.  The canonical shape is::

    {
      "host": "local-m4",
      "rounds": [
        {
          "round": 1,
          "mode": "profile",
          "nodes": 1698311,
          "process_cpu_seconds": 0.35
        }
      ],
      "profile_rounds": [
        {
          "round": 1,
          "sample_summary": {
            "sample_period_seconds": 0.001,
            "total_samples": 340,
            "symbols": [
              {"symbol": "chess::...", "flat_samples": 42}
            ]
          }
        }
      ]
    }

``sample_summary`` may instead contain ``flat_buckets`` (or
``flat_bucket_counts``), whose values are counts in the same arbitrary unit.
Without calibration, the analyzer estimates component CPU cost as::

    process CPU seconds * flat sample share / searched nodes

For a complete A-B-B-A schedule, the default instead pools the surrounding
control rounds (A0+A3) to obtain a block-local control CPU ns/node and assigns
that total to both profiled B share vectors. This prevents a profiler-induced
frequency change from biasing every absolute component cost. The measured B
CPU time remains in the output as an explicit diagnostic.

This is an on-CPU attribution.  It cannot attribute wall time during which a
VM or dyno was not scheduled.  Confidence intervals independently resample
complete A-B-B-A blocks within each host when the profiled round ids expose
that schedule, with complete-round resampling retained as a sensitivity mode.
Positional paired resampling is available only for experiments whose units
really are paired.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Mapping, Sequence


BUCKETS = (
    "nn_forward",
    "nn_accumulator",
    "transposition_table",
    "move_generation_attacks",
    "make_unmake_king_safety",
    "static_exchange_evaluation",
    "move_ordering",
    "search_control",
    "runtime",
    "other",
)

BUCKET_ALIASES = {
    "nn": "nn_forward",
    "nnue": "nn_forward",
    "forward": "nn_forward",
    "accumulator": "nn_accumulator",
    "tt": "transposition_table",
    "movegen": "move_generation_attacks",
    "move_generation": "move_generation_attacks",
    "attacks": "move_generation_attacks",
    "make_unmake": "make_unmake_king_safety",
    "king_safety": "make_unmake_king_safety",
    "see": "static_exchange_evaluation",
    "ordering": "move_ordering",
    "search": "search_control",
    "libc_unknown": "runtime",
    "unknown": "other",
}


def _patterns(*patterns: str) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)


# Rules are ordered: a specialized leaf such as static_exchange_eval must be
# classified before generic words such as "evaluate" or "move".
SYMBOL_RULES: tuple[tuple[str, tuple[re.Pattern[str], ...]], ...] = (
    (
        "static_exchange_evaluation",
        _patterns(
            r"static[_ ]exchange",
            r"\bsee(?:_|\b)",
            r"clear_see_piece",
            r"find_least_valuable_(?:pseudo_)?attacker",
        ),
    ),
    (
        "nn_forward",
        _patterns(
            r"VnniNetwork::evaluate",
            r"Avx2Network::evaluate",
            r"Neon.*Network::evaluate",
            r"PhaseCandidateKernel.*::evaluate",
            r"X86PhaseKernel::evaluate",
            r"forward_positional",
            r"PhaseQuantizedNnueModel::evaluate",
            r"evaluate_cp_rounded",
            r"uses_(?:accelerated|neon_dotprod)_kernel",
            r"forward_kernel_name",
        ),
    ),
    (
        "nn_accumulator",
        _patterns(
            r"PhaseQuantizedNnueAccumulator",
            r"NnueAccumulator",
            r"accumulator.*(?:rebuild|update|copy|push|pop)",
        ),
    ),
    (
        "transposition_table",
        _patterns(
            r"TranspositionTable",
            r"(?:^|::)probe_tt",
            r"(?:^|::)store_tt",
            r"\btt_(?:probe|store|lookup|replace)",
        ),
    ),
    (
        "move_ordering",
        _patterns(
            r"HistoryTable",
            r"CounterHistory",
            r"CounterMove",
            r"KillerMove",
            r"sort_scored",
            r"ordered_.*moves",
            r"priority_quiet",
            r"reward_quiet_cutoff",
            r"make_(?:qsearch_)?scored",
        ),
    ),
    (
        "move_generation_attacks",
        _patterns(
            r"generate_legal",
            r"generate_.*moves",
            r"(?:bishop|rook|queen|king|knight|pawn)_attacks",
            r"is_square_attacked",
            r"is_pseudo_move_legal",
            r"is_(?:quiet_non_promotion_)?move_legal_by_attack_check",
            r"gives_check",
            r"try_emit_legal",
            r"(?:^|::)in_check(?:\(|$)",
            r"(?:^|::)pop_lsb",
            r"(?:^|::)popcount",
        ),
    ),
    (
        "make_unmake_king_safety",
        _patterns(
            r"(?:^|::)(?:un)?make_move",
            r"make_move_impl",
            r"make_state_snapshot",
            r"(?:capture|move|clear)_piece_fast",
            r"(?:^|::)set_piece",
            r"king_safety",
            r"castling_rights",
            r"zobrist::",
        ),
    ),
    (
        "search_control",
        _patterns(
            r"NnueSearcher",
            r"(?:^|::)negamax",
            r"(?:^|::)quiescence",
            r"null_move",
            r"history_draw",
            r"RepetitionStack",
            r"ScopedRepetitionPush",
            r"search_best_move",
        ),
    ),
    (
        "runtime",
        _patterns(
            r"^\[.*(?:libc|libm|libgcc|kernel|unknown).*$",
            r"(?:^|::)(?:malloc|free|memcpy|memmove|memset)(?:\b|@)",
            r"^_platform_(?:bzero|memcpy|memmove|memset)$",
            r"^_+chkstk_darwin$",
            r"^_?(?:exp|exp2|log|log2|log10)(?:f|l)?$",
            r"^_?(?:start|init)$",
            r"^main$",
        ),
    ),
)


def _strip_template_arguments(symbol: str) -> str:
    """Remove balanced C++ template arguments without losing the leaf name.

    Clang often prints a callback's complete type inside a leaf function's
    template arguments.  Matching the unmodified text can therefore classify
    ``generate_legal...<...ordered_moves...>`` as move ordering.  Removing
    balanced angle-bracket contents preserves names such as
    ``PhaseCandidateKernel<...>::evaluate`` while discarding nested callback
    implementation details.
    """

    result: list[str] = []
    depth = 0
    for character in symbol:
        if character == "<":
            depth += 1
            continue
        if character == ">" and depth > 0:
            depth -= 1
            continue
        if depth == 0:
            result.append(character)
    return "".join(result)


def _leaf_classification_subject(symbol: str) -> str:
    """Return the outermost demangled leaf name used by bucket rules."""

    # Protect this namespace marker from the argument-list delimiter below.
    normalized = symbol.replace("(anonymous namespace)", "anonymous_namespace")
    without_templates = _strip_template_arguments(normalized)
    return without_templates.split("(", 1)[0].strip()


def classify_symbol(symbol: str) -> str:
    """Map a demangled flat-profile symbol into one exclusive bucket."""

    subject = _leaf_classification_subject(symbol)
    for bucket, patterns in SYMBOL_RULES:
        if any(pattern.search(subject) for pattern in patterns):
            return bucket
    return "other"


_DURATION_SCALES = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}


def _duration_seconds(value: str, unit: str) -> float:
    try:
        scale = _DURATION_SCALES[unit]
    except KeyError as error:
        raise ValueError(f"unsupported pprof duration unit: {unit}") from error
    return float(value) * scale


def parse_pprof_flat_report(report: str) -> dict[str, Any]:
    """Convert a ``pprof -top -flat -functions`` report to sample_summary.

    Per-round reports must be generated with node/edge fractions set to zero
    so every flat leaf is present. Rows with zero flat time are harmless.
    """

    total_match = re.search(
        r"\bof\s+([0-9.]+)(ns|us|ms|s)\s+total$", report, re.MULTILINE
    )
    if total_match is None:
        raise ValueError("cannot parse total sampled time from pprof report")
    total_seconds = _duration_seconds(*total_match.groups())
    if total_seconds <= 0.0:
        raise ValueError("pprof report has no positive sampled time")
    symbols = []
    row_pattern = re.compile(
        r"^\s*([0-9.]+)(ns|us|ms|s)\s+[0-9.]+%\s+[0-9.]+%"
        r"\s+[0-9.]+(?:ns|us|ms|s)\s+[0-9.]+%\s{2,}(.+)$"
    )
    for line in report.splitlines():
        match = row_pattern.match(line)
        if match is None:
            continue
        seconds = _duration_seconds(match.group(1), match.group(2))
        if seconds > 0.0:
            symbols.append({"symbol": match.group(3), "flat_seconds": seconds})
    if not symbols:
        raise ValueError("pprof report contains no positive flat symbol rows")
    visible_total = sum(row["flat_seconds"] for row in symbols)
    # pprof normally prints exact multiples of the sample period. Permit one
    # final-display quantum of rounding; a larger excess is rejected later.
    if visible_total > total_seconds:
        excess = visible_total - total_seconds
        display_quantum = min(row["flat_seconds"] for row in symbols)
        if excess <= display_quantum + 1e-12:
            total_seconds = visible_total
    return {
        "total_sampled_seconds": total_seconds,
        "symbols": symbols,
        "source": "pprof flat function report",
    }


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot take a percentile of an empty sequence")
    ordered = sorted(values)
    index = probability * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _positive_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{field} must be finite and positive")
    return result


def _nonnegative_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field} must be finite and non-negative")
    return result


def _first(mapping: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _canonical_bucket(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
    normalized = BUCKET_ALIASES.get(normalized, normalized)
    if normalized not in BUCKETS:
        raise ValueError(f"unknown pre-normalized bucket: {name}")
    return normalized


def _value_from_row(row: Any, fields: Sequence[str], label: str) -> float:
    if isinstance(row, (int, float)) and not isinstance(row, bool):
        return _nonnegative_number(row, label)
    if not isinstance(row, Mapping):
        raise ValueError(f"{label} must be a number or object")
    value = _first(row, fields)
    if value is None:
        raise ValueError(f"{label} lacks one of {', '.join(fields)}")
    return _nonnegative_number(value, label)


def _extract_sample_weights(
    summary: Mapping[str, Any],
) -> tuple[dict[str, float], float, str]:
    """Return exclusive bucket weights, total weight, and source unit."""

    bucket_rows = _first(
        summary, ("flat_buckets", "flat_bucket_counts", "buckets")
    )
    declared_weight_ns = _first(summary, ("total_weight_ns",))
    declared_samples = _first(summary, ("total_samples", "sample_count"))
    declared_seconds = _first(
        summary,
        ("total_sampled_seconds", "sampled_seconds", "total_flat_seconds"),
    )
    if declared_weight_ns is not None:
        # xctrace rows can have unequal duration weights.  Their exact integer
        # nanoseconds are authoritative even when audit-only row counts are
        # present in the same document.
        fields = ("flat_weight_ns", "weight_ns")
        total: float | None = _positive_number(
            declared_weight_ns, "declared total_weight_ns"
        )
        unit = "nanoseconds"
        if declared_seconds is not None:
            seconds = _positive_number(
                declared_seconds, "declared sampled seconds"
            )
            if not math.isclose(
                total * 1e-9, seconds, rel_tol=1e-9, abs_tol=1e-12
            ):
                raise ValueError(
                    "total_weight_ns is inconsistent with total_sampled_seconds"
                )
    elif declared_samples is not None:
        fields = ("flat_samples", "samples", "count")
        total = _positive_number(declared_samples, "declared sample count")
        unit = "samples"
    elif declared_seconds is not None:
        fields = ("flat_seconds", "seconds")
        total = _positive_number(declared_seconds, "declared sampled seconds")
        unit = "seconds"
    else:
        total = None
        fields = ()
        unit = "flat_weight" if bucket_rows is not None else "seconds"

    weights = {bucket: 0.0 for bucket in BUCKETS}
    if bucket_rows is not None:
        if not isinstance(bucket_rows, Mapping):
            raise ValueError("flat_buckets must be an object")
        bucket_fields = fields or (
            "flat_seconds",
            "seconds",
            "flat_samples",
            "samples",
            "count",
            "flat_weight_ns",
            "weight_ns",
        )
        for raw_name, row in bucket_rows.items():
            bucket = _canonical_bucket(str(raw_name))
            weights[bucket] += _value_from_row(
                row, bucket_fields, f"bucket {raw_name}"
            )
    else:
        symbol_rows = summary.get("symbols")
        if symbol_rows is None:
            raise ValueError("sample summary has neither symbols nor flat_buckets")
        if isinstance(symbol_rows, Mapping):
            iterable = [
                {"symbol": symbol, **(dict(row) if isinstance(row, Mapping) else {"count": row})}
                for symbol, row in symbol_rows.items()
            ]
        elif isinstance(symbol_rows, list):
            iterable = symbol_rows
        else:
            raise ValueError("symbols must be an array or object")
        symbol_fields = fields or ("flat_seconds", "seconds")
        for index, row in enumerate(iterable):
            if not isinstance(row, Mapping) or not isinstance(row.get("symbol"), str):
                raise ValueError(f"symbol row {index} lacks a string symbol")
            weight = _value_from_row(row, symbol_fields, f"symbol row {index}")
            weights[classify_symbol(row["symbol"])] += weight

    summed = sum(weights.values())
    total = summed if total is None else total
    tolerance = max(1e-9, total * 1e-9)
    if summed > total + tolerance:
        raise ValueError(
            f"flat weights ({summed}) exceed declared sample total ({total})"
        )
    if summed > total:
        # Avoid shares summing slightly above one after benign float rounding.
        total = summed
    # Sampling tools can omit unresolved/truncated leaf rows.  Preserve their
    # cost explicitly instead of silently renormalizing the visible symbols.
    if summed < total:
        weights["other"] += total - summed
    return weights, total, unit


def _has_sample_payload(row: Mapping[str, Any]) -> bool:
    return any(
        key in row
        for key in (
            "sample_summary",
            "sampling",
            "symbols",
            "flat_buckets",
            "flat_bucket_counts",
            "buckets",
        )
    )


def _round_id(row: Mapping[str, Any]) -> Any:
    value = _first(row, ("round", "round_id", "id"))
    if value is None or isinstance(value, (dict, list)):
        raise ValueError("profile round lacks a scalar round identifier")
    return value


_CPU_FIELDS = ("process_cpu_seconds", "cpu_seconds", "cpu_time_seconds")


def _round_cpu_seconds(row: Mapping[str, Any], label: str) -> float:
    return _positive_number(_first(row, _CPU_FIELDS), label)


def _profile_rows_from_document(
    document: Mapping[str, Any], raw_rounds: Sequence[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    explicit_profiles = _first(document, ("profile_rounds", "profiled_rounds"))
    if explicit_profiles is None:
        return [row for row in raw_rounds if _has_sample_payload(row)]
    if not isinstance(explicit_profiles, list):
        raise ValueError("profile_rounds must be an array")
    if not all(isinstance(row, Mapping) for row in explicit_profiles):
        raise ValueError("each profile round must be an object")
    return list(explicit_profiles)


def _abba_control_blocks(
    metadata: Mapping[Any, Mapping[str, Any]],
    profile_rows: Sequence[Mapping[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Validate A-B-B-A metadata and derive block-local control totals."""

    profile_ids = []
    for row in profile_rows:
        identifier = _round_id(row)
        if (
            isinstance(identifier, bool)
            or not isinstance(identifier, int)
            or identifier < 0
            or identifier % 4 not in (1, 2)
        ):
            raise ValueError(
                "ABBA control calibration requires profiled round ids 4k+1 and 4k+2"
            )
        profile_ids.append(identifier)
    if len(profile_ids) != len(set(profile_ids)):
        raise ValueError("ABBA control calibration found duplicate profiled rounds")

    blocks: dict[int, dict[str, Any]] = {}
    for block in sorted({identifier // 4 for identifier in profile_ids}):
        expected_profiles = [4 * block + 1, 4 * block + 2]
        actual_profiles = sorted(
            identifier for identifier in profile_ids if identifier // 4 == block
        )
        if actual_profiles != expected_profiles:
            raise ValueError(
                f"ABBA control calibration block {block} lacks both B rounds"
            )
        control_ids = [4 * block, 4 * block + 3]
        missing = [identifier for identifier in control_ids if identifier not in metadata]
        if missing:
            raise ValueError(
                f"ABBA control calibration block {block} lacks controls {missing}"
            )
        controls = [metadata[identifier] for identifier in control_ids]
        for identifier, row in zip(control_ids, controls):
            if row.get("mode", "control") != "control":
                raise ValueError(
                    f"ABBA calibration round {identifier} is not a control"
                )
        for identifier in expected_profiles:
            if identifier not in metadata:
                raise ValueError(
                    f"ABBA control calibration lacks profile metadata round {identifier}"
                )
            if metadata[identifier].get("mode", "profile") != "profile":
                raise ValueError(
                    f"ABBA calibration round {identifier} is not profiled"
                )
        control_nodes = sum(
            _positive_number(row.get("nodes"), f"control round {identifier} nodes")
            for identifier, row in zip(control_ids, controls)
        )
        control_cpu_seconds = sum(
            _round_cpu_seconds(row, f"control round {identifier} CPU seconds")
            for identifier, row in zip(control_ids, controls)
        )
        blocks[block] = {
            "block": block,
            "control_round_ids": control_ids,
            "profile_round_ids": expected_profiles,
            "control_nodes": control_nodes,
            "control_process_cpu_seconds": control_cpu_seconds,
            "control_cpu_ns_per_node": control_cpu_seconds * 1e9 / control_nodes,
        }
    if not blocks:
        raise ValueError("ABBA control calibration found no complete blocks")
    return blocks


def _control_calibration_probe(document: Mapping[str, Any]) -> dict[str, Any]:
    """Report availability without turning an auto fallback into an error."""

    try:
        raw_rounds = document.get("rounds", [])
        if not isinstance(raw_rounds, list) or not all(
            isinstance(row, Mapping) for row in raw_rounds
        ):
            raise ValueError("rounds must be an array of objects")
        metadata = {_round_id(row): row for row in raw_rounds}
        if len(metadata) != len(raw_rounds):
            raise ValueError("duplicate round metadata")
        profile_rows = _profile_rows_from_document(document, raw_rounds)
        blocks = _abba_control_blocks(metadata, profile_rows)
        return {"available": True, "blocks": len(blocks), "reason": None}
    except ValueError as error:
        return {"available": False, "blocks": 0, "reason": str(error)}


def parse_host_document(
    document: Mapping[str, Any],
    fallback_label: str,
    *,
    cpu_calibration: str = "auto",
) -> dict[str, Any]:
    """Validate a host document and normalize its profiled rounds."""

    if cpu_calibration not in {"auto", "abba-control", "profile-measured"}:
        raise ValueError(
            "cpu calibration must be auto, abba-control, or profile-measured"
        )
    raw_rounds = document.get("rounds", [])
    if not isinstance(raw_rounds, list):
        raise ValueError("rounds must be an array")
    metadata: dict[Any, dict[str, Any]] = {}
    for row in raw_rounds:
        if not isinstance(row, Mapping):
            raise ValueError("each round must be an object")
        identifier = _round_id(row)
        if identifier in metadata:
            raise ValueError(f"duplicate round metadata: {identifier}")
        metadata[identifier] = dict(row)

    profile_rows = _profile_rows_from_document(document, raw_rounds)
    if not profile_rows:
        raise ValueError("host document contains no profiled rounds")

    control_blocks: dict[int, dict[str, Any]] = {}
    resolved_calibration = cpu_calibration
    fallback_reason = None
    if cpu_calibration in {"auto", "abba-control"}:
        try:
            control_blocks = _abba_control_blocks(metadata, profile_rows)
            resolved_calibration = "abba-control"
        except ValueError as error:
            if cpu_calibration == "abba-control":
                raise
            resolved_calibration = "profile-measured"
            fallback_reason = str(error)

    normalized = []
    seen: set[Any] = set()
    for profile_row in profile_rows:
        identifier = _round_id(profile_row)
        if identifier in seen:
            raise ValueError(f"duplicate profiled round: {identifier}")
        seen.add(identifier)
        merged = {**metadata.get(identifier, {}), **dict(profile_row)}
        nodes = _positive_number(merged.get("nodes"), f"round {identifier} nodes")
        cpu_seconds = _round_cpu_seconds(
            merged, f"round {identifier} process CPU seconds"
        )
        summary = _first(merged, ("sample_summary", "sampling"))
        if summary is None:
            summary = merged
        if not isinstance(summary, Mapping):
            raise ValueError(f"round {identifier} sample_summary must be an object")
        weights, total_weight, weight_unit = _extract_sample_weights(summary)
        shares = {bucket: weights[bucket] / total_weight for bucket in BUCKETS}

        if resolved_calibration == "abba-control":
            block = identifier // 4
            control_ns_per_node = control_blocks[block]["control_cpu_ns_per_node"]
            attribution_cpu_seconds = control_ns_per_node * nodes / 1e9
        else:
            block = identifier // 4 if isinstance(identifier, int) else None
            control_ns_per_node = None
            attribution_cpu_seconds = cpu_seconds
        normalized.append(
            {
                "round": identifier,
                "block": block,
                "nodes": nodes,
                "process_cpu_seconds": cpu_seconds,
                "measured_profile_cpu_seconds": cpu_seconds,
                "attribution_cpu_seconds": attribution_cpu_seconds,
                "control_cpu_ns_per_node": control_ns_per_node,
                "profile_to_control_cpu_ns_per_node_ratio": (
                    (cpu_seconds * 1e9 / nodes) / control_ns_per_node
                    if control_ns_per_node is not None
                    else None
                ),
                "sample_total_weight": total_weight,
                "sample_weight_unit": weight_unit,
                "bucket_sample_weights": weights,
                "bucket_sample_shares": shares,
            }
        )

    block_metadata = []
    if resolved_calibration == "abba-control":
        by_round = {row["round"]: row for row in normalized}
        for _block, control in sorted(control_blocks.items()):
            profiles = [by_round[identifier] for identifier in control["profile_round_ids"]]
            profile_nodes = sum(row["nodes"] for row in profiles)
            profile_cpu = sum(row["measured_profile_cpu_seconds"] for row in profiles)
            measured_ns_per_node = profile_cpu * 1e9 / profile_nodes
            block_metadata.append(
                {
                    **control,
                    "measured_profile_nodes": profile_nodes,
                    "measured_profile_process_cpu_seconds": profile_cpu,
                    "measured_profile_cpu_ns_per_node": measured_ns_per_node,
                    "profile_to_control_cpu_ns_per_node_ratio": (
                        measured_ns_per_node / control["control_cpu_ns_per_node"]
                    ),
                }
            )

    label = _first(document, ("host", "host_label", "label")) or fallback_label
    return {
        "host": str(label),
        "rounds": normalized,
        "calibration": {
            "requested": cpu_calibration,
            "resolved": resolved_calibration,
            "enabled": resolved_calibration == "abba-control",
            "method": (
                "for each 4k A, 4k+1 B, 4k+2 B, 4k+3 A block, pool A CPU/nodes "
                "and assign its CPU ns/node total to each B sample-share vector"
                if resolved_calibration == "abba-control"
                else "use measured profiled-round process CPU directly"
            ),
            "fallback_reason": fallback_reason,
            "blocks": block_metadata,
        },
    }


def _aggregate_rounds(rounds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    nodes = sum(float(row["nodes"]) for row in rounds)
    measured_cpu_seconds = sum(
        float(row["measured_profile_cpu_seconds"]) for row in rounds
    )
    attribution_cpu_seconds = sum(
        float(row["attribution_cpu_seconds"]) for row in rounds
    )
    if (
        nodes <= 0.0
        or measured_cpu_seconds <= 0.0
        or attribution_cpu_seconds <= 0.0
    ):
        raise ValueError("cannot aggregate empty or non-positive rounds")
    total_ns_per_node = attribution_cpu_seconds * 1e9 / nodes
    measured_ns_per_node = measured_cpu_seconds * 1e9 / nodes
    calibrated = all(row.get("control_cpu_ns_per_node") is not None for row in rounds)
    buckets: dict[str, Any] = {}
    for bucket in BUCKETS:
        attributed_cpu = sum(
            float(row["attribution_cpu_seconds"])
            * float(row["bucket_sample_shares"][bucket])
            for row in rounds
        )
        raw_weight = sum(
            float(row["bucket_sample_weights"][bucket]) for row in rounds
        )
        raw_total = sum(float(row["sample_total_weight"]) for row in rounds)
        buckets[bucket] = {
            "attributed_cpu_seconds": attributed_cpu,
            "estimated_cpu_seconds": attributed_cpu,
            "cpu_ns_per_node": attributed_cpu * 1e9 / nodes,
            "estimated_cpu_share": attributed_cpu / attribution_cpu_seconds,
            "pooled_raw_sample_share": raw_weight / raw_total,
        }
    return {
        "rounds": len(rounds),
        "round_ids": [row["round"] for row in rounds],
        "nodes": nodes,
        "process_cpu_seconds": measured_cpu_seconds,
        "measured_profile_process_cpu_seconds": measured_cpu_seconds,
        "attribution_cpu_seconds": attribution_cpu_seconds,
        "measured_profile_cpu_ns_per_node": measured_ns_per_node,
        "cpu_ns_per_node": total_ns_per_node,
        "profile_to_control_cpu_ns_per_node_ratio": (
            measured_ns_per_node / total_ns_per_node if calibrated else None
        ),
        "buckets": buckets,
    }


def summarize_host(host: Mapping[str, Any]) -> dict[str, Any]:
    result = _aggregate_rounds(host["rounds"])
    result["host"] = host["host"]
    blocks = host["calibration"]["blocks"]
    block_ratios = [
        row["profile_to_control_cpu_ns_per_node_ratio"] for row in blocks
    ]
    control_totals = [row["control_cpu_ns_per_node"] for row in blocks]
    result["calibration"] = {
        **host["calibration"],
        "cpu_total_source": host["calibration"]["resolved"],
        "measured_profile_cpu_ns_per_node": result[
            "measured_profile_cpu_ns_per_node"
        ],
        "attributed_cpu_ns_per_node": result["cpu_ns_per_node"],
        "profile_to_control_cpu_ns_per_node_ratio": result[
            "profile_to_control_cpu_ns_per_node_ratio"
        ],
        "block_profile_to_control_ratio_median": (
            percentile(block_ratios, 0.5) if block_ratios else None
        ),
        "block_profile_to_control_ratio_range": (
            [min(block_ratios), max(block_ratios)] if block_ratios else None
        ),
        "block_control_cpu_ns_per_node_range": (
            [min(control_totals), max(control_totals)] if control_totals else None
        ),
    }
    return result


def _comparison(
    local_summary: Mapping[str, Any], heroku_summary: Mapping[str, Any]
) -> dict[str, Any]:
    total_gap = (
        float(heroku_summary["cpu_ns_per_node"])
        - float(local_summary["cpu_ns_per_node"])
    )
    result: dict[str, Any] = {}
    names = ("total", *BUCKETS)
    for name in names:
        if name == "total":
            local_value = float(local_summary["cpu_ns_per_node"])
            heroku_value = float(heroku_summary["cpu_ns_per_node"])
        else:
            local_value = float(local_summary["buckets"][name]["cpu_ns_per_node"])
            heroku_value = float(heroku_summary["buckets"][name]["cpu_ns_per_node"])
        gap = heroku_value - local_value
        result[name] = {
            "local_cpu_ns_per_node": local_value,
            "heroku_cpu_ns_per_node": heroku_value,
            "slowdown_ratio": heroku_value / local_value if local_value > 0.0 else None,
            "gap_cpu_ns_per_node": gap,
            "gap_share": gap / total_gap if total_gap != 0.0 else None,
        }
    return result


def _interval(values: Sequence[float]) -> list[float] | None:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return None
    return [percentile(finite, 0.025), percentile(finite, 0.975)]


def _abba_block_units(
    rounds: Sequence[Mapping[str, Any]],
) -> list[list[Mapping[str, Any]]]:
    """Group the two profiled B rounds from each A-B-B-A block."""

    grouped: dict[int, dict[int, Mapping[str, Any]]] = {}
    for row in rounds:
        identifier = row["round"]
        if (
            isinstance(identifier, bool)
            or not isinstance(identifier, int)
            or identifier < 0
        ):
            raise ValueError(
                "ABBA-block bootstrap requires non-negative integer round ids"
            )
        residue = identifier % 4
        if residue not in (1, 2):
            raise ValueError(
                "ABBA-block bootstrap requires profiled round ids congruent "
                "to 1 or 2 modulo 4"
            )
        block = identifier // 4
        by_residue = grouped.setdefault(block, {})
        if residue in by_residue:
            raise ValueError(f"duplicate profiled round in ABBA block {block}")
        by_residue[residue] = row

    units = []
    for block in sorted(grouped):
        by_residue = grouped[block]
        if set(by_residue) != {1, 2}:
            raise ValueError(
                f"ABBA block {block} must contain both profiled B rounds"
            )
        units.append([by_residue[1], by_residue[2]])
    if not units:
        raise ValueError("ABBA-block bootstrap has no complete blocks")
    return units


def _resolve_bootstrap_unit(
    local_rounds: Sequence[Mapping[str, Any]],
    heroku_rounds: Sequence[Mapping[str, Any]],
    requested: str | None,
) -> str:
    if requested not in {None, "round", "abba-block"}:
        raise ValueError("bootstrap unit must be round or abba-block")
    if requested is not None:
        return requested
    try:
        _abba_block_units(local_rounds)
        _abba_block_units(heroku_rounds)
    except ValueError:
        return "round"
    return "abba-block"


def _bootstrap(
    local_rounds: Sequence[Mapping[str, Any]],
    heroku_rounds: Sequence[Mapping[str, Any]],
    *,
    replicates: int,
    seed: int,
    mode: str,
    unit: str,
) -> dict[str, Any]:
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    if mode not in {"independent", "paired"}:
        raise ValueError("bootstrap mode must be independent or paired")
    if unit == "round":
        local_units = [[row] for row in local_rounds]
        heroku_units = [[row] for row in heroku_rounds]
        unit_label = "complete-round"
    elif unit == "abba-block":
        local_units = _abba_block_units(local_rounds)
        heroku_units = _abba_block_units(heroku_rounds)
        unit_label = "complete-ABBA-block"
    else:
        raise ValueError("bootstrap unit must be round or abba-block")
    if mode == "paired" and len(local_units) != len(heroku_units):
        raise ValueError("paired bootstrap requires equal sampling-unit counts")

    metric_names = ("total", *BUCKETS)
    values = {
        name: {
            "local_cpu_ns_per_node": [],
            "heroku_cpu_ns_per_node": [],
            "slowdown_ratio": [],
            "gap_cpu_ns_per_node": [],
            "gap_share": [],
        }
        for name in metric_names
    }
    generator = random.Random(seed)
    for _ in range(replicates):
        if mode == "paired":
            indices = [generator.randrange(len(local_units)) for _ in local_units]
            selected_local = [local_units[index] for index in indices]
            selected_heroku = [heroku_units[index] for index in indices]
        else:
            selected_local = [
                local_units[generator.randrange(len(local_units))]
                for _ in local_units
            ]
            selected_heroku = [
                heroku_units[generator.randrange(len(heroku_units))]
                for _ in heroku_units
            ]
        local_sample = [row for current in selected_local for row in current]
        heroku_sample = [row for current in selected_heroku for row in current]
        comparison = _comparison(
            _aggregate_rounds(local_sample), _aggregate_rounds(heroku_sample)
        )
        for name in metric_names:
            for metric, value in comparison[name].items():
                if value is not None and math.isfinite(float(value)):
                    values[name][metric].append(float(value))

    intervals = {}
    valid_replicates = {}
    for name in metric_names:
        intervals[name] = {
            metric: _interval(metric_values)
            for metric, metric_values in values[name].items()
        }
        valid_replicates[name] = {
            metric: len(metric_values)
            for metric, metric_values in values[name].items()
        }
    method = (
        f"paired {unit_label} percentile bootstrap"
        if mode == "paired"
        else f"independent {unit_label} percentile bootstrap within each host"
    )
    return {
        "method": method,
        "mode": mode,
        "unit": unit,
        "local_sampling_units": len(local_units),
        "heroku_sampling_units": len(heroku_units),
        "profiled_rounds_per_unit": 1 if unit == "round" else 2,
        "replicates": replicates,
        "seed": seed,
        "interval95": intervals,
        "valid_replicates": valid_replicates,
        "interpretation": "conditional descriptive intervals for this workload and these hosts",
    }


def analyze(
    local_document: Mapping[str, Any],
    heroku_document: Mapping[str, Any],
    *,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 20_260_825,
    bootstrap_mode: str = "independent",
    bootstrap_unit: str | None = None,
    cpu_calibration: str = "auto",
) -> dict[str, Any]:
    if cpu_calibration not in {"auto", "abba-control", "profile-measured"}:
        raise ValueError(
            "cpu calibration must be auto, abba-control, or profile-measured"
        )
    local_probe = _control_calibration_probe(local_document)
    heroku_probe = _control_calibration_probe(heroku_document)
    if cpu_calibration == "auto":
        resolved_calibration = (
            "abba-control"
            if local_probe["available"] and heroku_probe["available"]
            else "profile-measured"
        )
    else:
        resolved_calibration = cpu_calibration
    if resolved_calibration == "abba-control" and not (
        local_probe["available"] and heroku_probe["available"]
    ):
        raise ValueError(
            "ABBA control calibration must be available on both hosts: "
            f"local={local_probe}, heroku={heroku_probe}"
        )

    # Resolve once for both hosts. Applying control totals to only one side
    # would turn a measurement-method difference into an apparent slowdown.
    local = parse_host_document(
        local_document, "local", cpu_calibration=resolved_calibration
    )
    heroku = parse_host_document(
        heroku_document, "heroku", cpu_calibration=resolved_calibration
    )
    local_summary = summarize_host(local)
    heroku_summary = summarize_host(heroku)
    resolved_bootstrap_unit = _resolve_bootstrap_unit(
        local["rounds"], heroku["rounds"], bootstrap_unit
    )
    bootstrap = _bootstrap(
        local["rounds"],
        heroku["rounds"],
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
        mode=bootstrap_mode,
        unit=resolved_bootstrap_unit,
    )
    bootstrap["requested_unit"] = bootstrap_unit or "auto"
    return {
        "schema_version": 1,
        "metric_definition": (
            "component_cpu_ns_per_node = attributed_total_cpu_seconds * "
            "round_flat_sample_share * 1e9 / nodes; attributed total is the "
            "pooled surrounding A controls for each ABBA block when calibrated"
        ),
        "scope": (
            "on-CPU attribution; scheduler wait, architecture, compiler, and hosting "
            "effects are not separately identified"
        ),
        "cpu_calibration": {
            "requested": cpu_calibration,
            "resolved": resolved_calibration,
            "applied_consistently_to_both_hosts": True,
            "method": (
                "pooled surrounding A controls per complete ABBA block"
                if resolved_calibration == "abba-control"
                else "measured profile B CPU"
            ),
            "local_probe": local_probe,
            "heroku_probe": heroku_probe,
            "profile_measured_sensitivity": (
                "rerun with --cpu-calibration profile-measured"
                if resolved_calibration == "abba-control"
                else None
            ),
        },
        "buckets": list(BUCKETS),
        "local": local_summary,
        "heroku": heroku_summary,
        "comparison": _comparison(local_summary, heroku_summary),
        "bootstrap": bootstrap,
    }


def _load_json(path: Path) -> Mapping[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return document


def _round_from_report_name(path: Path) -> int:
    patterns = (r"(?:^|[-_])r(\d+)(?:[-_.]|$)", r"round[-_]?(\d+)")
    for pattern in patterns:
        match = re.search(pattern, path.name, re.IGNORECASE)
        if match is not None:
            return int(match.group(1))
    raise ValueError(f"cannot identify round from pprof report name: {path.name}")


def _document_from_pprof_directory(
    path: Path, fallback_label: str
) -> Mapping[str, Any]:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"profile directory lacks manifest.json: {path}")
    manifest = _load_json(manifest_path)
    report_paths = sorted((path / "pprof-rounds").glob("*.txt"))
    if not report_paths:
        report_paths = sorted(path.glob("pprof-*round*.txt"))
    if not report_paths:
        raise ValueError(
            f"profile directory has no canonical JSON or per-round pprof reports: {path}"
        )
    profile_rounds = []
    seen = set()
    for report_path in report_paths:
        round_index = _round_from_report_name(report_path)
        if round_index in seen:
            raise ValueError(f"duplicate pprof report for round {round_index}")
        seen.add(round_index)
        profile_rounds.append(
            {
                "round": round_index,
                "sample_summary": parse_pprof_flat_report(
                    report_path.read_text(encoding="utf-8")
                ),
            }
        )
    expected = {
        row["round"]
        for row in manifest.get("rounds", [])
        if isinstance(row, Mapping) and row.get("mode") == "profile"
    }
    if expected and seen != expected:
        raise ValueError(
            "per-round pprof reports do not match manifest profile rounds: "
            f"expected {sorted(expected)}, got {sorted(seen)}"
        )
    return {
        "host": _first(manifest, ("host", "host_label", "label")) or fallback_label,
        "rounds": manifest.get("rounds", []),
        "profile_rounds": profile_rounds,
    }


def load_host_input(path: Path, fallback_label: str) -> Mapping[str, Any]:
    """Load canonical JSON, or adapt a profiler artifact directory."""

    if path.is_file():
        return _load_json(path)
    if not path.is_dir():
        raise ValueError(f"profile input does not exist: {path}")
    for name in (
        "profile-rounds.json",
        "cross-host-profile.json",
        "cross-host-profile-summary.json",
    ):
        candidate = path / name
        if candidate.is_file():
            return _load_json(candidate)
    return _document_from_pprof_directory(path, fallback_label)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Normalize and compare local/Heroku complete-round CPU profiles"
    )
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--heroku", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_825)
    parser.add_argument(
        "--bootstrap-mode", choices=("independent", "paired"), default="independent"
    )
    parser.add_argument(
        "--bootstrap-unit",
        choices=("round", "abba-block"),
        default=None,
        help=(
            "resampling unit; default uses complete ABBA blocks when both inputs "
            "have round ids 4k+1 and 4k+2, otherwise complete rounds"
        ),
    )
    parser.add_argument(
        "--cpu-calibration",
        choices=("auto", "abba-control", "profile-measured"),
        default="auto",
        help=(
            "CPU total used for hotspot attribution; auto uses surrounding "
            "ABBA controls only when complete on both hosts"
        ),
    )
    args = parser.parse_args()
    if args.bootstrap_replicates <= 0:
        parser.error("--bootstrap-replicates must be positive")
    result = analyze(
        load_host_input(args.local, "local"),
        load_host_input(args.heroku, "heroku"),
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_mode=args.bootstrap_mode,
        bootstrap_unit=args.bootstrap_unit,
        cpu_calibration=args.cpu_calibration,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

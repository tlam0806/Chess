#!/usr/bin/env python3
"""Verify the canonical promoted V43 profile encoded by a Lichess config."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from tools.benchmark_uci_platform import read_config_uci_options
except ModuleNotFoundError:  # Direct execution from a staged tools directory.
    from benchmark_uci_platform import read_config_uci_options


PRODUCTION_CONFIG_HASH = (
    "98b7732c9587da35554cc274a072a0a5b5f55902aaa77605e78c1ae13e88b4f2"
)


def canonical_profile_from_options(options: dict[str, object]) -> dict[str, object]:
    selective = {
        "enable_late_move_pruning": options["LmpEnabled"],
        "enable_lmr": True,
        "enable_main_search_see_pruning": options["MainSeeEnabled"],
        "enable_null_move": True,
        "enable_qsearch_see_pruning": options["QseeEnabled"],
        "enable_reverse_futility": options["RfpEnabled"],
        "late_move_pruning_base": options["LmpBase"],
        "late_move_pruning_depth_multiplier": options["LmpDepthMultiplier"],
        "late_move_pruning_max_depth": options["LmpMaxDepth"],
        "lmr_base": float(options["LmrBase"]),
        "lmr_divisor": float(options["LmrDivisor"]),
        "lmr_min_depth": options["LmrMinDepth"],
        "lmr_min_move_index": options["LmrMinMoveIndex"],
        "main_search_see_margin_per_depth": options["MainSeeMarginPerDepth"],
        "main_search_see_max_depth": options["MainSeeMaxDepth"],
        "null_move_min_depth": options["NullMoveMinDepth"],
        "null_move_reduction": options["NullMoveReduction"],
        "qsearch_see_threshold": options["QseeThreshold"],
        "reverse_futility_base_margin": options["RfpBaseMargin"],
        "reverse_futility_margin_per_depth": options["RfpMarginPerDepth"],
        "reverse_futility_max_depth": options["RfpMaxDepth"],
    }
    aspiration = {
        "delta_base_cp": options["AspirationDeltaBaseCp"],
        "delta_divisor": options["AspirationDeltaDivisor"],
        "enabled": options["AspirationEnabled"],
        "expansion_factor_per_mille": options["AspirationExpansionPermille"],
        "max_fail_high_reductions": options["AspirationMaxFailHighReductions"],
        "max_researches": options["AspirationMaxResearches"],
        "mean_score_clamp_cp": options["AspirationMeanClampCp"],
        "mean_score_new_weight_per_mille": options["AspirationMeanWeightPermille"],
        "min_depth": options["AspirationMinDepth"],
    }
    return {"aspiration": aspiration, "selective": selective}


def profile_hash(profile: dict[str, object]) -> str:
    encoded = json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def config_profile_hash(path: Path) -> str:
    options = read_config_uci_options(path)
    if options.get("TwofoldSearchDraw") is not True:
        raise ValueError("production V43 requires TwofoldSearchDraw=true")
    if options.get("ReuseStaleTtScores") is not False:
        raise ValueError("production V43 requires ReuseStaleTtScores=false")
    if options.get("ReuseDeeperTtScores") is not False:
        raise ValueError("production V43 requires ReuseDeeperTtScores=false")
    return profile_hash(canonical_profile_from_options(options))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--expect", default=PRODUCTION_CONFIG_HASH)
    args = parser.parse_args()
    actual = config_profile_hash(args.config)
    if actual != args.expect:
        raise SystemExit(
            f"V43 production profile mismatch: expected {args.expect}, got {actual}"
        )
    print(actual)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

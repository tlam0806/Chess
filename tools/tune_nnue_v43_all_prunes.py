#!/usr/bin/env python3
"""Final, resumable joint tuner for every selective V43 pruning policy.

This experiment intentionally avoids a Cartesian grid.  It first screens one
logical pruning block at a time, measures the important cross-block
interactions, uses a deterministic maximin global design, and then spends the
expensive depths only on the WDL/node Pareto set.  Two local generations retain
15 percent genuine global restarts so a successful early basin cannot trap the
run.  A final local-closure pass proves that the three preselected profiles have
no untested one-grid-step neighbour before a small aspiration refresh and a
sealed, rejection-only holdout.

The root teacher is immutable V36.  Every candidate from preflight through
pruning selection uses the promoted V43 Balanced aspiration policy identically.
Aspiration is allowed to vary only in the later refresh, after
Fast/Balanced/Safe selective profiles have been selected, and before the
sealed rejection-only holdout.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

if __package__:  # Package import in tests.
    from tools import tune_nnue_v43_aspiration as infra
else:  # Direct execution deliberately prefers the snapshotted sibling.
    import tune_nnue_v43_aspiration as infra  # type: ignore


SCHEMA_VERSION = 2
EXPERIMENT = "v43-final-joint-all-prunes-balanced-baseline-v2"
DATASET_SCHEMA = infra.DATASET_SCHEMA
RANKING_TARGET_ABS_CP = infra.RANKING_TARGET_ABS_CP
WDL_FORMULA = infra.WDL_FORMULA
WDL_CALIBRATION_RUN = infra.WDL_CALIBRATION_RUN
CANDIDATE_ENGINE_PROFILE = infra.CANDIDATE_ENGINE_PROFILE
TWOFOLD_SEARCH_DRAW_ENABLED = True
# The V36 root cache is candidate-independent.  Preserve its original v1
# identity namespace so the Balanced-baseline schema bump does not rebuild
# byte-identical teacher artifacts; exact digest matching below still commits
# to the full teacher profile and rejects unverifiable foreign caches.
CONTROL_CACHE_IDENTITY_EXPERIMENT = "v43-final-joint-all-prunes-v1"
REQUIRED_DATASET_OVERLAP_KEYS = frozenset({
    "games_between_new_splits",
    "position_or_mirror_keys_between_new_splits",
    "games_with_prior_v42_v43",
    "position_or_mirror_keys_with_prior_v42_v43",
})

SELECTIVE_CONFIG_FIELDS = (
    "enable_lmr", "lmr_base", "lmr_divisor", "lmr_min_depth",
    "lmr_min_move_index", "enable_null_move", "null_move_min_depth",
    "null_move_reduction", "enable_reverse_futility",
    "reverse_futility_max_depth", "reverse_futility_base_margin",
    "reverse_futility_margin_per_depth", "enable_late_move_pruning",
    "late_move_pruning_max_depth", "late_move_pruning_base",
    "late_move_pruning_depth_multiplier", "enable_qsearch_see_pruning",
    "qsearch_see_threshold", "enable_main_search_see_pruning",
    "main_search_see_max_depth", "main_search_see_margin_per_depth",
)

ASPIRATION_CONFIG_FIELDS = infra.CONFIG_FIELDS

# Irrelevant children are reset to these values when their block is disabled.
# This is not merely cosmetic: it prevents thousands of hash-distinct policies
# that execute exactly the same search.
DISABLED_CHILD_DEFAULTS: dict[str, int | float] = {
    "lmr_base": 0.45,
    "lmr_divisor": 2.9,
    "lmr_min_depth": 3,
    "lmr_min_move_index": 6,
    "null_move_min_depth": 2,
    "null_move_reduction": 3,
    "reverse_futility_max_depth": 2,
    "reverse_futility_base_margin": 175,
    "reverse_futility_margin_per_depth": 275,
    "late_move_pruning_max_depth": 3,
    "late_move_pruning_base": 4,
    "late_move_pruning_depth_multiplier": 2,
    "qsearch_see_threshold": -75,
    "main_search_see_max_depth": 5,
    "main_search_see_margin_per_depth": 100,
}

BLOCKS: dict[str, tuple[str, ...]] = {
    "lmr": (
        "enable_lmr", "lmr_base", "lmr_divisor", "lmr_min_depth",
        "lmr_min_move_index",
    ),
    "nmp": (
        "enable_null_move", "null_move_min_depth", "null_move_reduction",
    ),
    "rfp": (
        "enable_reverse_futility", "reverse_futility_max_depth",
        "reverse_futility_base_margin",
        "reverse_futility_margin_per_depth",
    ),
    "lmp": (
        "enable_late_move_pruning", "late_move_pruning_max_depth",
        "late_move_pruning_base", "late_move_pruning_depth_multiplier",
    ),
    "qsee": ("enable_qsearch_see_pruning", "qsearch_see_threshold"),
    "main_see": (
        "enable_main_search_see_pruning", "main_search_see_max_depth",
        "main_search_see_margin_per_depth",
    ),
}

ENABLE_FIELD = {
    "lmr": "enable_lmr",
    "nmp": "enable_null_move",
    "rfp": "enable_reverse_futility",
    "lmp": "enable_late_move_pruning",
    "qsee": "enable_qsearch_see_pruning",
    "main_see": "enable_main_search_see_pruning",
}

GRIDS: dict[str, tuple[int | float | bool, ...]] = {
    "enable_lmr": (False, True),
    "lmr_base": tuple(round(value / 100, 2) for value in range(20, 81, 5)),
    "lmr_divisor": tuple(round(value / 100, 2) for value in range(180, 351, 5)),
    "lmr_min_depth": tuple(range(3, 8)),
    "lmr_min_move_index": tuple(range(2, 13)),
    "enable_null_move": (False, True),
    "null_move_min_depth": tuple(range(2, 7)),
    "null_move_reduction": (1, 2, 3, 4),
    "enable_reverse_futility": (False, True),
    "reverse_futility_max_depth": (1, 2, 3, 4),
    "reverse_futility_base_margin": tuple(range(50, 501, 25)),
    "reverse_futility_margin_per_depth": tuple(range(75, 451, 25)),
    "enable_late_move_pruning": (False, True),
    "late_move_pruning_max_depth": (1, 2, 3, 4, 5),
    "late_move_pruning_base": tuple(range(0, 17)),
    "late_move_pruning_depth_multiplier": tuple(range(0, 9)),
    "enable_qsearch_see_pruning": (False, True),
    "qsearch_see_threshold": tuple(range(-200, 51, 25)),
    "enable_main_search_see_pruning": (False, True),
    "main_search_see_max_depth": (1, 2, 3, 4, 5, 6),
    "main_search_see_margin_per_depth": tuple(range(50, 401, 25)),
}


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")


def _nearest(name: str, value: int | float | bool) -> int | float | bool:
    grid = GRIDS[name]
    if isinstance(value, bool):
        return value
    return min(grid, key=lambda item: abs(float(item) - float(value)))


@dataclass(frozen=True)
class SelectiveConfig:
    """All 21 V43 selective-search fields, including conditional switches."""

    enable_lmr: bool = True
    lmr_base: float = 0.45
    lmr_divisor: float = 2.9
    lmr_min_depth: int = 3
    lmr_min_move_index: int = 6
    enable_null_move: bool = True
    null_move_min_depth: int = 2
    null_move_reduction: int = 3
    enable_reverse_futility: bool = True
    reverse_futility_max_depth: int = 2
    reverse_futility_base_margin: int = 175
    reverse_futility_margin_per_depth: int = 275
    enable_late_move_pruning: bool = True
    late_move_pruning_max_depth: int = 3
    late_move_pruning_base: int = 4
    late_move_pruning_depth_multiplier: int = 2
    enable_qsearch_see_pruning: bool = True
    qsearch_see_threshold: int = -75
    enable_main_search_see_pruning: bool = False
    main_search_see_max_depth: int = 5
    main_search_see_margin_per_depth: int = 100

    def __post_init__(self) -> None:
        if tuple(field.name for field in fields(self)) != SELECTIVE_CONFIG_FIELDS:
            raise AssertionError("SelectiveConfig schema drift")
        for name in SELECTIVE_CONFIG_FIELDS:
            value = getattr(self, name)
            if name.startswith("enable_"):
                if not isinstance(value, bool):
                    raise ValueError(f"{name} must be bool")
            elif value not in GRIDS[name]:
                raise ValueError(f"{name}={value!r} is outside its frozen grid")

    def normalized(self) -> "SelectiveConfig":
        values = asdict(self)
        for block, enabled_field in ENABLE_FIELD.items():
            if not bool(values[enabled_field]):
                for name in BLOCKS[block]:
                    if name != enabled_field:
                        values[name] = DISABLED_CHILD_DEFAULTS[name]
        return SelectiveConfig(**values)

    def canonical(self) -> dict[str, bool | int | float]:
        raw = asdict(self.normalized())
        return {name: raw[name] for name in SELECTIVE_CONFIG_FIELDS}

    @property
    def hash(self) -> str:
        return hashlib.sha256(canonical_json(self.canonical())).hexdigest()

    def evaluator_args(self) -> list[str]:
        value = self.normalized()

        def switch(enabled: bool, stem: str) -> str:
            return f"--enable-{stem}" if enabled else f"--disable-{stem}"

        return [
            switch(value.enable_lmr, "lmr"),
            "--lmr-base", str(value.lmr_base),
            "--lmr-divisor", str(value.lmr_divisor),
            "--lmr-min-depth", str(value.lmr_min_depth),
            "--lmr-min-move-index", str(value.lmr_min_move_index),
            switch(value.enable_null_move, "null-move"),
            "--null-min-depth", str(value.null_move_min_depth),
            "--null-reduction", str(value.null_move_reduction),
            switch(value.enable_reverse_futility, "reverse-futility"),
            "--reverse-futility-max-depth",
            str(value.reverse_futility_max_depth),
            "--reverse-futility-base-margin",
            str(value.reverse_futility_base_margin),
            "--reverse-futility-margin-per-depth",
            str(value.reverse_futility_margin_per_depth),
            switch(value.enable_late_move_pruning, "late-move-pruning"),
            "--late-move-pruning-max-depth",
            str(value.late_move_pruning_max_depth),
            "--late-move-pruning-base", str(value.late_move_pruning_base),
            "--late-move-pruning-depth-multiplier",
            str(value.late_move_pruning_depth_multiplier),
            switch(value.enable_qsearch_see_pruning, "qsearch-see-pruning"),
            "--qsearch-see-threshold", str(value.qsearch_see_threshold),
            switch(value.enable_main_search_see_pruning,
                   "main-search-see-pruning"),
            "--main-search-see-max-depth",
            str(value.main_search_see_max_depth),
            "--main-search-see-margin-per-depth",
            str(value.main_search_see_margin_per_depth),
        ]

    @classmethod
    def from_dict(cls, value: dict) -> "SelectiveConfig":
        if tuple(value) != SELECTIVE_CONFIG_FIELDS and set(value) != set(
            SELECTIVE_CONFIG_FIELDS
        ):
            raise ValueError("non-canonical 21-field selective config")
        return cls(**{name: value[name] for name in SELECTIVE_CONFIG_FIELDS})


PRODUCTION_FAST_QSEE = SelectiveConfig()
PRODUCTION_BASELINE_NAME = "v43_production_fast_qsee_balanced_aspiration"


@dataclass(frozen=True)
class AspirationConfig:
    enabled: bool = False
    min_depth: int = 3
    delta_base_cp: int = 30
    delta_divisor: int = 10_000
    expansion_factor_per_mille: int = 1_750
    max_fail_high_reductions: int = 2
    mean_score_new_weight_per_mille: int = 500
    max_researches: int = 6
    mean_score_clamp_cp: int = 1_500

    def canonical(self) -> dict[str, bool | int]:
        raw = asdict(self)
        return {name: raw[name] for name in ASPIRATION_CONFIG_FIELDS}

    def evaluator_args(self) -> list[str]:
        return infra.Config.from_dict(self.canonical()).evaluator_args()

    @property
    def hash(self) -> str:
        return hashlib.sha256(canonical_json(self.canonical())).hexdigest()

    @classmethod
    def from_dict(cls, value: dict) -> "AspirationConfig":
        if set(value) != set(ASPIRATION_CONFIG_FIELDS):
            raise ValueError("non-canonical aspiration config")
        # Reuse the production validator for all numeric ranges.
        infra.Config.from_dict(value)
        return cls(**{name: value[name] for name in ASPIRATION_CONFIG_FIELDS})


LEGACY_FIXED50 = AspirationConfig(enabled=False)
BALANCED_V43_ASPIRATION = AspirationConfig(
    enabled=True,
    min_depth=2,
    delta_base_cp=68,
    delta_divisor=33_700,
    expansion_factor_per_mille=2_290,
    max_fail_high_reductions=1,
    mean_score_new_weight_per_mille=370,
    max_researches=6,
    mean_score_clamp_cp=1_500,
)
PRODUCTION_BASELINE_ASPIRATION = BALANCED_V43_ASPIRATION
ASPIRATION_ANCHORS = (
    LEGACY_FIXED50,
    AspirationConfig(True, 3, 75, 5_400, 2_950, 0, 170),   # Safe.
    BALANCED_V43_ASPIRATION,
    AspirationConfig(True, 4, 10, 28_000, 2_160, 3, 840),  # Fast.
)


@dataclass(frozen=True)
class Candidate:
    selective: SelectiveConfig = PRODUCTION_FAST_QSEE
    aspiration: AspirationConfig = PRODUCTION_BASELINE_ASPIRATION

    def canonical(self) -> dict:
        return {
            "selective": self.selective.canonical(),
            "aspiration": self.aspiration.canonical(),
        }

    @property
    def hash(self) -> str:
        return hashlib.sha256(canonical_json(self.canonical())).hexdigest()

    @property
    def label(self) -> str:
        if (
            self.selective.canonical() == PRODUCTION_FAST_QSEE.canonical()
            and self.aspiration.canonical()
                == PRODUCTION_BASELINE_ASPIRATION.canonical()
        ):
            return f"{PRODUCTION_BASELINE_NAME}-{self.hash[:12]}"
        return f"v43-final-{self.hash[:12]}"

    @classmethod
    def from_dict(cls, value: dict) -> "Candidate":
        if set(value) != {"selective", "aspiration"}:
            raise ValueError("non-canonical candidate")
        return cls(
            SelectiveConfig.from_dict(value["selective"]),
            AspirationConfig.from_dict(value["aspiration"]),
        )


@dataclass(frozen=True)
class Proposal:
    candidate: Candidate
    origin: str


PRUNE_TUNING_STAGES = frozenset({
    "preflight",
    "interactions",
    "global",
    "halving",
    "refinement_1",
    "refinement_2",
    "closure_1",
    "closure_2",
    "deep_tune",
    "selection",
})


def is_prune_tuning_stage(stage: str) -> bool:
    return stage in PRUNE_TUNING_STAGES or stage.startswith("screen_")


def validate_stage_aspiration_policy(
    stage: str,
    proposals: Sequence[Proposal],
) -> None:
    """Fail closed if any pruning candidate escapes the frozen baseline."""
    if not is_prune_tuning_stage(stage):
        return
    expected = PRODUCTION_BASELINE_ASPIRATION.canonical()
    mismatches = [
        proposal.candidate.hash
        for proposal in proposals
        if proposal.candidate.aspiration.canonical() != expected
    ]
    if mismatches:
        raise RuntimeError(
            f"prune-tuning stage {stage} must use the frozen V43 Balanced "
            f"aspiration policy; mismatched candidates: {mismatches[:3]}")


def lmr_behavior_signature(config: SelectiveConfig) -> bytes:
    value = config.normalized()
    signature: list[int] = []
    # C++ uses zero-based searched_move_count.  Cover every legal move index
    # (the theoretical chess maximum is 218 legal moves) and the evaluator's
    # full supported depth domain, so floating parameters that diverge only in
    # a late/wide node are never collapsed.
    for depth in range(3, 65):
        for move_index in range(0, 218):
            if (
                not value.enable_lmr
                or depth < value.lmr_min_depth
                or move_index < value.lmr_min_move_index
            ):
                signature.append(0)
                continue
            reduction = int(
                value.lmr_base
                + math.log(depth) * math.log(move_index + 1)
                / value.lmr_divisor
            )
            signature.append(max(1, min(reduction, depth - 2)))
    # Reduction <=62 over this domain, so bytes is exact and keeps the global
    # behaviour-dedupe pool compact.
    return bytes(signature)


def rfp_behavior_signature(config: SelectiveConfig) -> tuple[int, ...]:
    value = config.normalized()
    return tuple(
        (
            value.reverse_futility_base_margin
            + value.reverse_futility_margin_per_depth * depth
        )
        if value.enable_reverse_futility
        and depth <= value.reverse_futility_max_depth
        else -1
        for depth in range(1, 9)
    )


def lmp_behavior_signature(config: SelectiveConfig) -> tuple[int, ...]:
    value = config.normalized()
    return tuple(
        value.late_move_pruning_base
        + value.late_move_pruning_depth_multiplier * depth * depth
        if value.enable_late_move_pruning
        and depth <= value.late_move_pruning_max_depth
        else -1
        for depth in range(1, 9)
    )


def behavior_signature(config: SelectiveConfig) -> tuple:
    value = config.normalized()
    return (
        lmr_behavior_signature(value),
        (
            value.enable_null_move,
            value.null_move_min_depth if value.enable_null_move else 0,
            value.null_move_reduction if value.enable_null_move else 0,
        ),
        rfp_behavior_signature(value),
        lmp_behavior_signature(value),
        (
            value.enable_qsearch_see_pruning,
            value.qsearch_see_threshold
            if value.enable_qsearch_see_pruning else 0,
        ),
        (
            value.enable_main_search_see_pruning,
            value.main_search_see_max_depth
            if value.enable_main_search_see_pruning else 0,
            value.main_search_see_margin_per_depth
            if value.enable_main_search_see_pruning else 0,
        ),
    )


def candidate_behavior_signature(candidate: Candidate) -> tuple:
    return (
        behavior_signature(candidate.selective),
        tuple(candidate.aspiration.canonical().values()),
    )


def unique_proposals(proposals: Iterable[Proposal]) -> list[Proposal]:
    result: list[Proposal] = []
    seen_hashes: set[str] = set()
    seen_behaviors: set[tuple] = set()
    for proposal in proposals:
        candidate = Candidate(
            proposal.candidate.selective.normalized(),
            proposal.candidate.aspiration,
        )
        signature = candidate_behavior_signature(candidate)
        if candidate.hash in seen_hashes or signature in seen_behaviors:
            continue
        seen_hashes.add(candidate.hash)
        seen_behaviors.add(signature)
        result.append(Proposal(candidate, proposal.origin))
    return result


BLOCK_SCREEN_COUNTS = {
    "lmr": 32,
    "nmp": 17,
    "rfp": 24,
    "lmp": 24,
    "qsee": 9,
    "main_see": 21,
}
INTERACTION_PAIRS = (
    ("lmr", "lmp"),
    ("nmp", "rfp"),
    ("qsee", "main_see"),
)


def _random_grid_value(name: str, rng: random.Random):
    return rng.choice(GRIDS[name])


def random_selective_config(
    rng: random.Random,
    *,
    mutable_blocks: Sequence[str] = tuple(BLOCKS),
    base: SelectiveConfig = PRODUCTION_FAST_QSEE,
) -> SelectiveConfig:
    values = asdict(base.normalized())
    for block in mutable_blocks:
        enabled_field = ENABLE_FIELD[block]
        # Main SEE is the only production-disabled block and therefore needs
        # balanced on/off exploration.  Existing proven blocks are ablated in
        # 10% of global restarts rather than being gratuitously disabled half
        # the time.
        if block == "main_see":
            values[enabled_field] = rng.random() < 0.60
        elif len(mutable_blocks) > 1:
            values[enabled_field] = rng.random() >= 0.10
        else:
            values[enabled_field] = True
        if values[enabled_field]:
            for name in BLOCKS[block]:
                if name != enabled_field:
                    values[name] = _random_grid_value(name, rng)
    return SelectiveConfig(**values).normalized()


def _parameter_vector(config: SelectiveConfig) -> tuple[float, ...]:
    """Normalized design coordinates; disabled child dimensions collapse."""
    value = config.normalized()
    coordinates: list[float] = []
    for block, names in BLOCKS.items():
        enabled = bool(getattr(value, ENABLE_FIELD[block]))
        coordinates.append(1.0 if enabled else 0.0)
        for name in names:
            if name == ENABLE_FIELD[block]:
                continue
            if not enabled:
                coordinates.append(0.0)
                continue
            grid = GRIDS[name]
            index = grid.index(getattr(value, name))
            coordinates.append(index / max(1, len(grid) - 1))
    return tuple(coordinates)


def _distance_squared(left: SelectiveConfig, right: SelectiveConfig) -> float:
    return sum(
        (a - b) ** 2
        for a, b in zip(_parameter_vector(left), _parameter_vector(right))
    )


def deterministic_maximin(
    count: int,
    seed: int,
    sampler: Callable[[random.Random], SelectiveConfig],
    anchors: Sequence[SelectiveConfig] = (),
) -> list[SelectiveConfig]:
    """Greedy maximin design over a deterministic, behaviour-deduped pool."""
    if count < 0:
        raise ValueError("count must be non-negative")
    if count == 0:
        return []
    rng = random.Random(seed)
    pool_target = max(256, count * 16)
    pool: list[SelectiveConfig] = []
    # Raw canonical identity is enough while constructing the design pool.
    # Full engine-behaviour dedupe is applied to the selected proposals; doing
    # a 3k-cell LMR signature for thousands of throwaway pool points would
    # dominate tuner startup.
    seen: set[str] = set()
    for anchor in anchors:
        normalized = anchor.normalized()
        if normalized.hash not in seen:
            seen.add(normalized.hash)
            pool.append(normalized)
    attempts = 0
    while len(pool) < pool_target and attempts < pool_target * 80:
        attempts += 1
        candidate = sampler(rng).normalized()
        if candidate.hash in seen:
            continue
        seen.add(candidate.hash)
        pool.append(candidate)
    if len(pool) < count:
        raise RuntimeError(
            f"only {len(pool)} behaviour-distinct policies for maximin {count}")

    selected = list(pool[:min(len(anchors), count)])
    remaining = [item for item in pool if item not in selected]
    if not selected:
        selected.append(remaining.pop(0))
    vectors = {item.hash: _parameter_vector(item) for item in pool}

    def distance(left: SelectiveConfig, right: SelectiveConfig) -> float:
        return sum(
            (a - b) ** 2
            for a, b in zip(vectors[left.hash], vectors[right.hash]))

    minimum_distance = {
        item.hash: min(distance(item, chosen) for chosen in selected)
        for item in remaining}
    while len(selected) < count:
        # Hash is an explicit deterministic tie breaker, independent of dict
        # or set iteration order.
        chosen = max(
            remaining,
            key=lambda item: (minimum_distance[item.hash], item.hash),
        )
        selected.append(chosen)
        remaining.remove(chosen)
        for item in remaining:
            minimum_distance[item.hash] = min(
                minimum_distance[item.hash],
                distance(item, chosen),
            )
    return selected


def _disable_block(config: SelectiveConfig, block: str) -> SelectiveConfig:
    return replace(config, **{ENABLE_FIELD[block]: False}).normalized()


def all_prunes_disabled() -> SelectiveConfig:
    return SelectiveConfig(**{
        **asdict(PRODUCTION_FAST_QSEE),
        **{enabled: False for enabled in ENABLE_FIELD.values()},
    }).normalized()


def preflight_proposals() -> list[Proposal]:
    values = [
        Proposal(Candidate(PRODUCTION_FAST_QSEE), "anchor:production-fast-qsee"),
        Proposal(Candidate(all_prunes_disabled()), "anchor:all-prunes-off"),
    ]
    for block in BLOCKS:
        values.append(Proposal(
            Candidate(_disable_block(PRODUCTION_FAST_QSEE, block)),
            f"ablation:{block}-off",
        ))
    aggressive = SelectiveConfig(
        lmr_base=0.8, lmr_divisor=1.8, lmr_min_depth=3,
        lmr_min_move_index=2, null_move_min_depth=2,
        null_move_reduction=4, reverse_futility_max_depth=4,
        reverse_futility_base_margin=50,
        reverse_futility_margin_per_depth=75,
        late_move_pruning_max_depth=5, late_move_pruning_base=0,
        late_move_pruning_depth_multiplier=0,
        qsearch_see_threshold=50,
        enable_main_search_see_pruning=True,
        main_search_see_max_depth=6,
        main_search_see_margin_per_depth=50,
    )
    conservative = SelectiveConfig(
        lmr_base=0.2, lmr_divisor=3.5, lmr_min_depth=7,
        lmr_min_move_index=12, null_move_min_depth=6,
        null_move_reduction=1, reverse_futility_max_depth=1,
        reverse_futility_base_margin=500,
        reverse_futility_margin_per_depth=450,
        late_move_pruning_max_depth=1, late_move_pruning_base=16,
        late_move_pruning_depth_multiplier=8,
        qsearch_see_threshold=-200,
        enable_main_search_see_pruning=True,
        main_search_see_max_depth=1,
        main_search_see_margin_per_depth=400,
    )
    values.extend((
        Proposal(Candidate(aggressive), "extreme:aggressive"),
        Proposal(Candidate(conservative), "extreme:conservative"),
    ))
    return unique_proposals(values)


def block_screen_proposals(block: str, count: int, seed: int) -> list[Proposal]:
    if block not in BLOCKS:
        raise ValueError(f"unknown block {block!r}")
    anchors = [
        PRODUCTION_FAST_QSEE,
        _disable_block(PRODUCTION_FAST_QSEE, block),
    ]
    raw_cardinality = 1 + math.prod(
        len(GRIDS[name]) for name in BLOCKS[block]
        if name != ENABLE_FIELD[block])
    configs = deterministic_maximin(
        min(count + 8, raw_cardinality),
        seed,
        lambda rng: random_selective_config(
            rng, mutable_blocks=(block,), base=PRODUCTION_FAST_QSEE),
        anchors,
    )
    proposals = unique_proposals([
        Proposal(Candidate(config), f"screen:{block}:maximin")
        for config in configs
    ])
    if len(proposals) < count:
        raise RuntimeError(f"insufficient distinct block policies for {block}")
    return proposals[:count]


def interaction_proposals(count: int, seed: int) -> list[Proposal]:
    if count < len(INTERACTION_PAIRS):
        raise ValueError("interaction count is smaller than pair count")
    quotient, remainder = divmod(count, len(INTERACTION_PAIRS))
    proposals: list[Proposal] = []
    for index, pair in enumerate(INTERACTION_PAIRS):
        pair_count = quotient + (1 if index < remainder else 0)
        configs = deterministic_maximin(
            pair_count + 8,
            seed ^ (0x1A2B3C + index * 0x1021),
            lambda rng, pair=pair: random_selective_config(
                rng, mutable_blocks=pair, base=PRODUCTION_FAST_QSEE),
            (PRODUCTION_FAST_QSEE,),
        )
        pair_proposals = unique_proposals(
            Proposal(Candidate(config), f"interaction:{pair[0]}x{pair[1]}")
            for config in configs
        )
        if index:
            baseline_hash = Candidate(PRODUCTION_FAST_QSEE).hash
            pair_proposals = [
                item for item in pair_proposals
                if item.candidate.hash != baseline_hash]
        if len(pair_proposals) < pair_count:
            raise RuntimeError(f"insufficient interaction policies for {pair}")
        proposals.extend(pair_proposals[:pair_count])
    result = unique_proposals(proposals)
    if len(result) != count:
        raise RuntimeError("cross-pair interaction design contains duplicates")
    return result


def global_proposals(count: int, seed: int) -> list[Proposal]:
    anchors = [PRODUCTION_FAST_QSEE, all_prunes_disabled()]
    anchors.extend(
        _disable_block(PRODUCTION_FAST_QSEE, block) for block in BLOCKS)
    configs = deterministic_maximin(
        count + 8,
        seed,
        lambda rng: random_selective_config(rng),
        anchors,
    )
    proposals = unique_proposals([
        Proposal(Candidate(config), "global:deterministic-maximin")
        for config in configs
    ])
    if len(proposals) < count:
        raise RuntimeError("insufficient behavior-distinct global policies")
    return proposals[:count]


def config_from_record(record: dict) -> Candidate:
    candidate = Candidate.from_dict(record["config"])
    if candidate.hash != record.get("config_hash"):
        raise RuntimeError("record candidate hash mismatch")
    return candidate


def result_metrics(entry: dict) -> tuple[float, float]:
    result = entry["result"]
    return float(result["mean_wdl_loss"]), float(result["node_ratio"])


def dominates(left: dict, right: dict) -> bool:
    left_loss, left_nodes = result_metrics(left)
    right_loss, right_nodes = result_metrics(right)
    return (
        left_loss <= right_loss
        and left_nodes <= right_nodes
        and (left_loss < right_loss or left_nodes < right_nodes)
    )


def frontier(entries: Sequence[dict]) -> list[dict]:
    return [
        entry for entry in entries
        if not any(
            dominates(other, entry) for other in entries if other is not entry
        )
    ]


def knee(entries: Sequence[dict]) -> dict:
    if not entries:
        raise ValueError("cannot choose knee of empty frontier")
    if len(entries) <= 2:
        return min(entries, key=lambda entry: (
            sum(result_metrics(entry)), entry["config_hash"]))
    losses = [result_metrics(entry)[0] for entry in entries]
    nodes = [result_metrics(entry)[1] for entry in entries]
    loss_min, loss_max = min(losses), max(losses)
    node_min, node_max = min(nodes), max(nodes)

    def key(entry: dict) -> tuple[float, float, float]:
        loss, node = result_metrics(entry)
        normalized_loss = (
            (loss - loss_min) / (loss_max - loss_min)
            if loss_max > loss_min else 0.0)
        normalized_node = (
            (node - node_min) / (node_max - node_min)
            if node_max > node_min else 0.0)
        return (
            1.0 - normalized_loss - normalized_node,
            -normalized_loss,
            -normalized_node,
        )
    return max(entries, key=lambda entry: (*key(entry), entry["config_hash"]))


def _evenly_spaced(entries: Sequence[dict], count: int) -> list[dict]:
    ordered = sorted(entries, key=lambda entry: result_metrics(entry)[1])
    if len(ordered) <= count:
        return ordered
    if count == 1:
        return [knee(ordered)]
    indexes = {
        round(index * (len(ordered) - 1) / (count - 1))
        for index in range(count)
    }
    return [ordered[index] for index in sorted(indexes)]


def successive_halving_survivors(
    entries: Sequence[dict], max_count: int,
) -> list[dict]:
    """Keep Pareto diversity, then fill by deterministic scalar ranks."""
    if max_count <= 0:
        raise ValueError("max_count must be positive")
    pareto = frontier(entries)
    # Never discard an exact Pareto survivor merely to hit a convenient rung
    # size.  ``max_count`` is a fill target, not a frontier cap.
    if len(pareto) >= max_count:
        return list(pareto)
    survivors = list(pareto)
    seen = {entry["config_hash"] for entry in survivors}
    losses = [result_metrics(entry)[0] for entry in entries]
    nodes = [result_metrics(entry)[1] for entry in entries]
    loss_min, loss_max = min(losses), max(losses)
    node_min, node_max = min(nodes), max(nodes)

    def scalar(entry: dict) -> tuple[float, float, str]:
        loss, node = result_metrics(entry)
        normalized_loss = (
            (loss - loss_min) / (loss_max - loss_min)
            if loss_max > loss_min else 0.0)
        normalized_node = (
            (node - node_min) / (node_max - node_min)
            if node_max > node_min else 0.0)
        return normalized_loss + normalized_node, normalized_loss, entry["config_hash"]

    for entry in sorted(entries, key=scalar):
        if entry["config_hash"] not in seen:
            survivors.append(entry)
            seen.add(entry["config_hash"])
        if len(survivors) == max_count:
            break
    return survivors


def _neighbor_values(name: str, value, radius: int = 1) -> list:
    grid = GRIDS[name]
    index = grid.index(value)
    return [
        grid[index + delta]
        for delta in range(-radius, radius + 1)
        if delta and 0 <= index + delta < len(grid)
    ]


def mutate_selective(
    config: SelectiveConfig,
    mode: str,
    rng: random.Random,
) -> SelectiveConfig:
    values = asdict(config.normalized())
    if mode == "single":
        block_names = [rng.choice(tuple(BLOCKS))]
        change_count = 1
    elif mode == "same-block":
        block_names = [rng.choice(tuple(BLOCKS))]
        change_count = 2
    elif mode == "cross-block":
        block_names = rng.sample(tuple(BLOCKS), 2)
        change_count = 2
    else:
        raise ValueError(f"unknown mutation mode {mode!r}")

    parameters: list[str] = []
    for block in block_names:
        enabled_field = ENABLE_FIELD[block]
        if not bool(values[enabled_field]):
            values[enabled_field] = True
        children = [name for name in BLOCKS[block] if name != enabled_field]
        parameters.append(rng.choice(children))
    while len(parameters) < change_count:
        block = block_names[0]
        children = [
            name for name in BLOCKS[block]
            if name != ENABLE_FIELD[block] and name not in parameters
        ]
        if not children:
            break
        parameters.append(rng.choice(children))
    for name in parameters:
        neighbors = _neighbor_values(name, values[name], radius=2)
        if neighbors:
            values[name] = rng.choice(neighbors)
    return SelectiveConfig(**values).normalized()


def crossover_selective(
    left: SelectiveConfig,
    right: SelectiveConfig,
    rng: random.Random,
) -> SelectiveConfig:
    left_values = asdict(left.normalized())
    right_values = asdict(right.normalized())
    inherited = 0
    for block, names in BLOCKS.items():
        if rng.random() < 0.5:
            inherited += 1
            for name in names:
                left_values[name] = right_values[name]
    if inherited in (0, len(BLOCKS)):
        block = rng.choice(tuple(BLOCKS))
        source = right_values if inherited == 0 else asdict(left.normalized())
        for name in BLOCKS[block]:
            left_values[name] = source[name]
    return SelectiveConfig(**left_values).normalized()


def refinement_method_counts(total: int) -> dict[str, int]:
    """Largest-remainder allocation; global restarts are always >=15%."""
    if total < 0:
        raise ValueError("total must be non-negative")
    weights = {
        "single": 0.25,
        "same-block": 0.30,
        "cross-block": 0.20,
        "crossover": 0.10,
        "random-restart": 0.15,
    }
    raw = {name: total * weight for name, weight in weights.items()}
    counts = {name: math.floor(value) for name, value in raw.items()}
    for name in sorted(
        weights,
        key=lambda item: (raw[item] - counts[item], item),
        reverse=True,
    )[:total - sum(counts.values())]:
        counts[name] += 1
    # Floating allocation can round 15% down.  The local-minimum guarantee is
    # a floor, so transfer one slot from the largest local bucket if needed.
    required_restarts = math.ceil(total * 0.15)
    while counts["random-restart"] < required_restarts:
        donor = max(
            (name for name in counts if name != "random-restart"),
            key=lambda name: counts[name],
        )
        counts[donor] -= 1
        counts["random-restart"] += 1
    return counts


def refinement_proposals(
    parent_entries: Sequence[dict],
    count: int,
    seed: int,
    seen_behaviors: set[tuple] | None = None,
) -> list[Proposal]:
    if not parent_entries and count:
        raise ValueError("refinement requires at least one parent")
    rng = random.Random(seed)
    ordered_parents = sorted(parent_entries, key=lambda entry: entry["config_hash"])
    parent_configs = [
        config_from_record(entry).selective
        for entry in sorted(frontier(ordered_parents),
                            key=lambda entry: entry["config_hash"])
    ]
    if not parent_configs:
        parent_configs = [
            config_from_record(entry).selective for entry in ordered_parents]
    methods = refinement_method_counts(count)
    schedule = [
        method for method, method_count in methods.items()
        for _ in range(method_count)
    ]
    rng.shuffle(schedule)
    seen = set(seen_behaviors or ())
    seen.update(
        behavior_signature(config_from_record(entry).selective)
        for entry in parent_entries)
    proposals: list[Proposal] = []
    for method in schedule:
        for _ in range(2_000):
            if method == "random-restart":
                candidate = random_selective_config(rng)
            elif method == "crossover":
                if len(parent_configs) >= 2:
                    left, right = rng.sample(parent_configs, 2)
                    candidate = crossover_selective(left, right, rng)
                else:
                    candidate = random_selective_config(rng)
            else:
                candidate = mutate_selective(
                    rng.choice(parent_configs), method, rng)
            signature = behavior_signature(candidate)
            if method == "crossover" and signature in seen:
                # A narrowed frontier may differ in only one block, making
                # every pure crossover equal a parent.  Preserve the declared
                # crossover allocation but add a local novelty mutation.
                candidate = mutate_selective(candidate, "single", rng)
                signature = behavior_signature(candidate)
            if signature in seen:
                continue
            seen.add(signature)
            proposals.append(Proposal(
                Candidate(candidate, PRODUCTION_BASELINE_ASPIRATION),
                f"refine:{method}",
            ))
            break
        else:
            raise RuntimeError(f"unable to generate unique {method} refinement")
    return proposals


def named_profiles(entries: Sequence[dict]) -> tuple[tuple[str, dict], ...]:
    """Preselect Fast/Balanced/Safe using selection only."""
    if not entries:
        return ()
    full = frontier(entries)
    production_hash = Candidate(
        PRODUCTION_FAST_QSEE, PRODUCTION_BASELINE_ASPIRATION).hash
    challengers = [
        entry for entry in full if entry["config_hash"] != production_hash]
    if not challengers:
        challengers = list(full)
    fastest = min(challengers, key=lambda entry: (
        result_metrics(entry)[1], result_metrics(entry)[0],
        entry["config_hash"],
    ))
    safest = min(challengers, key=lambda entry: (
        result_metrics(entry)[0], result_metrics(entry)[1],
        entry["config_hash"],
    ))
    balanced = knee(challengers)
    return (
        ("fast", fastest),
        ("balanced", balanced),
        ("safe", safest),
    )


def local_closure_proposals(
    profiles: Sequence[tuple[str, dict]],
    seen_behaviors: set[tuple] | None = None,
) -> list[Proposal]:
    """Enumerate every one-grid-step neighbour, including enable toggles."""
    seen = set(seen_behaviors or ())
    proposals: list[Proposal] = []
    for profile_name, entry in profiles:
        parent = config_from_record(entry).selective.normalized()
        parent_values = asdict(parent)
        for block, names in BLOCKS.items():
            enabled_field = ENABLE_FIELD[block]
            toggled = replace(
                parent, **{enabled_field: not getattr(parent, enabled_field)}
            ).normalized()
            signature = behavior_signature(toggled)
            if signature not in seen:
                seen.add(signature)
                proposals.append(Proposal(
                    Candidate(toggled),
                    f"closure:{profile_name}:{enabled_field}:toggle",
                ))
            if not bool(parent_values[enabled_field]):
                continue
            for name in names:
                if name == enabled_field:
                    continue
                for neighbor in _neighbor_values(name, parent_values[name]):
                    child = replace(parent, **{name: neighbor}).normalized()
                    signature = behavior_signature(child)
                    if signature in seen:
                        continue
                    seen.add(signature)
                    proposals.append(Proposal(
                        Candidate(child),
                        f"closure:{profile_name}:{name}",
                    ))
    return proposals


def _mutate_aspiration(
    config: AspirationConfig, rng: random.Random,
) -> AspirationConfig:
    # The refresh is deliberately local and never changes the two guards.
    names = (
        "min_depth", "delta_base_cp", "delta_divisor",
        "expansion_factor_per_mille", "max_fail_high_reductions",
        "mean_score_new_weight_per_mille",
    )
    name = rng.choice(names)
    grids = {
        "min_depth": tuple(range(2, 6)),
        "delta_base_cp": tuple(range(10, 91)),
        "delta_divisor": tuple(range(4_000, 40_001, 100)),
        "expansion_factor_per_mille": tuple(range(1_300, 3_001, 10)),
        "max_fail_high_reductions": tuple(range(4)),
        "mean_score_new_weight_per_mille": tuple(range(0, 1_001, 10)),
    }
    values = asdict(config)
    values["enabled"] = True
    grid = grids[name]
    current = min(range(len(grid)), key=lambda index: abs(
        grid[index] - int(values[name])))
    deltas = [delta for delta in (-2, -1, 1, 2)
              if 0 <= current + delta < len(grid)]
    values[name] = grid[current + rng.choice(deltas)]
    return AspirationConfig(**values)


def aspiration_refresh_proposals(
    profiles: Sequence[tuple[str, dict]],
    per_profile: int,
    seed: int,
) -> list[Proposal]:
    if per_profile < len(ASPIRATION_ANCHORS):
        raise ValueError("per_profile is smaller than aspiration anchor set")
    rng = random.Random(seed)
    proposals: list[Proposal] = []
    for profile_name, entry in profiles:
        selective = config_from_record(entry).selective
        aspirations = list(ASPIRATION_ANCHORS)
        seen = {tuple(item.canonical().values()) for item in aspirations}
        while len(aspirations) < per_profile:
            parent = rng.choice(ASPIRATION_ANCHORS[1:])
            candidate = _mutate_aspiration(parent, rng)
            signature = tuple(candidate.canonical().values())
            if signature in seen:
                continue
            seen.add(signature)
            aspirations.append(candidate)
        proposals.extend(
            Proposal(
                Candidate(selective, aspiration),
                f"aspiration-refresh:{profile_name}",
            )
            for aspiration in aspirations
        )
    return unique_proposals(proposals)


def choose_refreshed_profiles(
    profiles: Sequence[tuple[str, dict]],
    refresh_entries: Sequence[dict],
    seed: int = 0,
    resamples: int = 400,
) -> tuple[tuple[str, dict], ...]:
    """Choose aspiration only on refresh; never inspect sealed holdout."""
    result: list[tuple[str, dict]] = []
    for profile_name, profile_entry in profiles:
        selective_hash = config_from_record(profile_entry).selective.hash
        group = [
            entry for entry in refresh_entries
            if config_from_record(entry).selective.hash == selective_hash
        ]
        if not group:
            raise RuntimeError(f"missing aspiration refresh group {profile_name}")
        candidates = (
            confidence_frontier(
                group, seed ^ int(selective_hash[:8], 16), resamples)
            if all(isinstance(entry.get("detail"), dict) for entry in group)
            else frontier(group)
        )
        result.append((profile_name, knee(candidates)))
    return tuple(result)


ACCEPTED_DEPTH_NONINFERIOR_TOLERANCE = 0.005
PROMOTION_MAX_ACCEPTED_DEPTH_REDUCTION = (
    PRODUCTION_BASELINE_ASPIRATION.max_fail_high_reductions)


def pre_refresh_selection_lineage(selection_entry: dict) -> dict[str, str]:
    if selection_entry.get("stage") != "selection":
        raise RuntimeError("pre-refresh lineage must come from selection")
    candidate = config_from_record(selection_entry)
    if candidate.aspiration != PRODUCTION_BASELINE_ASPIRATION:
        raise RuntimeError(
            "pre-refresh selection lineage did not use Balanced aspiration")
    return {
        "pre_refresh_selection_config_hash": candidate.hash,
        "pre_refresh_selective_hash": candidate.selective.hash,
    }


def accepted_depth_safeguard(
    candidate_entry: dict,
    baseline_entry: dict,
) -> dict:
    """Compare a final candidate with its same-rung production anchor.

    A failed safeguard is a diagnostic promotion veto, not a hard strength
    rejection and never an excuse to inspect another sealed-holdout profile.
    """
    expected_baseline = Candidate(
        PRODUCTION_FAST_QSEE, PRODUCTION_BASELINE_ASPIRATION)
    if config_from_record(baseline_entry).hash != expected_baseline.hash:
        raise RuntimeError("accepted-depth baseline is not production Balanced")
    for key in (
        "stage", "dataset_sha256", "depth", "control_cache_identity_sha256",
    ):
        candidate_value = candidate_entry.get(key)
        baseline_value = baseline_entry.get(key)
        if (
            candidate_value is not None
            and baseline_value is not None
            and candidate_value != baseline_value
        ):
            raise RuntimeError(
                f"accepted-depth comparison is not same-rung at {key}")
    candidate_ratio = float(
        candidate_entry["result"]["aspiration_accepted_depth_ratio"])
    baseline_ratio = float(
        baseline_entry["result"]["aspiration_accepted_depth_ratio"])
    if not (
        math.isfinite(candidate_ratio)
        and math.isfinite(baseline_ratio)
        and 0.0 <= candidate_ratio <= 1.0
        and 0.0 <= baseline_ratio <= 1.0
    ):
        raise RuntimeError("invalid accepted-depth ratio in promotion comparison")
    max_reduction = int(candidate_entry["result"][
        "aspiration_max_accepted_depth_reduction"])
    delta = candidate_ratio - baseline_ratio
    ratio_pass = delta >= -ACCEPTED_DEPTH_NONINFERIOR_TOLERANCE
    reduction_cap_pass = (
        0 <= max_reduction <= PROMOTION_MAX_ACCEPTED_DEPTH_REDUCTION)
    selection_candidate = config_from_record(candidate_entry)
    return {
        "comparison_rung": candidate_entry.get("stage"),
        "selection_candidate_config_hash": selection_candidate.hash,
        "selection_candidate_selective_hash": selection_candidate.selective.hash,
        "same_rung_production_baseline_hash": expected_baseline.hash,
        "same_rung_production_baseline_accepted_depth_ratio": baseline_ratio,
        "candidate_accepted_depth_ratio": candidate_ratio,
        "accepted_depth_ratio_delta_vs_production_baseline": delta,
        "noninferiority_tolerance": ACCEPTED_DEPTH_NONINFERIOR_TOLERANCE,
        "accepted_depth_ratio_noninferior": ratio_pass,
        "max_accepted_depth_reduction": max_reduction,
        "max_accepted_depth_reduction_cap": (
            PROMOTION_MAX_ACCEPTED_DEPTH_REDUCTION),
        "max_accepted_depth_reduction_pass": reduction_cap_pass,
        "pass": ratio_pass and reduction_cap_pass,
    }


def gauntlet_selective_profile(name: str, config: SelectiveConfig) -> str:
    """Serialize the gauntlet's name-plus-all-21-fields profile form."""
    value = config.normalized()

    def flag(enabled: bool) -> str:
        return "1" if enabled else "0"

    fields = (
        name,
        str(value.lmr_base),
        str(value.lmr_divisor),
        str(value.lmr_min_depth),
        str(value.lmr_min_move_index),
        str(value.null_move_min_depth),
        str(value.null_move_reduction),
        flag(value.enable_reverse_futility),
        str(value.reverse_futility_max_depth),
        str(value.reverse_futility_base_margin),
        str(value.reverse_futility_margin_per_depth),
        flag(value.enable_late_move_pruning),
        str(value.late_move_pruning_max_depth),
        str(value.late_move_pruning_base),
        str(value.late_move_pruning_depth_multiplier),
        flag(value.enable_qsearch_see_pruning),
        str(value.qsearch_see_threshold),
        flag(value.enable_main_search_see_pruning),
        str(value.main_search_see_max_depth),
        str(value.main_search_see_margin_per_depth),
        flag(value.enable_lmr),
        flag(value.enable_null_move),
    )
    return ",".join(fields)


def gauntlet_aspiration_profile(name: str, config: AspirationConfig) -> str:
    values = config.canonical()
    fields = (
        name,
        *("1" if values[field] is True else
          "0" if values[field] is False else str(values[field])
          for field in ASPIRATION_CONFIG_FIELDS),
    )
    return ",".join(fields)


def selfplay_confirmation_args(
    profile_name: str,
    candidate_entry: dict,
    baseline_entry: dict,
    eligible: bool,
) -> dict:
    candidate = config_from_record(candidate_entry)
    baseline = config_from_record(baseline_entry)
    expected_baseline = Candidate(
        PRODUCTION_FAST_QSEE, PRODUCTION_BASELINE_ASPIRATION)
    if baseline.hash != expected_baseline.hash:
        raise RuntimeError("selfplay control is not production Balanced")
    candidate_name = f"v43_final_{profile_name}_{candidate.hash[:10]}"
    control_name = "v43_balanced_aspiration_baseline_control"
    arguments = [
        "--profile", gauntlet_selective_profile(
            candidate_name, candidate.selective),
        "--profile", gauntlet_selective_profile(
            control_name, baseline.selective),
        "--aspiration-profile", gauntlet_aspiration_profile(
            candidate_name, candidate.aspiration),
        "--aspiration-profile", gauntlet_aspiration_profile(
            control_name, baseline.aspiration),
        "--twofold-search-profile", candidate_name,
        "--twofold-search-profile", control_name,
        "--tt-mb", str(CANDIDATE_ENGINE_PROFILE["tt_mb"]),
    ]
    return {
        "candidate_name": candidate_name,
        "control_name": control_name,
        "production_baseline_name": PRODUCTION_BASELINE_NAME,
        "candidate_config_hash": candidate.hash,
        "control_config_hash": baseline.hash,
        "suggested_output_filename": (
            f"games_{profile_name}_{candidate.hash[:12]}_vs_"
            f"{baseline.hash[:12]}.jsonl"),
        "arguments": arguments,
        "twofold_search_draw_enabled_for_both": True,
        "reuse_stale_tt_scores_for_both": False,
        "reuse_deeper_tt_scores_for_both": False,
        "tt_depth_policy_for_both": "Exact",
        "tt_generation_policy_for_both": "CurrentOnly",
        "eligible_for_equal_time_selfplay": eligible,
    }


Budget = infra.Budget
EvaluationFailure = infra.EvaluationFailure
EvaluationBudgetExhausted = infra.EvaluationBudgetExhausted
CacheBudgetExhausted = infra.CacheBudgetExhausted
atomic_write_json = infra.atomic_write_json
append_jsonl = infra.append_jsonl
load_jsonl = infra.load_jsonl
file_fingerprint = infra.file_fingerprint
sha256_file = infra.sha256_file


def _content_identity(path: Path) -> dict[str, int | str]:
    fingerprint = file_fingerprint(path)
    return {"size": fingerprint["size"], "sha256": fingerprint["sha256"]}


def control_cache_spec(
    run_dir: Path,
    cache_key: str,
    binary: Path,
    model: Path,
    dataset: Path,
    depth: int,
) -> dict:
    dataset_schema = infra.validate_dataset_schema(dataset)
    identity = {
        "schema": infra.CONTROL_CACHE_SCHEMA,
        "experiment": CONTROL_CACHE_IDENTITY_EXPERIMENT,
        "candidate_independent_v36_root_teacher": True,
        "evaluator": _content_identity(binary),
        "model": _content_identity(model),
        "ordered_dataset": _content_identity(dataset),
        "dataset_schema": DATASET_SCHEMA,
        "depth": depth,
        "offset": 0,
        "count": dataset_schema["count"],
        "teacher_profile": infra.CONTROL_TEACHER_PROFILE,
    }
    return {
        "cache_key": cache_key,
        "path": str((run_dir / f"control-root-{cache_key}-d{depth}.tsv").resolve()),
        "identity": identity,
        "identity_sha256": hashlib.sha256(canonical_json(identity)).hexdigest(),
    }


def _cache_header(path: Path) -> tuple[dict[str, str], list[str]]:
    lines = path.read_text().splitlines()
    if len(lines) < 9 or lines[0] != infra.CONTROL_CACHE_SCHEMA:
        raise RuntimeError(f"truncated or wrong-schema control cache: {path}")
    headers: dict[str, str] = {}
    for line in lines[1:8]:
        parts = line.split("\t", 1)
        if len(parts) != 2:
            raise RuntimeError(f"bad control cache header: {line!r}")
        headers[parts[0]] = parts[1]
    if lines[8].split("\t") != list(infra.CONTROL_CACHE_COLUMNS):
        raise RuntimeError("control cache column schema mismatch")
    return headers, lines[9:]


def validate_control_cache(path: Path, spec: dict, dataset: Path) -> dict:
    headers, rows = _cache_header(path)
    identity = spec["identity"]
    expected = {
        "identity_sha256": spec["identity_sha256"],
        "evaluator_sha256": str(identity["evaluator"]["sha256"]),
        "model_sha256": str(identity["model"]["sha256"]),
        "dataset_sha256": str(identity["ordered_dataset"]["sha256"]),
        "depth": str(identity["depth"]),
        "offset": str(identity["offset"]),
        "count": str(identity["count"]),
    }
    if headers != expected:
        raise RuntimeError(f"control cache identity header mismatch: {path}")
    if _content_identity(dataset) != identity["ordered_dataset"]:
        raise RuntimeError("control cache dataset changed")
    if len(rows) != int(identity["count"]):
        raise RuntimeError("control cache row count mismatch")
    dataset_rows = dataset.read_text().splitlines()
    if len(dataset_rows) != len(rows):
        raise RuntimeError("control cache and dataset count differ")
    control_nodes = fallbacks = conflicts = unresolved = 0
    for index, row in enumerate(rows):
        columns = row.split("\t")
        if len(columns) != len(infra.CONTROL_CACHE_COLUMNS):
            raise RuntimeError(f"bad control cache row {index}")
        values = dict(zip(infra.CONTROL_CACHE_COLUMNS, columns))
        _, expected_hash, expected_ply, _ = infra.parse_dataset_row(
            dataset_rows[index], dataset, index + 1)
        if (
            int(values["index"]) != index
            or values["sample_hash"] != expected_hash
            or int(values["calibration_ply"]) != expected_ply
            or int(values["result_depth"]) != int(identity["depth"])
            or int(values["stopped"]) != 0
        ):
            raise RuntimeError(f"control cache row identity mismatch at {index}")
        control_nodes += int(values["nodes"])
        fallbacks += int(values["full_window_fallbacks"])
        conflicts += int(values["range_conflicts"])
        unresolved += int(values["unresolved_ranges"])
    if control_nodes <= 0 or min(fallbacks, conflicts, unresolved) < 0:
        raise RuntimeError("invalid control cache counters")
    if unresolved:
        raise RuntimeError("control cache contains unresolved V36 root")
    return {
        "count": len(rows),
        "depth": int(identity["depth"]),
        "control_nodes": control_nodes,
        "control_exact_searches": len(rows),
        "control_exact_full_window_fallbacks": fallbacks,
        "control_exact_range_conflicts": conflicts,
        "control_exact_unresolved_ranges": unresolved,
    }


def _control_cache_args(spec: dict, *, write: bool = False) -> list[str]:
    identity = spec["identity"]
    return [
        "--write-control-cache" if write else "--control-cache",
        str(spec["path"]),
        "--control-cache-identity-sha256", spec["identity_sha256"],
        "--control-cache-evaluator-sha256",
        str(identity["evaluator"]["sha256"]),
        "--control-cache-model-sha256", str(identity["model"]["sha256"]),
        "--control-cache-dataset-sha256",
        str(identity["ordered_dataset"]["sha256"]),
    ]


def _external_cache_candidate(
    directory: Path,
    spec: dict,
) -> Path | None:
    if not directory.exists():
        return None
    identity = spec["identity"]
    expected_headers = {
        "identity_sha256": spec["identity_sha256"],
        "evaluator_sha256": str(identity["evaluator"]["sha256"]),
        "model_sha256": str(identity["model"]["sha256"]),
        "dataset_sha256": str(identity["ordered_dataset"]["sha256"]),
        "depth": str(identity["depth"]),
        "offset": str(identity["offset"]),
        "count": str(identity["count"]),
    }
    for path in sorted(directory.glob("control-root-*.tsv")):
        try:
            headers, rows = _cache_header(path)
        except RuntimeError:
            continue
        # The identity digest commits to every V36 teacher semantic, not only
        # the convenient content/depth headers.  Never adopt a foreign digest:
        # doing so would make teacher-profile drift unverifiable.
        if headers == expected_headers and len(rows) == int(identity["count"]):
            return path
    return None


def ensure_control_cache(
    records: list[dict],
    log_path: Path,
    spec: dict,
    binary: Path,
    model: Path,
    dataset: Path,
    budget: Budget,
    existing_cache_dir: Path | None = None,
) -> dict:
    key = spec["cache_key"]
    ready = [
        record for record in records
        if record.get("kind") == "control_cache_ready"
        and record.get("cache_key") == key
    ]
    if len(ready) > 1:
        raise RuntimeError(f"multiple cache descriptors for {key}")
    if ready:
        descriptor = ready[0]["control_cache"]
        if file_fingerprint(Path(descriptor["path"])) != descriptor["artifact"]:
            raise RuntimeError(f"immutable cache changed for {key}")
        summary = validate_control_cache(Path(descriptor["path"]), descriptor, dataset)
        if summary != descriptor["summary"]:
            raise RuntimeError(f"control cache summary changed for {key}")
        return descriptor
    if budget.exhausted:
        raise CacheBudgetExhausted(f"budget exhausted before cache {key}")

    path = Path(spec["path"])
    external = (
        _external_cache_candidate(existing_cache_dir, spec)
        if existing_cache_dir is not None else None)
    if external is not None and not path.exists():
        # Hard-linking preserves bytes and avoids duplicating multi-GB caches.
        # If filesystems differ, fall back to a streaming copy via copyfile.
        try:
            os.link(external, path)
        except OSError:
            import shutil
            shutil.copyfile(external, path)
    elif path.exists():
        # Recover the narrow crash window after an external cache was linked
        # but before its ready record was appended.  Validation below requires
        # the exact semantic digest, including the frozen teacher profile.
        validate_control_cache(path, spec, dataset)
    if not path.exists():
        command = [
            str(binary), "--dataset", str(dataset), "--model", str(model),
            "--depth", str(spec["identity"]["depth"]),
            *_control_cache_args(spec, write=True),
            *infra.candidate_engine_evaluator_args(),
            *PRODUCTION_FAST_QSEE.evaluator_args(),
            *PRODUCTION_BASELINE_ASPIRATION.evaluator_args(),
            "--enable-twofold-search-draw",
        ]
        try:
            completed = subprocess.run(
                command, text=True, capture_output=True,
                timeout=max(0.001, budget.remaining_sec),
            )
        except subprocess.TimeoutExpired as error:
            raise CacheBudgetExhausted(f"budget exhausted building {key}") from error
        if completed.returncode:
            raise RuntimeError(
                f"cache build failed for {key}: {completed.stderr[-2000:]}")
    summary = validate_control_cache(path, spec, dataset)
    descriptor = {
        **spec,
        "artifact": file_fingerprint(path),
        "summary": summary,
    }
    record = {
        "kind": "control_cache_ready",
        "schema_version": SCHEMA_VERSION,
        "cache_key": key,
        "control_cache": descriptor,
        "budget_elapsed_sec": budget.elapsed_sec,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    append_jsonl(log_path, record)
    records.append(record)
    return descriptor


def validate_aspiration_telemetry(
    result: dict,
    aspiration: AspirationConfig,
    count: int,
    depth: int,
) -> None:
    accepted = float(result["aspiration_accepted_depth_ratio"])
    if not aspiration.enabled:
        nonzero = {
            name: result[name] for name in infra.TELEMETRY_FIELDS
            if int(result[name]) != 0
        }
        if nonzero or accepted != 1.0:
            raise RuntimeError(
                "disabled aspiration emitted adaptive telemetry: "
                f"{sorted(nonzero)}")
        return

    completed_iterations = int(result["aspiration_completed_iterations"])
    final_mean_score_count = int(result["aspiration_final_mean_score_count"])
    narrow_iterations = int(result["aspiration_narrow_iterations"])
    narrow_attempts = int(result["aspiration_narrow_attempts"])
    initial_successes = int(result["aspiration_initial_window_successes"])
    accepted_reduced = int(
        result["aspiration_accepted_reduced_depth_iterations"])
    nominal_depth_sum = int(
        result["aspiration_accepted_narrow_nominal_depth_sum"])
    search_depth_sum = int(
        result["aspiration_accepted_narrow_search_depth_sum"])
    max_reduction = int(result["aspiration_max_accepted_depth_reduction"])
    if completed_iterations != count * depth:
        raise RuntimeError("adaptive aspiration iteration count mismatch")
    if final_mean_score_count != count:
        raise RuntimeError("adaptive aspiration mean-score count mismatch")
    if not math.isfinite(accepted) or not 0.0 <= accepted <= 1.0:
        raise RuntimeError("invalid adaptive accepted-depth ratio")
    if narrow_attempts < narrow_iterations:
        raise RuntimeError("aspiration attempts are fewer than narrow iterations")
    if initial_successes > narrow_iterations:
        raise RuntimeError(
            "aspiration initial successes exceed narrow iterations")
    if accepted_reduced > narrow_iterations:
        raise RuntimeError(
            "accepted reduced-depth count exceeds narrow iterations")
    if search_depth_sum > nominal_depth_sum:
        raise RuntimeError(
            "accepted aspiration search depth exceeds nominal depth")
    expected_accepted = (
        search_depth_sum / nominal_depth_sum
        if nominal_depth_sum > 0 else 1.0)
    if not math.isclose(
        accepted, expected_accepted, rel_tol=1e-12, abs_tol=1e-15,
    ):
        raise RuntimeError(
            "aspiration accepted-depth ratio disagrees with depth sums")
    if not 0 <= max_reduction <= aspiration.max_fail_high_reductions:
        raise RuntimeError(
            "accepted aspiration depth reduction exceeds configured cap")
    if depth >= aspiration.min_depth and narrow_iterations <= 0:
        raise RuntimeError("enabled aspiration never opened a narrow window")
    if depth >= aspiration.min_depth and narrow_attempts <= 0:
        raise RuntimeError("enabled aspiration never attempted narrow search")


def validate_result(
    result: dict,
    candidate: Candidate,
    depth: int,
    control_cache: dict,
) -> None:
    required = {
        "count", "ranking_count", "ranking_target_abs_cp",
        "include_all_in_objective", "objective", "wdl_formula",
        "wdl_calibration_run", "mean_wdl_loss", "p95_wdl_loss",
        "objective_loss", "node_ratio", "control_nodes", "candidate_nodes",
        "critical_mistakes", "candidate_search_mode", "candidate_time_ms",
        "candidate_mean_depth", "candidate_completed_depth_count",
        "candidate_stopped_pct", "dataset_ply_schema",
        "dataset_explicit_ply_count",
        "dataset_explicit_ply_position_match_count", "selective_config",
        "aspiration_config", "twofold_search_draw_enabled",
        "control_root_source", "control_time_source", "time_ratio",
        "control_cache_identity_sha256", "control_live_root_searches",
        "control_cached_root_results", "strict_best_score_violations",
        "strict_best_score_max_excess_cp",
        "candidate_repetition_history_aware_searches",
        "aspiration_accepted_depth_ratio", *CANDIDATE_ENGINE_PROFILE,
        *infra.TELEMETRY_FIELDS,
    }
    missing = sorted(required - set(result))
    if missing:
        raise RuntimeError(f"evaluator result missing fields: {missing}")
    count = int(result["count"])
    if count <= 0 or int(result["ranking_count"]) != count:
        raise RuntimeError("evaluator excluded an all-position objective row")
    if (
        int(result["ranking_target_abs_cp"]) != RANKING_TARGET_ABS_CP
        or result["include_all_in_objective"] is not True
        or result["objective"] != "wdl"
        or result["wdl_formula"] != WDL_FORMULA
        or result["wdl_calibration_run"] != WDL_CALIBRATION_RUN
    ):
        raise RuntimeError("evaluator objective/calibration profile drift")
    if (
        result["dataset_ply_schema"] != DATASET_SCHEMA
        or int(result["dataset_explicit_ply_count"]) != count
        or int(result["dataset_explicit_ply_position_match_count"]) != count
    ):
        raise RuntimeError("evaluator did not consume explicit dataset ply")
    if (
        result["candidate_search_mode"] != "iterative_depth"
        or int(result["candidate_time_ms"]) != 0
        or int(result["candidate_completed_depth_count"]) != count
        or float(result["candidate_stopped_pct"]) != 0.0
        or not math.isclose(float(result["candidate_mean_depth"]), depth)
    ):
        raise RuntimeError("candidate did not finish requested fixed depth")
    for name, expected in CANDIDATE_ENGINE_PROFILE.items():
        if result[name] != expected:
            raise RuntimeError(f"V43 engine profile drift at {name}")
    if result["selective_config"] != candidate.selective.canonical():
        raise RuntimeError("evaluator selective config differs from request")
    if result["aspiration_config"] != candidate.aspiration.canonical():
        raise RuntimeError("evaluator aspiration config differs from request")
    if result["twofold_search_draw_enabled"] is not True:
        raise RuntimeError("twofold search-draw policy drift")
    if (
        result["control_root_source"] != "immutable_cache"
        or result["control_time_source"] != "not_measured_cached"
        or result["time_ratio"] is not None
        or result["control_cache_identity_sha256"]
            != control_cache["identity_sha256"]
        or int(result["control_live_root_searches"]) != 0
        or int(result["control_cached_root_results"]) != count
    ):
        raise RuntimeError("evaluator bypassed or mismatched immutable cache")
    if (
        int(result["strict_best_score_violations"]) != 0
        or int(result["strict_best_score_max_excess_cp"]) != 0
    ):
        raise RuntimeError("strict child score exceeded strict root best")
    if int(result["candidate_repetition_history_aware_searches"]) != count:
        raise RuntimeError("candidate bypassed repetition-aware search")
    metrics = (
        "mean_wdl_loss", "p95_wdl_loss", "objective_loss", "node_ratio")
    if any(not math.isfinite(float(result[name])) for name in metrics):
        raise RuntimeError("non-finite objective metric")
    if not math.isclose(
        float(result["objective_loss"]), float(result["mean_wdl_loss"]),
        rel_tol=2e-6, abs_tol=1e-12,
    ):
        raise RuntimeError("objective_loss differs from mean_wdl_loss")
    control_nodes = int(result["control_nodes"])
    candidate_nodes = int(result["candidate_nodes"])
    if min(control_nodes, candidate_nodes) <= 0 or not math.isclose(
        float(result["node_ratio"]), candidate_nodes / control_nodes,
        rel_tol=1e-12, abs_tol=1e-15,
    ):
        raise RuntimeError("node ratio disagrees with exact totals")
    summary = control_cache["summary"]
    if (
        control_nodes != int(summary["control_nodes"])
        or count != int(summary["count"])
        or depth != int(summary["depth"])
    ):
        raise RuntimeError("evaluator cache totals differ from descriptor")
    if (
        int(result.get("aspiration_range_conflict_fallbacks", 0)) != 0
        or int(result.get("aspiration_unresolved_ranges", 0)) != 0
    ):
        raise RuntimeError("V43 scalar search produced unresolved/range conflict")

    validate_aspiration_telemetry(
        result, candidate.aspiration, count=count, depth=depth)


def _read_detail_rows(path: Path) -> tuple[dict, ...]:
    rows = infra.load_jsonl(path)
    required = {
        "schema", "index", "global_index", "category", "sample_hash",
        "calibration_ply", "phase", "static_target_cp", "ranking_sample",
        "safety_sample", "control_move", "candidate_move", "control_score",
        "candidate_strict_score", "control_root_nodes", "candidate_nodes",
        "candidate_depth", "candidate_stopped", "move_match", "cp_regret",
        "wdl_loss", "win_to_draw", "win_to_loss", "self_mated",
        "critical_mistake",
    }
    for index, row in enumerate(rows):
        if row.get("schema") != "nnue_selective_position_details_jsonl_v1":
            raise RuntimeError(f"detail schema mismatch at row {index}")
        missing = required - set(row)
        extra = set(row) - required
        if missing or extra:
            raise RuntimeError(
                f"detail row fields differ at {index}: "
                f"missing={sorted(missing)} extra={sorted(extra)}")
    return tuple(rows)


def _load_detail_rows(path: Path) -> tuple[dict, ...]:
    return _read_detail_rows(path)


def _compact_detail_metrics(detail: dict):
    artifact = detail["artifact"]
    path = Path(artifact["path"])
    if sha256_file(path) != artifact["sha256"]:
        raise RuntimeError("immutable detail sidecar digest changed")
    rows = _read_detail_rows(path)
    try:
        import numpy as np
    except ModuleNotFoundError as error:
        raise RuntimeError("paired bootstrap requires numpy") from error
    order = hashlib.sha256()
    for row in rows:
        order.update(str(row["sample_hash"]).encode("ascii"))
        order.update(b"\0")
    return (
        order.hexdigest(),
        np.asarray([float(row["wdl_loss"]) for row in rows], dtype=np.float64),
        np.asarray([int(row["candidate_nodes"]) for row in rows], dtype=np.int64),
    )


def validate_detail_sidecar(path: Path, dataset: Path, expected_count: int) -> dict:
    rows = _load_detail_rows(path)
    dataset_rows = dataset.read_text().splitlines()
    if len(rows) != expected_count or len(dataset_rows) != expected_count:
        raise RuntimeError("detail sidecar count mismatch")
    for index, (row, line) in enumerate(zip(rows, dataset_rows)):
        _, sample_hash, _, _ = infra.parse_dataset_row(line, dataset, index + 1)
        if int(row.get("index", -1)) != index:
            raise RuntimeError("detail index mismatch")
        if row["sample_hash"] != sample_hash:
            raise RuntimeError("detail sample hash/order mismatch")
        if (
            not math.isfinite(float(row["wdl_loss"]))
            or int(row["control_root_nodes"]) <= 0
            or int(row["candidate_nodes"]) <= 0
        ):
            raise RuntimeError("invalid detail metric")
    return {
        "schema": "nnue_selective_position_details_jsonl_v1",
        "count": len(rows),
        "artifact": file_fingerprint(path),
    }


def evaluate(
    binary: Path,
    dataset: Path,
    model: Path,
    depth: int,
    candidate: Candidate,
    control_cache: dict,
    detail_path: Path | None,
    deadline: float | None,
) -> tuple[dict, dict | None]:
    command = [
        str(binary), "--dataset", str(dataset), "--model", str(model),
        "--depth", str(depth), "--objective", "wdl",
        "--include-all-in-objective",
        "--ranking-target-abs-cp", str(RANKING_TARGET_ABS_CP),
        *_control_cache_args(control_cache),
        *infra.candidate_engine_evaluator_args(),
        *candidate.selective.evaluator_args(),
        *candidate.aspiration.evaluator_args(),
        "--enable-twofold-search-draw",
    ]
    if detail_path is not None:
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        command.extend(("--write-detail-jsonl", str(detail_path)))
    try:
        completed = subprocess.run(
            command, text=True, capture_output=True,
            timeout=(
                max(0.001, deadline - time.monotonic())
                if deadline is not None else None),
        )
    except subprocess.TimeoutExpired as error:
        if deadline is not None:
            raise EvaluationBudgetExhausted("shared stage deadline reached") from error
        raise EvaluationFailure("evaluator timed out", command=command) from error
    except OSError as error:
        raise EvaluationFailure(
            f"failed to launch evaluator: {error}", command=command) from error
    if completed.returncode:
        raise EvaluationFailure(
            "evaluation failed", command=command,
            returncode=completed.returncode,
            stdout=completed.stdout, stderr=completed.stderr)
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise EvaluationFailure(
            "evaluator produced invalid JSON", command=command,
            stdout=completed.stdout, stderr=completed.stderr) from error
    try:
        validate_result(result, candidate, depth, control_cache)
        detail = (
            validate_detail_sidecar(detail_path, dataset, int(result["count"]))
            if detail_path is not None else None)
        if detail is not None:
            if (
                result.get("detail_jsonl_schema") != detail["schema"]
                or int(result.get("detail_jsonl_count", -1)) != detail["count"]
                or result.get("detail_jsonl_written") is not True
            ):
                raise RuntimeError("aggregate detail sidecar echo mismatch")
    except (RuntimeError, TypeError, ValueError, KeyError) as error:
        raise EvaluationFailure(
            f"evaluator result validation failed: {error}", command=command,
            stdout=completed.stdout, stderr=completed.stderr) from error
    return result, detail


def _paired_bootstrap_dominates(
    left: dict,
    right: dict,
    *,
    seed: int,
    resamples: int = 400,
    confidence: float = 0.95,
    compact: dict[str, tuple] | None = None,
) -> bool:
    """Whether left dominates right in >=confidence paired resamples."""
    left_detail = left.get("detail")
    right_detail = right.get("detail")
    if not isinstance(left_detail, dict) or not isinstance(right_detail, dict):
        return dominates(left, right)
    compact = compact if compact is not None else {}
    if left["config_hash"] not in compact:
        compact[left["config_hash"]] = _compact_detail_metrics(left_detail)
    if right["config_hash"] not in compact:
        compact[right["config_hash"]] = _compact_detail_metrics(right_detail)
    left_metrics = compact[left["config_hash"]]
    right_metrics = compact[right["config_hash"]]
    left_order, left_loss, left_nodes = left_metrics
    right_order, right_loss, right_nodes = right_metrics
    if len(left_loss) != len(right_loss) or len(left_loss) == 0:
        raise RuntimeError("paired bootstrap sidecars differ in length")
    if left_order != right_order:
        raise RuntimeError("paired bootstrap sidecars differ in row order")
    try:
        import numpy as np
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "paired bootstrap requires numpy; runner preflight must install it"
        ) from error
    loss_array = left_loss - right_loss
    node_array = left_nodes - right_nodes
    rng = np.random.default_rng(seed)
    successes = 0
    count = len(loss_array)
    # Batch the index matrix so even an 8k-row selection stays under 4 MiB
    # per 64-resample batch while removing Python's inner row loop.
    for offset in range(0, resamples, 64):
        batch = min(64, resamples - offset)
        indexes = rng.integers(0, count, size=(batch, count), dtype=np.int32)
        loss_sums = loss_array[indexes].sum(axis=1)
        node_sums = node_array[indexes].sum(axis=1)
        success = (
            (loss_sums <= 0.0)
            & (node_sums <= 0)
            & ((loss_sums < 0.0) | (node_sums < 0)))
        successes += int(success.sum())
        failures = offset + batch - successes
        allowed_failures = math.floor(
            (1.0 - confidence) * resamples + 1e-12)
        if failures > allowed_failures:
            return False
    return successes >= math.ceil(confidence * resamples)


def confidence_frontier(
    entries: Sequence[dict], seed: int, resamples: int = 400,
) -> list[dict]:
    """Point candidates are removed only by 95% paired dominance."""
    result: list[dict] = []
    compact: dict[str, tuple] = {}
    for right in entries:
        rejected = False
        for left in entries:
            if left is right or not dominates(left, right):
                continue
            pair_seed = seed ^ int(left["config_hash"][:8], 16) \
                ^ int(right["config_hash"][:8], 16)
            if _paired_bootstrap_dominates(
                left, right, seed=pair_seed, resamples=resamples,
                compact=compact):
                rejected = True
                break
        if not rejected:
            result.append(right)
    return result


def proposal_payload(proposal: Proposal) -> dict:
    return {
        "config": proposal.candidate.canonical(),
        "config_hash": proposal.candidate.hash,
        "label": proposal.candidate.label,
        "origin": proposal.origin,
    }


def ensure_plan(
    records: list[dict],
    log_path: Path,
    stage: str,
    proposals: Sequence[Proposal],
    dataset: Path,
    depth: int,
    control_cache: dict,
    budget: Budget,
    write_details: bool,
) -> dict:
    candidates = unique_proposals(proposals)
    validate_stage_aspiration_policy(stage, candidates)
    expected = {
        "kind": "plan",
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "dataset": file_fingerprint(dataset),
        "depth": depth,
        "candidate_time_ms": 0,
        "write_detail_jsonl": write_details,
        "control_cache": control_cache,
        "candidates": [proposal_payload(item) for item in candidates],
    }
    matches = [
        record for record in records
        if record.get("kind") == "plan" and record.get("stage") == stage]
    if len(matches) > 1:
        raise RuntimeError(f"multiple saved plans for {stage}")
    if matches:
        actual = {key: matches[0].get(key) for key in expected}
        if canonical_json(actual) != canonical_json(expected):
            raise RuntimeError(f"saved plan mismatch for {stage}")
        return matches[0]
    record = {**expected, "budget_elapsed_sec": budget.elapsed_sec}
    append_jsonl(log_path, record)
    records.append(record)
    return record


def completed_results(records: Sequence[dict], stage: str) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for record in records:
        if record.get("kind") != "result" or record.get("stage") != stage:
            continue
        candidate = config_from_record(record)
        previous = result.get(candidate.hash)
        if previous is not None and canonical_json(previous) != canonical_json(record):
            raise RuntimeError(f"conflicting duplicate result {stage}/{candidate.hash}")
        result[candidate.hash] = record
    return result


def run_stage(
    records: list[dict],
    log_path: Path,
    plan: dict,
    binary: Path,
    model: Path,
    run_dir: Path,
    workers: int,
    budget: Budget,
) -> tuple[list[dict], bool]:
    stage = plan["stage"]
    done = completed_results(records, stage)
    proposals = [
        Proposal(Candidate.from_dict(item["config"]), item["origin"])
        for item in plan["candidates"]]
    validate_stage_aspiration_policy(stage, proposals)
    for raw, proposal in zip(plan["candidates"], proposals):
        if raw["config_hash"] != proposal.candidate.hash:
            raise RuntimeError(f"plan hash mismatch in {stage}")
    unexpected = set(done) - {item.candidate.hash for item in proposals}
    if unexpected:
        raise RuntimeError(f"unexpected saved result in {stage}: {min(unexpected)}")
    dataset = Path(plan["dataset"]["path"])
    cache = plan["control_cache"]
    if file_fingerprint(Path(cache["path"])) != cache["artifact"]:
        raise RuntimeError(f"control cache mutated before {stage}")
    validate_control_cache(Path(cache["path"]), cache, dataset)
    for entry in done.values():
        if (
            entry.get("dataset_sha256") != plan["dataset"]["sha256"]
            or int(entry.get("depth", -1)) != int(plan["depth"])
            or entry.get("control_cache_sha256") != cache["artifact"]["sha256"]
        ):
            raise RuntimeError(f"saved result metadata mismatch in {stage}")
        detail = entry.get("detail")
        if plan["write_detail_jsonl"]:
            if not isinstance(detail, dict):
                raise RuntimeError(f"saved detail missing in {stage}")
            if file_fingerprint(Path(detail["artifact"]["path"])) != detail["artifact"]:
                raise RuntimeError(f"immutable detail artifact changed in {stage}")
    pending = [item for item in proposals if item.candidate.hash not in done]
    while pending and not budget.exhausted:
        batch = pending[:workers]
        pending = pending[workers:]
        deadline = time.monotonic() + budget.remaining_sec

        def execute(proposal: Proposal):
            started = time.monotonic()
            detail_path = None
            if plan["write_detail_jsonl"]:
                # Attempt-unique means a process killed after evaluator publish
                # but before result append leaves only an ignorable orphan.
                nonce = f"{os.getpid()}-{time.time_ns()}-{random.getrandbits(32):08x}"
                detail_path = (
                    run_dir / "details" / stage
                    / f"{proposal.candidate.hash}-{nonce}.jsonl")
            result, detail = evaluate(
                binary, dataset, model, int(plan["depth"]),
                proposal.candidate, cache, detail_path, deadline)
            return proposal, result, detail, time.monotonic() - started

        expired: list[Proposal] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(workers, len(batch))) as pool:
            futures = {pool.submit(execute, proposal): proposal for proposal in batch}
            for future in concurrent.futures.as_completed(futures):
                try:
                    proposal, result, detail, wall_sec = future.result()
                except EvaluationBudgetExhausted:
                    expired.append(futures[future])
                    continue
                entry = {
                    "kind": "result", "schema_version": SCHEMA_VERSION,
                    "stage": stage,
                    "config": proposal.candidate.canonical(),
                    "config_hash": proposal.candidate.hash,
                    "label": proposal.candidate.label,
                    "origin": proposal.origin,
                    "dataset_sha256": plan["dataset"]["sha256"],
                    "depth": int(plan["depth"]), "candidate_time_ms": 0,
                    "control_cache_sha256": cache["artifact"]["sha256"],
                    "control_cache_identity_sha256": cache["identity_sha256"],
                    "detail": detail, "wall_sec": wall_sec,
                    "budget_elapsed_sec": budget.elapsed_sec,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "result": result,
                }
                append_jsonl(log_path, entry)
                records.append(entry)
                done[proposal.candidate.hash] = entry
                print(
                    f"{stage}_progress label={proposal.candidate.label} "
                    f"origin={proposal.origin} nodes={result['node_ratio']:.6f} "
                    f"wdl={result['mean_wdl_loss']:.9f} "
                    f"critical={result['critical_mistakes']}", flush=True)
        if expired:
            pending = [*expired, *pending]
            break
    complete = not pending
    if complete and not any(
        record.get("kind") == "stage_complete" and record.get("stage") == stage
        for record in records
    ):
        record = {
            "kind": "stage_complete", "schema_version": SCHEMA_VERSION,
            "stage": stage, "candidate_count": len(proposals),
            "result_count": len(done), "hard_reject_count": 0,
            "frontier_count": len(frontier(list(done.values()))),
            "budget_elapsed_sec": budget.elapsed_sec,
        }
        append_jsonl(log_path, record)
        records.append(record)
    return ([done[item.candidate.hash] for item in proposals
             if item.candidate.hash in done], complete)


def max_budget_elapsed(records: Sequence[dict]) -> float:
    return infra.max_budget_elapsed(records)


def ensure_budget_limit(
    records: list[dict], log_path: Path, requested: int,
) -> None:
    infra.ensure_budget_limit(records, log_path, requested)


def update_status(run_dir: Path, state: str, **extra) -> None:
    atomic_write_json(run_dir / "status.json", {
        "state": state,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **extra,
    })


def pause(
    args: argparse.Namespace,
    records: list[dict],
    log_path: Path,
    budget: Budget,
    stage: str,
) -> None:
    payload = {
        "state": "PAUSED", "stage": stage,
        "budget_elapsed_sec": budget.elapsed_sec,
        "cumulative_limit_sec": args.duration_sec,
        "resume_requires_larger_duration": True,
    }
    append_jsonl(log_path, {"kind": "paused", **payload})
    atomic_write_json(args.run_dir / "PAUSED.json", payload)
    update_status(args.run_dir, "PAUSED", **{
        key: value for key, value in payload.items() if key != "state"})
    print(json.dumps(payload, sort_keys=True), flush=True)


def _unique_entries(entries: Sequence[dict]) -> list[dict]:
    result: dict[str, dict] = {}
    for entry in entries:
        previous = result.get(entry["config_hash"])
        if previous is None:
            result[entry["config_hash"]] = entry
            continue
        # Same fixed dataset/depth must be bit-reproducible.  Keep the earlier
        # record but fail if objective totals disagree.
        if result_metrics(previous) != result_metrics(entry):
            raise RuntimeError(
                f"duplicate candidate has different result {entry['config_hash']}")
    return list(result.values())


def _proposals_from_entries(entries: Sequence[dict], origin: str) -> list[Proposal]:
    return [Proposal(config_from_record(entry), origin) for entry in entries]


def manifest_for(
    args: argparse.Namespace,
    sources: dict[str, Path],
    rungs: dict[str, Path],
    caches: dict[str, dict],
) -> dict:
    dataset_manifest = args.dataset_dir / "manifest.json"
    numpy_version = None
    if args.detail_sidecars:
        try:
            import numpy as np
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "detail-sidecar paired bootstrap requires numpy") from error
        numpy_version = np.__version__
    body = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "objective": "all-position-wdl-node-pareto",
        "hard_strength_rejects": False,
        "sealed_holdout_policy": "rejection-only-no-reselection",
        "dataset_schema": DATASET_SCHEMA,
        "dataset_manifest": (
            file_fingerprint(dataset_manifest)
            if dataset_manifest.exists() else None),
        "dataset_sources": {
            name: {
                **file_fingerprint(path),
                **infra.validate_dataset_schema(path),
                "category_counts": infra.category_counts(path),
            }
            for name, path in sources.items()
        },
        "rung_datasets": {
            name: {
                **file_fingerprint(path),
                **infra.validate_dataset_schema(path),
                "category_counts": infra.category_counts(path),
            }
            for name, path in rungs.items()
        },
        "tuner_source": file_fingerprint(Path(__file__)),
        "infra_source": file_fingerprint(Path(infra.__file__)),
        "binary": file_fingerprint(args.binary),
        "model": file_fingerprint(args.model),
        "candidate_engine_profile": CANDIDATE_ENGINE_PROFILE,
        "twofold_search_draw_enabled": True,
        "prune_stage_aspiration_locked": True,
        "prune_stage_aspiration_fixed_through_selection": True,
        "prune_stage_aspiration": {
            "name": "v43_balanced_aspiration",
            "config_sha256": PRODUCTION_BASELINE_ASPIRATION.hash,
            "config": PRODUCTION_BASELINE_ASPIRATION.canonical(),
        },
        "prune_tuning_stages": sorted(PRUNE_TUNING_STAGES),
        "prune_tuning_stage_prefixes": ["screen_"],
        "aspiration_refresh_after_prune_selection": True,
        "baseline_name": PRODUCTION_BASELINE_NAME,
        "production_baseline": Candidate(
            PRODUCTION_FAST_QSEE,
            PRODUCTION_BASELINE_ASPIRATION,
        ).canonical(),
        "production_baseline_config_hash": Candidate(
            PRODUCTION_FAST_QSEE,
            PRODUCTION_BASELINE_ASPIRATION,
        ).hash,
        "production_fast_qsee_selective_baseline": (
            PRODUCTION_FAST_QSEE.canonical()),
        "config_fields": list(SELECTIVE_CONFIG_FIELDS),
        "conditional_disabled_defaults": DISABLED_CHILD_DEFAULTS,
        "behavior_dedup": ["lmr-table", "rfp-margin-vector", "lmp-threshold-vector"],
        "blocks": {name: list(values) for name, values in BLOCKS.items()},
        "grids": {name: list(values) for name, values in GRIDS.items()},
        "block_screen_counts": (
            {name: min(2, count) for name, count in BLOCK_SCREEN_COUNTS.items()}
            if args.smoke_test else BLOCK_SCREEN_COUNTS),
        "interaction_pairs": [list(pair) for pair in INTERACTION_PAIRS],
        "candidate_counts": {
            "interaction": args.interaction_candidates,
            "global": args.global_candidates,
            "halving_fill_target": args.halving_candidates,
            "refinement_per_generation": args.refinement_candidates,
            "refinement_generations": 2,
            "aspiration_refresh_per_profile": args.aspiration_per_profile,
        },
        "refinement_method_counts": refinement_method_counts(
            args.refinement_candidates),
        "random_restart_floor": 0.15,
        "sizes": {
            "preflight": args.preflight_size,
            "screen": args.screen_size,
            "halving": args.halving_size,
            "closure": args.closure_size,
            "deep_tune": args.deep_tune_size,
            "selection": args.selection_size,
            "aspiration_refresh": args.aspiration_refresh_size,
            "holdout": args.holdout_size,
        },
        "depths": {
            "preflight": args.preflight_depth,
            "screen": args.screen_depth,
            "halving": args.halving_depth,
            "closure": args.closure_depth,
            "deep_tune": args.deep_tune_depth,
            "selection": args.selection_depth,
            "aspiration_refresh": args.aspiration_refresh_depth,
            "holdout": args.holdout_depth,
        },
        "control_cache_identities": caches,
        "control_cache_policy": {
            "teacher": "V36 strict root", "candidate_independent": True,
            "existing_cache_cli_supported": True,
            "immutable_artifact_sha256": True,
        },
        "detail_sidecars": {
            "enabled": args.detail_sidecars,
            "schema": "nnue_selective_position_details_jsonl_v1",
            "stages": [
                "halving", "refinement_1", "refinement_2", "closure_1",
                "closure_2", "deep_tune", "selection",
                "aspiration_refresh", "holdout"],
            "attempt_unique_no_clobber": True,
        },
        "dominance": {
            "point_estimate_early": True,
            "paired_bootstrap_expensive": args.detail_sidecars,
            "confidence": 0.95,
            "resamples": args.bootstrap_resamples,
            "seeded": True,
            "uncertainty_unavailable_when_sidecars_disabled": True,
        },
        "accepted_depth_promotion_safeguard": {
            "comparison_rung": "selection",
            "baseline": "production-fast-qsee-plus-balanced-aspiration",
            "baseline_config_hash": Candidate(
                PRODUCTION_FAST_QSEE,
                PRODUCTION_BASELINE_ASPIRATION,
            ).hash,
            "accepted_depth_ratio_noninferiority_tolerance": (
                ACCEPTED_DEPTH_NONINFERIOR_TOLERANCE),
            "max_accepted_depth_reduction": (
                PROMOTION_MAX_ACCEPTED_DEPTH_REDUCTION),
            "failures_are_diagnostic_promotion_vetoes": True,
            "holdout_profile_reselection": False,
        },
        "seed": args.seed,
        "testing_smoke_mode": args.smoke_test,
        "workers": args.workers,
        "runtime_profile": {
            "python": list(sys.version_info[:3]),
            "python_implementation": sys.implementation.name,
            "numpy": numpy_version,
            "numpy_rng": "default_rng-PCG64",
        },
        "budget_policy": infra.BUDGET_POLICY,
    }
    return {**body, "manifest_sha256": hashlib.sha256(canonical_json(body)).hexdigest()}


def validate_dataset_overlap_proof(overlap: object) -> None:
    if (
        not isinstance(overlap, dict)
        or set(overlap) != REQUIRED_DATASET_OVERLAP_KEYS
        or any(int(overlap[key]) != 0 for key in REQUIRED_DATASET_OVERLAP_KEYS)
    ):
        raise RuntimeError("sealed dataset overlap proof is missing or nonzero")


def validate_sealed_dataset_manifest(
    dataset_dir: Path,
    sources: dict[str, Path],
    allow_unsealed: bool,
) -> dict | None:
    path = dataset_dir / "manifest.json"
    if not path.exists():
        if allow_unsealed:
            return None
        raise RuntimeError("missing sealed V43 all-prunes manifest.json")
    try:
        manifest = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise RuntimeError("invalid sealed dataset manifest JSON") from error
    if allow_unsealed:
        # Tests may deliberately use a tiny synthetic manifest.  It is still
        # fingerprinted by the immutable run manifest.
        return manifest
    if (
        manifest.get("schema_version") != 3
        or manifest.get("format") != "nnue-v43-all-prunes-game-disjoint-v3"
    ):
        raise RuntimeError("wrong sealed V43 final-prune dataset format")
    expected_counts = {
        "tune": 8_000, "selection": 4_000,
        "aspiration_refresh": 4_000, "holdout": 4_000,
    }
    policy = manifest.get("selection_policy")
    expected_policy = {
        "split_unit": "game_id",
        "positions_per_game": 1,
        "phase_distribution": "natural (not stratified or reweighted)",
        "category": "phase_index",
        "output_ply": "zero-based absolute ply derived from six-field FEN",
    }
    if not isinstance(policy, dict) or any(
        policy.get(name) != expected for name, expected in expected_policy.items()
    ):
        raise RuntimeError("sealed dataset selection policy drift")
    splits = manifest.get("splits")
    split_files = manifest.get("split_files")
    if not isinstance(splits, dict) or not isinstance(split_files, dict):
        raise RuntimeError("sealed dataset lacks split manifests")
    for name, expected_count in expected_counts.items():
        stats = splits.get(name)
        artifact = split_files.get(name)
        source = sources[name]
        if not isinstance(stats, dict) or not isinstance(artifact, dict):
            raise RuntimeError(f"sealed dataset lacks split {name}")
        actual_count = int(infra.validate_dataset_schema(source)["count"])
        if (
            actual_count != expected_count
            or stats.get("count") != expected_count
            or stats.get("games") != expected_count
            or stats.get("canonical_position_and_mirror_keys") != expected_count
            or artifact.get("file") != source.name
            or artifact.get("bytes") != source.stat().st_size
            or artifact.get("sha256") != sha256_file(source)
        ):
            raise RuntimeError(f"sealed dataset split identity mismatch: {name}")
    overlap = manifest.get("overlap")
    exclusions = manifest.get("exclusions")
    validate_dataset_overlap_proof(overlap)
    if (
        not isinstance(exclusions, dict)
        or int(exclusions.get("unique_game_ids", -1)) != 16_000
        or int(exclusions.get(
            "canonical_position_and_mirror_keys", -1)) != 16_000
        or int(exclusions.get("selected_overlap_game_ids", -1)) != 0
        or int(exclusions.get(
            "selected_overlap_position_or_mirror_keys", -1)) != 0
    ):
        raise RuntimeError("sealed dataset leakage invariant failed")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--allow-unsealed-dataset", action="store_true")
    parser.add_argument("--smoke-test", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--existing-control-cache-dir", type=Path)
    parser.add_argument("--duration-sec", type=int, default=43_200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument(
        "--detail-sidecars", action=argparse.BooleanOptionalAction,
        default=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=400)
    parser.add_argument("--preflight-size", type=int, default=128)
    parser.add_argument("--screen-size", type=int, default=1_000)
    parser.add_argument("--halving-size", type=int, default=3_000)
    parser.add_argument("--closure-size", type=int, default=1_500)
    parser.add_argument("--deep-tune-size", type=int, default=8_000)
    parser.add_argument("--selection-size", type=int, default=4_000)
    parser.add_argument("--aspiration-refresh-size", type=int, default=4_000)
    parser.add_argument("--holdout-size", type=int, default=4_000)
    parser.add_argument("--preflight-depth", type=int, default=6)
    parser.add_argument("--screen-depth", type=int, default=6)
    parser.add_argument("--halving-depth", type=int, default=7)
    parser.add_argument("--closure-depth", type=int, default=7)
    parser.add_argument("--deep-tune-depth", type=int, default=7)
    parser.add_argument("--selection-depth", type=int, default=7)
    parser.add_argument("--aspiration-refresh-depth", type=int, default=7)
    parser.add_argument("--holdout-depth", type=int, default=8)
    parser.add_argument("--interaction-candidates", type=int, default=48)
    parser.add_argument("--global-candidates", type=int, default=64)
    parser.add_argument("--halving-candidates", type=int, default=40)
    parser.add_argument("--refinement-candidates", type=int, default=32)
    parser.add_argument("--aspiration-per-profile", type=int, default=8)
    args = parser.parse_args()
    if args.duration_sec <= 0 or args.workers <= 0:
        parser.error("duration and workers must be positive")
    for name in (
        "preflight_size", "screen_size", "halving_size", "closure_size",
        "deep_tune_size",
        "selection_size", "aspiration_refresh_size", "holdout_size",
        "interaction_candidates", "global_candidates", "halving_candidates",
        "refinement_candidates", "aspiration_per_profile",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "preflight_depth", "screen_depth", "halving_depth", "closure_depth",
        "deep_tune_depth",
        "selection_depth", "aspiration_refresh_depth", "holdout_depth",
    ):
        if getattr(args, name) < 2:
            parser.error(f"--{name.replace('_', '-')} must be at least 2")
    if args.aspiration_per_profile < len(ASPIRATION_ANCHORS):
        parser.error("aspiration-per-profile is smaller than anchor set")
    if args.bootstrap_resamples < 100:
        parser.error("bootstrap-resamples must be at least 100")
    return args


def _main() -> None:
    args = parse_args()
    args.binary = args.binary.resolve(strict=True)
    args.model = args.model.resolve(strict=True)
    args.dataset_dir = args.dataset_dir.resolve(strict=True)
    if args.existing_control_cache_dir is not None:
        args.existing_control_cache_dir = (
            args.existing_control_cache_dir.resolve(strict=True))
    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.run_dir = args.run_dir.resolve()
    update_status(args.run_dir, "RUNNING", stage="initializing")

    source_names = ("tune", "selection", "aspiration_refresh", "holdout")
    sources = {name: args.dataset_dir / f"{name}.tsv" for name in source_names}
    for path in sources.values():
        infra.validate_dataset_schema(path)
    infra.validate_dataset_hash_disjointness(sources)
    infra.validate_no_balanced_lineage_overlap(sources)
    validate_sealed_dataset_manifest(
        args.dataset_dir, sources, args.allow_unsealed_dataset)

    rungs = {
        "preflight": args.run_dir / "preflight.tsv",
        "screen": args.run_dir / "screen.tsv",
        "halving": args.run_dir / "halving.tsv",
        "closure": args.run_dir / "closure.tsv",
        "deep_tune": args.run_dir / "deep_tune.tsv",
        "selection": args.run_dir / "selection.tsv",
        "aspiration_refresh": args.run_dir / "aspiration_refresh.tsv",
        "holdout": args.run_dir / "holdout.tsv",
    }
    infra.ensure_subset(
        rungs["preflight"], sources["tune"], args.preflight_size,
        args.seed ^ 0x505245)
    infra.ensure_subset(
        rungs["screen"], sources["tune"], args.screen_size,
        args.seed ^ 0x534352)
    infra.ensure_subset(
        rungs["halving"], sources["tune"], args.halving_size,
        args.seed ^ 0x48414C)
    infra.ensure_subset(
        rungs["closure"], sources["tune"], args.closure_size,
        args.seed ^ 0x434C4F)
    infra.ensure_subset(
        rungs["deep_tune"], sources["tune"], args.deep_tune_size,
        args.seed ^ 0x444545)
    infra.ensure_subset(
        rungs["selection"], sources["selection"], args.selection_size,
        args.seed ^ 0x53454C)
    infra.ensure_subset(
        rungs["aspiration_refresh"], sources["aspiration_refresh"],
        args.aspiration_refresh_size, args.seed ^ 0x415350)
    infra.ensure_subset(
        rungs["holdout"], sources["holdout"], args.holdout_size,
        args.seed ^ 0x484F4C)
    depths = {
        "preflight": args.preflight_depth,
        "screen": args.screen_depth,
        "halving": args.halving_depth,
        "closure": args.closure_depth,
        "deep_tune": args.deep_tune_depth,
        "selection": args.selection_depth,
        "aspiration_refresh": args.aspiration_refresh_depth,
        "holdout": args.holdout_depth,
    }
    cache_specs = {
        name: control_cache_spec(
            args.run_dir, name, args.binary, args.model, path, depths[name])
        for name, path in rungs.items()
    }
    expected_manifest = manifest_for(args, sources, rungs, cache_specs)
    infra.ensure_manifest(args.run_dir / "manifest.json", expected_manifest)
    if (args.run_dir / "DONE").exists():
        if not (args.run_dir / "summary.json").exists():
            raise RuntimeError("DONE exists without summary.json")
        update_status(args.run_dir, "DONE", stage="complete")
        print((args.run_dir / "summary.json").read_text(), flush=True)
        return

    log_path = args.run_dir / "results.jsonl"
    records = load_jsonl(log_path)
    ensure_budget_limit(records, log_path, args.duration_sec)
    (args.run_dir / "PAUSED.json").unlink(missing_ok=True)
    budget = Budget(
        float(args.duration_sec), max_budget_elapsed(records), time.monotonic())
    prepared_caches: dict[str, dict] = {}

    def prepare_cache(name: str) -> dict | None:
        if name in prepared_caches:
            return prepared_caches[name]
        update_status(args.run_dir, "RUNNING", stage=f"control_cache:{name}")
        try:
            cache = ensure_control_cache(
                records, log_path, cache_specs[name], args.binary, args.model,
                rungs[name], budget, args.existing_control_cache_dir)
        except CacheBudgetExhausted:
            pause(args, records, log_path, budget, f"control_cache:{name}")
            return None
        prepared_caches[name] = cache
        return cache

    def stage(
        name: str,
        proposals: Sequence[Proposal],
        rung: str,
        *,
        details: bool = False,
    ) -> list[dict] | None:
        cache = prepare_cache(rung)
        if cache is None:
            return None
        update_status(args.run_dir, "RUNNING", stage=name)
        plan = ensure_plan(
            records, log_path, name, proposals, rungs[rung], depths[rung],
            cache, budget, details and args.detail_sidecars)
        outcomes, complete = run_stage(
            records, log_path, plan, args.binary, args.model, args.run_dir,
            args.workers, budget)
        if not complete:
            pause(args, records, log_path, budget, name)
            return None
        return outcomes

    initial_preflight = preflight_proposals()
    if args.smoke_test:
        initial_preflight = initial_preflight[:3]
    preflight = stage("preflight", initial_preflight, "preflight")
    if preflight is None:
        return

    early: list[dict] = []
    effective_block_counts = (
        {name: min(2, count) for name, count in BLOCK_SCREEN_COUNTS.items()}
        if args.smoke_test else BLOCK_SCREEN_COUNTS)
    for index, (block, count) in enumerate(effective_block_counts.items()):
        outcomes = stage(
            f"screen_{block}",
            block_screen_proposals(block, count, args.seed ^ (0x1000 + index)),
            "screen")
        if outcomes is None:
            return
        early.extend(outcomes)
    interactions = stage(
        "interactions",
        interaction_proposals(args.interaction_candidates, args.seed ^ 0x1A2B),
        "screen")
    if interactions is None:
        return
    global_entries = stage(
        "global",
        global_proposals(args.global_candidates, args.seed ^ 0x610B),
        "screen")
    if global_entries is None:
        return
    early = _unique_entries([*early, *interactions, *global_entries])
    survivors = successive_halving_survivors(early, args.halving_candidates)
    baseline_candidate = Candidate(
        PRODUCTION_FAST_QSEE, PRODUCTION_BASELINE_ASPIRATION)
    baseline_early = next(
        entry for entry in early
        if entry["config_hash"] == baseline_candidate.hash)
    if baseline_candidate.hash not in {
        entry["config_hash"] for entry in survivors
    }:
        survivors.append(baseline_early)
    halving = stage(
        "halving", _proposals_from_entries(survivors, "halving:survivor"),
        "halving", details=True)
    if halving is None:
        return

    seen = {
        behavior_signature(config_from_record(entry).selective)
        for entry in halving}
    refinement_1_plan = refinement_proposals(
        halving, args.refinement_candidates, args.seed ^ 0x524631, seen)
    refinement_1 = stage(
        "refinement_1", refinement_1_plan, "halving", details=True)
    if refinement_1 is None:
        return
    generation_1 = _unique_entries([*halving, *refinement_1])
    seen.update(
        behavior_signature(config_from_record(entry).selective)
        for entry in refinement_1)
    refinement_2_plan = refinement_proposals(
        generation_1, args.refinement_candidates, args.seed ^ 0x524632, seen)
    refinement_2 = stage(
        "refinement_2", refinement_2_plan, "halving", details=True)
    if refinement_2 is None:
        return
    refined = _unique_entries([*generation_1, *refinement_2])
    refined_frontier = confidence_frontier(
        refined, args.seed ^ 0x434631, args.bootstrap_resamples)
    preliminary_profiles = named_profiles(refined_frontier)

    closure_seen = {
        behavior_signature(config_from_record(entry).selective)
        for entry in refined}
    closure_1_proposals = [
        Proposal(baseline_candidate, "closure:production-anchor"),
        *_proposals_from_entries(refined_frontier, "closure:parent"),
        *local_closure_proposals(preliminary_profiles, closure_seen),
    ]
    if args.smoke_test:
        closure_1_proposals = closure_1_proposals[:8]
    closure_1 = stage(
        "closure_1", closure_1_proposals, "closure", details=True)
    if closure_1 is None:
        return
    closure_frontier_1 = confidence_frontier(
        closure_1, args.seed ^ 0x434632, args.bootstrap_resamples)
    closure_profiles_1 = named_profiles(closure_frontier_1)
    closure_seen.update(
        behavior_signature(config_from_record(entry).selective)
        for entry in closure_1)
    closure_2_proposals = local_closure_proposals(
        closure_profiles_1, closure_seen)
    if args.smoke_test:
        closure_2_proposals = closure_2_proposals[:6]
    if closure_2_proposals:
        closure_2 = stage(
            "closure_2", closure_2_proposals, "closure", details=True)
        if closure_2 is None:
            return
    else:
        closure_2 = []
    closure_all = _unique_entries([*closure_1, *closure_2])
    closure_frontier = confidence_frontier(
        closure_all, args.seed ^ 0x434633, args.bootstrap_resamples)
    closure_2_hashes = {entry["config_hash"] for entry in closure_2}
    closure_stationary = not (
        {entry["config_hash"] for entry in closure_frontier} & closure_2_hashes)

    deep_tune = stage(
        "deep_tune",
        [
            Proposal(baseline_candidate, "deep-tune:production-anchor"),
            *_proposals_from_entries(
                closure_frontier, "deep-tune:closure-frontier"),
        ],
        "deep_tune", details=True)
    if deep_tune is None:
        return
    deep_tune_frontier = confidence_frontier(
        deep_tune, args.seed ^ 0x444545, args.bootstrap_resamples)

    selection = stage(
        "selection",
        [
            Proposal(baseline_candidate, "selection:production-anchor"),
            *_proposals_from_entries(
                deep_tune_frontier, "selection:deep-tune-frontier"),
        ],
        "selection", details=True)
    if selection is None:
        return
    selection_frontier = confidence_frontier(
        selection, args.seed ^ 0x53454C, args.bootstrap_resamples)
    selected_profiles = named_profiles(selection_frontier)
    if not selected_profiles:
        raise RuntimeError("selection produced no named profiles")
    selection_by_hash = {
        entry["config_hash"]: entry for entry in selection}
    selection_baseline = selection_by_hash[baseline_candidate.hash]
    selection_depth_safeguards = {
        name: accepted_depth_safeguard(entry, selection_baseline)
        for name, entry in selected_profiles
    }
    selection_lineage = {
        name: pre_refresh_selection_lineage(entry)
        for name, entry in selected_profiles
    }

    refresh = stage(
        "aspiration_refresh",
        aspiration_refresh_proposals(
            selected_profiles, args.aspiration_per_profile,
            args.seed ^ 0x415350),
        "aspiration_refresh", details=True)
    if refresh is None:
        return
    refreshed_profiles = choose_refreshed_profiles(
        selected_profiles, refresh, args.seed ^ 0x415350,
        args.bootstrap_resamples)
    baseline = baseline_candidate
    holdout_proposals = [
        Proposal(baseline, "holdout:production-balanced-aspiration"),
        *(
            Proposal(config_from_record(entry), f"holdout:preselected-{name}")
            for name, entry in refreshed_profiles
        ),
    ]
    holdout = stage(
        "holdout", holdout_proposals, "holdout", details=True)
    if holdout is None:
        return
    holdout_frontier = confidence_frontier(
        holdout, args.seed ^ 0x484F4C, args.bootstrap_resamples)
    holdout_frontier_hashes = {entry["config_hash"] for entry in holdout_frontier}
    holdout_by_hash = {entry["config_hash"]: entry for entry in holdout}
    baseline_holdout = holdout_by_hash[baseline.hash]
    profiles_payload: dict[str, dict] = {}
    for name, refresh_entry in refreshed_profiles:
        candidate_hash = refresh_entry["config_hash"]
        result = holdout_by_hash[candidate_hash]
        distinct_challenger = candidate_hash != baseline.hash
        survives = candidate_hash in holdout_frontier_hashes
        # This comparison deliberately stays on the selection rung where both
        # lineages use the locked Balanced aspiration.  Comparing post-refresh
        # holdout ratios would confound pruning with the newly varied policy.
        depth_safeguard = selection_depth_safeguards[name]
        promotion_eligible = (
            distinct_challenger and survives and depth_safeguard["pass"])
        profiles_payload[name] = {
            "config": result["config"], "config_hash": candidate_hash,
            **selection_lineage[name],
            "result": result["result"],
            "distinct_from_production": distinct_challenger,
            "holdout_frontier_survivor": survives,
            "accepted_depth_safeguard": depth_safeguard,
            "holdout_accepted_depth_diagnostic": {
                "accepted_depth_ratio": result["result"][
                    "aspiration_accepted_depth_ratio"],
                "max_accepted_depth_reduction": result["result"][
                    "aspiration_max_accepted_depth_reduction"],
                "promotion_comparison": False,
                "reason": "post-refresh aspiration varies",
            },
            "promotion_eligible": promotion_eligible,
            "eligible_for_equal_time_selfplay": promotion_eligible,
            "status": (
                "production_equivalent" if not distinct_challenger else
                "rejected_by_sealed_holdout" if not survives else
                "holdout_survivor_depth_safeguard_failed"
                if not depth_safeguard["pass"] else
                "holdout_confirmed_challenger"),
        }
    selfplay_configs = {
        name: selfplay_confirmation_args(
            name,
            holdout_by_hash[payload["config_hash"]],
            baseline_holdout,
            payload["promotion_eligible"],
        )
        for name, payload in profiles_payload.items()
        if payload["distinct_from_production"]
        and payload["holdout_frontier_survivor"]
    }
    requires_equal_time_selfplay = any(
        payload["promotion_eligible"] for payload in profiles_payload.values())
    summary = {
        "kind": "complete", "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "budget_elapsed_sec": budget.elapsed_sec,
        "cumulative_limit_sec": args.duration_sec,
        "objective": "all-position WDL/node Pareto",
        "hard_strength_rejects": False,
        "prune_stage_aspiration_locked": True,
        "prune_stage_aspiration_config": (
            PRODUCTION_BASELINE_ASPIRATION.canonical()),
        "prune_stage_aspiration_config_sha256": (
            PRODUCTION_BASELINE_ASPIRATION.hash),
        "baseline_name": PRODUCTION_BASELINE_NAME,
        "production_baseline_config_hash": baseline.hash,
        "selection_preselected_profiles": [name for name, _ in selected_profiles],
        "aspiration_refresh_pre_holdout": True,
        "holdout_used_for_rejection_only": True,
        "holdout_reselected_profiles": False,
        "uncertainty": (
            {"method": "paired-bootstrap", "confidence": 0.95,
             "resamples": args.bootstrap_resamples}
            if args.detail_sidecars else
            {"method": "unavailable", "reason": "detail-sidecars-disabled"}),
        "early_candidate_count": len(early),
        "halving_candidate_count": len(halving),
        "refined_candidate_count": len(refined),
        "closure_candidate_count": len(closure_all),
        "closure_stationary_after_two_passes": closure_stationary,
        "closure_frontier_still_moved_at_pass_limit": not closure_stationary,
        "deep_tune_candidate_count": len(deep_tune),
        "deep_tune_frontier_count": len(deep_tune_frontier),
        "selection_frontier_count": len(selection_frontier),
        "named_profile_count": len(selected_profiles),
        "distinct_named_profile_count": len({
            entry["config_hash"] for _, entry in selected_profiles}),
        "sealed_candidate_count": len(holdout),
        "sealed_frontier_count": len(holdout_frontier),
        "profiles": profiles_payload,
        "production_balanced_aspiration": baseline_holdout,
        "accepted_depth_promotion_safeguard": {
            "comparison": "same-rung production Balanced selection anchor",
            "ratio_noninferiority_tolerance": (
                ACCEPTED_DEPTH_NONINFERIOR_TOLERANCE),
            "max_accepted_depth_reduction": (
                PROMOTION_MAX_ACCEPTED_DEPTH_REDUCTION),
            "failures_are_diagnostic_promotion_vetoes": True,
            "holdout_profile_reselection": False,
        },
        "requires_equal_time_selfplay": requires_equal_time_selfplay,
        "equal_time_selfplay_artifacts": {
            "gauntlet_binary": "bin/nnue_v43_time_gauntlet",
            "model": "inputs/phase_quantized_nnue.bin",
            "tt_mb": CANDIDATE_ENGINE_PROFILE["tt_mb"],
            "tt_bucket_size": CANDIDATE_ENGINE_PROFILE["tt_bucket_size"],
            "reuse_stale_tt_scores": False,
            "reuse_deeper_tt_scores": False,
            "tt_depth_policy": "Exact",
            "tt_generation_policy": "CurrentOnly",
            "twofold_search_draw_enabled_for_both": True,
            "full_selective_config_fields": list(SELECTIVE_CONFIG_FIELDS),
        },
        "equal_time_selfplay_config_args": selfplay_configs,
    }
    append_jsonl(log_path, summary)
    atomic_write_json(args.run_dir / "summary.json", summary)
    (args.run_dir / "DONE").touch()
    (args.run_dir / "PAUSED.json").unlink(missing_ok=True)
    update_status(args.run_dir, "DONE", stage="complete")
    print(json.dumps(summary, sort_keys=True), flush=True)


def main() -> None:
    try:
        _main()
    except Exception as error:
        # Best-effort monitor state; the traceback remains authoritative.
        try:
            argv = sys.argv if "sys" in globals() else []
            if "--run-dir" in argv:
                run_dir = Path(argv[argv.index("--run-dir") + 1]).resolve()
                if run_dir.exists():
                    update_status(run_dir, "FAILED", error=str(error))
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()

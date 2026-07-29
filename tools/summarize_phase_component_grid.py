#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    rows = []
    for summary_path in sorted(args.model_root.glob("*/summary.json")):
        label = summary_path.parent.name
        if label == "smoke":
            continue
        summary = json.loads(summary_path.read_text())
        errors = np.asarray(
            json.loads((summary_path.parent / "ranking_errors.json").read_text()),
            dtype=np.float64,
        )
        rows.append(
            (
                label,
                summary["selection"],
                summary["ranking"],
                errors,
                summary["saturation"],
            )
        )
    if len(rows) != 12:
        raise SystemExit(f"expected 12 completed configs, got {len(rows)}")
    rows.sort(key=lambda row: row[2]["cp_mae"])
    best_label, _best_selection, _best_ranking, best_errors, _best_sat = rows[0]
    rng = np.random.default_rng(20260720)

    lines = [
        "# Eight-phase component-supervised 50M grid",
        "",
        f"Best ranking config: `{best_label}`.",
        "",
        "| Rank | Config | Selection CP MAE | Ranking CP MAE | Slope | Paired delta vs best (95% block-bootstrap CI) |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for rank, (label, selection, ranking, errors, _saturation) in enumerate(rows, 1):
        delta = errors - best_errors
        block_means = delta.reshape(1000, -1).mean(axis=1)
        bootstrap = block_means[
            rng.integers(0, len(block_means), size=(5000, len(block_means)))
        ].mean(axis=1)
        low, high = np.quantile(bootstrap, [0.025, 0.975])
        lines.append(
            f"| {rank} | `{label}` | {selection['cp_mae']:.4f} | "
            f"{ranking['cp_mae']:.4f} | {ranking['slope']:.4f} | "
            f"{delta.mean():+.4f} [{low:+.4f}, {high:+.4f}] |"
        )

    lines.extend(
        ["", "## Best config by phase", "", "| Phase | Samples | CP MAE |", "|---:|---:|---:|"]
    )
    for phase in rows[0][2]["phase"]:
        mae = "n/a" if phase["cp_mae"] is None else f"{phase['cp_mae']:.4f}"
        lines.append(f"| {phase['phase']} | {phase['samples']} | {mae} |")

    lines.extend(
        [
            "",
            "## Best config saturation",
            "",
            "| Phase | Layer | Clip rate | Zero rate |",
            "|---:|---:|---:|---:|",
        ]
    )
    for phase in rows[0][4]["phase"]:
        for layer in phase["layers"]:
            lines.append(
                f"| {phase['phase']} | {layer['layer']} | "
                f"{100 * layer['clip_rate']:.4f}% | "
                f"{100 * layer['zero_rate']:.4f}% |"
            )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "best": best_label,
                "selection_cp_mae": rows[0][1]["cp_mae"],
                "ranking_cp_mae": rows[0][2]["cp_mae"],
                "slope": rows[0][2]["slope"],
                "report": str(args.report),
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

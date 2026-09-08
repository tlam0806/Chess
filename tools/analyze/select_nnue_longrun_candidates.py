#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_events(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            result.append(event)
    return result


def ranks(rows: list[dict[str, Any]], key: str, reverse: bool = False) -> dict[str, int]:
    ordered = sorted(rows, key=lambda row: float(row[key]), reverse=reverse)
    return {str(row["branch"]): index + 1 for index, row in enumerate(ordered)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select long-run candidates using validation, calibration, tail MAE and saturation"
    )
    parser.add_argument("--tag", required=True)
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", type=int, default=4)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for path in sorted(args.log_dir.glob(f"{args.tag}_*.log")):
        branch = path.stem.removeprefix(f"{args.tag}_")
        if "__" not in branch:
            continue
        family = branch.split("__", 1)[0]
        events = load_events(path)
        epoch_events = [event for event in events if event.get("event") == "epoch"]
        if not epoch_events:
            continue
        best = min(epoch_events, key=lambda event: float(event["val_cp"]))
        epoch = int(best["epoch"])
        calibration = next(
            event
            for event in events
            if event.get("event") == "calibration"
            and event.get("split") == "val"
            and int(event.get("epoch", -1)) == epoch
        )
        saturation = next(
            event
            for event in events
            if event.get("event") == "hidden_saturation"
            and int(event.get("epoch", -1)) == epoch
        )
        bins = best["val_cp_bins"]
        high_bins = [row for row in bins if float(row["min_abs_target_cp"]) >= 1000.0]
        high_samples = sum(int(row["samples"]) for row in high_bins)
        high_mae = sum(float(row["cp_mae"]) * int(row["samples"]) for row in high_bins) / high_samples
        rows.append(
            {
                "branch": branch,
                "family": family,
                "best_epoch": epoch,
                "val_cp": float(best["val_cp"]),
                "high_mae": high_mae,
                "slope": float(calibration["slope"]),
                "max_clip_rate": max(float(row["clip_rate"]) for row in saturation["stats"]),
            }
        )
    if len(rows) < args.count:
        raise RuntimeError(f"only {len(rows)} complete candidates for requested count {args.count}")

    component_ranks = {
        "val": ranks(rows, "val_cp"),
        "tail": ranks(rows, "high_mae"),
        "slope": ranks(rows, "slope", reverse=True),
        "clip": ranks(rows, "max_clip_rate"),
    }
    for row in rows:
        branch = str(row["branch"])
        row["rank_sum"] = sum(group[branch] for group in component_ranks.values())

    selected: list[dict[str, Any]] = []
    for family in sorted({str(row["family"]) for row in rows}):
        family_rows = [row for row in rows if row["family"] == family]
        selected.append(min(family_rows, key=lambda row: (int(row["rank_sum"]), float(row["val_cp"]))))
    for row in sorted(rows, key=lambda item: (int(item["rank_sum"]), float(item["val_cp"]))):
        if row not in selected:
            selected.append(row)
        if len(selected) >= args.count:
            break
    selected = selected[: args.count]
    report = {"tag": args.tag, "selected": selected, "all_candidates": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for row in selected:
        print(row["branch"])


if __name__ == "__main__":
    main()

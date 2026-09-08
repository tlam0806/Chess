from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from chess_nnue.training_targets import TRAINING_PIPELINE_VERSION

PIPELINE_VERSION = TRAINING_PIPELINE_VERSION


def _activation_health(saturation: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(saturation, dict):
        return None
    stats = saturation.get("stats")
    if not isinstance(stats, list) or not stats:
        return None
    zero_rates = [float(item.get("zero_rate", 0.0)) for item in stats]
    clip_rates = [float(item.get("clip_rate", 0.0)) for item in stats]
    max_zero_rate = max(zero_rates)
    max_clip_rate = max(clip_rates)
    return {
        "healthy": max_zero_rate < 0.99 and max_clip_rate < 0.50,
        "max_zero_rate": max_zero_rate,
        "max_clip_rate": max_clip_rate,
        "dead_layers": [
            int(float(item.get("layer", index + 1)))
            for index, item in enumerate(stats)
            if float(item.get("zero_rate", 0.0)) >= 0.99
        ],
        "severely_clipped_layers": [
            int(float(item.get("layer", index + 1)))
            for index, item in enumerate(stats)
            if float(item.get("clip_rate", 0.0)) >= 0.50
        ],
    }


def _finite_metric(item: dict[str, Any], key: str) -> float:
    try:
        value = float(item[key])
    except (KeyError, TypeError, ValueError):
        return float("inf")
    return value if math.isfinite(value) else float("inf")


def parse_quantized_training_log(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {"error": "missing log"}

    start: dict[str, Any] | None = None
    epochs: list[dict[str, Any]] = []
    saturations: list[dict[str, Any]] = []
    training_complete: dict[str, Any] | None = None
    final_test: dict[str, Any] | None = None
    rejected: dict[str, Any] | None = None
    with log_path.open("r", encoding="utf-8") as log:
        for line in log:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = item.get("event")
            if event == "start":
                start = item
            elif event == "epoch":
                epochs.append(item)
            elif event == "hidden_saturation":
                saturations.append(item)
            elif event == "training_complete":
                training_complete = item
            elif event == "final_test":
                final_test = item
            elif event == "config_rejected":
                rejected = item

    provenance = None if start is None else start.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("pipeline_version") != PIPELINE_VERSION:
        return {
            "error": "legacy or missing pipeline provenance",
            "required_pipeline_version": PIPELINE_VERSION,
        }
    initial_saturation = next(
        (item for item in saturations if int(item.get("epoch", -1)) == 0),
        None,
    )
    if rejected is not None:
        return {
            "pipeline_version": PIPELINE_VERSION,
            "provenance": provenance,
            "completed": training_complete is not None,
            "rejected": True,
            "rejection_reasons": rejected.get("reasons", []),
            "initial_saturation": initial_saturation,
            "activation_health": _activation_health(initial_saturation),
            "test_evaluated": False,
        }
    if not epochs:
        return {"error": "no epochs", "provenance": provenance}
    if any(_finite_metric(epoch, "val_loss") == float("inf") for epoch in epochs):
        return {"error": "epoch missing val_loss", "provenance": provenance}

    last = epochs[-1]
    best_epoch = int(last.get("best_epoch", 0))
    best = next((epoch for epoch in epochs if int(epoch.get("epoch", 0)) == best_epoch), None)
    if best is None:
        return {
            "error": "reported best validation epoch is missing from log",
            "best_epoch": best_epoch,
            "provenance": provenance,
        }
    reported_best_loss = _finite_metric(last, "best_val_loss")
    if reported_best_loss == float("inf"):
        return {"error": "epoch missing best_val_loss", "provenance": provenance}
    if abs(reported_best_loss - _finite_metric(best, "val_loss")) > 1e-7:
        return {
            "error": "reported best validation loss does not match best epoch",
            "best_epoch": best_epoch,
            "provenance": provenance,
        }
    best_saturation = next(
        (item for item in saturations if int(item.get("epoch", 0)) == best_epoch),
        None,
    )

    result: dict[str, Any] = {
        "pipeline_version": PIPELINE_VERSION,
        "provenance": provenance,
        "best_epoch": best_epoch,
        "best_val_loss": reported_best_loss,
        "val_cp_at_best_loss": best.get("val_cp"),
        "val_cp_bins_at_best_loss": best.get("val_cp_bins"),
        "last_epoch": last.get("epoch"),
        "last_val_loss": last.get("val_loss"),
        "last_val_cp": last.get("val_cp"),
        "last_lr": last.get("lr"),
        "best_saturation": best_saturation,
        "initial_saturation": initial_saturation,
        "activation_health": _activation_health(best_saturation),
        "last_saturation": saturations[-1] if saturations else None,
        "completed": training_complete is not None or final_test is not None,
        "test_evaluated": final_test is not None,
    }
    if training_complete is not None:
        selected_epoch = int(training_complete.get("selected_epoch", 0))
        if selected_epoch != best_epoch:
            return {
                "error": "training_complete selected epoch does not match best validation epoch",
                "best_epoch": best_epoch,
                "selected_epoch": selected_epoch,
                "provenance": provenance,
            }
        if training_complete.get("test_evaluated") is not False:
            return {
                "error": "training_complete must explicitly report test_evaluated=false",
                "best_epoch": best_epoch,
                "provenance": provenance,
            }
        if final_test is not None:
            return {
                "error": "log contains both skipped-test completion and final_test",
                "best_epoch": best_epoch,
                "provenance": provenance,
            }
    if final_test is not None:
        selected_epoch = int(final_test.get("selected_epoch", 0))
        if selected_epoch != best_epoch:
            return {
                "error": "final_test selected epoch does not match best validation epoch",
                "best_epoch": best_epoch,
                "selected_epoch": selected_epoch,
                "provenance": provenance,
            }
        if (
            _finite_metric(final_test, "test_loss") == float("inf")
            or _finite_metric(final_test, "test_val_cp") == float("inf")
        ):
            return {
                "error": "final_test contains missing or non-finite metrics",
                "best_epoch": best_epoch,
                "provenance": provenance,
            }
        try:
            test_samples = int(final_test.get("test_samples", 0))
        except (TypeError, ValueError):
            test_samples = 0
        if test_samples <= 0:
            return {
                "error": "final_test contains no samples",
                "best_epoch": best_epoch,
                "provenance": provenance,
            }
        result.update(
            {
                "final_test_loss": final_test.get("test_loss"),
                "final_test_cp": final_test.get("test_val_cp"),
                "final_test_samples": test_samples,
            }
        )
    return result

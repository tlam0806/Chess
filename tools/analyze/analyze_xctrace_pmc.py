#!/usr/bin/env python3
"""Summarize raw Instruments CPU-counter samples for one target process."""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


DEFAULT_EVENTS = (
    "FIXED_CYCLES",
    "FIXED_INSTRUCTIONS",
    "ARM_L1D_CACHE_RD",
    "ARM_L1D_CACHE_REFILL",
    "ARM_L1D_CACHE_LMISS_RD",
)


def resolve(element: ET.Element, values: dict[str, object], factory):
    reference = element.get("ref")
    if reference is not None:
        return values[reference]
    value = factory(element)
    identifier = element.get("id")
    if identifier is not None:
        values[identifier] = value
    return value


def parse_probe_count(path: Path | None) -> int | None:
    if path is None:
        return None
    match = re.search(r"\bprobes=(\d+)", path.read_text())
    return int(match.group(1)) if match else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("xml", type=Path)
    parser.add_argument("--process", required=True)
    parser.add_argument("--benchmark-output", type=Path)
    parser.add_argument("--probes", type=int)
    parser.add_argument("--events", nargs="+", default=DEFAULT_EVENTS)
    args = parser.parse_args()

    root = ET.parse(args.xml).getroot()
    thread_values: dict[str, bool] = {}
    core_values: dict[str, str] = {}
    counter_values: dict[str, tuple[int, ...]] = {}
    totals = [0] * len(args.events)
    samples = 0
    core_samples: Counter[str] = Counter()

    for row in root.iter("row"):
        thread = row.find("thread")
        counters = row.find("pmc-events")
        if thread is None:
            continue
        is_target = resolve(
            thread,
            thread_values,
            lambda element: args.process in element.get("fmt", ""),
        )
        core = row.find("core")
        core_name = None
        if core is not None:
            core_name = resolve(
                core,
                core_values,
                lambda element: element.get("fmt", "unknown"),
            )
        if counters is None:
            continue
        counter_tuple = resolve(
            counters,
            counter_values,
            lambda element: tuple(int(value) for value in (element.text or "").split()),
        )
        if not is_target:
            continue
        if len(counter_tuple) != len(totals):
            raise ValueError(
                f"expected {len(totals)} counters, found {len(counter_tuple)}"
            )
        for index, value in enumerate(counter_tuple):
            totals[index] += value
        if core_name is not None:
            core_samples[core_name] += 1
        samples += 1

    probes = args.probes or parse_probe_count(args.benchmark_output)
    event_totals = dict(zip(args.events, totals, strict=True))
    result: dict[str, object] = {
        "process": args.process,
        "samples": samples,
        "core_samples": dict(core_samples),
        "totals": event_totals,
    }
    if probes is not None:
        result["probes"] = probes
        result["per_probe"] = {
            event: total / probes for event, total in event_totals.items()
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Profile the canonical V41 UCI workload with a native macOS sampler.

The Heroku profiler uses gperftools, which is not a useful local twin on
Apple Silicon.  This harness keeps the benchmark protocol identical while
using either Instruments Time Profiler through ``xctrace`` (the default) or
the legacy ``/usr/bin/sample`` backend:

* the canonical suite, deployment options, engine and model are hashed;
* every position/round gets a fresh engine process;
* control and profiled rounds use an A-B-B-A schedule;
* model loading, UCI setup and position setup are outside the timed region;
* the profiler is armed before ``go`` and timing ends at ``bestmove``;
* wall time, live process CPU time, search signatures and raw reports survive.

``xctrace`` provides an exact Darwin-notification start barrier.  The exported
``time-profile`` table is reduced to exclusive leaf weights, including an
explicit ``other`` symbol for rows without a stack.  A compact JSON export is
validated before bulky trace bundles are removed; a small requested subset is
retained for audit.  ``sample`` has no equivalent programmatic signal, so its
fallback path keeps the existing PTY attach barrier and audited stdin-idle
filter.
"""

from __future__ import annotations

import argparse
import ast
import ctypes
import datetime as dt
import errno
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import pty
import queue
import re
import select
import shutil
import signal
import statistics
import struct
import subprocess
import sys
import time
import threading
import uuid
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Any, Sequence


PROGRESS_SENTINEL = "CHESS_LOCAL_PROFILE_PROGRESS="
RESULT_SENTINEL = "CHESS_LOCAL_PROFILE_RESULT="
ERROR_SENTINEL = "CHESS_LOCAL_PROFILE_ERROR="
OPTION_RE = re.compile(r"^option name (.+?) type ")
UCI_MOVE_RE = re.compile(r"(?:[a-h][1-8]){2}[nbrq]?")
SAMPLE_ROOT_RE = re.compile(r"^ {4}(\d+) Thread_")
SAMPLE_FLAT_RE = re.compile(r"^ {8}(.+?)\s{2,}(\d+)\s*$")
SAMPLE_IMAGE_SUFFIX_RE = re.compile(r"\s+\(in .+\)$")
SAMPLE_CALL_NODE_RE = re.compile(
    r"^(?P<prefix>[ !:+|]*)(?P<count>\d+)\s+(?P<rest>.+)$"
)
IDLE_STDIN_LEAF_SYMBOLS = frozenset({"__read_nocancel", "__read", "read"})
XCTRACE_NO_STACK_SYMBOL = "[xctrace no stack]"
XCTRACE_SEARCH_ROOT_SYMBOL_PREFIX = (
    "chess::NnueSearcherV38::search_best_move_impl("
)
XCTRACE_MIN_SEARCH_CPU_COVERAGE = 0.90
XCTRACE_MAX_SEARCH_CPU_COVERAGE = 1.10
XCTRACE_TARGET_MIN_SEARCH_CPU_COVERAGE = 0.95
XCTRACE_TARGET_MAX_SEARCH_CPU_COVERAGE = 1.05
XCTRACE_TIME_PROFILE_XPATH = (
    '/trace-toc/run[@number="1"]/data/table[@schema="time-profile"]'
)


class ProfileError(RuntimeError):
    """A profiling, UCI protocol or integrity failure."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def utc_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def directory_tree_sha256(path: Path) -> str:
    """Hash file names and contents in a trace bundle deterministically."""

    if not path.is_dir():
        raise ProfileError(f"expected directory bundle: {path}")
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = child.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with child.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def directory_size_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def write_validated_json(path: Path, payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(serialized, encoding="utf-8")
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProfileError(f"cannot validate compact profile JSON: {path}") from error
    if loaded != payload:
        raise ProfileError(f"compact profile JSON failed round-trip validation: {path}")
    return sha256_file(path)


def write_durable_json(path: Path, payload: dict[str, Any]) -> str:
    """Atomically replace a JSON checkpoint and fsync file plus directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProfileError(f"cannot validate durable checkpoint: {path}") from error
    if loaded != payload:
        raise ProfileError(f"durable checkpoint failed validation: {path}")
    return sha256_file(path)


def case_key(round_index: int, sequence: int, position_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", position_id):
        raise ProfileError(f"unsafe position id for checkpoint: {position_id!r}")
    return f"r{round_index:02d}-{sequence:02d}-{position_id}"


def load_case_checkpoints(case_dir: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted(case_dir.glob("*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ProfileError(f"invalid case checkpoint: {path}") from error
        if document.get("schema_version") != 1:
            raise ProfileError(f"unsupported case checkpoint schema: {path}")
        key = document.get("case_key")
        row = document.get("row")
        if not isinstance(key, str) or not isinstance(row, dict):
            raise ProfileError(f"malformed case checkpoint: {path}")
        if path.stem != key:
            raise ProfileError(f"case checkpoint filename/key mismatch: {path}")
        if key in rows:
            raise ProfileError(f"duplicate case checkpoint: {key}")
        rows[key] = row
    return rows


def validate_case_checkpoint_row(
    key: str,
    row: dict[str, Any],
    expected: dict[str, Any],
    requested_depth: int,
) -> None:
    for field in ("round", "sequence", "mode", "position_id", "fen"):
        if row.get(field) != expected[field]:
            raise ProfileError(f"checkpoint {key} has mismatched {field}")
    if row.get("reported_depth") != requested_depth:
        raise ProfileError(f"checkpoint {key} did not complete requested depth")
    sample = row.get("sample")
    if expected["mode"] == "profile":
        if not isinstance(sample, dict):
            raise ProfileError(f"profile checkpoint {key} has no sample payload")
        raw_path_value = sample.get("raw_path")
        raw_sha256 = sample.get("raw_sha256")
        if not isinstance(raw_path_value, str) or not isinstance(raw_sha256, str):
            raise ProfileError(f"profile checkpoint {key} has no raw artifact hash")
        raw_path = Path(raw_path_value)
        if not raw_path.is_file() or sha256_file(raw_path) != raw_sha256:
            raise ProfileError(f"profile checkpoint {key} raw artifact changed")
        trace_path_value = sample.get("trace_path")
        if trace_path_value is not None:
            trace_path = Path(str(trace_path_value))
            expected_tree_hash = sample.get("trace_tree_sha256")
            if (
                not trace_path.is_dir()
                or not isinstance(expected_tree_hash, str)
                or directory_tree_sha256(trace_path) != expected_tree_hash
            ):
                raise ProfileError(f"profile checkpoint {key} trace bundle changed")
    elif sample is not None:
        raise ProfileError(f"control checkpoint {key} unexpectedly has samples")


def validate_resume_contract(
    saved: dict[str, Any], current: dict[str, Any]
) -> None:
    if saved == current:
        return
    differing = sorted(
        key
        for key in set(saved) | set(current)
        if saved.get(key) != current.get(key)
    )
    raise ProfileError(
        "resume contract differs from checkpoint in: " + ", ".join(differing)
    )


def archive_uncheckpointed_raw_artifacts(
    raw_dir: Path,
    raw_stem: str,
    failed_attempts_dir: Path,
    attempt_number: int,
) -> list[str]:
    """Move retry-stem artifacts aside; never delete evidence of a failure."""

    matches = sorted(
        path
        for path in raw_dir.iterdir()
        if path.name == raw_stem or path.name.startswith(raw_stem + ".")
    )
    if not matches:
        return []
    destination = failed_attempts_dir / (
        f"attempt-{attempt_number:02d}-{raw_stem}-{uuid.uuid4().hex[:8]}"
    )
    destination.mkdir(parents=True, exist_ok=False)
    moved: list[str] = []
    for path in matches:
        target = destination / path.name
        shutil.move(str(path), str(target))
        moved.append(str(target))
    return moved


def option_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def parse_yaml_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        raise ProfileError("empty YAML scalar in uci_options")
    if value[0] in {'"', "'"}:
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError) as error:
            raise ProfileError(f"invalid quoted YAML scalar: {value}") from error
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "~"}:
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def read_config_uci_options(path: Path) -> dict[str, Any]:
    """Read the deployment YAML's simple scalar ``uci_options`` mapping."""
    result: dict[str, Any] = {}
    section_indent: int | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if section_indent is None:
            if stripped == "uci_options:":
                section_indent = indent
            continue
        if indent <= section_indent:
            break
        if ":" not in stripped:
            raise ProfileError(f"invalid uci_options YAML line: {raw_line}")
        name, value = stripped.split(":", 1)
        result[name.strip()] = parse_yaml_scalar(value)
    if section_indent is None or not result:
        raise ProfileError(f"config has no uci_options mapping: {path}")
    return result


def validate_config_options(
    config_path: Path,
    spec_options: dict[str, Any],
) -> dict[str, Any]:
    config_options = read_config_uci_options(config_path)
    normalized_spec = {
        name: option_value(value) for name, value in spec_options.items()
    }
    normalized_config = {
        name: option_value(value) for name, value in config_options.items()
    }
    if normalized_config != normalized_spec:
        missing = sorted(set(normalized_spec) - set(normalized_config))
        extra = sorted(set(normalized_config) - set(normalized_spec))
        changed = sorted(
            name
            for name in set(normalized_spec) & set(normalized_config)
            if normalized_spec[name] != normalized_config[name]
        )
        raise ProfileError(
            "profile suite/config UCI options differ: "
            f"missing={missing}, extra={extra}, changed={changed}"
        )
    return {"status": "match", "options": normalized_config}


def parse_info_line(line: str) -> dict[str, Any] | None:
    tokens = line.split()
    if not tokens or tokens[0] != "info":
        return None
    parsed: dict[str, Any] = {"raw_info": line}
    index = 1
    while index < len(tokens):
        token = tokens[index]
        try:
            if token == "depth" and index + 1 < len(tokens):
                parsed["reported_depth"] = int(tokens[index + 1])
                index += 2
                continue
            if token == "nodes" and index + 1 < len(tokens):
                parsed["nodes"] = int(tokens[index + 1])
                index += 2
                continue
            if token == "score" and index + 2 < len(tokens):
                parsed["score_type"] = tokens[index + 1]
                parsed["score_value"] = int(tokens[index + 2])
                index += 3
                continue
        except ValueError as error:
            raise ProfileError(f"malformed UCI info line: {line}") from error
        index += 1
    required = {"reported_depth", "nodes", "score_type", "score_value"}
    return parsed if required.issubset(parsed) else None


def mach_ticks_to_seconds(ticks: int, numerator: int, denominator: int) -> float:
    if denominator <= 0:
        raise ValueError("Mach timebase denominator must be positive")
    return ticks * (numerator / denominator) / 1_000_000_000.0


@functools.lru_cache(maxsize=1)
def mach_timebase() -> tuple[int, int]:
    class MachTimebaseInfo(ctypes.Structure):
        _fields_ = [
            ("numerator", ctypes.c_uint32),
            ("denominator", ctypes.c_uint32),
        ]

    libsystem = ctypes.CDLL(None)
    function = libsystem.mach_timebase_info
    function.argtypes = [ctypes.POINTER(MachTimebaseInfo)]
    function.restype = ctypes.c_int
    info = MachTimebaseInfo()
    if function(ctypes.byref(info)) != 0 or info.denominator == 0:
        raise ProfileError("mach_timebase_info failed")
    return int(info.numerator), int(info.denominator)


def process_cpu_seconds(pid: int) -> float:
    """Return live user+system CPU seconds for ``pid`` via ``libproc``."""
    try:
        libproc = ctypes.CDLL("libproc.dylib", use_errno=True)
    except OSError as error:
        raise ProfileError("libproc.dylib is required for process CPU timing") from error
    function = libproc.proc_pid_rusage
    function.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    function.restype = ctypes.c_int
    # rusage_info_v4 is currently 280 bytes.  Keep spare space so a future SDK
    # cannot make a read overflow this private Python buffer.
    buffer = ctypes.create_string_buffer(512)
    if function(pid, 4, ctypes.byref(buffer)) != 0:  # RUSAGE_INFO_V4
        error_number = ctypes.get_errno()
        raise ProfileError(
            f"proc_pid_rusage({pid}) failed: {os.strerror(error_number)}"
        )
    user_ticks, system_ticks = struct.unpack_from("@QQ", buffer.raw, 16)
    numerator, denominator = mach_timebase()
    return mach_ticks_to_seconds(user_ticks + system_ticks, numerator, denominator)


def position_order(
    positions: Sequence[dict[str, str]], round_index: int
) -> list[dict[str, str]]:
    if not positions:
        return []
    offset = (round_index // 2) % len(positions)
    rotated = list(positions[offset:]) + list(positions[:offset])
    return list(reversed(rotated)) if round_index % 2 else rotated


def signature(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["reported_depth"],
        row["score_type"],
        row["score_value"],
        row["nodes"],
        row["bestmove"],
    )


def parse_sample_report(text: str, interval_ms: int) -> dict[str, Any]:
    """Parse exact top-of-stack counts from a macOS ``sample`` report.

    The collapsed-flat section omits symbols with fewer than five samples,
    but the call graph retains those branches.  A call-graph node is a leaf
    when the next node's numeric count column is not farther right.  Summing
    those leaves reconstructs the exact flat distribution, including 1--4
    sample symbols.
    """
    if interval_ms <= 0:
        raise ValueError("sample interval must be positive")
    root_samples = 0
    in_call_graph = False
    in_flat_section = False
    call_nodes: list[tuple[int, int, str]] = []
    collapsed_symbols: defaultdict[str, int] = defaultdict(int)
    for line in text.splitlines():
        if line == "Call graph:":
            in_call_graph = True
            continue
        if line.startswith("Total number in stack"):
            in_call_graph = False
        root_match = SAMPLE_ROOT_RE.match(line)
        if root_match:
            root_samples += int(root_match.group(1))
        if in_call_graph:
            node_match = SAMPLE_CALL_NODE_RE.match(line)
            if node_match is not None:
                rest = node_match.group("rest")
                symbol = rest.split("  (in ", 1)[0].strip()
                call_nodes.append(
                    (
                        node_match.start("count"),
                        int(node_match.group("count")),
                        symbol,
                    )
                )
        if line.startswith("Sort by top of stack, same collapsed"):
            in_flat_section = True
            continue
        if in_flat_section and line.startswith("Binary Images:"):
            break
        if not in_flat_section:
            continue
        match = SAMPLE_FLAT_RE.match(line)
        if match is None:
            continue
        symbol = SAMPLE_IMAGE_SUFFIX_RE.sub("", match.group(1)).strip()
        if symbol:
            collapsed_symbols[symbol] += int(match.group(2))
    if root_samples <= 0:
        raise ProfileError("sample report contains no thread-root sample count")
    symbols: defaultdict[str, int] = defaultdict(int)
    for index, (count_column, count, symbol) in enumerate(call_nodes):
        is_leaf = index + 1 == len(call_nodes) or call_nodes[index + 1][0] <= count_column
        if is_leaf:
            symbols[symbol] += count
    leaf_samples = sum(symbols.values())
    if leaf_samples != root_samples:
        raise ProfileError(
            "call-graph leaf counts do not match roots: "
            f"leaves={leaf_samples}, roots={root_samples}"
        )
    period = interval_ms / 1000.0
    excluded_idle_samples = sum(
        count for name, count in symbols.items() if name in IDLE_STDIN_LEAF_SYMBOLS
    )
    search_samples = root_samples - excluded_idle_samples
    # A very small position can finish between two 1 ms profiler ticks.  Keep
    # that case as a real zero-sample observation; the enclosing complete
    # round is rejected below if all positions together still have no search
    # samples.  The case's measured CPU time and nodes remain in the round.
    if search_samples < 0:
        raise ProfileError("idle samples exceed the call-graph root count")
    search_symbols = {
        name: count
        for name, count in symbols.items()
        if name not in IDLE_STDIN_LEAF_SYMBOLS
    }
    symbol_rows = [
        {
            "symbol": name,
            "flat_samples": count,
            "flat_seconds": count * period,
        }
        for name, count in sorted(
            search_symbols.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    return {
        "sample_period_seconds": period,
        "total_samples": search_samples,
        "total_sampled_thread_seconds": search_samples * period,
        "flat_samples_listed": sum(search_symbols.values()),
        "raw_total_samples": root_samples,
        "raw_flat_samples_listed": sum(collapsed_symbols.values()),
        "callgraph_leaf_samples": leaf_samples,
        "excluded_pre_go_idle_samples": excluded_idle_samples,
        "idle_leaf_symbols": sorted(IDLE_STDIN_LEAF_SYMBOLS),
        "symbols": symbol_rows,
        "note": (
            "pre-go stdin-read arming samples are excluded; symbols come from "
            "call-graph leaves, which retain counts below the collapsed-flat threshold"
        ),
    }


def parse_xctrace_time_profile_xml(
    text: str, included_pids: set[int] | None = None
) -> dict[str, Any]:
    """Parse exclusive/self weights from an exported Time Profiler table.

    ``xctrace export`` de-duplicates XML values through global ``id``/``ref``
    attributes.  The first frame in each resolved tagged backtrace is the leaf
    (exclusive/self) frame.  Rows without a usable stack retain both their row
    count and weight under ``XCTRACE_NO_STACK_SYMBOL`` so downstream analysis
    never silently renormalizes missing samples away.
    """

    try:
        root = ET.fromstring(text)
    except ET.ParseError as error:
        raise ProfileError(f"invalid xctrace Time Profiler XML: {error}") from error

    schemas = [
        element
        for element in root.iter("schema")
        if element.attrib.get("name") == "time-profile"
    ]
    if len(schemas) != 1:
        raise ProfileError(
            "xctrace export must contain exactly one time-profile schema"
        )

    elements_by_id: dict[str, ET.Element] = {}
    for element in root.iter():
        identifier = element.attrib.get("id")
        if identifier is None:
            continue
        if identifier in elements_by_id:
            raise ProfileError(f"duplicate xctrace XML id: {identifier}")
        elements_by_id[identifier] = element

    def resolve(element: ET.Element | None) -> ET.Element | None:
        seen: set[str] = set()
        while element is not None and "ref" in element.attrib:
            reference = element.attrib["ref"]
            if reference in seen:
                raise ProfileError(f"cyclic xctrace XML reference: {reference}")
            seen.add(reference)
            try:
                element = elements_by_id[reference]
            except KeyError as error:
                raise ProfileError(
                    f"unresolved xctrace XML reference: {reference}"
                ) from error
        return element

    def resolved_child(element: ET.Element | None, tag: str) -> ET.Element | None:
        element = resolve(element)
        return resolve(element.find(tag)) if element is not None else None

    symbol_rows: defaultdict[str, int] = defaultdict(int)
    symbol_weights_ns: defaultdict[str, int] = defaultdict(int)
    row_weights_ns: list[int] = []
    sample_times_ns: list[int] = []
    all_rows = 0
    all_weight_ns = 0
    unknown_pid_rows = 0
    unknown_pid_weight_ns = 0
    pid_rows: defaultdict[int, int] = defaultdict(int)
    pid_weights_ns: defaultdict[int, int] = defaultdict(int)
    target_pid_rows: defaultdict[int, int] = defaultdict(int)
    target_pid_weights_ns: defaultdict[int, int] = defaultdict(int)
    selected_pid_rows: defaultdict[int, int] = defaultdict(int)
    selected_pid_weights_ns: defaultdict[int, int] = defaultdict(int)
    target_non_search_rows = 0
    target_non_search_weight_ns = 0
    process_thread_pid_mismatch_rows = 0
    process_thread_pid_mismatch_weight_ns = 0
    no_stack_rows = 0
    no_stack_weight_ns = 0
    for row_index, row in enumerate(root.iter("row")):
        weight = resolved_child(row, "weight")
        if weight is None or weight.text is None:
            raise ProfileError(f"xctrace row {row_index} has no sample weight")
        try:
            weight_ns = int(weight.text.strip())
        except ValueError as error:
            raise ProfileError(
                f"xctrace row {row_index} has invalid sample weight"
            ) from error
        if weight_ns <= 0:
            raise ProfileError(
                f"xctrace row {row_index} has non-positive sample weight"
            )
        all_rows += 1
        all_weight_ns += weight_ns

        def process_pid(process: ET.Element | None, source: str) -> int | None:
            pid_element = resolved_child(process, "pid")
            if pid_element is None or not pid_element.text:
                return None
            try:
                return int(pid_element.text.strip())
            except ValueError as error:
                raise ProfileError(
                    f"xctrace row {row_index} has invalid {source} process id"
                ) from error

        # Never infer a PID from the human-readable ``fmt`` attribute.  Resolve
        # both schema paths independently and prove they agree whenever the
        # export supplies both.
        direct_pid = process_pid(resolved_child(row, "process"), "row")
        thread = resolved_child(row, "thread")
        thread_pid = process_pid(resolved_child(thread, "process"), "thread")
        if (
            direct_pid is not None
            and thread_pid is not None
            and direct_pid != thread_pid
        ):
            process_thread_pid_mismatch_rows += 1
            process_thread_pid_mismatch_weight_ns += weight_ns
            continue
        pid = direct_pid if direct_pid is not None else thread_pid
        if pid is None:
            unknown_pid_rows += 1
            unknown_pid_weight_ns += weight_ns
        else:
            pid_rows[pid] += 1
            pid_weights_ns[pid] += weight_ns
        tagged_backtrace = resolved_child(row, "tagged-backtrace")
        backtrace = resolved_child(tagged_backtrace, "backtrace")
        frames: list[ET.Element] = []
        if backtrace is not None:
            frames = [
                resolved
                for frame in backtrace.findall("frame")
                if (resolved := resolve(frame)) is not None
            ]
        frame_names = [
            name
            for frame in frames
            if (name := frame.attrib.get("name")) is not None
        ]
        has_search_root = any(
            name.startswith(XCTRACE_SEARCH_ROOT_SYMBOL_PREFIX)
            for name in frame_names
        )
        if included_pids is not None:
            if pid not in included_pids:
                continue
            assert pid is not None
            target_pid_rows[pid] += 1
            target_pid_weights_ns[pid] += weight_ns
            if not has_search_root:
                target_non_search_rows += 1
                target_non_search_weight_ns += weight_ns
                continue

        row_weights_ns.append(weight_ns)
        if pid is not None:
            selected_pid_rows[pid] += 1
            selected_pid_weights_ns[pid] += weight_ns

        sample_time = resolved_child(row, "sample-time")
        if sample_time is not None and sample_time.text:
            try:
                sample_times_ns.append(int(sample_time.text.strip()))
            except ValueError as error:
                raise ProfileError(
                    f"xctrace row {row_index} has invalid sample time"
                ) from error

        leaf: ET.Element | None = None
        if frames:
            leaf = frames[0]
        symbol = leaf.attrib.get("name") if leaf is not None else None
        if not symbol:
            symbol = XCTRACE_NO_STACK_SYMBOL
            no_stack_rows += 1
            no_stack_weight_ns += weight_ns
        symbol_rows[symbol] += 1
        symbol_weights_ns[symbol] += weight_ns

    if process_thread_pid_mismatch_rows:
        raise ProfileError(
            "xctrace row/thread process PID mismatch: "
            f"rows={process_thread_pid_mismatch_rows}, "
            f"weight_ns={process_thread_pid_mismatch_weight_ns}"
        )

    total_rows = len(row_weights_ns)
    total_weight_ns = sum(row_weights_ns)
    listed_weight_ns = sum(symbol_weights_ns.values())
    if listed_weight_ns != total_weight_ns:
        raise ProfileError(
            "xctrace exclusive weights do not match total row weight: "
            f"flat={listed_weight_ns}, total={total_weight_ns}"
        )
    if included_pids is not None:
        target_weight_ns = sum(target_pid_weights_ns.values())
        if total_weight_ns + target_non_search_weight_ns != target_weight_ns:
            raise ProfileError(
                "xctrace target weight partition is not exact: "
                f"selected={total_weight_ns}, "
                f"non_search={target_non_search_weight_ns}, "
                f"target={target_weight_ns}"
            )
        if total_rows + target_non_search_rows != sum(target_pid_rows.values()):
            raise ProfileError("xctrace target row partition is not exact")
        if no_stack_rows:
            raise ProfileError(
                "xctrace selected search-stack rows contain a no-stack sentinel"
            )
    symbolized_rows = total_rows - no_stack_rows
    symbolized_weight_ns = total_weight_ns - no_stack_weight_ns
    requested_pids = sorted(included_pids) if included_pids is not None else None
    observed_requested_pids = (
        sorted(pid for pid in included_pids if selected_pid_rows.get(pid, 0) > 0)
        if included_pids is not None
        else None
    )
    symbols = [
        {
            "symbol": name,
            "flat_samples": symbol_rows[name],
            "flat_weight_ns": weight_ns,
            "flat_seconds": weight_ns / 1_000_000_000.0,
        }
        for name, weight_ns in sorted(
            symbol_weights_ns.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    return {
        "sample_period_seconds": (
            statistics.median(row_weights_ns) / 1_000_000_000.0
            if row_weights_ns
            else None
        ),
        "total_sampled_seconds": total_weight_ns / 1_000_000_000.0,
        "total_weight_ns": total_weight_ns,
        "sample_rows": total_rows,
        "symbolized_rows": symbolized_rows,
        "symbolized_weight_ns": symbolized_weight_ns,
        "no_stack_rows": no_stack_rows,
        "no_stack_weight_ns": no_stack_weight_ns,
        "no_stack_symbol": XCTRACE_NO_STACK_SYMBOL,
        "flat_weight_ns_listed": listed_weight_ns,
        "first_sample_time_ns": min(sample_times_ns) if sample_times_ns else None,
        "last_sample_time_ns": max(sample_times_ns) if sample_times_ns else None,
        "sample_time_span_ns": (
            max(sample_times_ns) - min(sample_times_ns)
            if len(sample_times_ns) >= 2
            else 0
        ),
        "symbols": symbols,
        "pid_filter": {
            "enabled": included_pids is not None,
            "kind": (
                "pid-and-search-root-stack"
                if included_pids is not None
                else "none"
            ),
            "search_root_symbol_prefix": (
                XCTRACE_SEARCH_ROOT_SYMBOL_PREFIX
                if included_pids is not None
                else None
            ),
            "requested_pids": requested_pids,
            "observed_requested_pids": observed_requested_pids,
            "observed_target_pids": (
                sorted(pid for pid in included_pids if target_pid_rows.get(pid, 0) > 0)
                if included_pids is not None
                else None
            ),
            "missing_requested_pids": (
                sorted(set(included_pids) - set(observed_requested_pids or []))
                if included_pids is not None
                else None
            ),
            "all_rows": all_rows,
            "all_weight_ns": all_weight_ns,
            "included_rows": total_rows,
            "included_weight_ns": total_weight_ns,
            "excluded_rows": all_rows - total_rows,
            "excluded_weight_ns": all_weight_ns - total_weight_ns,
            "target_pid_rows": sum(target_pid_rows.values()),
            "target_pid_weight_ns": sum(target_pid_weights_ns.values()),
            "target_pid_non_search_rows": target_non_search_rows,
            "target_pid_non_search_weight_ns": target_non_search_weight_ns,
            "unknown_pid_rows": unknown_pid_rows,
            "unknown_pid_weight_ns": unknown_pid_weight_ns,
            "process_thread_pid_mismatch_rows": 0,
            "process_thread_pid_mismatch_weight_ns": 0,
            "observed_pid_count": len(pid_rows),
            "included_pid_rows": [
                {
                    "pid": pid,
                    "rows": selected_pid_rows.get(pid, 0),
                    "weight_ns": selected_pid_weights_ns.get(pid, 0),
                }
                for pid in (requested_pids or sorted(pid_rows))
                if selected_pid_rows.get(pid, 0) > 0
            ],
            "target_pid_rows_by_pid": [
                {
                    "pid": pid,
                    "rows": target_pid_rows.get(pid, 0),
                    "weight_ns": target_pid_weights_ns.get(pid, 0),
                }
                for pid in (requested_pids or [])
                if target_pid_rows.get(pid, 0) > 0
            ],
        },
        "source": "xctrace time-profile exclusive leaf weights",
        "note": (
            "first resolved frame is the exclusive leaf; batch exports require "
            "both an exact target PID and the audited search root in the full "
            "stack; no-stack rows are preserved as other only for unfiltered "
            "per-case exports"
        ),
    }


def merge_xctrace_summaries(
    summaries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Merge exact Time Profiler weights without converting them to counts."""

    symbol_weights_ns: defaultdict[str, int] = defaultdict(int)
    symbol_rows: defaultdict[str, int] = defaultdict(int)
    total_weight_ns = 0
    sample_rows = 0
    symbolized_rows = 0
    symbolized_weight_ns = 0
    no_stack_rows = 0
    no_stack_weight_ns = 0
    for summary in summaries:
        total_weight_ns += int(summary["total_weight_ns"])
        sample_rows += int(summary["sample_rows"])
        symbolized_rows += int(summary["symbolized_rows"])
        symbolized_weight_ns += int(summary["symbolized_weight_ns"])
        no_stack_rows += int(summary["no_stack_rows"])
        no_stack_weight_ns += int(summary["no_stack_weight_ns"])
        for row in summary["symbols"]:
            name = str(row["symbol"])
            symbol_rows[name] += int(row["flat_samples"])
            symbol_weights_ns[name] += int(row["flat_weight_ns"])
    if sum(symbol_weights_ns.values()) != total_weight_ns:
        raise ProfileError("merged xctrace leaf weights do not match total weight")
    return {
        "sample_period_seconds": (
            statistics.median(
                float(summary["sample_period_seconds"])
                for summary in summaries
                if summary.get("sample_period_seconds") is not None
            )
            if any(
                summary.get("sample_period_seconds") is not None
                for summary in summaries
            )
            else None
        ),
        "total_sampled_seconds": total_weight_ns / 1_000_000_000.0,
        "total_weight_ns": total_weight_ns,
        "sample_rows": sample_rows,
        "symbolized_rows": symbolized_rows,
        "symbolized_weight_ns": symbolized_weight_ns,
        "no_stack_rows": no_stack_rows,
        "no_stack_weight_ns": no_stack_weight_ns,
        "no_stack_symbol": XCTRACE_NO_STACK_SYMBOL,
        "flat_weight_ns_listed": total_weight_ns,
        "symbols": [
            {
                "symbol": name,
                "flat_samples": symbol_rows[name],
                "flat_weight_ns": weight_ns,
                "flat_seconds": weight_ns / 1_000_000_000.0,
            }
            for name, weight_ns in sorted(
                symbol_weights_ns.items(), key=lambda item: (-item[1], item[0])
            )
        ],
        "profiles": len(summaries),
        "source": "merged xctrace time-profile exclusive leaf weights",
        "note": "sum of exact per-process Time Profiler row weights",
    }


def merge_sample_summaries(
    summaries: Sequence[dict[str, Any]], interval_ms: int
) -> dict[str, Any]:
    counts: defaultdict[str, int] = defaultdict(int)
    total_samples = 0
    flat_samples_listed = 0
    raw_total_samples = 0
    raw_flat_samples_listed = 0
    excluded_idle_samples = 0
    for summary in summaries:
        total_samples += int(summary["total_samples"])
        flat_samples_listed += int(summary["flat_samples_listed"])
        raw_total_samples += int(summary["raw_total_samples"])
        raw_flat_samples_listed += int(summary["raw_flat_samples_listed"])
        excluded_idle_samples += int(summary["excluded_pre_go_idle_samples"])
        for row in summary["symbols"]:
            counts[str(row["symbol"])] += int(row["flat_samples"])
    period = interval_ms / 1000.0
    return {
        "sample_period_seconds": period,
        "total_samples": total_samples,
        "total_sampled_thread_seconds": total_samples * period,
        "flat_samples_listed": flat_samples_listed,
        "raw_total_samples": raw_total_samples,
        "raw_flat_samples_listed": raw_flat_samples_listed,
        "callgraph_leaf_samples": raw_total_samples,
        "excluded_pre_go_idle_samples": excluded_idle_samples,
        "idle_leaf_symbols": sorted(IDLE_STDIN_LEAF_SYMBOLS),
        "symbols": [
            {
                "symbol": name,
                "flat_samples": count,
                "flat_seconds": count * period,
            }
            for name, count in sorted(
                counts.items(), key=lambda item: (-item[1], item[0])
            )
        ],
        "profiles": len(summaries),
        "note": "sum of exact per-process call-graph leaf counts after audited idle filtering",
    }


def aggregate_round(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    nodes = sum(int(row["nodes"]) for row in rows)
    elapsed_ns = sum(int(row["elapsed_ns"]) for row in rows)
    process_cpu = sum(float(row["process_cpu_seconds"]) for row in rows)
    return {
        "nodes": nodes,
        "elapsed_ns": elapsed_ns,
        "process_cpu_seconds": process_cpu,
        "nps": nodes * 1_000_000_000.0 / elapsed_ns,
        "cpu_to_wall": process_cpu / (elapsed_ns / 1_000_000_000.0),
        "process_cpu_ns_per_node": process_cpu * 1_000_000_000.0 / nodes,
    }


def xctrace_record_command(
    xcrun: Path,
    pid: int,
    notification_name: str,
    trace_path: Path,
    time_limit_seconds: int,
) -> list[str]:
    if time_limit_seconds <= 0:
        raise ValueError("xctrace time limit must be positive")
    return [
        str(xcrun),
        "xctrace",
        "record",
        "--template",
        "Time Profiler",
        "--attach",
        str(pid),
        "--notify-tracing-started",
        notification_name,
        "--output",
        str(trace_path),
        "--time-limit",
        f"{time_limit_seconds}s",
        "--no-prompt",
    ]


def xctrace_all_processes_record_command(
    xcrun: Path,
    notification_name: str,
    trace_path: Path,
    time_limit_seconds: int,
) -> list[str]:
    if time_limit_seconds <= 0:
        raise ValueError("xctrace time limit must be positive")
    return [
        str(xcrun),
        "xctrace",
        "record",
        "--template",
        "Time Profiler",
        "--all-processes",
        "--notify-tracing-started",
        notification_name,
        "--output",
        str(trace_path),
        "--time-limit",
        f"{time_limit_seconds}s",
        "--no-prompt",
    ]


class MacSampler:
    """One attached ``sample`` process with a PTY-backed ready barrier."""

    def __init__(
        self,
        executable: Path,
        pid: int | None,
        output: Path,
        interval_ms: int,
        maximum_duration_seconds: int,
        attach_timeout_seconds: float,
    ) -> None:
        self.output = output
        self.attach_timeout_seconds = attach_timeout_seconds
        self.started_ns = time.perf_counter_ns()
        master, slave = pty.openpty()
        self.master_fd = master
        try:
            self.process = subprocess.Popen(
                [
                    str(executable),
                    str(pid),
                    str(maximum_duration_seconds),
                    str(interval_ms),
                    "-mayDie",
                    "-fullPaths",
                    "-file",
                    str(output),
                ],
                stdin=subprocess.DEVNULL,
                stdout=slave,
                stderr=slave,
                close_fds=True,
            )
        finally:
            os.close(slave)
        self.console = bytearray()
        self.ready_ns: int | None = None

    def _read_available(self, timeout: float) -> None:
        readable, _, _ = select.select([self.master_fd], [], [], max(0.0, timeout))
        if not readable:
            return
        try:
            chunk = os.read(self.master_fd, 65536)
        except OSError as error:
            if error.errno != errno.EIO:
                raise
            chunk = b""
        self.console.extend(chunk)

    def wait_until_attached(self) -> int:
        deadline = time.monotonic() + self.attach_timeout_seconds
        while b"Sampling process " not in self.console:
            if self.process.poll() is not None:
                self._read_available(0)
                raise ProfileError(
                    "sample exited before attach: "
                    + self.console.decode("utf-8", errors="replace")[-2000:]
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.abort()
                raise ProfileError("timed out waiting for sample to attach")
            self._read_available(min(0.1, remaining))
        self.ready_ns = time.perf_counter_ns()
        return self.ready_ns

    def stop(self, timeout_seconds: float) -> tuple[str, int]:
        stop_call_ns = time.perf_counter_ns()
        deadline = time.monotonic() + timeout_seconds
        # With -mayDie the profiler normally stops itself when the fresh
        # target exits.  Give it a short grace window to finish symbolization;
        # an immediate SIGINT can race that phase and make an otherwise
        # complete report exit via SIGKILL on macOS 26.
        grace_deadline = min(deadline, time.monotonic() + 2.0)
        while self.process.poll() is None and time.monotonic() < grace_deadline:
            self._read_available(min(0.1, grace_deadline - time.monotonic()))
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
        while self.process.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                tail = self.console.decode("utf-8", errors="replace")[-2000:]
                self.abort()
                raise ProfileError(
                    "sample did not flush after SIGINT; console tail: " + tail
                )
            self._read_available(min(0.1, remaining))
        while True:
            before = len(self.console)
            self._read_available(0)
            if len(self.console) == before:
                break
        os.close(self.master_fd)
        console = self.console.decode("utf-8", errors="replace")
        if self.process.returncode != 0:
            raise ProfileError(
                f"sample failed ({self.process.returncode}): {console[-2000:]}"
            )
        if not self.output.is_file() or self.output.stat().st_size == 0:
            raise ProfileError("sample produced no raw report")
        return console, stop_call_ns

    def abort(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait(timeout=5)
        try:
            os.close(self.master_fd)
        except OSError:
            pass


class XcTraceSampler:
    """One attached Instruments Time Profiler with an exact start notify."""

    def __init__(
        self,
        xcrun: Path,
        notifyutil: Path,
        pid: int | None,
        trace_path: Path,
        compact_path: Path,
        attach_timeout_seconds: float,
        notification_registration_delay_ms: int,
        time_limit_seconds: int,
        retain_trace: bool,
        all_processes: bool = False,
    ) -> None:
        if trace_path.exists() or compact_path.exists():
            raise ProfileError(f"xctrace output already exists: {trace_path}")
        if trace_path.parent != compact_path.parent:
            raise ProfileError("xctrace trace and compact output must share a directory")
        self.xcrun = xcrun
        self.notifyutil = notifyutil
        self.trace_path = trace_path
        self.compact_path = compact_path
        self.export_path = compact_path.with_suffix(".time-profile.xml")
        self.attach_timeout_seconds = attach_timeout_seconds
        self.retain_trace = retain_trace
        self.time_limit_seconds = time_limit_seconds
        self.notification_name = "dev.chess.profile." + uuid.uuid4().hex
        self.started_ns = time.perf_counter_ns()
        self.ready_ns: int | None = None
        self.notification_output = ""
        self.record_console = ""
        self.waiter: subprocess.Popen[str] | None = subprocess.Popen(
            [str(notifyutil), "-1", self.notification_name],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        # notifyutil has no registration acknowledgement.  Keep it alive for a
        # short audited interval before xctrace can possibly post the signal.
        if notification_registration_delay_ms:
            time.sleep(notification_registration_delay_ms / 1000.0)
        if self.waiter.poll() is not None:
            output, _ = self.waiter.communicate()
            self.waiter = None
            raise ProfileError(
                "notifyutil exited before xctrace launch: " + output[-2000:]
            )
        if all_processes:
            self.command = xctrace_all_processes_record_command(
                xcrun,
                self.notification_name,
                trace_path,
                time_limit_seconds,
            )
        else:
            if pid is None:
                raise ProfileError("per-case xctrace requires a target pid")
            self.command = xctrace_record_command(
                xcrun,
                pid,
                self.notification_name,
                trace_path,
                time_limit_seconds,
            )
        self.record_started_ns = time.perf_counter_ns()
        self.process: subprocess.Popen[str] | None = subprocess.Popen(
            self.command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    def wait_until_attached(self) -> int:
        deadline = time.monotonic() + self.attach_timeout_seconds
        assert self.waiter is not None
        assert self.process is not None
        while self.waiter.poll() is None:
            if self.process.poll() is not None:
                console, _ = self.process.communicate()
                self.record_console = console
                self.process = None
                self.waiter.kill()
                self.waiter.wait(timeout=5)
                self.waiter = None
                raise ProfileError(
                    "xctrace exited before start notification: " + console[-2000:]
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.abort()
                raise ProfileError("timed out waiting for xctrace start notification")
            time.sleep(min(0.01, remaining))
        notification_output, _ = self.waiter.communicate()
        self.notification_output = notification_output
        self.waiter = None
        if self.notification_name not in notification_output:
            self.abort()
            raise ProfileError(
                "notifyutil returned without the xctrace start notification: "
                + notification_output[-2000:]
            )
        if self.process.poll() is not None:
            console, _ = self.process.communicate()
            self.record_console = console
            self.process = None
            raise ProfileError(
                "xctrace exited immediately after start notification: "
                + console[-2000:]
            )
        self.ready_ns = time.perf_counter_ns()
        return self.ready_ns

    def stop(
        self,
        timeout_seconds: float,
        included_pids: set[int] | None = None,
        pid_groups: dict[str, set[int]] | None = None,
        interrupt_recording: bool = False,
    ) -> dict[str, Any]:
        stop_call_ns = time.perf_counter_ns()
        if self.process is None:
            raise ProfileError("xctrace recorder is not running")
        if interrupt_recording:
            if self.process.poll() is not None:
                raise ProfileError(
                    "xctrace batch recorder was not alive at the SIGINT barrier"
                )
            self.process.send_signal(signal.SIGINT)
        try:
            console, _ = self.process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as error:
            self.abort()
            raise ProfileError("xctrace did not flush after its time limit") from error
        returncode = self.process.returncode
        self.process = None
        self.record_console = console
        recording_completed_ns = time.perf_counter_ns()
        if returncode != 0:
            raise ProfileError(f"xctrace failed ({returncode}): {console[-2000:]}")
        reached_time_limit = "Reached specified time limit" in console
        if interrupt_recording and reached_time_limit:
            raise ProfileError(
                "xctrace batch reached its time limit instead of the SIGINT barrier"
            )
        if not self.trace_path.is_dir():
            raise ProfileError("xctrace produced no trace bundle")

        export_command = [
            str(self.xcrun),
            "xctrace",
            "export",
            "--input",
            str(self.trace_path),
            "--xpath",
            XCTRACE_TIME_PROFILE_XPATH,
            "--output",
            str(self.export_path),
        ]
        export_started_ns = time.perf_counter_ns()
        try:
            exported = subprocess.run(
                export_command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise ProfileError("xctrace export timed out") from error
        export_completed_ns = time.perf_counter_ns()
        if exported.returncode != 0:
            raise ProfileError(
                f"xctrace export failed ({exported.returncode}): "
                + exported.stdout[-2000:]
            )
        if not self.export_path.is_file() or self.export_path.stat().st_size == 0:
            raise ProfileError("xctrace export produced no Time Profiler XML")
        export_text = self.export_path.read_text(encoding="utf-8", errors="strict")
        summary = parse_xctrace_time_profile_xml(export_text, included_pids)
        group_summaries = {
            name: parse_xctrace_time_profile_xml(export_text, pids)
            for name, pids in (pid_groups or {}).items()
        }
        trace_bundle_bytes = directory_size_bytes(self.trace_path)
        trace_tree_sha256 = directory_tree_sha256(self.trace_path)
        compact_payload = {
            "schema_version": 1,
            "source": "xctrace Time Profiler",
            "trace_retained": self.retain_trace,
            "trace_bundle_bytes": trace_bundle_bytes,
            "trace_tree_sha256": trace_tree_sha256,
            "export_xml_sha256": sha256_bytes(export_text.encode("utf-8")),
            "record_command": self.command,
            "export_command": export_command,
            "notification_name": self.notification_name,
            "notification_output": self.notification_output,
            "record_console": console,
            "record_stop": {
                "requested": "sigint" if interrupt_recording else "time-limit",
                "reached_time_limit": reached_time_limit,
            },
            "export_console": exported.stdout,
            "timing": {
                "profiler_launch_to_ready_ns": (
                    int(self.ready_ns) - self.started_ns
                    if self.ready_ns is not None
                    else None
                ),
                "record_process_launch_to_ready_ns": (
                    int(self.ready_ns) - self.record_started_ns
                    if self.ready_ns is not None
                    else None
                ),
                "record_flush_elapsed_ns": recording_completed_ns - stop_call_ns,
                "export_elapsed_ns": export_completed_ns - export_started_ns,
            },
            "summary": summary,
            "group_summaries": group_summaries,
        }
        compact_sha256 = write_validated_json(self.compact_path, compact_payload)
        # The compact export has round-tripped successfully and carries the XML
        # and bundle hashes.  Only now is it safe to discard bulky intermediates.
        self.export_path.unlink()
        trace_path: str | None = str(self.trace_path)
        if not self.retain_trace:
            shutil.rmtree(self.trace_path)
            trace_path = None
        return {
            "stop_call_ns": stop_call_ns,
            "recording_completed_ns": recording_completed_ns,
            "export_completed_ns": export_completed_ns,
            "compact_path": str(self.compact_path),
            "compact_sha256": compact_sha256,
            "trace_path": trace_path,
            "trace_retained": self.retain_trace,
            "trace_bundle_bytes": trace_bundle_bytes,
            "trace_tree_sha256": trace_tree_sha256,
            "summary": summary,
            "group_summaries": group_summaries,
            "timing": compact_payload["timing"],
            "record_stop": compact_payload["record_stop"],
        }

    def abort(self) -> None:
        if self.waiter is not None:
            if self.waiter.poll() is None:
                self.waiter.kill()
            self.waiter.wait(timeout=5)
            self.waiter = None
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
            self.process = None


class EngineSession:
    def __init__(
        self,
        command: Sequence[str],
        cwd: Path,
        timeout_seconds: float,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        if self.process.stdin is None or self.process.stdout is None:
            raise ProfileError("failed to open UCI subprocess pipes")
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._reader = threading.Thread(target=self._read_output, daemon=True)
        self._reader.start()

    def _read_output(self) -> None:
        assert self.process.stdout is not None
        try:
            for line in self.process.stdout:
                self._lines.put(line.rstrip("\r\n"))
        finally:
            self._lines.put(None)

    def send(self, command: str) -> None:
        if self.process.poll() is not None:
            raise ProfileError(
                f"engine exited with {self.process.returncode} before {command!r}"
            )
        assert self.process.stdin is not None
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def read_line(self, deadline: float) -> str:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ProfileError("timed out waiting for UCI output")
        try:
            line = self._lines.get(timeout=remaining)
        except queue.Empty as error:
            raise ProfileError("timed out waiting for UCI output") from error
        if line is None:
            raise ProfileError(f"unexpected UCI EOF ({self.process.poll()})")
        return line

    def initialize(
        self,
        expected_name: str,
        options: dict[str, Any],
    ) -> dict[str, Any]:
        self.send("uci")
        deadline = time.monotonic() + self.timeout_seconds
        engine_name = None
        advertised: set[str] = set()
        info_strings: list[str] = []
        while True:
            line = self.read_line(deadline)
            if line.startswith("id name "):
                engine_name = line.removeprefix("id name ")
            option_match = OPTION_RE.match(line)
            if option_match:
                advertised.add(option_match.group(1))
            if line.startswith("info string "):
                info_strings.append(line.removeprefix("info string "))
            if line == "uciok":
                break
        if engine_name != expected_name:
            raise ProfileError(f"unexpected engine name: {engine_name!r}")
        missing = sorted(set(options) - advertised)
        if missing:
            raise ProfileError(f"engine is missing UCI options: {missing}")
        for name, value in options.items():
            self.send(f"setoption name {name} value {option_value(value)}")
        self.send("isready")
        deadline = time.monotonic() + self.timeout_seconds
        rejected = []
        while True:
            line = self.read_line(deadline)
            if "ignored invalid option" in line:
                rejected.append(line)
            if line == "readyok":
                break
        if rejected:
            raise ProfileError("engine rejected options: " + "; ".join(rejected))
        kernel = None
        for info in info_strings:
            if info.startswith("nnue_kernel="):
                kernel = info.split("=", 1)[1]
        return {"engine_name": engine_name, "nnue_kernel": kernel}

    def prepare_position(self, fen: str) -> None:
        self.send("ucinewgame")
        self.send(f"position fen {fen}")
        self.send("isready")
        deadline = time.monotonic() + self.timeout_seconds
        while self.read_line(deadline) != "readyok":
            pass

    def search(
        self,
        depth: int,
        sampler: MacSampler | XcTraceSampler | None,
        sample_interval_ms: int,
        sample_flush_timeout: float,
        sample_arm_delay_ms: int,
    ) -> dict[str, Any]:
        sampler_ready_ns = sampler.wait_until_attached() if sampler else None
        if isinstance(sampler, MacSampler) and sample_arm_delay_ms:
            # The CLI prints its attach message before the sampling helper is
            # fully armed.  Keep the engine blocked in std::getline here; the
            # resulting __read_nocancel samples are removed explicitly by
            # parse_sample_report and retained in the raw-count audit fields.
            time.sleep(sample_arm_delay_ms / 1000.0)
        cpu_started = process_cpu_seconds(self.process.pid)
        wall_started_ns = time.perf_counter_ns()
        self.send(f"go depth {depth}")
        go_sent_ns = time.perf_counter_ns()
        deadline = time.monotonic() + self.timeout_seconds
        latest_info = None
        bestmove = None
        while bestmove is None:
            line = self.read_line(deadline)
            parsed = parse_info_line(line)
            if parsed is not None:
                latest_info = parsed
            if line.startswith("bestmove "):
                parts = line.split()
                if len(parts) < 2:
                    raise ProfileError(f"malformed bestmove line: {line}")
                bestmove = parts[1]
        bestmove_received_ns = time.perf_counter_ns()
        cpu_elapsed = process_cpu_seconds(self.process.pid) - cpu_started
        elapsed_ns = bestmove_received_ns - wall_started_ns
        sample_data = None
        if sampler is not None:
            if isinstance(sampler, XcTraceSampler):
                # Do not rely on attached-target exit to terminate Instruments:
                # rare xctrace runs fail to observe it.  Time Profiler has an
                # explicit time limit and records only Running rows, so the
                # engine can safely remain blocked in std::getline after
                # bestmove while xctrace finalizes and exports the trace.
                stopped = sampler.stop(sample_flush_timeout)
                target_exit_requested_ns = time.perf_counter_ns()
                self.send("quit")
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
                summary = stopped["summary"]
                sampled_seconds = float(summary["total_sampled_seconds"])
                sample_data = {
                    "backend": "xctrace",
                    "raw_path": stopped["compact_path"],
                    "raw_sha256": stopped["compact_sha256"],
                    "trace_path": stopped["trace_path"],
                    "trace_retained": stopped["trace_retained"],
                    "trace_bundle_bytes": stopped["trace_bundle_bytes"],
                    "trace_tree_sha256": stopped["trace_tree_sha256"],
                    "attach_to_go_ns": go_sent_ns - int(sampler_ready_ns),
                    "profiler_launch_to_ready_ns": stopped["timing"][
                        "profiler_launch_to_ready_ns"
                    ],
                    "record_process_launch_to_ready_ns": stopped["timing"][
                        "record_process_launch_to_ready_ns"
                    ],
                    "bestmove_to_target_exit_request_ns": (
                        target_exit_requested_ns - bestmove_received_ns
                    ),
                    "bestmove_to_sampler_stop_call_ns": (
                        stopped["stop_call_ns"] - bestmove_received_ns
                    ),
                    "record_flush_elapsed_ns": stopped["timing"][
                        "record_flush_elapsed_ns"
                    ],
                    "export_elapsed_ns": stopped["timing"]["export_elapsed_ns"],
                    "sampled_to_process_cpu_ratio": (
                        sampled_seconds / cpu_elapsed if cpu_elapsed > 0 else None
                    ),
                    "sampled_to_wall_ratio": (
                        sampled_seconds / (elapsed_ns / 1_000_000_000.0)
                        if elapsed_ns > 0
                        else None
                    ),
                    "engine_kept_alive_until_profiler_complete": True,
                    "summary": summary,
                }
            else:
                # `sample` acknowledges SIGINT promptly but its helper can keep
                # observing a live idle target while symbols are processed.
                # `-mayDie` stops it with the target; its audited pre-go idle
                # rows are removed by parse_sample_report.
                target_exit_requested_ns = time.perf_counter_ns()
                self.send("quit")
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
                console, sampler_stop_ns = sampler.stop(sample_flush_timeout)
                raw_text = sampler.output.read_text(
                    encoding="utf-8", errors="replace"
                )
                console_path = sampler.output.with_suffix(".console.txt")
                console_path.write_text(console, encoding="utf-8")
                summary = parse_sample_report(raw_text, sample_interval_ms)
                sampled_seconds = float(summary["total_sampled_thread_seconds"])
                sample_data = {
                    "backend": "sample",
                    "raw_path": str(sampler.output),
                    "raw_sha256": sha256_file(sampler.output),
                    "console_path": str(console_path),
                    "console_sha256": sha256_file(console_path),
                    "attach_to_go_ns": go_sent_ns - int(sampler_ready_ns),
                    "profiler_launch_to_ready_ns": (
                        int(sampler_ready_ns) - sampler.started_ns
                    ),
                    "bestmove_to_target_exit_request_ns": (
                        target_exit_requested_ns - bestmove_received_ns
                    ),
                    "bestmove_to_sampler_stop_call_ns": (
                        sampler_stop_ns - bestmove_received_ns
                    ),
                    "sampled_to_process_cpu_ratio": (
                        sampled_seconds / cpu_elapsed if cpu_elapsed > 0 else None
                    ),
                    "sampled_to_wall_ratio": (
                        sampled_seconds / (elapsed_ns / 1_000_000_000.0)
                        if elapsed_ns > 0
                        else None
                    ),
                    "summary": summary,
                }
        if latest_info is None:
            raise ProfileError("bestmove arrived without a complete info line")
        if latest_info["nodes"] <= 0:
            raise ProfileError("engine reported a non-positive node count")
        if not UCI_MOVE_RE.fullmatch(bestmove):
            raise ProfileError(f"invalid bestmove: {bestmove!r}")
        return {
            **latest_info,
            "bestmove": bestmove,
            "go_sent_ns": go_sent_ns,
            "bestmove_received_ns": bestmove_received_ns,
            "elapsed_ns": elapsed_ns,
            "process_cpu_seconds": cpu_elapsed,
            "nps": latest_info["nodes"] * 1_000_000_000.0 / elapsed_ns,
            "cpu_to_wall": cpu_elapsed / (elapsed_ns / 1_000_000_000.0),
            "sample": sample_data,
        }

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self.send("quit")
                self.process.wait(timeout=5)
            except (ProfileError, subprocess.TimeoutExpired):
                self.process.kill()
                self.process.wait(timeout=5)
        if self.process.stdin is not None:
            self.process.stdin.close()
        if self.process.stdout is not None:
            self.process.stdout.close()
        self._reader.join(timeout=1)

    def __enter__(self) -> "EngineSession":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def command_output(command: Sequence[str]) -> str | None:
    result = subprocess.run(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def machine_metadata() -> dict[str, Any]:
    sysctl: dict[str, str | None] = {}
    for key in (
        "hw.model",
        "hw.ncpu",
        "hw.physicalcpu",
        "hw.logicalcpu",
        "hw.memsize",
        "machdep.cpu.brand_string",
    ):
        sysctl[key] = command_output(["/usr/sbin/sysctl", "-n", key])
    return {
        "platform": platform.platform(),
        "uname": list(platform.uname()),
        "python": platform.python_version(),
        "sysctl": sysctl,
        "os_build": command_output(["/usr/bin/sw_vers", "-buildVersion"]),
        "power_source": command_output(["/usr/bin/pmset", "-g", "batt"]),
    }


def git_metadata(worktree: Path) -> dict[str, Any]:
    head = command_output(["git", "-C", str(worktree), "rev-parse", "HEAD"])
    status = command_output(["git", "-C", str(worktree), "status", "--porcelain"])
    return {
        "head": head,
        "worktree_dirty": bool(status),
        "status_porcelain_sha256": (
            sha256_bytes(status.encode("utf-8")) if status else None
        ),
    }


def median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--suite", type=Path, default=Path("benchmarks/uci_platform_v1.json")
    )
    parser.add_argument("--engine-cwd", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="disable per-case durable checkpoints for an uninterrupted run",
    )
    parser.add_argument(
        "--profiler",
        choices=("xctrace", "sample"),
        default="xctrace" if platform.system() == "Darwin" else "sample",
    )
    parser.add_argument("--sample", type=Path, default=Path("/usr/bin/sample"))
    parser.add_argument("--xcrun", type=Path, default=Path("/usr/bin/xcrun"))
    parser.add_argument(
        "--notifyutil", type=Path, default=Path("/usr/bin/notifyutil")
    )
    parser.add_argument("--retain-xctrace-count", type=int, default=2)
    parser.add_argument(
        "--xctrace-scope", choices=("batch", "per-case"), default="batch"
    )
    parser.add_argument("--xctrace-time-limit-seconds", type=int, default=2)
    parser.add_argument("--xctrace-batch-time-limit-seconds", type=int, default=30)
    parser.add_argument(
        "--xctrace-notify-registration-delay-ms", type=int, default=100
    )
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--sample-interval-ms", type=int, default=1)
    parser.add_argument("--positions-limit", type=int)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--attach-timeout", type=float, default=10.0)
    parser.add_argument("--sample-flush-timeout", type=float, default=180.0)
    parser.add_argument("--sample-arm-delay-ms", type=int, default=100)
    return parser.parse_args()


def run_xctrace_batch_no_checkpoint(
    args: argparse.Namespace,
    started_utc: str,
    engine: Path,
    model: Path,
    config: Path,
    suite_path: Path,
    suite_bytes: bytes,
    suite: dict[str, Any],
    positions: list[dict[str, str]],
    engine_cwd: Path,
    xcrun_executable: Path,
    notifyutil_executable: Path,
    config_validation: dict[str, Any],
) -> Path:
    """Run one all-processes Time Profiler session per adjacent B-B pair."""

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else Path(f"logs/nnue_v41_local_xctrace_batch_{utc_stamp()}")
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    raw_dir = output_dir / "raw"
    snapshot_dir = output_dir / "snapshots"
    raw_dir.mkdir()
    snapshot_dir.mkdir()
    shutil.copy2(suite_path, snapshot_dir / "suite.json")
    shutil.copy2(config, snapshot_dir / "config.yml")
    shutil.copy2(Path(__file__), snapshot_dir / "harness.py")

    observations: list[dict[str, Any]] = []
    rounds: list[dict[str, Any]] = []
    profile_rounds: list[dict[str, Any]] = []
    batch_records: list[dict[str, Any]] = []
    handshake: dict[str, Any] | None = None
    total_cases = args.blocks * 4 * len(positions)
    completed = 0

    def initialize_session() -> EngineSession:
        nonlocal handshake
        session = EngineSession([str(engine), str(model)], engine_cwd, args.timeout)
        try:
            current = session.initialize(
                suite["expected_engine_name"], suite["uci_options"]
            )
        except Exception:
            session.close()
            raise
        if handshake is None:
            handshake = current
        elif current != handshake:
            session.close()
            raise ProfileError("UCI handshake changed between cases")
        return session

    def emit_progress(row: dict[str, Any]) -> None:
        nonlocal completed
        completed += 1
        print(
            PROGRESS_SENTINEL
            + json.dumps(
                {
                    "completed": completed,
                    "total": total_cases,
                    "round": row["round"],
                    "mode": row["mode"],
                    "position": row["position_id"],
                },
                separators=(",", ":"),
            ),
            flush=True,
        )

    def run_control_round(round_index: int) -> dict[str, Any]:
        round_rows: list[dict[str, Any]] = []
        for sequence, position in enumerate(position_order(positions, round_index)):
            with initialize_session() as session:
                session.prepare_position(position["fen"])
                result = session.search(
                    args.depth,
                    None,
                    args.sample_interval_ms,
                    args.sample_flush_timeout,
                    args.sample_arm_delay_ms,
                )
            row = {
                "round": round_index,
                "sequence": sequence,
                "mode": "control",
                "position_id": position["id"],
                "fen": position["fen"],
                **result,
            }
            observations.append(row)
            round_rows.append(row)
            emit_progress(row)
        return {
            "round": round_index,
            "mode": "control",
            **aggregate_round(round_rows),
            "profile_count": 0,
        }

    def run_profile_pair(block_index: int) -> list[dict[str, Any]]:
        first_round = block_index * 4 + 1
        pair_rounds = (first_round, first_round + 1)
        recorder: XcTraceSampler | None = None
        try:
            raw_stem = (
                f"b{block_index:02d}-r{pair_rounds[0]:02d}-r{pair_rounds[1]:02d}"
            )
            recorder = XcTraceSampler(
                xcrun_executable,
                notifyutil_executable,
                None,
                raw_dir / f"{raw_stem}.trace",
                raw_dir / f"{raw_stem}.xctrace.json",
                args.attach_timeout,
                args.xctrace_notify_registration_delay_ms,
                args.xctrace_batch_time_limit_seconds,
                block_index < args.retain_xctrace_count,
                all_processes=True,
            )
            recorder_ready_ns = recorder.wait_until_attached()
            rows_by_round: dict[int, list[dict[str, Any]]] = {
                round_index: [] for round_index in pair_rounds
            }
            recorder_alive_go_barriers = 0
            target_pids: list[int] = []
            seen_pids: set[int] = set()
            pids_by_round: dict[str, set[int]] = {
                str(round_index): set() for round_index in pair_rounds
            }
            # Match control and Heroku process/cache semantics exactly: launch,
            # initialize, search and close one fresh engine at a time.  The
            # all-process recorder surrounds the pair, but only stack rows
            # rooted in search_best_move_impl survive export filtering.
            for round_index in pair_rounds:
                for sequence, position in enumerate(
                    position_order(positions, round_index)
                ):
                    if recorder.process is None or recorder.process.poll() is not None:
                        raise ProfileError(
                            "xctrace batch recorder ended before all searches completed"
                        )
                    with initialize_session() as session:
                        pid = session.process.pid
                        if pid in seen_pids:
                            raise ProfileError(
                                f"xctrace batch target PID was reused: {pid}"
                            )
                        seen_pids.add(pid)
                        target_pids.append(pid)
                        pids_by_round[str(round_index)].add(pid)
                        session.prepare_position(position["fen"])
                        if (
                            recorder.process is None
                            or recorder.process.poll() is not None
                        ):
                            raise ProfileError(
                                "xctrace batch recorder ended before a go barrier"
                            )
                        recorder_alive_go_barriers += 1
                        result = session.search(
                            args.depth,
                            None,
                            args.sample_interval_ms,
                            args.sample_flush_timeout,
                            args.sample_arm_delay_ms,
                        )
                    row = {
                        "round": round_index,
                        "sequence": sequence,
                        "mode": "profile",
                        "position_id": position["id"],
                        "fen": position["fen"],
                        "profile_target_pid": pid,
                        **result,
                    }
                    observations.append(row)
                    rows_by_round[round_index].append(row)
                    emit_progress(row)
            if len(target_pids) != len(positions) * 2:
                raise ProfileError("xctrace batch captured an incomplete target PID set")
            if pids_by_round[str(pair_rounds[0])] & pids_by_round[
                str(pair_rounds[1])
            ]:
                raise ProfileError("xctrace B-round target PID sets are not disjoint")
            if recorder.process is None or recorder.process.poll() is not None:
                raise ProfileError(
                    "xctrace batch recorder ended before the SIGINT barrier"
                )
            stopped = recorder.stop(
                args.sample_flush_timeout,
                included_pids=set(target_pids),
                pid_groups=pids_by_round,
                interrupt_recording=True,
            )
            recorder = None
            overall_filter = stopped["summary"]["pid_filter"]
            if int(overall_filter["included_rows"]) <= 0:
                raise ProfileError("batch Time Profiler export contains no target rows")
            if set(overall_filter["requested_pids"]) != set(target_pids):
                raise ProfileError("batch Time Profiler pid filter changed target set")
            if set(overall_filter["observed_target_pids"]) != set(target_pids):
                raise ProfileError(
                    "batch Time Profiler did not observe every fresh target process"
                )
            if overall_filter["missing_requested_pids"]:
                raise ProfileError(
                    "batch Time Profiler has target processes without a sampled "
                    "search_best_move_impl stack"
                )
            pair_rows = [
                row for round_index in pair_rounds for row in rows_by_round[round_index]
            ]
            pair_search_cpu_seconds = sum(
                float(row["process_cpu_seconds"]) for row in pair_rows
            )
            pair_sampled_seconds = float(
                stopped["summary"]["total_sampled_seconds"]
            )
            pair_search_cpu_coverage = (
                pair_sampled_seconds / pair_search_cpu_seconds
            )
            if not (
                XCTRACE_MIN_SEARCH_CPU_COVERAGE
                <= pair_search_cpu_coverage
                <= XCTRACE_MAX_SEARCH_CPU_COVERAGE
            ):
                raise ProfileError(
                    "batch Time Profiler search-stack coverage is outside the "
                    "audited gate: "
                    f"{pair_search_cpu_coverage:.3f} not in "
                    f"[{XCTRACE_MIN_SEARCH_CPU_COVERAGE:.2f}, "
                    f"{XCTRACE_MAX_SEARCH_CPU_COVERAGE:.2f}]"
                )

            result_rounds: list[dict[str, Any]] = []
            for round_index in pair_rounds:
                sample_summary = stopped["group_summaries"][str(round_index)]
                pid_filter = sample_summary["pid_filter"]
                expected_round_pids = pids_by_round[str(round_index)]
                if set(pid_filter["requested_pids"]) != expected_round_pids:
                    raise ProfileError(
                        f"profile round {round_index} target PID set changed"
                    )
                if set(pid_filter["observed_target_pids"]) != expected_round_pids:
                    raise ProfileError(
                        f"profile round {round_index} did not observe every target PID"
                    )
                if pid_filter["missing_requested_pids"]:
                    raise ProfileError(
                        f"profile round {round_index} has a target without search samples"
                    )
                sampled_seconds = float(sample_summary["total_sampled_seconds"])
                if sampled_seconds <= 0:
                    raise ProfileError(
                        f"profile round {round_index} contains no target samples"
                    )
                round_rows = rows_by_round[round_index]
                round_summary = {
                    "round": round_index,
                    "mode": "profile",
                    **aggregate_round(round_rows),
                    "profile_count": len(round_rows),
                }
                round_search_cpu_coverage = (
                    sampled_seconds / round_summary["process_cpu_seconds"]
                )
                sample_summary["sampled_to_process_cpu_ratio"] = (
                    round_search_cpu_coverage
                )
                sample_summary["search_cpu_coverage_hard_gate_passed"] = (
                    XCTRACE_MIN_SEARCH_CPU_COVERAGE
                    <= round_search_cpu_coverage
                    <= XCTRACE_MAX_SEARCH_CPU_COVERAGE
                )
                sample_summary["search_cpu_coverage_target_met"] = (
                    XCTRACE_TARGET_MIN_SEARCH_CPU_COVERAGE
                    <= round_search_cpu_coverage
                    <= XCTRACE_TARGET_MAX_SEARCH_CPU_COVERAGE
                )
                if not sample_summary["search_cpu_coverage_hard_gate_passed"]:
                    raise ProfileError(
                        f"profile round {round_index} search-stack coverage is "
                        "outside the hard gate: "
                        f"{round_search_cpu_coverage:.3f} not in "
                        f"[{XCTRACE_MIN_SEARCH_CPU_COVERAGE:.2f}, "
                        f"{XCTRACE_MAX_SEARCH_CPU_COVERAGE:.2f}]"
                    )
                sample_summary["sampled_to_wall_ratio"] = sampled_seconds / (
                    round_summary["elapsed_ns"] / 1_000_000_000.0
                )
                round_summary["sample_summary"] = sample_summary
                profile_rounds.append(
                    {
                        "round": round_index,
                        "nodes": round_summary["nodes"],
                        "elapsed_ns": round_summary["elapsed_ns"],
                        "process_cpu_seconds": round_summary[
                            "process_cpu_seconds"
                        ],
                        "sample_summary": sample_summary,
                    }
                )
                result_rounds.append(round_summary)
                per_pid = {
                    int(item["pid"]): item
                    for item in pid_filter["included_pid_rows"]
                }
                for row in round_rows:
                    pid = int(row["profile_target_pid"])
                    pid_sample = per_pid.get(pid, {"rows": 0, "weight_ns": 0})
                    pid_seconds = int(pid_sample["weight_ns"]) / 1_000_000_000.0
                    row["sample"] = {
                        "backend": "xctrace-batch",
                        "batch": block_index,
                        "target_pid": pid,
                        "raw_path": stopped["compact_path"],
                        "raw_sha256": stopped["compact_sha256"],
                        "trace_path": stopped["trace_path"],
                        "trace_retained": stopped["trace_retained"],
                        "trace_bundle_bytes": stopped["trace_bundle_bytes"],
                        "trace_tree_sha256": stopped["trace_tree_sha256"],
                        "profiler_launch_to_ready_ns": stopped["timing"][
                            "profiler_launch_to_ready_ns"
                        ],
                        "batch_recorder_ready_to_go_ns": (
                            int(row["go_sent_ns"]) - recorder_ready_ns
                        ),
                        "pid_sample_rows": int(pid_sample["rows"]),
                        "pid_sampled_seconds": pid_seconds,
                        "sampled_to_process_cpu_ratio": (
                            pid_seconds / float(row["process_cpu_seconds"])
                            if float(row["process_cpu_seconds"]) > 0
                            else None
                        ),
                    }
            batch_records.append(
                {
                    "batch": block_index,
                    "rounds": list(pair_rounds),
                    "target_pids": target_pids,
                    "raw_path": stopped["compact_path"],
                    "raw_sha256": stopped["compact_sha256"],
                    "trace_path": stopped["trace_path"],
                    "trace_retained": stopped["trace_retained"],
                    "timing": stopped["timing"],
                    "record_stop": stopped["record_stop"],
                    "recorder_alive_go_barriers": recorder_alive_go_barriers,
                    "recorder_alive_at_sigint_barrier": True,
                    "search_process_cpu_seconds": pair_search_cpu_seconds,
                    "search_sampled_seconds": pair_sampled_seconds,
                    "search_cpu_coverage": pair_search_cpu_coverage,
                    "filter": overall_filter,
                }
            )
            return result_rounds
        finally:
            if recorder is not None:
                recorder.abort()

    for block_index in range(args.blocks):
        rounds.append(run_control_round(block_index * 4))
        rounds.extend(run_profile_pair(block_index))
        rounds.append(run_control_round(block_index * 4 + 3))

    signatures: dict[str, set[tuple[Any, ...]]] = {
        position["id"]: set() for position in positions
    }
    for row in observations:
        signatures[row["position_id"]].add(signature(row))
    mismatches = {
        position_id: [list(item) for item in sorted(values, key=str)]
        for position_id, values in signatures.items()
        if len(values) != 1
    }
    if mismatches:
        raise ProfileError(f"fixed-depth signature mismatch: {mismatches}")
    if any(row["reported_depth"] != args.depth for row in observations):
        raise ProfileError("engine did not complete the requested depth")

    observations_path = output_dir / "observations.json"
    observations_path.write_text(
        json.dumps(observations, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    machine = machine_metadata()
    profile_rounds_path = output_dir / "profile-rounds.json"
    profile_rounds_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "host": "local-macos",
                "machine": machine,
                "rounds": rounds,
                "profile_rounds": profile_rounds,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    profiled = [row for row in rounds if row["mode"] == "profile"]
    controls = [row for row in rounds if row["mode"] == "control"]
    profile_median_nps = median([row["nps"] for row in profiled])
    control_median_nps = median([row["nps"] for row in controls])
    profile_median_cpu_ns_per_node = median(
        [row["process_cpu_ns_per_node"] for row in profiled]
    )
    control_median_cpu_ns_per_node = median(
        [row["process_cpu_ns_per_node"] for row in controls]
    )
    assert profile_median_cpu_ns_per_node is not None
    assert control_median_cpu_ns_per_node is not None
    sampling_cpu_overhead_fraction = (
        profile_median_cpu_ns_per_node / control_median_cpu_ns_per_node - 1.0
    )
    # A profiler-induced speed-up is just as invalidating as a slow-down for
    # absolute component costs: either direction means the sampled process is
    # running in a materially different CPU/frequency/scheduler state.
    cpu_timing_distortion_fraction = abs(sampling_cpu_overhead_fraction)
    cpu_overhead_valid = cpu_timing_distortion_fraction <= 0.05
    cpu_overhead_target_met = cpu_timing_distortion_fraction <= 0.03
    xctrace_path_value = command_output(
        [str(xcrun_executable), "--find", "xctrace"]
    )
    xctrace_path = Path(xctrace_path_value) if xctrace_path_value else None
    finished_utc = utc_now()
    manifest = {
        "schema_version": 1,
        "status": "valid" if cpu_overhead_valid else "invalid",
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "method": {
            "profiler": "macos-xctrace-Time-Profiler",
            "xctrace_scope": "batch-all-processes",
            "recorder_sessions": args.blocks,
            "cases_per_session": len(positions) * 2,
            "timer": "wall=perf_counter_ns; cpu=proc_pid_rusage(RUSAGE_INFO_V4)",
            "sample_interval_ms": 1,
            "sample_interval_source": "Time Profiler default 1000us setting",
            "scope": (
                "after the all-processes start notification, each fresh engine is "
                "spawned, initialized, prepared, searched and closed before the "
                "next case; exported rows require both its captured PID and an "
                "exact search_best_move_impl ancestor in the resolved full stack"
            ),
            "time_profile_xpath": XCTRACE_TIME_PROFILE_XPATH,
            "batch_time_limit_seconds": args.xctrace_batch_time_limit_seconds,
            "schedule": "ABBA with one recorder spanning each adjacent B-B pair",
            "blocks": args.blocks,
            "depth": args.depth,
            "process_isolation": "fresh process per position per round",
            "checkpoint": "disabled by --no-checkpoint",
            "search_stack_symbol_prefix": XCTRACE_SEARCH_ROOT_SYMBOL_PREFIX,
            "search_cpu_coverage_gate": [
                XCTRACE_MIN_SEARCH_CPU_COVERAGE,
                XCTRACE_MAX_SEARCH_CPU_COVERAGE,
            ],
            "search_cpu_coverage_target": [
                XCTRACE_TARGET_MIN_SEARCH_CPU_COVERAGE,
                XCTRACE_TARGET_MAX_SEARCH_CPU_COVERAGE,
            ],
            "sampling_cpu_overhead_target_fraction": 0.03,
            "sampling_cpu_overhead_hard_limit_fraction": 0.05,
        },
        "identity": {
            "engine_path": str(engine),
            "engine_sha256": sha256_file(engine),
            "model_path": str(model),
            "model_sha256": sha256_file(model),
            "config_path": str(config),
            "config_sha256": sha256_file(config),
            "suite_path": str(suite_path),
            "suite_sha256": sha256_bytes(suite_bytes),
            "harness_sha256": sha256_file(Path(__file__)),
            "profiler_backend": "xctrace-batch",
            "xcrun_path": str(xcrun_executable),
            "xcrun_sha256": sha256_file(xcrun_executable),
            "xctrace_path": str(xctrace_path) if xctrace_path else None,
            "xctrace_sha256": (
                sha256_file(xctrace_path)
                if xctrace_path is not None and xctrace_path.is_file()
                else None
            ),
            "notifyutil_path": str(notifyutil_executable),
            "notifyutil_sha256": sha256_file(notifyutil_executable),
            "git": git_metadata(engine_cwd),
        },
        "handshake": handshake,
        "config_validation": config_validation,
        "machine": machine,
        "integrity": {
            "positions": len(positions),
            "observations": len(observations),
            "raw_profiles": len(batch_records),
            "recorder_sessions": len(batch_records),
            "target_processes": sum(len(row["target_pids"]) for row in batch_records),
            "unknown_pid_rows": sum(
                int(row["filter"]["unknown_pid_rows"]) for row in batch_records
            ),
            "unknown_pid_weight_ns": sum(
                int(row["filter"]["unknown_pid_weight_ns"])
                for row in batch_records
            ),
            "selected_search_rows": sum(
                int(row["filter"]["included_rows"]) for row in batch_records
            ),
            "selected_search_weight_ns": sum(
                int(row["filter"]["included_weight_ns"])
                for row in batch_records
            ),
            "target_pid_non_search_rows": sum(
                int(row["filter"]["target_pid_non_search_rows"])
                for row in batch_records
            ),
            "target_pid_non_search_weight_ns": sum(
                int(row["filter"]["target_pid_non_search_weight_ns"])
                for row in batch_records
            ),
            "missing_target_pids": sorted(
                {
                    int(pid)
                    for row in batch_records
                    for pid in row["filter"]["missing_requested_pids"]
                }
            ),
            "signature_mismatches": mismatches,
            "requested_depth_completed": True,
            "checkpoint_enabled": False,
            "process_thread_pid_mismatch_rows": sum(
                int(row["filter"]["process_thread_pid_mismatch_rows"])
                for row in batch_records
            ),
            "selected_no_stack_rows": sum(
                int(row["sample_summary"]["no_stack_rows"])
                for row in profiled
            ),
            "target_weight_partition_exact": all(
                int(row["filter"]["included_weight_ns"])
                + int(row["filter"]["target_pid_non_search_weight_ns"])
                == int(row["filter"]["target_pid_weight_ns"])
                for row in batch_records
            ),
            "target_pid_sets_unique_disjoint_exact": True,
            "recorder_alive_at_every_go_and_sigint": all(
                int(row["recorder_alive_go_barriers"])
                == len(positions) * 2
                and bool(row["recorder_alive_at_sigint_barrier"])
                for row in batch_records
            ),
            "recorder_reached_time_limit": any(
                bool(row["record_stop"]["reached_time_limit"])
                for row in batch_records
            ),
            "search_cpu_coverage_gate_passed": all(
                XCTRACE_MIN_SEARCH_CPU_COVERAGE
                <= float(row["search_cpu_coverage"])
                <= XCTRACE_MAX_SEARCH_CPU_COVERAGE
                for row in batch_records
            ),
            "search_cpu_coverage_target_met": all(
                bool(row["sample_summary"]["search_cpu_coverage_target_met"])
                for row in profiled
            ),
            "sampling_cpu_overhead_valid": cpu_overhead_valid,
        },
        "rounds": rounds,
        "profile_rounds": profile_rounds,
        "batch_records": batch_records,
        "summary": {
            "profile_median_nps": profile_median_nps,
            "control_median_nps": control_median_nps,
            "sampling_slowdown_fraction": (
                1.0 - profile_median_nps / control_median_nps
                if profile_median_nps and control_median_nps
                else None
            ),
            "profile_median_process_cpu_ns_per_node": (
                profile_median_cpu_ns_per_node
            ),
            "control_median_process_cpu_ns_per_node": (
                control_median_cpu_ns_per_node
            ),
            "sampling_cpu_overhead_fraction": sampling_cpu_overhead_fraction,
            "sampling_cpu_timing_distortion_fraction": (
                cpu_timing_distortion_fraction
            ),
            "sampling_cpu_overhead_target_met": cpu_overhead_target_met,
            "sampling_cpu_overhead_hard_limit_passed": cpu_overhead_valid,
            "profile_median_cpu_to_wall": median(
                [row["cpu_to_wall"] for row in profiled]
            ),
            "control_median_cpu_to_wall": median(
                [row["cpu_to_wall"] for row in controls]
            ),
            "median_profiler_launch_to_ready_ns": median(
                [
                    float(row["timing"]["profiler_launch_to_ready_ns"])
                    for row in batch_records
                ]
            ),
            "median_sampled_to_process_cpu_ratio": median(
                [
                    float(row["sample_summary"]["sampled_to_process_cpu_ratio"])
                    for row in profiled
                ]
            ),
        },
        "artifacts": {
            "observations": {
                "path": observations_path.name,
                "sha256": sha256_file(observations_path),
            },
            "profile_rounds": {
                "path": profile_rounds_path.name,
                "sha256": sha256_file(profile_rounds_path),
            },
            "suite_snapshot": {
                "path": "snapshots/suite.json",
                "sha256": sha256_file(snapshot_dir / "suite.json"),
            },
            "config_snapshot": {
                "path": "snapshots/config.yml",
                "sha256": sha256_file(snapshot_dir / "config.yml"),
            },
            "harness_snapshot": {
                "path": "snapshots/harness.py",
                "sha256": sha256_file(snapshot_dir / "harness.py"),
            },
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not cpu_overhead_valid:
        raise ProfileError(
            "batch xctrace CPU timing distortion exceeds the 5% validity "
            f"limit: signed={sampling_cpu_overhead_fraction:.2%}, "
            f"absolute={cpu_timing_distortion_fraction:.2%}; "
            f"manifest={manifest_path}"
        )
    print(RESULT_SENTINEL + str(manifest_path), flush=True)
    return manifest_path


def run(args: argparse.Namespace) -> Path:
    invocation_started_utc = utc_now()
    if platform.system() != "Darwin":
        raise ProfileError("local profiler requires macOS")
    if args.depth <= 0 or args.blocks <= 0 or args.sample_interval_ms <= 0:
        raise ProfileError("depth, blocks and sample interval must be positive")
    if args.sample_arm_delay_ms < 0:
        raise ProfileError("sample arm delay cannot be negative")
    if args.retain_xctrace_count < 0:
        raise ProfileError("retained xctrace count cannot be negative")
    if args.xctrace_time_limit_seconds <= 0:
        raise ProfileError("xctrace time limit must be a positive integer")
    if args.xctrace_batch_time_limit_seconds <= 0:
        raise ProfileError("xctrace batch time limit must be a positive integer")
    if args.xctrace_notify_registration_delay_ms < 0:
        raise ProfileError("xctrace notification registration delay cannot be negative")
    if args.resume and args.no_checkpoint:
        raise ProfileError("--resume and --no-checkpoint are mutually exclusive")
    if (
        args.profiler == "xctrace"
        and args.xctrace_scope == "batch"
        and not args.no_checkpoint
    ):
        raise ProfileError("batch xctrace currently requires --no-checkpoint")
    if (
        args.timeout <= 0
        or args.attach_timeout <= 0
        or args.sample_flush_timeout <= 0
    ):
        raise ProfileError("timeouts must be positive")
    required_paths = [args.engine, args.model, args.config, args.suite]
    if args.profiler == "sample":
        required_paths.append(args.sample)
    else:
        required_paths.extend((args.xcrun, args.notifyutil))
    for path in required_paths:
        if not path.is_file():
            raise ProfileError(f"required file is missing: {path}")
    engine = args.engine.resolve()
    model = args.model.resolve()
    config = args.config.resolve()
    suite_path = args.suite.resolve()
    sample_executable = args.sample.resolve() if args.profiler == "sample" else None
    xcrun_executable = args.xcrun.resolve() if args.profiler == "xctrace" else None
    notifyutil_executable = (
        args.notifyutil.resolve() if args.profiler == "xctrace" else None
    )
    engine_cwd = args.engine_cwd.resolve()
    suite_bytes = suite_path.read_bytes()
    suite = json.loads(suite_bytes)
    positions = list(suite["positions"])
    if args.positions_limit is not None:
        if args.positions_limit <= 0:
            raise ProfileError("positions limit must be positive")
        positions = positions[: args.positions_limit]
    if not positions:
        raise ProfileError("suite contains no positions")
    config_validation = validate_config_options(config, suite["uci_options"])
    if not engine_cwd.is_dir():
        raise ProfileError(f"engine working directory is missing: {engine_cwd}")

    harness_path = Path(__file__).resolve()
    if args.profiler == "xctrace":
        assert xcrun_executable is not None
        assert notifyutil_executable is not None
        xctrace_path_value = command_output(
            [str(xcrun_executable), "--find", "xctrace"]
        )
        if xctrace_path_value is None or not Path(xctrace_path_value).is_file():
            raise ProfileError("xcrun could not locate xctrace")
        profiler_files = {
            "xcrun_path": str(xcrun_executable),
            "xcrun_sha256": sha256_file(xcrun_executable),
            "xctrace_path": str(Path(xctrace_path_value).resolve()),
            "xctrace_sha256": sha256_file(Path(xctrace_path_value)),
            "notifyutil_path": str(notifyutil_executable),
            "notifyutil_sha256": sha256_file(notifyutil_executable),
        }
    else:
        assert sample_executable is not None
        profiler_files = {
            "sample_path": str(sample_executable),
            "sample_sha256": sha256_file(sample_executable),
        }
    if args.profiler == "xctrace" and args.xctrace_scope == "batch":
        assert xcrun_executable is not None
        assert notifyutil_executable is not None
        return run_xctrace_batch_no_checkpoint(
            args,
            invocation_started_utc,
            engine,
            model,
            config,
            suite_path,
            suite_bytes,
            suite,
            positions,
            engine_cwd,
            xcrun_executable,
            notifyutil_executable,
            config_validation,
        )
    run_contract = {
        "schema_version": 1,
        "identity": {
            "engine_path": str(engine),
            "engine_sha256": sha256_file(engine),
            "model_path": str(model),
            "model_sha256": sha256_file(model),
            "config_path": str(config),
            "config_sha256": sha256_file(config),
            "suite_path": str(suite_path),
            "suite_sha256": sha256_bytes(suite_bytes),
            "harness_path": str(harness_path),
            "harness_sha256": sha256_file(harness_path),
            "engine_cwd": str(engine_cwd),
            **profiler_files,
        },
        "workload": {
            "depth": args.depth,
            "blocks": args.blocks,
            "positions_limit": args.positions_limit,
            "positions": positions,
        },
        "profiler": {
            "backend": args.profiler,
            "sample_interval_ms": args.sample_interval_ms,
            "sample_arm_delay_ms": args.sample_arm_delay_ms,
            "retain_xctrace_count": args.retain_xctrace_count,
            "xctrace_time_limit_seconds": args.xctrace_time_limit_seconds,
            "xctrace_notify_registration_delay_ms": (
                args.xctrace_notify_registration_delay_ms
            ),
            "timeout": args.timeout,
            "attach_timeout": args.attach_timeout,
            "sample_flush_timeout": args.sample_flush_timeout,
        },
    }
    modes = [False, True, True, False] * args.blocks
    expected_cases: dict[str, dict[str, Any]] = {}
    ordered_case_keys: list[str] = []
    profile_ordinals: dict[str, int] = {}
    next_profile_ordinal = 0
    for round_index, profiled in enumerate(modes):
        for sequence, position in enumerate(position_order(positions, round_index)):
            key = case_key(round_index, sequence, position["id"])
            expected_cases[key] = {
                "round": round_index,
                "sequence": sequence,
                "mode": "profile" if profiled else "control",
                "position_id": position["id"],
                "fen": position["fen"],
            }
            ordered_case_keys.append(key)
            if profiled:
                profile_ordinals[key] = next_profile_ordinal
                next_profile_ordinal += 1

    if args.resume and args.output_dir is None:
        raise ProfileError("--resume requires an explicit --output-dir")
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else Path(f"logs/nnue_v41_local_{args.profiler}_{utc_stamp()}")
    ).resolve()
    raw_dir = output_dir / "raw"
    snapshot_dir = output_dir / "snapshots"
    checkpoint_enabled = not args.no_checkpoint
    checkpoint_dir = output_dir / "checkpoint"
    case_checkpoint_dir = checkpoint_dir / "cases"
    run_checkpoint_path = checkpoint_dir / "run.json"
    failed_attempts_dir = output_dir / "failed-attempts"
    if args.resume:
        if not output_dir.is_dir() or not run_checkpoint_path.is_file():
            raise ProfileError(f"resume checkpoint is missing: {run_checkpoint_path}")
        try:
            run_checkpoint = json.loads(
                run_checkpoint_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise ProfileError("invalid run checkpoint") from error
        if run_checkpoint.get("schema_version") != 1:
            raise ProfileError("unsupported run checkpoint schema")
        saved_contract = run_checkpoint.get("contract")
        if not isinstance(saved_contract, dict):
            raise ProfileError("run checkpoint has no resume contract")
        validate_resume_contract(saved_contract, run_contract)
        completed_rows = load_case_checkpoints(case_checkpoint_dir)
        unexpected = sorted(set(completed_rows) - set(expected_cases))
        if unexpected:
            raise ProfileError(f"unexpected completed checkpoint cases: {unexpected}")
        for key, row in completed_rows.items():
            validate_case_checkpoint_row(key, row, expected_cases[key], args.depth)
        now = utc_now()
        attempts = run_checkpoint.get("attempts")
        if not isinstance(attempts, list):
            raise ProfileError("run checkpoint attempts are malformed")
        for attempt in attempts:
            if isinstance(attempt, dict) and attempt.get("status") == "running":
                attempt["status"] = "interrupted"
                attempt["finished_utc"] = now
        run_checkpoint["resume_count"] = int(
            run_checkpoint.get("resume_count", 0)
        ) + 1
        attempt_number = len(attempts) + 1
        attempts.append(
            {
                "attempt": attempt_number,
                "resumed": True,
                "started_utc": invocation_started_utc,
                "completed_cases_at_start": len(completed_rows),
                "archived_retry_artifacts": [],
                "status": "running",
            }
        )
        started_utc = str(run_checkpoint["started_utc"])
        handshake = run_checkpoint.get("handshake")
        if handshake is not None and not isinstance(handshake, dict):
            raise ProfileError("run checkpoint handshake is malformed")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        raw_dir.mkdir()
        snapshot_dir.mkdir()
        if checkpoint_enabled:
            checkpoint_dir.mkdir()
            case_checkpoint_dir.mkdir()
            failed_attempts_dir.mkdir()
        shutil.copy2(suite_path, snapshot_dir / "suite.json")
        shutil.copy2(config, snapshot_dir / "config.yml")
        shutil.copy2(harness_path, snapshot_dir / "harness.py")
        completed_rows = {}
        attempt_number = 1
        started_utc = invocation_started_utc
        handshake = None
        run_checkpoint = {
            "schema_version": 1,
            "status": "in_progress",
            "started_utc": started_utc,
            "updated_utc": invocation_started_utc,
            "finished_utc": None,
            "resume_count": 0,
            "completed_cases": 0,
            "total_cases": len(ordered_case_keys),
            "handshake": None,
            "contract": run_contract,
            "attempts": [
                {
                    "attempt": attempt_number,
                    "resumed": False,
                    "started_utc": invocation_started_utc,
                    "completed_cases_at_start": 0,
                    "archived_retry_artifacts": [],
                    "status": "running",
                }
            ],
        }
    raw_dir.mkdir(exist_ok=True)
    if checkpoint_enabled:
        case_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        failed_attempts_dir.mkdir(exist_ok=True)
    run_checkpoint["updated_utc"] = utc_now()
    run_checkpoint["completed_cases"] = len(completed_rows)
    if checkpoint_enabled:
        write_durable_json(run_checkpoint_path, run_checkpoint)

    observations: list[dict[str, Any]] = []
    rounds: list[dict[str, Any]] = []
    profile_rounds: list[dict[str, Any]] = []
    total_cases = len(modes) * len(positions)
    completed = len(completed_rows)
    maximum_sample_duration = max(2, math.ceil(args.timeout) + 5)
    for round_index, profiled in enumerate(modes):
        round_rows: list[dict[str, Any]] = []
        for sequence, position in enumerate(position_order(positions, round_index)):
            key = case_key(round_index, sequence, position["id"])
            if key in completed_rows:
                row = completed_rows[key]
                observations.append(row)
                round_rows.append(row)
                continue
            sampler: MacSampler | XcTraceSampler | None = None
            raw_stem = key
            if args.resume:
                moved = archive_uncheckpointed_raw_artifacts(
                    raw_dir,
                    raw_stem,
                    failed_attempts_dir,
                    attempt_number,
                )
                if moved:
                    run_checkpoint["attempts"][-1][
                        "archived_retry_artifacts"
                    ].extend(moved)
                    run_checkpoint["updated_utc"] = utc_now()
                    write_durable_json(run_checkpoint_path, run_checkpoint)
            try:
                with EngineSession(
                    [str(engine), str(model)], engine_cwd, args.timeout
                ) as session:
                    current_handshake = session.initialize(
                        suite["expected_engine_name"], suite["uci_options"]
                    )
                    if handshake is None:
                        handshake = current_handshake
                    elif current_handshake != handshake:
                        raise ProfileError("UCI handshake changed between cases")
                    if checkpoint_enabled and run_checkpoint.get("handshake") is None:
                        run_checkpoint["handshake"] = handshake
                        run_checkpoint["updated_utc"] = utc_now()
                        write_durable_json(run_checkpoint_path, run_checkpoint)
                    session.prepare_position(position["fen"])
                    if profiled:
                        if args.profiler == "xctrace":
                            assert xcrun_executable is not None
                            assert notifyutil_executable is not None
                            sampler = XcTraceSampler(
                                xcrun_executable,
                                notifyutil_executable,
                                session.process.pid,
                                raw_dir / f"{raw_stem}.trace",
                                raw_dir / f"{raw_stem}.xctrace.json",
                                args.attach_timeout,
                                args.xctrace_notify_registration_delay_ms,
                                args.xctrace_time_limit_seconds,
                                profile_ordinals[key] < args.retain_xctrace_count,
                            )
                        else:
                            assert sample_executable is not None
                            sampler = MacSampler(
                                sample_executable,
                                session.process.pid,
                                raw_dir / f"{raw_stem}.sample.txt",
                                args.sample_interval_ms,
                                maximum_sample_duration,
                                args.attach_timeout,
                            )
                    result = session.search(
                        args.depth,
                        sampler,
                        args.sample_interval_ms,
                        args.sample_flush_timeout,
                        args.sample_arm_delay_ms,
                    )
                    sampler = None
            finally:
                if sampler is not None:
                    sampler.abort()
            row = {
                "round": round_index,
                "sequence": sequence,
                "mode": "profile" if profiled else "control",
                "position_id": position["id"],
                "fen": position["fen"],
                **result,
            }
            validate_case_checkpoint_row(key, row, expected_cases[key], args.depth)
            if checkpoint_enabled:
                case_document = {
                    "schema_version": 1,
                    "case_key": key,
                    "attempt": attempt_number,
                    "completed_utc": utc_now(),
                    "row": row,
                }
                write_durable_json(
                    case_checkpoint_dir / f"{key}.json", case_document
                )
            completed_rows[key] = row
            completed += 1
            run_checkpoint["handshake"] = handshake
            run_checkpoint["completed_cases"] = completed
            run_checkpoint["updated_utc"] = utc_now()
            if checkpoint_enabled:
                write_durable_json(run_checkpoint_path, run_checkpoint)
            observations.append(row)
            round_rows.append(row)
            print(
                PROGRESS_SENTINEL
                + json.dumps(
                    {
                        "completed": completed,
                        "total": total_cases,
                        "round": round_index,
                        "mode": row["mode"],
                        "position": position["id"],
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
        round_summary = {
            "round": round_index,
            "mode": "profile" if profiled else "control",
            **aggregate_round(round_rows),
            "profile_count": sum(row["sample"] is not None for row in round_rows),
        }
        if profiled:
            case_summaries = [row["sample"]["summary"] for row in round_rows]
            if args.profiler == "xctrace":
                sample_summary = merge_xctrace_summaries(case_summaries)
                sampled_seconds = float(sample_summary["total_sampled_seconds"])
            else:
                sample_summary = merge_sample_summaries(
                    case_summaries, args.sample_interval_ms
                )
                sampled_seconds = float(
                    sample_summary["total_sampled_thread_seconds"]
                )
            if sampled_seconds <= 0:
                raise ProfileError(
                    f"profile round {round_index} contains no search samples"
                )
            sample_summary["sampled_to_process_cpu_ratio"] = (
                sampled_seconds / round_summary["process_cpu_seconds"]
                if round_summary["process_cpu_seconds"] > 0
                else None
            )
            sample_summary["sampled_to_wall_ratio"] = sampled_seconds / (
                round_summary["elapsed_ns"] / 1_000_000_000.0
            )
            round_summary["sample_summary"] = sample_summary
            profile_rounds.append(
                {
                    "round": round_index,
                    "nodes": round_summary["nodes"],
                    "elapsed_ns": round_summary["elapsed_ns"],
                    "process_cpu_seconds": round_summary["process_cpu_seconds"],
                    "sample_summary": sample_summary,
                }
            )
        rounds.append(round_summary)

    missing_checkpoints = sorted(set(expected_cases) - set(completed_rows))
    if missing_checkpoints:
        raise ProfileError(f"run ended with missing case checkpoints: {missing_checkpoints}")
    signatures: dict[str, set[tuple[Any, ...]]] = {
        position["id"]: set() for position in positions
    }
    for row in observations:
        signatures[row["position_id"]].add(signature(row))
    mismatches = {
        position_id: [list(item) for item in sorted(values, key=str)]
        for position_id, values in signatures.items()
        if len(values) != 1
    }
    if mismatches:
        raise ProfileError(f"fixed-depth signature mismatch: {mismatches}")
    if any(row["reported_depth"] != args.depth for row in observations):
        raise ProfileError("engine did not complete the requested depth")

    observations_path = output_dir / "observations.json"
    observations_path.write_text(
        json.dumps(observations, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    machine = machine_metadata()
    profile_rounds_path = output_dir / "profile-rounds.json"
    profile_rounds_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "host": "local-macos",
                "machine": machine,
                "rounds": rounds,
                "profile_rounds": profile_rounds,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    finished_utc = utc_now()
    current_attempt = run_checkpoint["attempts"][-1]
    current_attempt["status"] = "complete"
    current_attempt["finished_utc"] = finished_utc
    current_attempt["completed_cases_at_end"] = len(completed_rows)
    run_checkpoint["status"] = "complete"
    run_checkpoint["finished_utc"] = finished_utc
    run_checkpoint["updated_utc"] = finished_utc
    run_checkpoint["completed_cases"] = len(completed_rows)
    run_checkpoint["handshake"] = handshake
    if checkpoint_enabled:
        write_durable_json(run_checkpoint_path, run_checkpoint)
    profiled = [row for row in rounds if row["mode"] == "profile"]
    controls = [row for row in rounds if row["mode"] == "control"]
    profile_median_nps = median([row["nps"] for row in profiled])
    control_median_nps = median([row["nps"] for row in controls])
    if args.profiler == "xctrace":
        assert xcrun_executable is not None
        assert notifyutil_executable is not None
        xctrace_path_value = command_output(
            [str(xcrun_executable), "--find", "xctrace"]
        )
        xctrace_path = Path(xctrace_path_value) if xctrace_path_value else None
        profiler_method = {
            "profiler": "macos-xctrace-Time-Profiler",
            "timer": "wall=perf_counter_ns; cpu=proc_pid_rusage(RUSAGE_INFO_V4)",
            "sample_interval_ms": 1,
            "sample_interval_source": "Time Profiler default 1000us setting",
            "scope": (
                "xctrace --notify-tracing-started barrier immediately before go; "
                "after bestmove the engine remains alive blocked on stdin until "
                "the recorder time limit and export complete, then exits; Time "
                "Profiler exports only Running rows and the first frame is "
                "exclusive/self"
            ),
            "time_profile_xpath": XCTRACE_TIME_PROFILE_XPATH,
            "no_stack_symbol": XCTRACE_NO_STACK_SYMBOL,
            "retain_xctrace_count": args.retain_xctrace_count,
            "notification_registration_delay_ms": (
                args.xctrace_notify_registration_delay_ms
            ),
            "time_limit_seconds": args.xctrace_time_limit_seconds,
            "checkpoint_enabled": checkpoint_enabled,
            "schedule": "ABBA",
            "blocks": args.blocks,
            "depth": args.depth,
            "process_isolation": "fresh process per position per round",
        }
        profiler_identity = {
            "profiler_backend": "xctrace",
            "xcrun_path": str(xcrun_executable),
            "xcrun_sha256": sha256_file(xcrun_executable),
            "xctrace_path": str(xctrace_path) if xctrace_path else None,
            "xctrace_sha256": (
                sha256_file(xctrace_path)
                if xctrace_path is not None and xctrace_path.is_file()
                else None
            ),
            "xctrace_version": command_output(
                [str(xcrun_executable), "xctrace", "version"]
            ),
            "notifyutil_path": str(notifyutil_executable),
            "notifyutil_sha256": sha256_file(notifyutil_executable),
        }
    else:
        assert sample_executable is not None
        profiler_method = {
            "profiler": "macos-/usr/bin/sample",
            "timer": "wall=perf_counter_ns; cpu=proc_pid_rusage(RUSAGE_INFO_V4)",
            "sample_interval_ms": args.sample_interval_ms,
            "scope": (
                f"{args.sample_arm_delay_ms}ms pre-go stdin arming window is "
                "filtered by idle leaf; fresh target exits immediately after "
                "bestmove (-mayDie)"
            ),
            "sample_arm_delay_ms": args.sample_arm_delay_ms,
            "idle_filter_leaf_symbols": sorted(IDLE_STDIN_LEAF_SYMBOLS),
            "schedule": "ABBA",
            "blocks": args.blocks,
            "depth": args.depth,
            "process_isolation": "fresh process per position per round",
            "raw_flat_row_threshold": 5,
            "checkpoint_enabled": checkpoint_enabled,
        }
        profiler_identity = {
            "profiler_backend": "sample",
            "sample_path": str(sample_executable),
            "sample_sha256": sha256_file(sample_executable),
        }
    manifest = {
        "schema_version": 1,
        "status": "valid",
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "method": profiler_method,
        "identity": {
            "engine_path": str(engine),
            "engine_sha256": sha256_file(engine),
            "model_path": str(model),
            "model_sha256": sha256_file(model),
            "config_path": str(config),
            "config_sha256": sha256_file(config),
            "suite_path": str(suite_path),
            "suite_sha256": sha256_bytes(suite_bytes),
            "harness_sha256": sha256_file(Path(__file__)),
            **profiler_identity,
            "git": git_metadata(engine_cwd),
        },
        "handshake": handshake,
        "config_validation": config_validation,
        "resume": {
            "checkpoint_enabled": checkpoint_enabled,
            "resume_count": run_checkpoint["resume_count"],
            "attempts": run_checkpoint["attempts"],
            "checkpoint_cases": len(completed_rows) if checkpoint_enabled else 0,
        },
        "machine": machine,
        "integrity": {
            "positions": len(positions),
            "observations": len(observations),
            "raw_profiles": sum(row["sample"] is not None for row in observations),
            "retained_trace_bundles": sum(
                bool(row["sample"].get("trace_retained"))
                for row in observations
                if row["sample"] is not None
            ),
            "signature_mismatches": mismatches,
            "requested_depth_completed": True,
            "checkpoint_enabled": checkpoint_enabled,
            "checkpoint_cases": len(completed_rows) if checkpoint_enabled else 0,
        },
        "rounds": rounds,
        "profile_rounds": profile_rounds,
        "summary": {
            "profile_median_nps": profile_median_nps,
            "control_median_nps": control_median_nps,
            "sampling_slowdown_fraction": (
                1.0 - profile_median_nps / control_median_nps
                if profile_median_nps and control_median_nps
                else None
            ),
            "profile_median_cpu_to_wall": median(
                [row["cpu_to_wall"] for row in profiled]
            ),
            "control_median_cpu_to_wall": median(
                [row["cpu_to_wall"] for row in controls]
            ),
            "median_attach_to_go_ns": median(
                [
                    float(row["sample"]["attach_to_go_ns"])
                    for row in observations
                    if row["sample"] is not None
                ]
            ),
            "median_profiler_launch_to_ready_ns": median(
                [
                    float(row["sample"]["profiler_launch_to_ready_ns"])
                    for row in observations
                    if row["sample"] is not None
                    and row["sample"].get("profiler_launch_to_ready_ns") is not None
                ]
            ),
            "median_sampled_to_process_cpu_ratio": median(
                [
                    float(row["sample"]["sampled_to_process_cpu_ratio"])
                    for row in observations
                    if row["sample"] is not None
                    and row["sample"].get("sampled_to_process_cpu_ratio") is not None
                ]
            ),
            "median_sampled_to_wall_ratio": median(
                [
                    float(row["sample"]["sampled_to_wall_ratio"])
                    for row in observations
                    if row["sample"] is not None
                    and row["sample"].get("sampled_to_wall_ratio") is not None
                ]
            ),
            "median_bestmove_to_target_exit_request_ns": median(
                [
                    float(row["sample"]["bestmove_to_target_exit_request_ns"])
                    for row in observations
                    if row["sample"] is not None
                ]
            ),
        },
        "artifacts": {
            "observations": {
                "path": observations_path.name,
                "sha256": sha256_file(observations_path),
            },
            "profile_rounds": {
                "path": profile_rounds_path.name,
                "sha256": sha256_file(profile_rounds_path),
            },
            "suite_snapshot": {
                "path": "snapshots/suite.json",
                "sha256": sha256_file(snapshot_dir / "suite.json"),
            },
            "config_snapshot": {
                "path": "snapshots/config.yml",
                "sha256": sha256_file(snapshot_dir / "config.yml"),
            },
            "harness_snapshot": {
                "path": "snapshots/harness.py",
                "sha256": sha256_file(snapshot_dir / "harness.py"),
            },
            **(
                {
                    "run_checkpoint": {
                        "path": str(run_checkpoint_path.relative_to(output_dir)),
                        "sha256": sha256_file(run_checkpoint_path),
                    }
                }
                if checkpoint_enabled
                else {}
            ),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(RESULT_SENTINEL + str(manifest_path), flush=True)
    return manifest_path


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(
            ERROR_SENTINEL
            + json.dumps(
                {"type": type(error).__name__, "error": str(error)},
                separators=(",", ":"),
            ),
            file=sys.stderr,
            flush=True,
        )
        raise

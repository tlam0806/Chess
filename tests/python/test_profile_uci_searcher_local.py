from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "profile_uci_searcher_local",
    ROOT / "tools" / "benchmark" / "profile_uci_searcher_local.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_parse_args_supports_uninterrupted_no_checkpoint_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "profile_uci_searcher_local.py",
            "--engine",
            "engine",
            "--model",
            "model",
            "--config",
            "config",
            "--no-checkpoint",
        ],
    )
    args = MODULE.parse_args()
    assert args.no_checkpoint is True
    assert args.resume is False
    assert args.xctrace_scope == "batch"


def test_parse_info_line_requires_complete_signature() -> None:
    assert MODULE.parse_info_line("info depth 8 nodes 123 score cp -17 nps 10") == {
        "raw_info": "info depth 8 nodes 123 score cp -17 nps 10",
        "reported_depth": 8,
        "nodes": 123,
        "score_type": "cp",
        "score_value": -17,
    }
    assert MODULE.parse_info_line("info depth 8 nodes 123") is None


def test_position_order_rotates_then_reverses() -> None:
    positions = [{"id": name, "fen": name} for name in "abcd"]
    assert [row["id"] for row in MODULE.position_order(positions, 0)] == list("abcd")
    assert [row["id"] for row in MODULE.position_order(positions, 1)] == list("dcba")
    assert [row["id"] for row in MODULE.position_order(positions, 2)] == list("bcda")
    assert [row["id"] for row in MODULE.position_order(positions, 3)] == list("adcb")


def test_parse_sample_report_and_merge() -> None:
    report = """\
Analysis of sampling engine (pid 42) every 1 millisecond
Call graph:
    9 Thread_1 DispatchQueue_1
      9 root
        9 chess::forward  (in engine) + 1 [0x1]
    5 Thread_2
      5 root
        5 chess::search<int, 2>  (in engine) + 1 [0x2]
Total number in stack (recursive counted multiple, when >=5):

Sort by top of stack, same collapsed (when >= 5):
        chess::forward  (in engine)        7
        chess::search<int, 2>  (in engine)        5
        chess::forward  (in engine)        2

Binary Images:
"""
    parsed = MODULE.parse_sample_report(report, 1)
    assert parsed["total_samples"] == 14
    assert parsed["total_sampled_thread_seconds"] == pytest.approx(0.014)
    assert parsed["flat_samples_listed"] == 14
    assert [row["symbol"] for row in parsed["symbols"]] == [
        "chess::forward",
        "chess::search<int, 2>",
    ]
    assert [row["flat_samples"] for row in parsed["symbols"]] == [9, 5]
    assert [row["flat_seconds"] for row in parsed["symbols"]] == pytest.approx(
        [0.009, 0.005]
    )
    merged = MODULE.merge_sample_summaries([parsed, parsed], 1)
    assert merged["profiles"] == 2
    assert merged["total_samples"] == 28
    assert merged["raw_total_samples"] == 28
    assert merged["excluded_pre_go_idle_samples"] == 0
    assert merged["symbols"][0]["flat_samples"] == 18


def test_parse_sample_report_rejects_empty_profile() -> None:
    with pytest.raises(MODULE.ProfileError, match="thread-root"):
        MODULE.parse_sample_report("no call graph", 1)


def test_parse_sample_report_excludes_audited_pre_go_stdin_idle() -> None:
    report = """\
Call graph:
    25 Thread_1
      20 root
        20 __read_nocancel  (in libsystem_kernel.dylib) + 1 [0x1]
      5 root
        5 chess::search  (in engine) + 1 [0x2]
Sort by top of stack, same collapsed (when >= 5):
        __read_nocancel  (in libsystem_kernel.dylib)        20
        chess::search  (in engine)        5
Binary Images:
"""
    parsed = MODULE.parse_sample_report(report, 1)
    assert parsed["raw_total_samples"] == 25
    assert parsed["excluded_pre_go_idle_samples"] == 20
    assert parsed["total_samples"] == 5
    assert parsed["symbols"] == [
        {"symbol": "chess::search", "flat_samples": 5, "flat_seconds": 0.005}
    ]


def test_parse_sample_report_allows_sub_tick_search_case() -> None:
    report = """\
Call graph:
    7 Thread_1
      7 root
        7 __read_nocancel  (in libsystem_kernel.dylib) + 1 [0x1]
Sort by top of stack, same collapsed (when >= 5):
        __read_nocancel  (in libsystem_kernel.dylib)        7
Binary Images:
"""
    parsed = MODULE.parse_sample_report(report, 1)
    assert parsed["raw_total_samples"] == 7
    assert parsed["excluded_pre_go_idle_samples"] == 7
    assert parsed["total_samples"] == 0
    assert parsed["symbols"] == []


def test_parse_xctrace_time_profile_resolves_refs_and_keeps_no_stack() -> None:
    report = """\
<trace-query-result>
  <node>
    <schema name="time-profile"/>
    <row>
      <sample-time id="1">1000000</sample-time>
      <weight id="2">1000000</weight>
      <tagged-backtrace id="3"><backtrace id="4">
        <frame id="5" name="chess::forward"/>
        <frame id="6" name="chess::search"/>
      </backtrace></tagged-backtrace>
    </row>
    <row>
      <sample-time id="7">2000000</sample-time>
      <weight ref="2"/>
      <tagged-backtrace id="8"><backtrace id="9">
        <frame ref="5"/><frame ref="6"/>
      </backtrace></tagged-backtrace>
    </row>
    <row>
      <sample-time id="10">3000000</sample-time>
      <weight id="11">2000000</weight>
      <sentinel/>
    </row>
  </node>
</trace-query-result>
"""
    parsed = MODULE.parse_xctrace_time_profile_xml(report)
    assert parsed["total_sampled_seconds"] == pytest.approx(0.004)
    assert parsed["total_weight_ns"] == 4_000_000
    assert parsed["sample_rows"] == 3
    assert parsed["symbolized_rows"] == 2
    assert parsed["no_stack_rows"] == 1
    assert parsed["no_stack_weight_ns"] == 2_000_000
    symbols = {row["symbol"]: row for row in parsed["symbols"]}
    assert symbols["chess::forward"]["flat_samples"] == 2
    assert symbols["chess::forward"]["flat_seconds"] == pytest.approx(0.002)
    assert symbols[MODULE.XCTRACE_NO_STACK_SYMBOL]["flat_samples"] == 1
    assert symbols[MODULE.XCTRACE_NO_STACK_SYMBOL]["flat_seconds"] == pytest.approx(
        0.002
    )

    merged = MODULE.merge_xctrace_summaries([parsed, parsed])
    assert merged["profiles"] == 2
    assert merged["total_sampled_seconds"] == pytest.approx(0.008)
    assert merged["sample_rows"] == 6
    assert merged["no_stack_weight_ns"] == 4_000_000
    assert sum(row["flat_weight_ns"] for row in merged["symbols"]) == 8_000_000


def test_parse_xctrace_time_profile_rejects_unresolved_reference() -> None:
    report = """\
<trace-query-result><schema name="time-profile"/><row>
  <sample-time id="1">1</sample-time><weight id="2">1000000</weight>
  <tagged-backtrace><backtrace><frame ref="missing"/></backtrace></tagged-backtrace>
</row></trace-query-result>
"""
    with pytest.raises(MODULE.ProfileError, match="unresolved"):
        MODULE.parse_xctrace_time_profile_xml(report)


def test_parse_xctrace_batch_filters_exact_pid_and_search_root_stack() -> None:
    report = f"""\
<trace-query-result>
  <node>
    <schema name="time-profile"/>
    <process id="p42"><pid id="pid42">42</pid></process>
    <process id="p43"><pid id="pid43">43</pid></process>
    <process id="p99"><pid id="pid99">99</pid></process>
    <thread id="t42"><process ref="p42"/></thread>
    <thread id="t43"><process ref="p43"/></thread>
    <thread id="t99"><process ref="p99"/></thread>
    <row>
      <sample-time id="st1">1000000</sample-time><weight id="w1">1000000</weight>
      <process ref="p42"/><thread ref="t42"/>
      <tagged-backtrace><backtrace>
        <frame name="chess::forward"/>
        <frame name="{MODULE.XCTRACE_SEARCH_ROOT_SYMBOL_PREFIX}args)"/>
      </backtrace></tagged-backtrace>
    </row>
    <row>
      <sample-time id="st2">2000000</sample-time><weight ref="w1"/>
      <process ref="p42"/><thread ref="t42"/>
      <tagged-backtrace><backtrace>
        <frame name="chess::load_model"/><frame name="main"/>
      </backtrace></tagged-backtrace>
    </row>
    <row>
      <sample-time id="st3">3000000</sample-time><weight ref="w1"/>
      <thread ref="t43"/>
      <tagged-backtrace><backtrace>
        <frame name="chess::Position::make_move"/>
        <frame name="{MODULE.XCTRACE_SEARCH_ROOT_SYMBOL_PREFIX}args)"/>
      </backtrace></tagged-backtrace>
    </row>
    <row>
      <sample-time id="st4">4000000</sample-time><weight ref="w1"/>
      <process ref="p99"/><thread ref="t99"/>
      <tagged-backtrace><backtrace>
        <frame name="system_leaf"/>
        <frame name="{MODULE.XCTRACE_SEARCH_ROOT_SYMBOL_PREFIX}args)"/>
      </backtrace></tagged-backtrace>
    </row>
    <row>
      <sample-time id="st5">5000000</sample-time><weight ref="w1"/>
      <tagged-backtrace><backtrace>
        <frame name="unknown_system_leaf"/>
      </backtrace></tagged-backtrace>
    </row>
  </node>
</trace-query-result>
"""
    parsed = MODULE.parse_xctrace_time_profile_xml(report, {42, 43})
    assert parsed["sample_rows"] == 2
    assert parsed["total_weight_ns"] == 2_000_000
    assert parsed["no_stack_rows"] == 0
    assert {row["symbol"] for row in parsed["symbols"]} == {
        "chess::forward",
        "chess::Position::make_move",
    }
    filtering = parsed["pid_filter"]
    assert filtering["kind"] == "pid-and-search-root-stack"
    assert filtering["all_rows"] == 5
    assert filtering["target_pid_rows"] == 3
    assert filtering["target_pid_weight_ns"] == 3_000_000
    assert filtering["target_pid_non_search_rows"] == 1
    assert filtering["target_pid_non_search_weight_ns"] == 1_000_000
    assert filtering["unknown_pid_rows"] == 1
    assert filtering["requested_pids"] == [42, 43]
    assert filtering["observed_requested_pids"] == [42, 43]
    assert filtering["missing_requested_pids"] == []
    assert filtering["process_thread_pid_mismatch_rows"] == 0


def test_parse_xctrace_rejects_row_and_thread_process_pid_mismatch() -> None:
    report = f"""\
<trace-query-result><schema name="time-profile"/>
  <process id="p42"><pid>42</pid></process>
  <process id="p43"><pid>43</pid></process>
  <thread id="t43"><process ref="p43"/></thread>
  <row><sample-time>1</sample-time><weight>1000000</weight>
    <process ref="p42"/><thread ref="t43"/>
    <tagged-backtrace><backtrace>
      <frame name="leaf"/>
      <frame name="{MODULE.XCTRACE_SEARCH_ROOT_SYMBOL_PREFIX}args)"/>
    </backtrace></tagged-backtrace>
  </row>
</trace-query-result>
"""
    with pytest.raises(MODULE.ProfileError, match="PID mismatch"):
        MODULE.parse_xctrace_time_profile_xml(report, {42})


def test_compact_profile_json_is_round_trip_validated(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    payload = {"schema_version": 1, "symbols": [{"name": "chess::forward"}]}
    digest = MODULE.write_validated_json(path, payload)
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert digest == MODULE.sha256_file(path)


def test_xctrace_record_command_has_explicit_time_limit(tmp_path: Path) -> None:
    command = MODULE.xctrace_record_command(
        Path("/usr/bin/xcrun"),
        42,
        "dev.chess.test",
        tmp_path / "case.trace",
        1,
    )
    assert command[command.index("--time-limit") + 1] == "1s"
    with pytest.raises(ValueError, match="positive"):
        MODULE.xctrace_record_command(
            Path("/usr/bin/xcrun"), 42, "dev.chess.test", tmp_path / "bad.trace", 0
        )


def test_xctrace_all_processes_command_has_notify_and_safety_limit(
    tmp_path: Path,
) -> None:
    command = MODULE.xctrace_all_processes_record_command(
        Path("/usr/bin/xcrun"),
        "dev.chess.batch",
        tmp_path / "batch.trace",
        30,
    )
    assert "--all-processes" in command
    assert "--attach" not in command
    assert command[command.index("--notify-tracing-started") + 1] == (
        "dev.chess.batch"
    )
    assert command[command.index("--time-limit") + 1] == "30s"


def test_xctrace_search_finalizes_recorder_before_quitting_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeProcess:
        pid = 42

        def wait(self, timeout: float) -> None:
            events.append("engine-wait")

        def kill(self) -> None:
            events.append("engine-kill")

    session = MODULE.EngineSession.__new__(MODULE.EngineSession)
    session.process = FakeProcess()
    session.timeout_seconds = 5.0
    lines = iter(
        [
            "info depth 8 score cp 7 nodes 100",
            "bestmove e2e4",
        ]
    )
    session.read_line = lambda _deadline: next(lines)
    session.send = lambda command: events.append(command)

    sampler = MODULE.XcTraceSampler.__new__(MODULE.XcTraceSampler)
    monkeypatch.setattr(
        MODULE.XcTraceSampler,
        "wait_until_attached",
        lambda _self: 1,
    )

    def stop(_self: object, _timeout: float) -> dict[str, object]:
        events.append("xctrace-stop")
        return {
            "stop_call_ns": 2,
            "compact_path": "/tmp/case.json",
            "compact_sha256": "compact",
            "trace_path": None,
            "trace_retained": False,
            "trace_bundle_bytes": 1,
            "trace_tree_sha256": "trace",
            "summary": {"total_sampled_seconds": 0.1},
            "timing": {
                "profiler_launch_to_ready_ns": 1,
                "record_process_launch_to_ready_ns": 1,
                "record_flush_elapsed_ns": 1,
                "export_elapsed_ns": 1,
            },
        }

    monkeypatch.setattr(MODULE.XcTraceSampler, "stop", stop)
    cpu_values = iter([1.0, 1.1])
    monkeypatch.setattr(MODULE, "process_cpu_seconds", lambda _pid: next(cpu_values))
    result = session.search(8, sampler, 1, 5.0, 0)
    assert events.index("xctrace-stop") < events.index("quit")
    assert result["sample"]["engine_kept_alive_until_profiler_complete"] is True
    assert result["nodes"] == 100


def test_case_checkpoint_round_trip_and_retry_artifact_quarantine(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "checkpoint" / "cases"
    raw_dir = tmp_path / "raw"
    failed_dir = tmp_path / "failed-attempts"
    raw_dir.mkdir()
    failed_dir.mkdir()
    key = MODULE.case_key(1, 2, "startpos")
    row = {
        "round": 1,
        "sequence": 2,
        "mode": "control",
        "position_id": "startpos",
        "fen": "fen",
        "reported_depth": 8,
        "sample": None,
    }
    MODULE.write_durable_json(
        case_dir / f"{key}.json",
        {"schema_version": 1, "case_key": key, "row": row},
    )
    loaded = MODULE.load_case_checkpoints(case_dir)
    assert loaded == {key: row}
    MODULE.validate_case_checkpoint_row(
        key,
        row,
        {
            "round": 1,
            "sequence": 2,
            "mode": "control",
            "position_id": "startpos",
            "fen": "fen",
        },
        8,
    )

    retry_stem = MODULE.case_key(3, 0, "kiwipete")
    partial_trace = raw_dir / f"{retry_stem}.trace"
    partial_trace.mkdir()
    (partial_trace / "partial").write_text("trace", encoding="utf-8")
    (raw_dir / f"{retry_stem}.xctrace.json").write_text("{}", encoding="utf-8")
    unrelated = raw_dir / "unrelated.json"
    unrelated.write_text("{}", encoding="utf-8")
    moved = MODULE.archive_uncheckpointed_raw_artifacts(
        raw_dir, retry_stem, failed_dir, 2
    )
    assert len(moved) == 2
    assert not partial_trace.exists()
    assert unrelated.exists()
    assert all(Path(path).exists() for path in moved)


def test_resume_contract_requires_exact_profiler_arguments() -> None:
    contract = {"identity": {"engine_sha256": "a"}, "profiler": {"limit": 2}}
    MODULE.validate_resume_contract(contract, contract.copy())
    with pytest.raises(MODULE.ProfileError, match="profiler"):
        MODULE.validate_resume_contract(
            contract,
            {"identity": {"engine_sha256": "a"}, "profiler": {"limit": 1}},
        )


def test_aggregate_round_uses_process_cpu_time() -> None:
    result = MODULE.aggregate_round(
        [
            {"nodes": 100, "elapsed_ns": 1000, "process_cpu_seconds": 0.0000008},
            {"nodes": 200, "elapsed_ns": 2000, "process_cpu_seconds": 0.0000016},
        ]
    )
    assert result["nodes"] == 300
    assert result["elapsed_ns"] == 3000
    assert result["process_cpu_seconds"] == pytest.approx(0.0000024)
    assert result["process_cpu_ns_per_node"] == pytest.approx(8.0)


def test_mach_timebase_ticks_are_converted_before_cpu_seconds() -> None:
    # Apple Silicon commonly uses a 125/3 ns Mach timebase.  Treating raw
    # rusage ticks as nanoseconds would under-report CPU by 41.67x.
    assert MODULE.mach_ticks_to_seconds(24_000_000, 125, 3) == pytest.approx(1.0)

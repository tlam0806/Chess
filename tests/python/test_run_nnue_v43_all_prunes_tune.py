from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "tools" / "tune" / "run_nnue_v43_all_prunes_tune.sh"
BALANCED_TUNER_SOURCE = '''\
from dataclasses import dataclass

SCHEMA_VERSION = 2
EXPERIMENT = "v43-final-joint-all-prunes-balanced-baseline-v2"

@dataclass(frozen=True)
class _Aspiration:
    hash: str = (
        "906570ae8ede9e87540847228cd75eef707a64aa5873163be665e220c8b2515e"
    )

    def canonical(self):
        return {
            "enabled": True,
            "min_depth": 2,
            "delta_base_cp": 68,
            "delta_divisor": 33700,
            "expansion_factor_per_mille": 2290,
            "max_fail_high_reductions": 1,
            "mean_score_new_weight_per_mille": 370,
            "max_researches": 6,
            "mean_score_clamp_cp": 1500,
        }

PRODUCTION_BASELINE_ASPIRATION = _Aspiration()

@dataclass(frozen=True)
class _Candidate:
    hash: str = (
        "e770be6261511ed0bc391852c27d69cbe203204c3b5db96e8cc329daf7772d48"
    )

def Candidate():
    return _Candidate()
'''


def write_executable(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(0o755)


def write_mini_dataset(path: Path) -> None:
    path.mkdir()
    split_files: dict[str, dict[str, object]] = {}
    splits: dict[str, dict[str, int]] = {}
    for index, split in enumerate(
        ("tune", "selection", "aspiration_refresh", "holdout")
    ):
        payload = (
            f"phase0\t{index:032x}\t0\t"
            "8/8/8/8/8/8/8/K6k w - - 0 1\n"
        ).encode()
        filename = f"{split}.tsv"
        (path / filename).write_bytes(payload)
        split_files[split] = {
            "file": filename,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        splits[split] = {"count": 1}
    (path / "manifest.json").write_text(json.dumps({
        "format": "nnue-v43-all-prunes-game-disjoint-v3",
        "schema_version": 3,
        "splits": splits,
        "split_files": split_files,
        "exclusions": {"unique_game_ids": 3},
        "overlap": {
            "games_between_new_splits": 0,
            "position_or_mirror_keys_between_new_splits": 0,
            "games_with_prior_v42_v43": 0,
            "position_or_mirror_keys_with_prior_v42_v43": 0,
        },
    }))


def test_contract_loader_imports_real_dataclass_tuner() -> None:
    tuner_path = REPO_ROOT / "tools/tune/tune_nnue_v43_all_prunes.py"
    loader = r'''
import importlib.util
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(path.parent))
spec = importlib.util.spec_from_file_location("contract_real_tuner", path)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
print(module.SCHEMA_VERSION)
print(module.PRODUCTION_BASELINE_ASPIRATION.hash)
print(module.Candidate().hash)
'''
    completed = subprocess.run(
        [sys.executable, "-c", loader, str(tuner_path)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == [
        "2",
        "906570ae8ede9e87540847228cd75eef707a64aa5873163be665e220c8b2515e",
        "e770be6261511ed0bc391852c27d69cbe203204c3b5db96e8cc329daf7772d48",
    ]


def test_runner_snapshots_every_input_and_resume_is_fail_closed(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    write_mini_dataset(dataset)
    model = tmp_path / "model.bin"
    model.write_bytes(b"v43-final-model")
    tuner_source = tmp_path / "tuner.py"
    tuner_source.write_text(BALANCED_TUNER_SOURCE)
    tuner_infra_source = tmp_path / "tuner-infra.py"
    tuner_infra_source.write_text("immutable-v43-tuner-infra-v1\n")
    builder_source = tmp_path / "builder.py"
    builder_source.write_text("immutable-v43-final-builder-v1\n")

    mock_bin = tmp_path / "mock-bin"
    mock_bin.mkdir()
    build_log = tmp_path / "build.log"
    invocation_log = tmp_path / "invocations.log"
    argument_log = tmp_path / "arguments.log"
    write_executable(mock_bin / "cmake", r'''#!/bin/bash
set -euo pipefail
printf '%s\n' "$*" >> "${MOCK_BUILD_LOG:?}"
build_dir=""
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--build" ]]; then
    build_dir="$2"
    break
  fi
  shift
done
if [[ -n "$build_dir" ]]; then
  mkdir -p "$build_dir"
  cp /usr/bin/true "$build_dir/evaluate_nnue_v43_selective"
  cp /usr/bin/true "$build_dir/nnue_v43_time_gauntlet"
fi
''')
    write_executable(mock_bin / "mock-python", r'''#!/bin/bash
set -euo pipefail
if [[ "${1:-}" == "-c" ]]; then
  exec "${REAL_TEST_PYTHON:?}" "$@"
fi
script="$1"
shift
printf '%s\n' "$script" >> "${MOCK_TUNER_INVOCATIONS:?}"
printf '%s\n' "$*" >> "${MOCK_TUNER_ARGUMENTS:?}"
run_dir=""
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--run-dir" ]]; then
    run_dir="$2"
    shift 2
  else
    shift
  fi
done
[[ -n "$run_dir" ]]
printf '%s\n' '{"state":"PAUSED","reason":"duration_budget"}' \
  > "$run_dir/status.json"
''')
    write_executable(mock_bin / "caffeinate", "#!/bin/bash\nexit 0\n")

    run_dir = tmp_path / "run"
    environment = os.environ.copy()
    environment.update({
        "PATH": f"{mock_bin}{os.pathsep}{environment['PATH']}",
        "REAL_TEST_PYTHON": sys.executable,
        "MOCK_BUILD_LOG": str(build_log),
        "MOCK_TUNER_INVOCATIONS": str(invocation_log),
        "MOCK_TUNER_ARGUMENTS": str(argument_log),
        "NNUE_V43_ALL_PRUNES_DATASET_DIR": str(dataset),
        "NNUE_V43_ALL_PRUNES_MODEL": str(model),
        "NNUE_V43_ALL_PRUNES_TUNER_SOURCE": str(tuner_source),
        "NNUE_V43_ALL_PRUNES_TUNER_INFRA_SOURCE": str(tuner_infra_source),
        "NNUE_V43_ALL_PRUNES_DATASET_BUILDER_SOURCE": str(builder_source),
        "NNUE_V43_ALL_PRUNES_BUILD_DIR": str(tmp_path / "build-v43-final"),
        "NNUE_GLOBAL_BUILD_LOCK_FILE": str(tmp_path / "global-build.lock"),
        "NNUE_V43_ALL_PRUNES_PYTHON": str(mock_bin / "mock-python"),
        "NNUE_V43_ALL_PRUNES_WORKERS": "1",
        "NNUE_V43_ALL_PRUNES_BUILD_JOBS": "1",
        "NNUE_V43_ALL_PRUNES_DURATION_SEC": "36000",
    })

    dataset_manifest_path = dataset / "manifest.json"
    sealed_manifest_text = dataset_manifest_path.read_text()
    missing_overlap = json.loads(sealed_manifest_text)
    missing_overlap["overlap"] = {}
    dataset_manifest_path.write_text(json.dumps(missing_overlap))
    rejected_unproved_dataset = subprocess.run(
        [str(RUNNER), str(run_dir)],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
    )
    assert rejected_unproved_dataset.returncode != 0
    assert "overlap proof is incomplete" in rejected_unproved_dataset.stderr
    assert not build_log.exists()
    assert not invocation_log.exists()
    dataset_manifest_path.write_text(sealed_manifest_text)

    first = subprocess.run(
        [str(RUNNER), str(run_dir)],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
    )
    assert first.returncode == 0, first.stderr
    assert (run_dir / "JOB_PAUSED").exists()
    assert not (run_dir / "JOB_FAILED").exists()
    assert not (run_dir / "RUNNING").exists()
    assert not (run_dir / "runner.pid").exists()

    expected_snapshots = (
        "bin/evaluate_nnue_v43_selective",
        "bin/nnue_v43_time_gauntlet",
        "bin/tune_nnue_v43_all_prunes.py",
        "bin/tune_nnue_v43_aspiration.py",
        "bin/build_nnue_v43_all_prunes_dataset.py",
        "inputs/phase_quantized_nnue.bin",
        "inputs/dataset/tune.tsv",
        "inputs/dataset/selection.tsv",
        "inputs/dataset/aspiration_refresh.tsv",
        "inputs/dataset/holdout.tsv",
        "inputs/dataset/manifest.json",
    )
    contract = json.loads((run_dir / "inputs/run_contract.json").read_text())
    assert set(contract["files"]) == set(expected_snapshots)
    assert contract["schema_version"] == 2
    assert contract["experiment"] == (
        "nnue-v43-final-all-prunes-balanced-baseline-v2")
    assert contract["tuner_schema_version"] == 2
    assert contract["tuner_experiment"] == (
        "v43-final-joint-all-prunes-balanced-baseline-v2")
    assert contract["prune_stage_aspiration"] == {
        "name": "v43_balanced_aspiration",
        "config_sha256": (
            "906570ae8ede9e87540847228cd75eef707a64aa5873163be665e220c8b2515e"),
        "config": {
            "enabled": True,
            "min_depth": 2,
            "delta_base_cp": 68,
            "delta_divisor": 33_700,
            "expansion_factor_per_mille": 2_290,
            "max_fail_high_reductions": 1,
            "mean_score_new_weight_per_mille": 370,
            "max_researches": 6,
            "mean_score_clamp_cp": 1_500,
        },
        "fixed_identically_through_selection": True,
    }
    assert contract["production_baseline_config_hash"] == (
        "e770be6261511ed0bc391852c27d69cbe203204c3b5db96e8cc329daf7772d48")
    for relative in expected_snapshots:
        assert (run_dir / relative).is_file()
    assert contract["seed"] == 20260831
    assert contract["workers"] == 1
    assert (run_dir / "inputs/cumulative_duration_sec.txt").read_text() == "36000\n"
    assert (run_dir / "bin/tune_nnue_v43_all_prunes.py").read_text() == (
        BALANCED_TUNER_SOURCE)

    calls = build_log.read_text().splitlines()
    assert len(calls) == 2
    assert "--target evaluate_nnue_v43_selective nnue_v43_time_gauntlet" in calls[1]
    arguments = argument_log.read_text().splitlines()[0]
    assert "--duration-sec 36000" in arguments
    assert "--workers 1" in arguments
    assert f"--dataset-dir {run_dir / 'inputs/dataset'}" in arguments

    # A v1/fixed50-era contract must never launch its snapshotted tuner under
    # the new Balanced-baseline runner semantics.
    contract_path = run_dir / "inputs/run_contract.json"
    balanced_contract_text = contract_path.read_text()
    legacy_contract = json.loads(balanced_contract_text)
    legacy_contract.update({
        "schema_version": 1,
        "experiment": "nnue-v43-final-all-prunes",
    })
    contract_path.write_text(json.dumps(legacy_contract))
    rejected_legacy = subprocess.run(
        [str(RUNNER), str(run_dir)],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
    )
    assert rejected_legacy.returncode != 0
    assert "unsupported immutable run contract" in rejected_legacy.stderr
    assert invocation_log.read_text().splitlines() == [
        str(run_dir / "bin/tune_nnue_v43_all_prunes.py")]
    contract_path.write_text(balanced_contract_text)

    # Worktree/source changes are irrelevant to a resume: only the validated
    # immutable snapshots may execute.
    tuner_source.write_text("must-not-replace-snapshot\n")
    tuner_infra_source.unlink()
    builder_source.unlink()
    model.write_bytes(b"must-not-replace-model")
    for path in dataset.iterdir():
        path.unlink()
    dataset.rmdir()
    environment["NNUE_V43_ALL_PRUNES_DURATION_SEC"] = "43200"
    second = subprocess.run(
        [str(RUNNER), str(run_dir)],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
    )
    assert second.returncode == 0, second.stderr
    assert build_log.read_text().splitlines() == calls
    assert invocation_log.read_text().splitlines() == [
        str(run_dir / "bin/tune_nnue_v43_all_prunes.py"),
        str(run_dir / "bin/tune_nnue_v43_all_prunes.py"),
    ]
    assert (run_dir / "inputs/cumulative_duration_sec.txt").read_text() == "43200\n"

    # A corrupted snapshot must stop before invoking the tuner.  The runner
    # may never silently rebuild or accept it during a resume.
    with (run_dir / "inputs/dataset/tune.tsv").open("ab") as destination:
        destination.write(b"corruption\n")
    third = subprocess.run(
        [str(RUNNER), str(run_dir)],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
    )
    assert third.returncode != 0
    assert "immutable artifact" in third.stderr
    assert (run_dir / "JOB_FAILED").exists()
    assert invocation_log.read_text().splitlines() == [
        str(run_dir / "bin/tune_nnue_v43_all_prunes.py"),
        str(run_dir / "bin/tune_nnue_v43_all_prunes.py"),
    ]


def test_runner_rejects_decreasing_cumulative_budget(tmp_path: Path) -> None:
    # The detailed lifecycle is covered above.  This static assertion guards
    # the resume invariant without repeating a mocked CMake build.
    text = RUNNER.read_text()
    assert "DURATION_SEC < previous_duration" in text
    assert "resume cumulative duration may not decrease" in text


def test_concurrent_runner_cannot_touch_active_run(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    write_mini_dataset(dataset)
    model = tmp_path / "model.bin"
    model.write_bytes(b"model")
    tuner = tmp_path / "tuner.py"
    tuner.write_text(BALANCED_TUNER_SOURCE)
    infra = tmp_path / "infra.py"
    infra.write_text("infra\n")
    builder = tmp_path / "builder.py"
    builder.write_text("builder\n")

    mock_bin = tmp_path / "mock-bin"
    mock_bin.mkdir()
    write_executable(mock_bin / "cmake", r'''#!/bin/bash
set -euo pipefail
build_dir=""
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--build" ]]; then
    build_dir="$2"
    break
  fi
  shift
done
if [[ -n "$build_dir" ]]; then
  mkdir -p "$build_dir"
  cp /usr/bin/true "$build_dir/evaluate_nnue_v43_selective"
  cp /usr/bin/true "$build_dir/nnue_v43_time_gauntlet"
fi
''')
    write_executable(mock_bin / "mock-python", r'''#!/bin/bash
set -euo pipefail
if [[ "${1:-}" == "-c" ]]; then
  exec "${REAL_TEST_PYTHON:?}" "$@"
fi
shift
run_dir=""
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--run-dir" ]]; then
    run_dir="$2"
    shift 2
  else
    shift
  fi
done
[[ -n "$run_dir" ]]
touch "${MOCK_TUNER_STARTED:?}"
while [[ ! -e "${MOCK_TUNER_RELEASE:?}" ]]; do
  sleep 0.02
done
printf '%s\n' '{"state":"PAUSED"}' > "$run_dir/status.json"
''')
    write_executable(mock_bin / "caffeinate", "#!/bin/bash\nexit 0\n")

    started = tmp_path / "tuner-started"
    release = tmp_path / "tuner-release"
    run_dir = tmp_path / "run"
    environment = os.environ.copy()
    environment.update({
        "PATH": f"{mock_bin}{os.pathsep}{environment['PATH']}",
        "REAL_TEST_PYTHON": sys.executable,
        "MOCK_TUNER_STARTED": str(started),
        "MOCK_TUNER_RELEASE": str(release),
        "NNUE_V43_ALL_PRUNES_DATASET_DIR": str(dataset),
        "NNUE_V43_ALL_PRUNES_MODEL": str(model),
        "NNUE_V43_ALL_PRUNES_TUNER_SOURCE": str(tuner),
        "NNUE_V43_ALL_PRUNES_TUNER_INFRA_SOURCE": str(infra),
        "NNUE_V43_ALL_PRUNES_DATASET_BUILDER_SOURCE": str(builder),
        "NNUE_V43_ALL_PRUNES_BUILD_DIR": str(tmp_path / "build"),
        "NNUE_GLOBAL_BUILD_LOCK_FILE": str(tmp_path / "global-build.lock"),
        "NNUE_V43_ALL_PRUNES_PYTHON": str(mock_bin / "mock-python"),
        "NNUE_V43_ALL_PRUNES_WORKERS": "1",
        "NNUE_V43_ALL_PRUNES_BUILD_JOBS": "1",
    })

    first = subprocess.Popen(
        [str(RUNNER), str(run_dir)],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 10.0
        while not started.exists() and time.monotonic() < deadline:
            if first.poll() is not None:
                stdout, stderr = first.communicate()
                raise AssertionError(
                    f"first runner exited early: {first.returncode}\n{stdout}\n{stderr}")
            time.sleep(0.02)
        assert started.exists(), "first runner never reached the tuner"
        assert (run_dir / "RUNNING").exists()
        active_pid = (run_dir / "runner.pid").read_text()

        second = subprocess.run(
            [str(RUNNER), str(run_dir)],
            cwd=REPO_ROOT,
            env=environment,
            text=True,
            capture_output=True,
            timeout=5,
        )
        assert second.returncode != 0
        assert "already holds" in second.stderr
        assert (run_dir / "RUNNING").exists()
        assert (run_dir / "runner.pid").read_text() == active_pid
        assert not (run_dir / "JOB_FAILED").exists()
        assert not (run_dir / "JOB_PAUSED").exists()
    finally:
        release.touch()
        try:
            first_stdout, first_stderr = first.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            first.terminate()
            first_stdout, first_stderr = first.communicate(timeout=10)
    assert first.returncode == 0, f"{first_stdout}\n{first_stderr}"
    assert (run_dir / "JOB_PAUSED").exists()
    assert not (run_dir / "RUNNING").exists()

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "tools" / "run_nnue_v42_aspiration_tune.sh"


def write_executable(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(0o755)


def test_runner_snapshots_tuner_and_resume_ignores_changed_source(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    for split in ("tune", "selection", "holdout"):
        (dataset / f"{split}.tsv").write_text(
            "phase0\t0123456789abcdef0123456789abcdef\t0\t"
            "8/8/8/8/8/8/8/K6k w - - 0 1\n"
        )
    (dataset / "manifest.json").write_text(json.dumps({
        "format": "nnue-v42-lichess-game-disjoint-v2",
        "schema_version": 2,
        "splits": {
            split: {"count": 1}
            for split in ("tune", "selection", "holdout")
        },
    }))

    model = tmp_path / "model.bin"
    model.write_bytes(b"model-v1")
    tuner_source = tmp_path / "source-tuner.py"
    tuner_source.write_text("snapshot-v1\n")

    mock_bin = tmp_path / "mock-bin"
    mock_bin.mkdir()
    build_log = tmp_path / "build.log"
    invocation_log = tmp_path / "tuner-invocations.log"
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
  cp /usr/bin/true "$build_dir/evaluate_nnue_v42_selective"
  cp /usr/bin/true "$build_dir/nnue_v42_time_gauntlet"
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
printf '%s\n' '{"kind":"budget_exhausted"}' > "$run_dir/PAUSED.json"
''')
    write_executable(mock_bin / "caffeinate", "#!/bin/bash\nexit 0\n")

    run_dir = tmp_path / "run"
    environment = os.environ.copy()
    environment.update({
        "PATH": f"{mock_bin}{os.pathsep}{environment['PATH']}",
        "REAL_TEST_PYTHON": sys.executable,
        "MOCK_BUILD_LOG": str(build_log),
        "MOCK_TUNER_INVOCATIONS": str(invocation_log),
        "NNUE_V42_ASPIRATION_DATASET_DIR": str(dataset),
        "NNUE_V42_ASPIRATION_MODEL": str(model),
        "NNUE_V42_ASPIRATION_TUNER_SOURCE": str(tuner_source),
        "NNUE_V42_ASPIRATION_BUILD_DIR": str(tmp_path / "build"),
        "NNUE_V42_ASPIRATION_BUILD_LOCK_FILE": str(tmp_path / "build.lock"),
        "NNUE_V42_ASPIRATION_PYTHON": str(mock_bin / "mock-python"),
        "NNUE_V42_ASPIRATION_CUMULATIVE_DURATION_SEC": "10",
        "NNUE_V42_ASPIRATION_WORKERS": "1",
        "NNUE_V42_ASPIRATION_BUILD_JOBS": "1",
    })

    first = subprocess.run(
        [str(RUNNER), str(run_dir)], cwd=REPO_ROOT, env=environment,
        text=True, capture_output=True)
    assert first.returncode == 0, first.stderr
    tuner_snapshot = run_dir / "bin" / "tune_nnue_v42_aspiration.py"
    assert tuner_snapshot.read_text() == "snapshot-v1\n"
    assert os.access(tuner_snapshot, os.X_OK)
    first_build_calls = build_log.read_text().splitlines()
    assert len(first_build_calls) == 2

    # Resume must execute the immutable run copy. A worktree/source edit is
    # irrelevant once all per-run artifacts have been sealed.
    tuner_source.write_text("snapshot-v2-that-must-not-be-read\n")
    second = subprocess.run(
        [str(RUNNER), str(run_dir)], cwd=REPO_ROOT, env=environment,
        text=True, capture_output=True)
    assert second.returncode == 0, second.stderr
    assert tuner_snapshot.read_text() == "snapshot-v1\n"
    assert build_log.read_text().splitlines() == first_build_calls

    tuner_source.unlink()
    third = subprocess.run(
        [str(RUNNER), str(run_dir)], cwd=REPO_ROOT, env=environment,
        text=True, capture_output=True)
    assert third.returncode == 0, third.stderr
    assert tuner_snapshot.read_text() == "snapshot-v1\n"
    assert build_log.read_text().splitlines() == first_build_calls
    assert invocation_log.read_text().splitlines() == [
        str(tuner_snapshot), str(tuner_snapshot), str(tuner_snapshot)]
    assert (run_dir / "JOB_PAUSED").exists()
    assert not (run_dir / "RUNNING").exists()
    assert not (run_dir / "runner.pid").exists()

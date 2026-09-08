#!/usr/bin/env bash
set -euo pipefail

repo_root="${REPO_ROOT:-/workspace/Chess}"
data_dir="${DATA_DIR:-$repo_root/data/robotmoon_test80_2024_500m_unique_v2_shards_1m}"
candidates="${WORKER_CANDIDATES:-8 16 20 25}"
samples="${BENCHMARK_SAMPLES:-4000000}"
result_json="$repo_root/logs/f2m_worker_benchmark.json"
choice_env="$repo_root/logs/f2m_worker_choice.env"
scratch="$(mktemp -d /tmp/f2m-worker-benchmark.XXXXXX)"

cleanup() {
  rm -rf "$scratch"
}
trap cleanup EXIT

cd "$repo_root"
mkdir -p "$repo_root/logs"
printf 'workers\tsamples\telapsed_sec\tsamples_per_sec\n' > "$scratch/results.tsv"

cpu_count="$(nproc)"
for workers in $candidates; do
  if (( workers >= cpu_count )); then
    printf 'skip workers=%s cpus=%s\n' "$workers" "$cpu_count" >&2
    continue
  fi
  trial="$scratch/workers-$workers"
  mkdir -p "$trial"
  # 10 divides the production corpus' 500 shards exactly, avoiding a tail
  # worker with fewer shards on the 1% validation/test splits.
  eval_workers=10
  printf 'benchmark workers=%s eval_workers=%s samples=%s\n' \
    "$workers" "$eval_workers" "$samples"
  if PYTHONUNBUFFERED=1 python3 tools/train/train_phase_nnue_until_overfit.py \
      --data "$data_dir" \
      --output-dir "$trial" \
      --arch F2M \
      --phase-layout independent \
      --epochs 1 \
      --patience 1 \
      --batch-size 8192 \
      --workers "$workers" \
      --eval-workers "$eval_workers" \
      --torch-threads 2 \
      --device cuda \
      --train-max-samples "$samples" \
      --val-max-samples 65536 \
      --test-max-samples 65536 \
      --train-probe-max-samples 65536 \
      --split-mod 100 --val-mod 98 --test-mod 99 \
      --shuffle-block-size 20000 \
      --lr-steps-per-epoch 488 \
      --lr-warmup-steps 100 \
      --epoch-peak-lrs 0.000500 \
      --epoch-min-lrs 0.000050 \
      --checkpoint-samples 0 \
      --progress-batches 0 \
      --saturation-batches 2 \
      --skip-final-test \
      --seed 20260824 > "$trial/run.log" 2>&1; then
    python3 - "$workers" "$trial/summary.json" >> "$scratch/results.tsv" <<'PY'
import json
import sys

workers = int(sys.argv[1])
summary = json.load(open(sys.argv[2], encoding="utf-8"))
train = summary["history"][0]["train"]
samples = int(train["samples"])
elapsed = float(train["elapsed_sec"])
print(f"{workers}\t{samples}\t{elapsed:.6f}\t{samples / elapsed:.3f}")
PY
  else
    printf 'benchmark_failed workers=%s\n' "$workers" >&2
    tail -n 40 "$trial/run.log" >&2
  fi
done

python3 - "$scratch/results.tsv" "$result_json" "$choice_env" <<'PY'
import csv
import json
import os
import sys
from pathlib import Path

tsv, result_path, env_path = map(Path, sys.argv[1:])
with tsv.open(encoding="utf-8") as stream:
    rows = [
        {
            "workers": int(row["workers"]),
            "samples": int(row["samples"]),
            "elapsed_sec": float(row["elapsed_sec"]),
            "samples_per_sec": float(row["samples_per_sec"]),
        }
        for row in csv.DictReader(stream, delimiter="\t")
    ]
if not rows:
    raise SystemExit("all worker benchmarks failed")
corpus_shards = 500
# Whole-shard ownership avoids duplicate decompression. For the production
# epoch, also require an even shard split so equal per-worker sample quotas do
# not strand records on workers that own fewer shards. A few percent of raw
# benchmark speed is not worth silently training on fewer positions.
eligible = [row for row in rows if corpus_shards % row["workers"] == 0]
winner = max(eligible or rows, key=lambda row: row["samples_per_sec"])
eval_workers = 10
payload = {
    "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
    "cpu_count": os.cpu_count(),
    "trials": rows,
    "selected_workers": winner["workers"],
    "selected_eval_workers": eval_workers,
    "selection_policy": "fastest candidate that evenly divides 500 shards",
}
result_path.write_text(json.dumps(payload, indent=2) + "\n")
env_path.write_text(
    f"WORKERS={winner['workers']}\nEVAL_WORKERS={eval_workers}\n"
)
print(json.dumps(payload, separators=(",", ":")))
PY

printf 'worker benchmark saved to %s\n' "$result_json"

#!/usr/bin/env bash
set -euo pipefail

repo_root="${REPO_ROOT:-/workspace/Chess}"
cd "$repo_root"

tag="${TAG:-old_score_huber200_f2m_independent_500m_20260824}"
data_dir="${DATA_DIR:-$repo_root/data/robotmoon_test80_2024_500m_unique_v2_shards_1m}"
data_manifest="${DATA_MANIFEST:-$repo_root/data/robotmoon_test80_2024_500m_unique_v2.manifest.json}"
output_dir="$repo_root/models/quantized_scale_grid/$tag"
log="$repo_root/logs/$tag.runpod.log"
status="$repo_root/logs/$tag.runpod.status"
report="$repo_root/reports/$tag.md"
workers="${WORKERS:-24}"
eval_workers="${EVAL_WORKERS:-10}"
torch_threads="${TORCH_THREADS:-2}"
epochs="${EPOCHS:-12}"

mkdir -p "$output_dir" "$repo_root/logs" "$repo_root/reports"
printf 'state=preflight tag=%s updated_at=%s\n' \
  "$tag" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$status"

on_exit() {
  code=$?
  if [[ $code -ne 0 ]]; then
    printf 'state=failed exit_code=%s updated_at=%s\n' \
      "$code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$status"
  fi
}
trap on_exit EXIT

python3 - "$data_dir" "$data_manifest" "$workers" <<'PY'
import json
import sys
from pathlib import Path

import torch

from nn.compact_board_data import compact_paths, validate_compact_dataset

data, manifest_path, workers = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
paths = validate_compact_dataset(data)
manifest = json.loads(manifest_path.read_text())
if len(paths) != 500 or manifest["output"]["records"] != 500_000_000:
    raise SystemExit("500M corpus manifest/shard count mismatch")
if manifest["validation"]["independent_128bit_fingerprint_duplicates"] != 0:
    raise SystemExit("500M corpus is not unique")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
cpu_count = __import__("os").cpu_count() or 1
if workers >= cpu_count:
    raise SystemExit(f"workers={workers} leaves no CPU for the trainer (cpus={cpu_count})")
print(
    "runpod_preflight=pass "
    f"gpu={torch.cuda.get_device_name(0)!r} cuda={torch.version.cuda!r} "
    f"cpus={cpu_count} workers={workers} shards={len(paths)}"
)
PY

printf 'state=training tag=%s workers=%s eval_workers=%s updated_at=%s\n' \
  "$tag" "$workers" "$eval_workers" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$status"

PYTHONUNBUFFERED=1 python3 tools/train_phase_nnue_until_overfit.py \
  --data "$data_dir" \
  --output-dir "$output_dir" \
  --arch F2M \
  --phase-layout independent \
  --epochs "$epochs" \
  --patience 2 \
  --min-delta-loss 0.000001 \
  --batch-size 8192 \
  --workers "$workers" \
  --eval-workers "$eval_workers" \
  --torch-threads "$torch_threads" \
  --device cuda \
  --train-max-samples 490000000 \
  --val-max-samples 5000000 \
  --test-max-samples 5000000 \
  --train-probe-max-samples 1000000 \
  --split-mod 100 --val-mod 98 --test-mod 99 \
  --shuffle-block-size 250000 \
  --lr-steps-per-epoch 59815 \
  --lr-warmup-steps 2000 \
  --epoch-peak-lrs \
    0.000500 0.000100 0.000060 0.000040 \
    0.000025 0.0000175 0.00001225 0.000008575 \
    0.000006 0.000005 0.000005 0.000005 \
  --epoch-min-lrs \
    0.000050 0.000060 0.000040 0.000025 \
    0.0000175 0.00001225 0.000008575 0.000006 \
    0.000005 0.000005 0.000005 0.000005 \
  --checkpoint-samples 50000000 \
  --progress-batches 500 \
  --saturation-batches 20 \
  --auto-resume \
  --seed 20260824 >> "$log" 2>&1

printf 'state=verifying tag=%s updated_at=%s\n' \
  "$tag" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$status"
python3 tools/verify_phase_training_checkpoint.py \
  --checkpoint "$output_dir/phase_component_best.pt" \
  --arch F2M \
  --phase-layout independent > "$output_dir/checkpoint_verification.json"
python3 tools/export_phase_quantized_nnue.py \
  --checkpoint "$output_dir/phase_component_best.pt" \
  --output "$output_dir/phase_quantized_nnue.bin" \
  --parity-data "$data_dir" \
  --parity-output "$output_dir/phase_quantized_nnue_parity.tsv" \
  --parity-samples 16384 > "$output_dir/export.log"

cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build-release --target phase_quantized_nnue_tests -j 8 >/dev/null
build-release/phase_quantized_nnue_tests \
  "$output_dir/phase_quantized_nnue.bin" \
  "$output_dir/phase_quantized_nnue_parity.tsv" \
  > "$output_dir/cpp_parity.log"

python3 - "$output_dir/summary.json" "$report" <<'PY'
import json
import sys
from pathlib import Path

summary_path, report_path = map(Path, sys.argv[1:])
summary = json.loads(summary_path.read_text())
lines = [
    "# F2M independent — 500M until overfit",
    "",
    f"Selected epoch: **{summary['selected_epoch']}**.",
    "",
    "| Epoch | Train Huber | Train CP MAE | Validation Huber | Validation CP MAE | Best |",
    "|---:|---:|---:|---:|---:|:---:|",
]
for item in summary["history"]:
    lines.append(
        f"| {item['epoch']} | {item['train']['loss']:.8f} | "
        f"{item['train']['cp_mae']:.3f} | "
        f"{item['selection']['objective_loss']:.8f} | "
        f"{item['selection']['cp_mae']:.3f} | "
        f"{'yes' if item['improved'] else 'no'} |"
    )
ranking = summary.get("ranking")
if ranking:
    lines.extend([
        "", "## Sealed test", "",
        f"- Samples: {ranking['samples']}",
        f"- Huber loss: {ranking['objective_loss']:.8f}",
        f"- CP MAE: {ranking['cp_mae']:.3f}",
        f"- WDL score-only loss: {ranking['wdl_score_only_loss']:.8f}",
    ])
report_path.write_text("\n".join(lines) + "\n")
PY

printf 'state=complete tag=%s report=%s updated_at=%s\n' \
  "$tag" "$report" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$status"
trap - EXIT

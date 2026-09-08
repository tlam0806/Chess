#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 CHECKPOINT DATA REPORT_DIR [DEVICE]" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
checkpoint="$1"
data="$2"
report_dir="$3"
device="${4:-cpu}"
python_bin="${PYTHON_BIN:-$repo_root/.venv/bin/python}"
cxx="${CXX:-c++}"

cd "$repo_root"
mkdir -p "$report_dir"

"$python_bin" tools/analyze/audit_nnue_float_quantized.py \
  --checkpoint "$checkpoint" \
  --data "$data" \
  --data-format cbin \
  --device "$device" \
  --batch-size 8192 \
  --workers 4 \
  --max-samples 500000 \
  --split all \
  --output "$report_dir/python_float_quantized.json" \
  > "$report_dir/python_float_quantized.stdout"

"$python_bin" tools/train/export_quantized_candidate_fixture.py \
  --checkpoint "$checkpoint" \
  --data "$data" \
  --data-format cbin \
  --device "$device" \
  --batch-size 4096 \
  --workers 4 \
  --max-samples 500000 \
  --split all \
  --output "$report_dir/cpp_parity_fixture.bin"

"$cxx" -std=c++20 -O2 -Wall -Wextra -Wpedantic \
  tools/analyze/evaluate_quantized_cpp_fixture.cpp \
  -o "$report_dir/evaluate_quantized_cpp_fixture"

"$report_dir/evaluate_quantized_cpp_fixture" \
  "$report_dir/cpp_parity_fixture.bin" \
  > "$report_dir/cpp_report.json"

"$python_bin" tools/analyze/summarize_nnue_inference_audit.py \
  --python-report "$report_dir/python_float_quantized.json" \
  --cpp-report "$report_dir/cpp_report.json" \
  --output "$report_dir/summary.md"

#!/usr/bin/env bash
set -euo pipefail

MODEL_IN="${1:-models/nnue_value_mix_666k_hardneg_lr1e4.pt}"
MODEL_BIN="${2:-models/nnue_value_mix_666k_hardneg_lr1e4.bin}"
TAG="${3:-depth7_seed63013}"

RAW="data/nnue_spectator_gap_${TAG}.jsonl"
RAW_LOG="data/nnue_spectator_gap_${TAG}.log"
MIX="data/nnue_spectator_gap_${TAG}_finetune_mix.jsonl"
MIX_LOG="data/nnue_spectator_gap_${TAG}_finetune_mix.log"
MODEL_OUT="models/nnue_value_mix_666k_spectator_${TAG}.pt"
MODEL_OUT_BIN="models/nnue_value_mix_666k_spectator_${TAG}.bin"
TRAIN_LOG="data/nnue_value_mix_666k_spectator_${TAG}_train.log"
EVAL_LOG="data/nnue_value_mix_666k_spectator_${TAG}_baseline_eval.log"
COMPARE="data/nnue_value_mix_666k_spectator_${TAG}_compare.jsonl"

./build/nnue_spectator_gap_export \
  --book data/opening_book_6plies.txt \
  --model "${MODEL_BIN}" \
  --output "${RAW}" \
  --depth 7 \
  --max-plies 90 \
  --max-samples 22000 \
  --min-gap 180 \
  --time-limit-seconds 10200 \
  --progress-interval 500 \
  --seed 63013 \
  2>&1 | tee "${RAW_LOG}"

.venv/bin/python tools/mix_jsonl_dataset.py \
  --output "${MIX}" \
  --seed 63014 \
  --source "${RAW}:40000" \
  --source data/nnue_hard_negative_depth8_seed13.jsonl:10000 \
  --source data/nnue_baseline_capture_test5k.jsonl:5000 \
  --source data/nnue_baseline_check_test5k.jsonl:5000 \
  --source data/nnue_baseline_tactical_test5k.jsonl:5000 \
  --source data/nnue_mix_train_600k.jsonl:25000 \
  2>&1 | tee "${MIX_LOG}"

.venv/bin/python -m nn.train_nnue_value \
  --train "${MIX}" \
  --val data/nnue_mix_val_20k.jsonl \
  --test data/nnue_baseline_tactical_test5k.jsonl \
  --init-checkpoint "${MODEL_IN}" \
  --epochs 3 \
  --batch-size 512 \
  --lr 0.0001 \
  --weight-decay 0.0001 \
  --device cpu \
  --output "${MODEL_OUT}" \
  2>&1 | tee "${TRAIN_LOG}"

.venv/bin/python tools/evaluate_nnue_value.py \
  --checkpoint "${MODEL_OUT}" \
  --batch-size 512 \
  --device cpu \
  data/nnue_baseline_random_test5k.jsonl \
  data/nnue_baseline_material_test5k.jsonl \
  data/nnue_baseline_capture_test5k.jsonl \
  data/nnue_baseline_check_test5k.jsonl \
  data/nnue_baseline_tactical_test5k.jsonl \
  2>&1 | tee "${EVAL_LOG}"

.venv/bin/python tools/export_nnue_value_net.py \
  --checkpoint "${MODEL_OUT}" \
  --output "${MODEL_OUT_BIN}" \
  --compare-input data/nnue_baseline_tactical_test5k.jsonl \
  --compare-output "${COMPARE}" \
  --compare-limit 1000

./build/nnue_value_tests "${MODEL_OUT_BIN}" "${COMPARE}"
./build/nnue_searcher_v10_tests "${MODEL_OUT_BIN}"

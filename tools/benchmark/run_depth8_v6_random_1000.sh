#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p data models

DATASET="data/value_depth8_v6_random_1000games.jsonl"
CHECKPOINT="models/value_net_depth8_v6_random_1000games.pt"
BINARY="models/value_net_depth8_v6_random_1000games.bin"
COMPARE="data/nn_value_depth8_v6_random_1000games_compare.jsonl"

echo "[pipeline] start $(date)"

echo "[export] start $(date)"
./build/dataset_export \
  --games 1000 \
  --depth 8 \
  --random-plies 160 \
  --max-plies 160 \
  --progress-interval 10 \
  --threads 4 \
  --seed 20260612 \
  --teacher v6 \
  --output "${DATASET}"
echo "[export] done $(date) lines=$(wc -l < "${DATASET}")"

echo "[train] start $(date)"
./.venv/bin/python -m chess_nnue.train_value \
  --train "${DATASET}" \
  --epochs 10 \
  --batch-size 1024 \
  --device cpu \
  --output "${CHECKPOINT}"
echo "[train] done $(date)"

echo "[export-bin] start $(date)"
./.venv/bin/python tools/train/export_value_net.py \
  --checkpoint "${CHECKPOINT}" \
  --output "${BINARY}" \
  --compare-input "${DATASET}" \
  --compare-output "${COMPARE}" \
  --compare-limit 10000

echo "[pipeline] done $(date)"

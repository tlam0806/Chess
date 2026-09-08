#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p build data models

echo "[stream 100k] start $(date)"

./build/dataset_export \
  --games 1000 \
  --depth 2 \
  --random-plies 4 \
  --max-plies 60 \
  --progress-interval 100 \
  --seed 2001 \
  --output data/value_val_1k_stream.jsonl

./build/dataset_export \
  --games 1000 \
  --depth 2 \
  --random-plies 4 \
  --max-plies 60 \
  --progress-interval 100 \
  --seed 3001 \
  --output data/value_test_baseline_1k_stream.jsonl

./build/dataset_export \
  --games 100000 \
  --depth 2 \
  --random-plies 4 \
  --max-plies 60 \
  --progress-interval 1000 \
  --seed 1001 \
  --output /dev/stdout \
  | ./.venv/bin/python -m chess_nnue.train_stream_value \
      --val data/value_val_1k_stream.jsonl \
      --test data/value_test_baseline_1k_stream.jsonl \
      --batch-size 1024 \
      --device cpu \
      --output models/value_net_stream_100k.pt \
      --log-interval 100000 \
      --eval-interval 1000000

echo "[stream 100k] done $(date)"

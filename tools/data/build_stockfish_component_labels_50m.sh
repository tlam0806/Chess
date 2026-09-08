#!/usr/bin/env bash
set -euo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SOURCE_TRAIN="data/stockfish_components_source_50m_v1"
SOURCE_VAL="data/stockfish_components_source_val_1m_v1"
COMMIT="ebcea3efe9c1b8748e080111c727c33c544d7e06"
NETWORK="build/stockfish_static_nnue_upstream/src/nn-0ee0657fb25e.nnue"
TAG="stockfish_components_50m_v1"
STATUS="logs/${TAG}.status"

mkdir -p logs data
printf 'state=running updated_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS"

label_component() {
  local component="$1"
  local source="$2"
  local output="$3"
  local manifest="$4"
  local workers="$5"
  .venv/bin/python tools/data/label_stockfish_static_nnue_shards.py \
    --source-dir "$source" \
    --output-dir "$output" \
    --label-component "$component" \
    --workers "$workers" \
    --records-per-shard 1000000 \
    --zstd-level 6 \
    --manifest "$manifest" \
    --stockfish-commit "$COMMIT" \
    --network "$NETWORK"
}

label_component psqt "$SOURCE_TRAIN" \
  data/stockfish_static_nnue_psqt_component_50m_v1 \
  data/stockfish_static_nnue_psqt_component_50m_v1.manifest.json 5
label_component positional "$SOURCE_TRAIN" \
  data/stockfish_static_nnue_positional_component_50m_v1 \
  data/stockfish_static_nnue_positional_component_50m_v1.manifest.json 5
label_component psqt "$SOURCE_VAL" \
  data/stockfish_static_nnue_psqt_component_val_1m_v1 \
  data/stockfish_static_nnue_psqt_component_val_1m_v1.manifest.json 1
label_component positional "$SOURCE_VAL" \
  data/stockfish_static_nnue_positional_component_val_1m_v1 \
  data/stockfish_static_nnue_positional_component_val_1m_v1.manifest.json 1

printf 'state=complete updated_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS"

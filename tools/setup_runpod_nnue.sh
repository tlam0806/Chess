#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if command -v apt-get >/dev/null 2>&1; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        build-essential \
        cmake \
        git \
        python3-venv \
        rsync \
        tmux \
        zstd
fi

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy zstandard

if ! python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("torch") is not None else 1)
PY
then
    python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch
fi

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY

python -m py_compile \
    nn/compact_board_data.py \
    nn/nnue_architectures.py \
    tools/train_quantized_nnue_architecture.py \
    tools/evaluate_quantized_nnue_checkpoint.py \
    tools/quantized_pipeline_preflight.py \
    tools/quantized_training_log.py \
    tools/run_quantized_scale_grid.py \
    tools/run_quantized_activation_stage2.py \
    nn/quantized_nnue_architectures.py

echo
echo "Ready. Example 12h scale-tune command:"
echo
echo "tmux new -s scale"
echo "source .venv/bin/activate"
echo "python tools/run_quantized_scale_grid.py --data /workspace/data/robotmoon_cbin_v2_shards --data-format cbin --architectures E2 F2 --hidden-scales '64 128 256' --output-scales '8 16 32' --device cuda --jobs 1 --epochs 5 --train-max-samples 2000000 --batch-size 8192 --lr 0.0005 --max-hours 12"

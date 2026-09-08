# Dense-model prototype

Before the current NNUE pipeline, the project used a small dense value network
and a standalone neural search engine. This document preserves the commands for
reconstructing that historical pipeline; it does not describe the production
NNUE.

Run the commands from the repository root.

## Dataset export

Build the dataset exporter:

```sh
cmake --build build --target dataset_export -j
mkdir -p data
```

Generate game states with random opening plies, then engine-selected moves:

```sh
build/dataset_export \
  --games 1000 \
  --depth 2 \
  --random-plies 4 \
  --max-plies 60 \
  --progress-interval 100 \
  --seed 1001 \
  --output data/value_games.jsonl
```

## Training from saved JSONL

```sh
PYTHONPATH=python .venv/bin/python -m chess_nnue.train_value \
  --train data/value_train.jsonl \
  --val data/value_val.jsonl \
  --test data/value_test_baseline.jsonl \
  --epochs 10 \
  --batch-size 1024 \
  --device cpu \
  --output models/value_net.pt
```

The `--test` file is a held-out baseline. It is not used for training.

## Stream training

For large self-play runs, training data should not be fully written to disk.

Use streaming:

```sh
build/dataset_export \
  --games 100000 \
  --depth 2 \
  --random-plies 4 \
  --max-plies 60 \
  --progress-interval 1000 \
  --seed 1001 \
  --output /dev/stdout \
  | env PYTHONPATH=python .venv/bin/python -m chess_nnue.train_stream_value \
      --val data/value_val_1k_stream.jsonl \
      --test data/value_test_baseline_1k_stream.jsonl \
      --batch-size 1024 \
      --device cpu \
      --output models/value_net_stream.pt \
      --log-interval 100000 \
      --eval-interval 1000000
```

This keeps train data out of disk. Only validation/test baseline files are
stored.

## Export for C++ inference

PyTorch is only needed for training and exporting. Runtime inference in the
engine uses a small C++ forward pass.

Export a trained checkpoint:

```sh
.venv/bin/python tools/train/export_value_net.py \
  --checkpoint models/value_net_stream_100k.pt \
  --output models/value_net_stream_100k.bin \
  --compare-input data/value_test_baseline_1k_stream.jsonl \
  --compare-output data/nn_value_compare_1000.jsonl \
  --compare-limit 1000
```

Build and run the C++ parity test:

```sh
cmake --build build --target nn_value_tests -j

build/nn_value_tests \
  models/value_net_stream_100k.bin \
  data/nn_value_compare_1000.jsonl
```

The test compares C++ inference against Python/PyTorch output on encoded
samples.

## Search engine

Build the standalone NN search engine:

```sh
cmake --build build --target nn_engine -j
```

Search one position:

```sh
build/nn_engine \
  --model models/value_net_stream_100k.bin \
  --depth 3 \
  --go-once
```

Use a custom FEN:

```sh
build/nn_engine \
  --model models/value_net_stream_100k.bin \
  --depth 3 \
  --fen "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1" \
  --go-once
```

Without `--go-once`, it starts an interactive CLI where the NN engine plays
Black.

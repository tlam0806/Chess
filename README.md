# Chess

A small C++20 chess engine project built from bitboards upward.

Current scope includes:

- Bitboard position representation
- FEN parsing
- Attack generation
- Pseudo-legal and legal move generation
- Make move, castling, en passant, promotion
- Perft validation
- Basic evaluation
- Negamax alpha-beta search
- Sparse board encoding for NN experiments
- A small browser UI for playing against the engine
- A PyTorch value-network training script

## Build

```sh
mkdir -p build
c++ -std=c++20 -O2 -Wall -Wextra -Wpedantic -Iinclude \
  src/bitboard.cpp src/position.cpp src/attacks.cpp src/move.cpp \
  src/perft.cpp src/evaluate.cpp src/search.cpp src/board_encoder.cpp \
  src/main.cpp -o build/chess
```

## Dataset Export

```sh
mkdir -p build data
c++ -std=c++20 -O2 -Wall -Wextra -Wpedantic -Iinclude \
  src/bitboard.cpp src/position.cpp src/attacks.cpp src/move.cpp \
  src/perft.cpp src/evaluate.cpp src/search.cpp src/board_encoder.cpp \
  tools/dataset_export.cpp -o build/dataset_export

build/dataset_export --games 10 --depth 2 --random-plies 8 --max-plies 80 \
  --output data/value_games.jsonl
```

## Train Value Network

```sh
python -m nn.train_value \
  --train data/value_games_train.jsonl \
  --val data/value_games_val.jsonl \
  --epochs 10 \
  --batch-size 64 \
  --device cpu \
  --output models/value_net.pt
```

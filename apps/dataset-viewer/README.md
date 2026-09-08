# Dataset viewer

This local browser app can play against the earlier NNUE engine, replay match
logs, and inspect generated datasets. Its engine API uses the retained V36
searcher, so build it from an experiments-enabled CMake tree:

```sh
cmake -S . -B build-experiments -DCMAKE_BUILD_TYPE=Release \
  -DCHESS_BUILD_EXPERIMENTS=ON
cmake --build build-experiments --target chess_engine_api -j
python apps/dataset-viewer/server.py
```

Then open <http://127.0.0.1:8765>. The app expects the default NNUE model and
local datasets under the repository's `models/` and `data/` compatibility
paths. Set `CHESS_ENGINE_API` to use a binary from another build directory.

# Tooling

Repository utilities are grouped by the job they perform:

| Directory | Purpose |
|---|---|
| [`analyze/`](analyze/) | Evaluate experiment output and produce summaries or evidence. |
| [`benchmark/`](benchmark/) | Measure engine performance; low-level profilers live in `benchmark/profile/`. |
| [`data/`](data/) | Generate, convert, shard and validate datasets or opening books. |
| [`debug/`](debug/) | Reproduce failures and inspect engine or protocol behavior. |
| [`match/`](match/) | Run engine matches, gauntlets and self-play. |
| [`ops/`](ops/) | Provide UCI entry points and local or hosted operational helpers. |
| [`train/`](train/) | Train, export and inspect neural-network checkpoints. |
| [`tune/`](tune/) | Search pruning, ordering and model configurations. |
| [`third_party/`](third_party/) | Keep vendored support code isolated from project-owned tools. |

Run Python and shell tools from the repository root so their imports and
relative paths resolve consistently. C++ utilities are exposed as named CMake
targets in [`cmake/Tools.cmake`](../cmake/Tools.cmake) or
[`cmake/Experiments.cmake`](../cmake/Experiments.cmake); user-facing UCI
targets are defined in [`cmake/Production.cmake`](../cmake/Production.cmake).
These research targets are excluded from the lean production configuration.
Configure their build tree with `-DCHESS_BUILD_EXPERIMENTS=ON`:

```sh
cmake -S . -B build-experiments -DCMAKE_BUILD_TYPE=Release \
  -DCHESS_BUILD_EXPERIMENTS=ON
```

The three low-level transposition-table profilers alter instrumented class
layouts, so each must use its own consistently compiled build tree. Select one
with `CHESS_TT_PROFILE_MODE`: `timing`, `probe-paths`, or `cache-lines`. For
example:

```sh
cmake -S . -B build-tt-timing -DCMAKE_BUILD_TYPE=Release \
  -DCHESS_BUILD_EXPERIMENTS=ON \
  -DCHESS_TT_PROFILE_MODE=timing
cmake --build build-tt-timing --target profile_v32_v34_tt_timing -j
```

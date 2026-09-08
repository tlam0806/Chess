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

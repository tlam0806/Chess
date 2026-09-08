# Current NNUE searchers

This directory contains the current V43 production implementation and the V44
candidate line. Their corresponding source files live under
`src/searchers/strict`.

The older correctness-first search lineage is retained under `experiments/`:
classical Strict V15-V35 and NNUE V36-V42. Those implementations remain part of
the build because milestone benchmarks and regression tests compare them
directly with later versions.

# Fast searchers

Searchers in this folder are the performance line.

Rules:
- Allow practical engine shortcuts such as deeper TT reuse (`entry.depth >= requested_depth`).
- This line may use PVS or other optimizations that are benchmarked against the strict line.
- Do not use this as the first place to debug exact fixed-depth semantics.

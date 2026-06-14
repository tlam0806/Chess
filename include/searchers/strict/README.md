# Strict searchers

Searchers in this folder are the correctness/debug line.

Rules:
- Prefer exact-depth TT score reuse (`entry.depth == requested_depth`).
- Keep behavior easier to compare against fixed-depth search.
- Add new search features here first when debugging score or move mismatches.

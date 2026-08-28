# Strict searchers

Searchers in this folder are the correctness/debug line.

Rules:
- Require exact-depth TT score reuse (`entry.depth == requested_depth`) for
  versions classified as Strict.
- Keep behavior easier to compare against fixed-depth search.
- Add new search features here first when debugging score or move mismatches.

V34 is intentionally excluded from this folder because its `AtLeast` TT policy
can change the score requested at a shallower fixed depth. V35 returns to
exact-depth TT score reuse and remains in the Strict line.

# Fast searchers

Searchers in this folder are the performance line.

Rules:
- Allow practical engine shortcuts such as deeper TT reuse (`entry.depth >= requested_depth`).
- This line may use PVS or other optimizations that are benchmarked against the strict line.
- Do not use this as the first place to debug exact fixed-depth semantics.

V34's compact single-bound TT experiment belongs here because it uses
`AtLeast` depth reuse. Its historical class name remains
`HeuristicSearcherV34`; the directory records its semantic contract without
renaming the retained benchmark version. Fast describes the relaxed contract;
it does not imply that V34 was promoted or proved stronger.

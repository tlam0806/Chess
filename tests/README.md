# Tests

Tests are split by runtime:

| Directory | Purpose |
|---|---|
| [`cpp/`](cpp/) | Engine rules, move generation, search, evaluation and table tests. |
| [`python/`](python/) | Unit and integration tests for data, training, tuning, benchmarking and operational tools. |

A model-independent C++ smoke suite can be run with:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target \
  perft_tests movegen_regression_tests \
  v43_single_bound_transposition_table_tests -j
ctest --test-dir build --output-on-failure \
  -R '^(perft_tests|movegen_regression_tests|v43_single_bound_transposition_table_tests)$'
```

Historical searcher tests and comparison harnesses require an experiments build:

```sh
cmake -S . -B build-experiments -DCMAKE_BUILD_TYPE=Release \
  -DCHESS_BUILD_EXPERIMENTS=ON
cmake --build build-experiments --target \
  search_tests nnue_searcher_v43_single_bound_tests -j
```

The V43 single-bound search test is in this opt-in group because it compares
the production searcher against the historical V41 implementation. The V43
control-cache and final-tune evaluator tests similarly retain a V36 control.

Most C++ tests are registered with CTest; a few artifact-specific or expensive
executables remain opt-in build targets.

From the repository root, install the declared test dependencies and run the
Python suite with:

```sh
python -m pip install -e ".[test]"
python -m pytest
```

The base package supplies PyTorch; the `test` extra adds NumPy, pytest and
python-chess. Matplotlib and boto3 are only needed by the plotting and RunPod
utilities and are available as the `plot` and `runpod` extras. The platform
`zstd` executable is required only for workflows that read or write `.zst`
corpora; it is not installed by pip. Some artifact-specific NNUE tests
additionally require local model or dataset files. CI intentionally keeps its
default gate independent of those resources.

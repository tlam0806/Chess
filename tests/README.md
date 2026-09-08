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
  perft_tests movegen_regression_tests search_tests \
  v43_single_bound_transposition_table_tests -j
ctest --test-dir build --output-on-failure \
  -R '^(perft_tests|movegen_regression_tests|search_tests|v43_single_bound_transposition_table_tests)$'
```

Most C++ tests are registered with CTest; a few artifact-specific or expensive
executables remain opt-in build targets.

After installing the dependencies for the subsystem under test, run the complete
Python suite with `python -m pytest tests/python`. The suite spans PyTorch,
NumPy, Matplotlib, python-chess, cloud SDKs and platform tools such as `zstd`;
some NNUE tests also require local model or dataset artifacts. CI intentionally
keeps its default gate independent of those resources.

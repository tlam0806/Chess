# Python package

[`chess_nnue/`](chess_nnue/) contains the reusable Python code behind the
neural-evaluation pipeline. The package includes compact corpus I/O, training
targets, dense-model history, NNUE architectures, checkpoint transforms and
training entry points.

The root `pyproject.toml` maps this directory as the package source. Install the
package and its NumPy/PyTorch dependencies from the repository root with:

```sh
python -m pip install -e ".[ml]"
```

The Python package declares PyTorch as a core dependency because its public
modules import it directly; this does not affect CMake-only engine builds. The
`ml` extra adds NumPy for training and analysis tools. Use `.[test]` to install
the Python test dependencies. Plotting utilities and the RunPod upload helper
have separate `.[plot]` and `.[runpod]` extras; extras can be combined, for
example `.[test,plot,runpod]`.

The compressed-corpus tools invoke the platform `zstd` executable. Install it
with the operating-system package manager when working with `.zst` data; it is
not a Python dependency.

Experiment orchestration remains under `tools/`; reusable model and data logic
belongs here so it can be imported consistently by tools and tests.

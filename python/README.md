# Python package

[`chess_nnue/`](chess_nnue/) contains the reusable Python code behind the
neural-evaluation pipeline. The package includes compact corpus I/O, training
targets, dense-model history, NNUE architectures, checkpoint transforms and
training entry points.

The root `pyproject.toml` maps this directory as the package source. After
provisioning PyTorch for the local platform, install the project package from
the repository root:

```sh
python -m pip install -e .
```

The package metadata currently installs the project code only; it does not
manage the heavyweight training dependencies.

Experiment orchestration remains under `tools/`; reusable model and data logic
belongs here so it can be imported consistently by tools and tests.

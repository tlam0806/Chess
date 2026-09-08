# Local artifacts

This directory is the local home for large, generated, or machine-specific
files that do not belong in Git:

- `datasets/` — training and benchmark corpora;
- `models/` — checkpoints and exported networks;
- `runs/` — logs and temporary run state;
- `reports/` — generated experiment reports;
- `notes/` — private development handoffs.

Existing scripts still accept the historical root paths (`data/`, `models/`,
`logs/`, and `reports/`). A developer workspace may keep those names as local
symlinks into this directory while the contents remain ignored.

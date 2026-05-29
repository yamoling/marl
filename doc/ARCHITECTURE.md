# Repository Architecture

Summary: This document describes the high-level layout of the repository and where to find main components.

Top-level layout
- `pyproject.toml`, `README.md`, `test.toml`: project metadata and quickstart.
- `src/`: primary Python entry points and package code.
- `src/marl/`: core multi-agent RL implementation and extensions.
- `src/ui/`: user-facing tools (visualization / web UI).
- `examples/`: runnable examples and quickstart scripts (`train_example.py`, `plot_results.py`).
- `maps/`: environment maps used for some experiments
- `logs/`: main experiment results and metadata (not committed, but expected to be created by users).
- `doc/`: documentation and reference material.
- `tests/`: unit and integration tests (often break, not actively maintained).

Key scripts and their roles
- `start_run.py`: launches experimental runs (trainer bootstrapping, config parsing). This script can be used to start new runs of an existing experiment.
- `serve.py`: serves results in a simple UI for browsing experiment outputs.
- `distil.py`, `tuning.py`: utilities for distillation and hyperparameter search.
All other scripts are playgrounds for testing new ideas, running experiments, and visualizing results. They are not expected to be stable or well-maintained, but can be used as starting points for new work.


Where to look for common tasks
- Create your own experiment in a new script in `src/` that imports the relevant trainer and environment, creates an `Experiment` object with the relevant parameters, and saves it to the `logs/` folder.
- Add new environment map: `maps/` (create a .toml and add to examples).
- Add a new algorithm: `src/marl/algos`.
- Run an existing experiment: `start_run.py`


References
- See `pyproject.toml` and `README.md` for setup and dependency information.

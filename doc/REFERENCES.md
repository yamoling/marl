# Additional Reference Documents and Next Topics

This file lists existing reference material and suggested future documents the team will find useful.

Existing docs to consult
- `README.md`: project overview and quickstart.
- `pyproject.toml`: dependency and packaging details.
- `archives/*/experiment.json`: canonical experiment metadata format used throughout the repo.
- `tests/`: examples of unit tests and expected structure.

Suggested reference documents to add (priority order)
1. Experiment runner guide — describe `start_run.py`, recommended flags, GPU selection, checkpointing, and expected outputs.
2. Experiment metadata spec — formalize the fields in `experiment.json` (config schema, URIs, commit hash, seed).
3. Data & artifact storage guide — where to store checkpoints, naming conventions, and retention policy.
4. Map format specification — document TOML keys used in `maps/` and best practices for environment authors.
5. Contribution guide — branch/PR workflow, code style, testing expectations, and release notes.
6. Performance troubleshooting — tips for profiling, common bottlenecks, and recommended profiling tools.

Where to host long docs
- Keep short guidance in `doc/` and larger playbooks in `papers/` or a `docs/website/` if the team wants rendered docs.

How to extend these docs
- Open a PR with the new markdown files under `doc/` and link them from `README.md`.

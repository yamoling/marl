# AGENTS: Guidance for AI coding agents

Purpose: provide concise, actionable guidance so AI coding agents become productive quickly in this repo.

Quick setup
- **Activate virtualenv:** source .venv/bin/activate
- **Install dependencies:** `uv sync -U` (see [README.md](README.md))

Run & test
- **Run an example experiment:** `python src/test.py`
- **Start a run from an existing experiment:** `python start_run.py` (see [doc/ARCHITECTURE.md](doc/ARCHITECTURE.md#L1))
- **Serve UI:** `python src/serve.py`
- **Run tests:** `pytest -q` (pytest configured in `pyproject.toml`) (note: poorly maintained)

Key locations
- **Project root:** [pyproject.toml](pyproject.toml) — dependencies and test config
- **Quickstart & overview:** [README.md](README.md)
- **Architecture notes:** [doc/ARCHITECTURE.md](doc/ARCHITECTURE.md)
- **Examples:** [examples/](examples/)
- **Core code:** [src/](src/) — package source under `src/marl/`
- **Tests:** [tests/](tests/)
- **Experiment scripts:** [src/test.py](src/test.py), [start_run.py](start_run.py)

Conventions & tips for agents
- The repo uses a `src/` layout; add new imports relative to `src` or update `pyproject.toml` if needed.
- Prefer linking to docs rather than copying them into this file.
- Keep changes minimal and consistent with existing style (line length 140, typed code, `basedpyright` configured).

When to update this file
- Add pointers here for new subsystems (e.g., `src/ui`, `examples/`, experiment runner) so future agents can discover them quickly.


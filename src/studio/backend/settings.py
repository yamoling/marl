import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DIST_DIR = PROJECT_ROOT / "src" / "studio" / "frontend" / "dist"
START_RUN_SCRIPT = PROJECT_ROOT / "scripts" / "start_run.py"

HEALTH_TIMEOUT_S = 20.0
"""Timeout of the full deserialization check (launch/replay capabilities)."""
LAUNCH_EARLY_FAILURE_S = 2.0
"""Window during which a failure of the spawned `start_run.py` is reported to the client."""
EVENTS_RUNNING_INTERVAL_S = 2.0
"""Polling interval of the running experiments' tables (SSE `run-progress`)."""
EVENTS_SCAN_INTERVAL_S = 10.0
"""Interval of the logs root scan (SSE `experiment-added/removed/changed`)."""
EVENTS_PING_INTERVAL_S = 15.0
SYSTEM_WS_INTERVAL_S = 0.5
MAX_SERIES_QUERIES = 200


def workspaces_file(isolated_root: Path | None = None) -> Path:
    """Keep explicitly supplied test roots isolated from the user's Studio settings. @ai-generated"""
    if isolated_root is not None:
        return isolated_root / ".studio-workspaces.json"
    return Path(os.environ.get("MARL_STUDIO_WORKSPACES", Path.home() / ".config" / "marl-studio" / "workspaces.json")).expanduser()


def logs_root() -> Path:
    """
    Root directory containing the experiments, independent of the server's cwd.
    Overridable with the `MARL_STUDIO_LOGS` environment variable (used by tests).

    @ai-generated
    """
    return Path(os.environ.get("MARL_STUDIO_LOGS", PROJECT_ROOT / "logs")).resolve()

"""Stopping runs, renaming and deleting experiments (raw-JSON implementation, no `marl` import)."""

import os
import shutil
import time
from pathlib import Path
from signal import SIGINT
from typing import Any

import orjson
import psutil

from ..data.library import Library
from ..data.records import EXPERIMENT_FILE, RUN_FILE, ExperimentRecord, RunRecord, run_dirs
from ..errors import bad_request, conflict, not_found
from ..security import launcher_of, run_process, safe_new_experiment_path, stop_verified_run

STOP_DEADLINE_S = 30.0


def stop_run(library: Library, record: ExperimentRecord, run: RunRecord) -> bool:
    """@ai-generated"""
    stopped = stop_verified_run(run.path, record.path, library.root_for(record.path))
    library.invalidate(record.id)
    return stopped


def stop_experiment(library: Library, record: ExperimentRecord, deadline_s: float = STOP_DEADLINE_S):
    """
    Stop every active run and its verified launcher, including queued runs that start after the
    current ones, until no run is active or the deadline is reached.

    @ai-generated
    """
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        active = [
            (rundir, p)
            for rundir in run_dirs(record.path)
            if (p := run_process(rundir, record.path, library.root_for(record.path))) is not None
        ]
        if not active:
            break
        launchers = {}
        for _, process in active:
            launcher = launcher_of(process, record.path, library.root_for(record.path))
            if launcher is not None and launcher.pid != process.pid:
                launchers[launcher.pid] = launcher
        for rundir, _ in active:
            stop_verified_run(rundir, record.path, library.root_for(record.path))
        for launcher in launchers.values():
            try:
                launcher.send_signal(SIGINT)
            except psutil.NoSuchProcess:
                pass
        time.sleep(1)
    library.invalidate(record.id)


def ensure_inactive(library: Library, record: ExperimentRecord):
    """409 if any run has a verified process; 403 if a pid cannot be verified. @ai-generated"""
    for rundir in run_dirs(record.path):
        if run_process(rundir, record.path, library.root_for(record.path)) is not None:
            raise conflict("The experiment has active runs", "runs-active")


def _stored_path(stored: Any, new_path: Path, root: Path) -> str:
    """New `logdir`/`rundir` value, in the style of the stored one (absolute, or relative to the root's parent). @ai-generated"""
    if isinstance(stored, str) and Path(stored).is_absolute():
        return str(new_path)
    return new_path.relative_to(root.parent).as_posix()


def _rewrite(path: Path, key: str, new_path: Path, root: Path):
    """Rewrite one key of a JSON object file atomically, other keys and their order untouched. @ai-generated"""
    try:
        raw = orjson.loads(path.read_bytes())
    except (OSError, orjson.JSONDecodeError):
        return
    if not isinstance(raw, dict):
        return
    raw[key] = _stored_path(raw.get(key), new_path, root)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(orjson.dumps(raw))
    os.replace(tmp, path)


def rename(library: Library, record: ExperimentRecord, new_id: Any) -> str:
    """
    Move the experiment directory to `new_id`, then rewrite `logdir` in experiment.json and `rundir`
    in each run.json. Works for degraded experiments. Refused while runs are active.

    @ai-generated
    """
    if not isinstance(new_id, str):
        raise bad_request("new_id must be a string")
    root = library.root_for(record.path)
    if library.resolve(new_id) is not None:
        raise conflict("Destination already exists in a loaded logdir", "destination-exists")
    target = safe_new_experiment_path(root, new_id)
    if target.exists() or target.is_symlink():
        raise conflict("Destination already exists", "destination-exists")
    if target.is_relative_to(record.path):
        raise bad_request("Cannot move an experiment into itself")
    if not target.parent.is_dir():
        raise not_found("Destination parent does not exist")
    ensure_inactive(library, record)
    shutil.move(record.path, target)
    if (target / EXPERIMENT_FILE).is_file():
        _rewrite(target / EXPERIMENT_FILE, "logdir", target, root)
    for rundir in run_dirs(target):
        if (rundir / RUN_FILE).is_file():
            _rewrite(rundir / RUN_FILE, "rundir", rundir, root)
    library.invalidate(record.id)
    new = target.relative_to(root).as_posix()
    library.invalidate(new)
    return new


def delete(library: Library, record: ExperimentRecord):
    """Delete the validated experiment directory; refused while runs are active. @ai-generated"""
    ensure_inactive(library, record)
    shutil.rmtree(record.path)
    library.invalidate(record.id)

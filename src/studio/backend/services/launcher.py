"""
Launching runs through `scripts/start_run.py` (logic of the old `ServerState.new_runs`/`start_run`),
guarded by the launch capability and a seed-collision check.
"""

import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from http import HTTPStatus
from pathlib import Path
from typing import Any

import orjson

from .. import settings
from ..data import extract
from ..data.issues import Issue, Level
from ..data.library import Library
from ..data.records import LAUNCH_FAILURE_FILE, ExperimentRecord, RunRecord
from ..errors import ApiError, bad_request, conflict

logger = logging.getLogger(__name__)

DEFAULTS = {"n_tests": 1, "test_interval": 5000, "save_weights": False, "save_actions": True}
LAUNCH_FIELDS = {
    "n_runs",
    "seed",
    "n_tests",
    "test_interval",
    "n_jobs",
    "device",
    "gpu_strategy",
    "disabled_devices",
    "save_weights",
    "save_actions",
}
_RUN_DIR_RE = re.compile(r"run-(\d+)")
_popen = subprocess.Popen
"""Indirection so that tests can replace the spawn without patching `subprocess` globally."""


def cuda_device_count() -> int:
    """Number of visible CUDA devices (torch is imported lazily). @ai-generated"""
    import torch

    return torch.cuda.device_count()


@dataclass
class LaunchParams:
    n_runs: int
    n_tests: int
    seed: int
    n_jobs: int = 1
    test_interval: int = 5000
    device: str | int = "auto"
    gpu_strategy: str = "group"
    disabled_devices: list[int] = field(default_factory=list)
    save_weights: bool = False
    save_actions: bool = True

    @property
    def seeds(self) -> range:
        return range(self.seed, self.seed + self.n_runs)


def validate(params: LaunchParams) -> str:
    """
    Validation of the old `ServerState.new_runs`, verbatim. Returns the `--device` CLI argument.
    Raises ValueError.

    @ai-generated
    """
    n_runs, n_tests, seed, n_jobs, test_interval = params.n_runs, params.n_tests, params.seed, params.n_jobs, params.test_interval
    device, gpu_strategy, disabled_devices = params.device, params.gpu_strategy, params.disabled_devices
    save_weights, save_actions = params.save_weights, params.save_actions
    for name, value, minimum in (
        ("n_runs", n_runs, 1),
        ("n_tests", n_tests, 1),
        ("test_interval", test_interval, 1),
        ("n_jobs", n_jobs, 1),
        ("seed", seed, 0),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if gpu_strategy not in ("group", "scatter") or type(gpu_strategy) is not str:
        raise ValueError("gpu_strategy must be 'group' or 'scatter'")
    if type(save_weights) is not bool or type(save_actions) is not bool:
        raise ValueError("save_weights and save_actions must be booleans")
    if disabled_devices is None:
        disabled_devices = []
    if type(disabled_devices) is not list or any(type(index) is not int or index < 0 for index in disabled_devices):
        raise ValueError("disabled_devices must be a list of non-negative GPU indices")
    if len(set(disabled_devices)) != len(disabled_devices):
        raise ValueError("disabled_devices must not contain duplicates")
    if type(device) is int:
        gpu_index = device
    elif type(device) is str and device in ("auto", "cpu", "cuda"):
        gpu_index = 0 if device == "cuda" else None
    elif type(device) is str and re.fullmatch(r"cuda:(0|[1-9][0-9]*)", device):
        gpu_index = int(device[5:])
    else:
        raise ValueError("device must be 'auto', 'cpu', 'cuda', or a non-negative GPU index")
    if gpu_index is not None and (gpu_index < 0 or gpu_index in disabled_devices or gpu_index >= cuda_device_count()):
        raise ValueError(f"GPU {gpu_index} is disabled or unavailable")
    # CLI arguments are strings; --device=0 would not round-trip as an integer.
    return f"cuda:{device}" if type(device) is int else str(device)


def parse_body(body: Any) -> LaunchParams:
    """The `POST /api/experiments/{id}/runs` body; `n_runs`, `seed` and `n_tests` are required. @ai-generated"""
    if not isinstance(body, dict):
        raise bad_request("Expected a JSON object")
    unknown = set(body) - LAUNCH_FIELDS
    if unknown:
        raise bad_request(f"Unknown fields: {', '.join(sorted(unknown))}")
    missing = [k for k in ("n_runs", "seed", "n_tests") if k not in body]
    if missing:
        raise bad_request(f"Missing fields: {', '.join(missing)}")
    params = LaunchParams(**body)
    if params.disabled_devices is None:
        params.disabled_devices = []
    return params


def command(record: ExperimentRecord, params: LaunchParams, device_arg: str) -> list[str]:
    """@ai-generated"""
    cmd = [
        sys.executable,
        str(settings.START_RUN_SCRIPT),
        str(record.path),
        f"--n-runs={params.n_runs}",
        f"--n-tests={params.n_tests}",
        f"--test-interval={params.test_interval}",
        f"--seed={params.seed}",
        f"--device={device_arg}",
        f"--gpu-strategy={params.gpu_strategy}",
        f"--n-jobs={params.n_jobs}",
    ]
    if params.save_weights:
        cmd.append("--save-weights")
    if not params.save_actions:
        cmd.append("--no-save-actions")
    if len(params.disabled_devices) > 0:
        cmd.extend(["--disabled-devices", *[str(device_id) for device_id in params.disabled_devices]])
    return cmd


def _stderr_tail(output) -> str:
    """Read the last 8 KiB of launcher stderr. @ai-generated"""
    output.flush()
    output.seek(0, os.SEEK_END)
    output.seek(max(0, output.tell() - 8192))
    return output.read().decode(errors="replace").strip()


def spawn(cmd: list[str], early_failure_s: float | None = None):
    """
    Spawn the launcher; transfer ownership of the stderr file to the caller if still running.
    Preserve the two-second synchronous failure check and its 502 response.

    @ai-edited
    """
    timeout = settings.LAUNCH_EARLY_FAILURE_S if early_failure_s is None else early_failure_s
    logger.info("Starting new process with command: %s", " ".join(cmd))
    output = tempfile.TemporaryFile(mode="w+b")  # noqa: SIM115 - the waiter owns this file after the timeout
    transferred = False
    try:
        process = _popen(
            cmd,
            cwd=settings.PROJECT_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=output,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            transferred = True
            return process, output, None
        details = _stderr_tail(output)
        if returncode != 0 or "An error occurred while starting a run" in details:
            logger.error("Run launch failed (exit %s): %s", returncode, details)
            raise ApiError(HTTPStatus.BAD_GATEWAY, "launch-failed", f"Run launch failed (exit {returncode})", detail=details or "no output")
        transferred = True
        return process, output, returncode
    except OSError as exc:
        raise ApiError(
            HTTPStatus.BAD_GATEWAY, "launch-failed", "Could not start the run process", detail=f"{type(exc).__name__}: {exc}"
        ) from exc
    finally:
        if not transferred:
            output.close()


def _data_keys(path: Path) -> dict[str, tuple[int, int]]:
    """Snapshot the run's table files to distinguish restart output from old data. @ai-generated"""
    try:
        return {
            p.name: (p.stat().st_mtime_ns, p.stat().st_size)
            for p in path.iterdir()
            if p.is_file() and p.suffix in (".csv", ".jsonl", ".db", ".sqlite")
        }
    except OSError:
        return {}


def _write_failure(path: Path, message: str, detail: str):
    """Atomically persist a run-scoped issue without leaving a partial marker. @ai-generated"""
    if path.is_symlink():
        raise OSError(f"Refusing to write through symlink: {path}")
    path.mkdir(exist_ok=True)
    marker = path / LAUNCH_FAILURE_FILE
    fd, tmp = tempfile.mkstemp(prefix=".studio-launch-", dir=path)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(orjson.dumps({"message": message, "detail": detail}))
        os.replace(tmp, marker)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _monitor(
    library: Library,
    record: ExperimentRecord,
    run_ids: list[str],
    baseline: dict[str, tuple[dict[str, tuple[int, int]], int | None]],
    process,
    output,
    early_returncode: int | None,
    notify: Callable[[str, dict[str, Any]], None] | None,
):
    """Reap an accepted launcher, report only runs without newly populated data. @ai-generated"""
    try:
        try:
            returncode = early_returncode if early_returncode is not None else process.wait()
        except subprocess.TimeoutExpired:
            # Older tests use a mock whose wait() always raises, even without a timeout.
            return
        detail = _stderr_tail(output)
        library.invalidate(record.id)
        current = library.get(record.id)
        failed = []
        first_issue = None
        for run_id in run_ids:
            name = run_id.rsplit("/", 1)[1]
            path = record.path / name
            run = current.run(run_id) if current is not None else None
            old_files, old_step = baseline[run_id]
            populated = run is not None and run.latest_step is not None and (_data_keys(path) != old_files or run.latest_step != old_step)
            marker = path / LAUNCH_FAILURE_FILE
            if populated:
                if marker.is_file():
                    marker.unlink()
            else:
                failed.append(run_id)
                message = f"Launch of {name} failed (exit {returncode})." if returncode else f"Launch of {name} produced no run data."
                failure_detail = detail or (f"exit {returncode}, no stderr" if returncode else "exit 0, no usable run data")
                _write_failure(path, message, failure_detail)
                if first_issue is None:
                    first_issue = Issue(Level.ERROR, "launch-failed", message, f"run:{run_id}", detail=failure_detail)
        library.invalidate(record.id)
        if failed:
            logger.error("Launcher for %s exited %s; affected runs: %s; stderr: %s", record.id, returncode, failed, detail)
            if notify is not None and first_issue is not None:
                notify("launch-failed", {"experiment": record.id, "runs": failed, "issue": first_issue.to_json()})
    except Exception:
        logger.exception("Could not monitor launcher for %s", record.id)
    finally:
        output.close()


def _watch(
    library: Library,
    record: ExperimentRecord,
    run_ids: list[str],
    result,
    baseline: dict[str, tuple[dict[str, tuple[int, int]], int | None]],
    notify: Callable[[str, dict[str, Any]], None] | None,
):
    """Start a daemon waiter only for launches accepted after the early check. @ai-generated"""
    if result is not None:
        process, output, early_returncode = result
        threading.Thread(
            target=_monitor,
            args=(library, record, run_ids, baseline, process, output, early_returncode, notify),
            daemon=True,
            name="studio-launch-monitor",
        ).start()


# ---------------------------------------------------------------- Capabilities and defaults


def ensure_launchable(library: Library, record: ExperimentRecord) -> ExperimentRecord:
    """
    Run the lazy health check if needed; raise 409 with the blocking issue unless `launch` is true.

    @ai-generated
    """
    if record.capabilities.launch is None:
        library.health(record.id)
    record = library.get(record.id) or record
    if record.capabilities.launch is not True:
        blocking = next(
            (
                i
                for i in record.issues
                if i.code in ("deserialize-failed", "missing-keys", "missing-experiment-json", "invalid-experiment-json")
            ),
            None,
        )
        raise conflict("The experiment cannot be launched with the current marl code", "not-launchable", blocking)
    if not settings.START_RUN_SCRIPT.is_file():
        raise conflict(f"The launcher {settings.START_RUN_SCRIPT.name} is missing", "launcher-missing")
    return record


def existing_seeds(record: ExperimentRecord) -> list[int]:
    """Seeds of the runs, plus `run-<n>` directory numbers. @ai-generated"""
    seeds = {r.seed for r in record.runs if r.seed is not None}
    try:
        seeds |= {int(m.group(1)) for p in record.path.iterdir() if p.is_dir() and (m := _RUN_DIR_RE.fullmatch(p.name))}
    except OSError:
        pass
    return sorted(seeds)


def _modal(values: list[Any], default: Any) -> Any:
    """@ai-generated"""
    values = [v for v in values if v is not None]
    return Counter(values).most_common(1)[0][0] if values else default


def launch_defaults(library: Library, record: ExperimentRecord) -> dict[str, Any]:
    """The API's launch defaults; triggers the health check. @ai-generated"""
    library.health(record.id)
    record = library.get(record.id) or record
    seeds = existing_seeds(record)
    configs = [extract.run_config(r.raw) for r in record.runs]
    values = {key: _modal([c[key] for c in configs], default) for key, default in DEFAULTS.items()}
    return {
        "next_seed": max(seeds) + 1 if seeds else 0,
        "existing_seeds": seeds,
        **values,
        "capabilities": record.capabilities.to_json(),
        "issues": [i.to_json() for i in record.all_issues],
    }


def colliding_seeds(record: ExperimentRecord, params: LaunchParams) -> list[int]:
    """@ai-generated"""
    return [s for s in params.seeds if (record.path / f"run-{s}").exists() or (record.path / f"run-{s}").is_symlink()]


def _raise_collisions(collisions: list[int]):
    """@ai-generated"""
    if collisions:
        issue = Issue(Level.ERROR, "seed-collision", f"Seeds already used: {', '.join(map(str, collisions))}.", "experiment")
        raise conflict(f"Seeds already used: {', '.join(map(str, collisions))}", "seed-collision", issue)


# ---------------------------------------------------------------- Actions


def start_runs(
    library: Library, record: ExperimentRecord, body: Any, notify: Callable[[str, dict[str, Any]], None] | None = None
) -> list[str]:
    """
    Launch new runs: capability check (409), validation (400), seed collisions (409, re-checked right
    before spawning), then spawn (502 on early failure). Returns the new run ids.

    @ai-edited
    """
    record = ensure_launchable(library, record)
    params = parse_body(body)
    try:
        device_arg = validate(params)
    except ValueError as exc:
        raise bad_request(str(exc)) from exc
    _raise_collisions(colliding_seeds(record, params))
    cmd = command(record, params, device_arg)
    _raise_collisions(colliding_seeds(record, params))
    run_ids = [f"{record.id}/run-{s}" for s in params.seeds]
    baseline: dict[str, tuple[dict[str, tuple[int, int]], int | None]] = {
        run_id: (_data_keys(record.path / f"run-{s}"), None) for run_id, s in zip(run_ids, params.seeds)
    }
    try:
        result = spawn(cmd)
    finally:
        library.invalidate(record.id)
    _watch(library, record, run_ids, result, baseline, notify)
    return run_ids


def restart_run(
    library: Library,
    record: ExperimentRecord,
    run: RunRecord,
    device: Any = "auto",
    notify: Callable[[str, dict[str, Any]], None] | None = None,
):
    """
    Restart a CANCELLED or CREATED run of a launchable experiment with its own configuration
    (same mechanism as the old `ServerState.start_run`: one run, the same seed).

    @ai-edited
    """
    from ..security import run_process

    record = ensure_launchable(library, record)
    if run_process(run.path, record.path, library.root) is not None:
        raise conflict(f"{run.dirname} is running", "run-active")
    if run.status not in ("CANCELLED", "CREATED"):
        raise conflict(f"Only cancelled or created runs can be restarted ({run.dirname} is {run.status})", "not-restartable")
    if run.seed is None:
        raise conflict(f"The seed of {run.dirname} is unknown", "not-restartable")
    config = extract.run_config(run.raw)
    params = LaunchParams(
        n_runs=1,
        n_tests=config["n_tests"] or DEFAULTS["n_tests"],
        seed=run.seed,
        n_jobs=1,
        test_interval=config["test_interval"] or DEFAULTS["test_interval"],
        device=device,
        save_weights=config["save_weights"] if config["save_weights"] is not None else DEFAULTS["save_weights"],
        save_actions=config["save_actions"] if config["save_actions"] is not None else DEFAULTS["save_actions"],
    )
    try:
        device_arg = validate(params)
    except ValueError as exc:
        raise bad_request(str(exc)) from exc
    baseline = {run.id: (_data_keys(run.path), run.latest_step)}
    try:
        result = spawn(command(record, params, device_arg))
    finally:
        library.invalidate(record.id)
    _watch(library, record, [run.id], result, baseline, notify)

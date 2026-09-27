"""
Tolerant records of experiments and runs, built from raw files only (never from `marl` classes).
"""

import os
import re
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import orjson
import psutil

from . import extract
from .cache import FileKey, file_key
from .issues import Capabilities, Issue, Level, ParamsCapability, exception_detail, health_of, issue_counts
from .sources import TableInfo, discover_tables, latest_step
from .sources.sqlite import SUFFIXES as SQLITE_SUFFIXES

RunStatus = Literal["CREATED", "RUNNING", "COMPLETED", "CANCELLED", "UNKNOWN"]
ExperimentStatus = Literal["CREATED", "RUNNING", "COMPLETED", "CANCELLED", "UNKNOWN", "EMPTY"]

EXPERIMENT_FILE = "experiment.json"
RUN_FILE = "run.json"
LAUNCH_FAILURE_FILE = ".studio-launch-failed.json"
PID_FILE = "pid"
LAUNCHER_SCRIPT = "start_run.py"
TABLE_SUFFIXES = (".csv", ".jsonl", *SQLITE_SUFFIXES)
REQUIRED_KEYS = ("trainer", "env", "test_env")
MAX_DEPTH = 3
_RUN_DIR_RE = re.compile(r"run-(\d+)")
_run_pool = ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1), thread_name_prefix="studio-runs")
"""Runs are probed in parallel: Polars releases the GIL while parsing."""


# ---------------------------------------------------------------- PID ownership


@dataclass(frozen=True)
class PidState:
    pid: int | None
    state: Literal["none", "verified", "unverifiable"]
    detail: str | None = None


NO_PID = PidState(None, "none")

PidChecker = Callable[[Path, Path, Path], PidState]
"""`(rundir, experiment dir, logs root) -> PidState`. Injectable so that tests can stub it."""


def verified_launcher(process: psutil.Process, expdir: Path, root: Path) -> psutil.Process | None:
    """
    The ancestor (or the process itself) whose command line runs `start_run.py` on `expdir`,
    given either as an absolute path or relative to the parent of the logs root (e.g. `logs/<id>`).

    @ai-generated
    """
    names = {str(expdir)}
    try:
        names.add(str(expdir.relative_to(root.parent)))
    except ValueError:
        pass
    for ancestor in (process, *process.parents()):
        command = ancestor.cmdline()
        if any(Path(arg).name == LAUNCHER_SCRIPT for arg in command) and any(arg in names for arg in command):
            return ancestor
    return None


def check_pid(rundir: Path, expdir: Path, root: Path) -> PidState:
    """
    Read the run's pid file safely (no symlink, bounded read) and verify that the process belongs
    to a launcher of this experiment. A stale pid file (dead process) counts as no process; the
    file is left untouched since the data layer never writes to the logs.

    @ai-generated
    """
    pid_file = rundir / PID_FILE
    if pid_file.is_symlink():
        return PidState(None, "unverifiable", "The pid file is a symbolic link")
    if not pid_file.exists():
        return PidState(None, "none")
    try:
        fd = os.open(pid_file, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            pid = int(os.read(fd, 64).strip())
        finally:
            os.close(fd)
        if pid <= 0:
            raise ValueError(f"Non-positive PID {pid}")
    except (OSError, ValueError) as exc:
        return PidState(None, "unverifiable", exception_detail(exc))
    try:
        process = psutil.Process(pid)
        if process.status() == psutil.STATUS_ZOMBIE:
            return PidState(None, "none")
        if verified_launcher(process, expdir, root) is not None:
            return PidState(pid, "verified")
    except psutil.NoSuchProcess:
        return PidState(None, "none")
    except psutil.Error as exc:
        return PidState(pid, "unverifiable", exception_detail(exc))
    return PidState(pid, "unverifiable", f"Process {pid} was not started by {LAUNCHER_SCRIPT} for this experiment")


# ---------------------------------------------------------------- Records


@dataclass
class ExperimentSummary:
    name: str
    algo: str | None
    env_name: str | None
    test_env_name: str | None
    n_steps: int | None
    created: str | None
    loggers: list[str]


@dataclass
class RunRecord:
    id: str
    dirname: str
    path: Path
    seed: int | None
    raw: dict[str, Any]
    tables: dict[str, TableInfo]
    status: RunStatus
    progress: float | None
    latest_step: int | None
    pid: int | None
    issues: list[Issue] = field(default_factory=list)

    @property
    def scope(self) -> str:
        return f"run:{self.id}"

    def to_json(self) -> dict[str, Any]:
        """The API's `RunSummary`. @ai-generated"""
        return {
            "id": self.id,
            "dirname": self.dirname,
            "seed": self.seed,
            "status": self.status,
            "progress": self.progress,
            "latest_step": self.latest_step,
            "pid": self.pid,
            "config": extract.run_config(self.raw),
            "issues": [i.to_json() for i in self.issues],
        }


@dataclass
class ExperimentRecord:
    id: str
    """POSIX path relative to the logs root."""
    path: Path
    raw: dict[str, Any]
    summary: ExperimentSummary
    runs: list[RunRecord]
    issues: list[Issue]
    """Experiment-level issues only; see `all_issues`."""
    capabilities: Capabilities
    mtime_key: tuple

    @property
    def all_issues(self) -> list[Issue]:
        """Experiment-level issues followed by every run-level issue. @ai-generated"""
        return [*self.issues, *(i for run in self.runs for i in run.issues)]

    @property
    def status(self) -> ExperimentStatus:
        """Aggregated status (api-contract.md). @ai-generated"""
        statuses = {run.status for run in self.runs}
        if not statuses:
            return "EMPTY"
        if "RUNNING" in statuses:
            return "RUNNING"
        if len(statuses) == 1:
            status = next(iter(statuses))
            if status in ("COMPLETED", "CREATED", "UNKNOWN"):
                return status
        return "CANCELLED"

    @property
    def progress(self) -> float | None:
        """Mean progress of the runs with a known progress. @ai-generated"""
        values = [run.progress for run in self.runs if run.progress is not None]
        return sum(values) / len(values) if values else None

    @property
    def running_runs(self) -> int:
        return sum(run.status == "RUNNING" for run in self.runs)

    def run(self, run_id_or_dirname: str) -> RunRecord | None:
        """@ai-generated"""
        for run in self.runs:
            if run_id_or_dirname in (run.id, run.dirname):
                return run
        return None

    def to_summary_json(self) -> dict[str, Any]:
        """The API's `ExperimentSummary`. @ai-generated"""
        issues = self.all_issues
        return {
            "id": self.id,
            "name": self.summary.name,
            "algo": self.summary.algo,
            "env": self.summary.env_name,
            "created": self.summary.created,
            "n_steps": self.summary.n_steps,
            "status": self.status,
            "progress": self.progress,
            "health": health_of(issues),
            "issue_counts": issue_counts(issues),
            "n_runs": len(self.runs),
            "running_runs": self.running_runs,
        }


# ---------------------------------------------------------------- Discovery


def _is_hidden(path: Path) -> bool:
    return path.name.startswith(".") or path.name == "__pycache__"


def _is_run_file(name: str) -> bool:
    return name in (RUN_FILE, LAUNCH_FAILURE_FILE) or name.endswith(TABLE_SUFFIXES)


def _scan_experiment_dir(path: Path) -> tuple[list[os.DirEntry], list[tuple[os.DirEntry, list[os.DirEntry]]]]:
    """
    One `scandir` of an experiment directory, then one per child directory (`DirEntry.is_dir/is_file`
    use the cached file type). Returns the database entries and the `(run dir entry, file entries)`
    of the run directories, sorted by name.

    @ai-generated
    """
    try:
        with os.scandir(path) as it:
            entries = sorted(it, key=lambda e: e.name)
    except OSError:
        return [], []
    databases = [e for e in entries if e.name.endswith(SQLITE_SUFFIXES)]
    children = [e for e in entries if e.is_dir() and not e.name.startswith(".") and e.name != "__pycache__"]
    return databases, _scan_children(children)


def _scan_run_dirs(path: Path) -> list[tuple[os.DirEntry, list[os.DirEntry]]]:
    """`(run dir entry, file entries)` of the run directories of `path`, sorted by name. @ai-generated"""
    return _scan_experiment_dir(path)[1]


def _scan_children(children: list[os.DirEntry]) -> list[tuple[os.DirEntry, list[os.DirEntry]]]:
    """@ai-generated"""
    result = []
    for child in children:
        try:
            with os.scandir(child.path) as it:
                files = sorted((e for e in it if e.is_file() or e.is_symlink()), key=lambda e: e.name)
        except OSError:
            continue
        if any(_is_run_file(e.name) for e in files):
            result.append((child, files))
    return result


def _has_run_dir(path: str) -> bool:
    """Whether a child directory of `path` is a run directory (early exit, no sorting). @ai-generated"""
    try:
        with os.scandir(path) as it:
            children = [e.path for e in it if e.is_dir() and not e.name.startswith(".") and e.name != "__pycache__"]
    except OSError:
        return False
    for child in children:
        try:
            with os.scandir(child) as it:
                if any(_is_run_file(e.name) and (e.is_file() or e.is_symlink()) for e in it):
                    return True
        except OSError:
            continue
    return False


def is_run_dir(path: Path) -> bool:
    """A directory with a `run.json` or at least one table file. @ai-generated"""
    try:
        return any(e.is_file() and _is_run_file(e.name) for e in os.scandir(path))
    except OSError:
        return False


def run_dirs(path: Path) -> list[Path]:
    """@ai-generated"""
    return [Path(entry.path) for entry, _ in _scan_run_dirs(path)]


def is_experiment_dir(path: Path) -> bool:
    """@ai-generated"""
    return (path / EXPERIMENT_FILE).is_file() or len(run_dirs(path)) > 0


def discover(root: Path, max_depth: int = MAX_DEPTH) -> list[Path]:
    """
    Experiment directories below `root`, up to `max_depth` levels deep, without descending into
    experiments. Hidden directories and `__pycache__` are skipped.

    @ai-generated
    """
    found = list[Path]()

    def walk(directory: str, depth: int):
        try:
            with os.scandir(directory) as it:
                children = sorted(
                    (e for e in it if e.is_dir(follow_symlinks=False) and not e.name.startswith(".") and e.name != "__pycache__"),
                    key=lambda e: e.name,
                )
        except OSError:
            return
        for child in children:
            if os.path.isfile(os.path.join(child.path, EXPERIMENT_FILE)) or _has_run_dir(child.path):
                found.append(Path(child.path))
            elif depth < max_depth:
                walk(child.path, depth + 1)

    walk(str(root), 1)
    return found


@dataclass(frozen=True)
class Fingerprint:
    key: tuple
    pids: dict[str, PidState]
    files: dict[str, dict[str, FileKey]]
    """Files of each run directory: name -> `file_key` (stats are only taken for run.json and tables)."""


def _entry_key(entry: os.DirEntry) -> FileKey:
    """Same value as `file_key(entry.path)`, without building a `Path`. @ai-generated"""
    try:
        st = entry.stat()
    except OSError:
        return None
    return (entry.path, st.st_mtime_ns, st.st_size)


def scan_fingerprint(path: Path, root: Path, pid_checker: PidChecker = check_pid) -> Fingerprint:
    """
    Cache key of an experiment: stats of experiment.json, databases, run.json and table files, plus
    the pid states (a process can die without touching any file). Also returns the pid states and
    the file names of each run, so that building the record does not list directories again.

    @ai-generated
    """
    databases, runs = _scan_experiment_dir(path)
    parts: list[Any] = [file_key(path / EXPERIMENT_FILE), *(_entry_key(e) for e in databases)]
    pids = dict[str, PidState]()
    files = dict[str, dict[str, FileKey]]()
    for run, entries in runs:
        keys = {
            e.name: _entry_key(e) if e.name in (RUN_FILE, LAUNCH_FAILURE_FILE) or e.name.endswith(TABLE_SUFFIXES) else None for e in entries
        }
        pid = pid_checker(Path(run.path), path, root) if PID_FILE in keys else NO_PID
        pids[run.name] = pid
        files[run.name] = keys
        parts.append((run.name, tuple(keys.values()), pid))
    return Fingerprint(tuple(parts), pids, files)


def fingerprint(path: Path, root: Path, pid_checker: PidChecker = check_pid) -> tuple[tuple, dict[str, PidState]]:
    """
    Cache key of an experiment: stats of experiment.json, run.json, pid and table files, plus
    the pid states (a process can die without touching any file). Also returns the pid states.

    @ai-generated
    """
    fp = scan_fingerprint(path, root, pid_checker)
    return fp.key, fp.pids


def _read_json(path: Path) -> tuple[Any, Exception | None]:
    """@ai-generated"""
    try:
        return orjson.loads(path.read_bytes()), None
    except (OSError, orjson.JSONDecodeError) as exc:
        return None, exc


def _path_matches(stored: Any, actual: Path) -> bool:
    """
    Whether a stored `logdir`/`rundir` designates `actual`: an absolute path resolving to it, or a
    relative path whose parts are a suffix of the actual path (e.g. `logs/<id>`).

    @ai-generated
    """
    if not isinstance(stored, str) or not stored:
        return False
    p = Path(stored)
    if p.is_absolute():
        try:
            return p.resolve() == actual.resolve()
        except OSError:
            return False
    parts = tuple(x for x in p.parts if x != ".")
    return len(parts) > 0 and actual.parts[-len(parts) :] == parts


def _run_status(pid: PidState, latest: int | None, n_steps: int | None) -> RunStatus:
    """@ai-generated"""
    if pid.state == "verified":
        return "RUNNING"
    if latest is None:
        return "CREATED"
    if n_steps is None:
        return "UNKNOWN"
    if latest >= n_steps:
        return "COMPLETED"
    return "CANCELLED"


def build_run(
    rundir: Path, experiment_id: str, experiment_n_steps: int | None, pid: PidState, files: dict[str, FileKey] | None = None
) -> RunRecord:
    """Record of one run; `files` maps its file names to their `file_key`, when already scanned. @ai-generated"""
    run_id = f"{experiment_id}/{rundir.name}"
    scope = f"run:{run_id}"
    issues = list[Issue]()
    raw: dict[str, Any] = {}
    run_file = rundir / RUN_FILE
    if run_file.exists():
        parsed, exc = _read_json(run_file)
        if isinstance(parsed, dict):
            raw = parsed
        else:
            detail = exception_detail(exc) if exc is not None else "run.json is not a JSON object"
            issues.append(Issue(Level.ERROR, "invalid-run-json", f"run.json of {rundir.name} is unreadable.", scope, RUN_FILE, detail))
    else:
        issues.append(Issue(Level.WARNING, "invalid-run-json", f"{rundir.name} has no run.json.", scope, RUN_FILE))

    seed = extract.seed(raw)
    if seed is None:
        match = _RUN_DIR_RE.fullmatch(rundir.name)
        if match is not None:
            seed = int(match.group(1))
            issues.append(Issue(Level.INFO, "seed-inferred", f"The seed of {rundir.name} was inferred from its directory name.", scope))

    stored = raw.get("rundir")
    if stored is not None and not _path_matches(stored, rundir):
        issues.append(
            Issue(
                Level.INFO,
                "logdir-mismatch",
                f"run.json points to {stored}; the actual location is used.",
                scope,
                f"{rundir.name}/{RUN_FILE}",
            )
        )

    if pid.state == "unverifiable":
        issues.append(
            Issue(
                Level.WARNING,
                "pid-unverifiable",
                f"The process of {rundir.name} cannot be verified.",
                scope,
                f"{rundir.name}/{PID_FILE}",
                pid.detail,
            )
        )

    failure = rundir / LAUNCH_FAILURE_FILE
    if failure.is_file() and not failure.is_symlink():
        parsed, exc = _read_json(failure)
        if isinstance(parsed, dict) and isinstance(parsed.get("message"), str) and isinstance(parsed.get("detail"), str):
            issues.append(Issue(Level.ERROR, "launch-failed", parsed["message"], scope, None, parsed["detail"]))
        else:
            issues.append(
                Issue(
                    Level.ERROR,
                    "launch-failed",
                    "The launch failed; its details are unreadable.",
                    scope,
                    None,
                    exception_detail(exc) if exc else None,
                )
            )

    tables, table_issues = discover_tables(rundir, scope, files)
    issues.extend(table_issues)
    steps = [s for s in (latest_step(t) for t in tables.values()) if s is not None]
    latest = max(steps) if steps else None
    n_steps = extract.n_steps(raw) or experiment_n_steps
    progress = None
    if n_steps:
        progress = min(1.0, max(0.0, (latest or 0) / n_steps))
    return RunRecord(
        id=run_id,
        dirname=rundir.name,
        path=rundir,
        seed=seed,
        raw=raw,
        tables=tables,
        status=_run_status(pid, latest, n_steps),
        progress=progress,
        latest_step=latest,
        pid=pid.pid if pid.state == "verified" else None,
        issues=issues,
    )


def _sort_key(run: RunRecord):
    return (run.seed is None, run.seed if run.seed is not None else 0, run.dirname)


def build_experiment(path: Path, root: Path, pid_checker: PidChecker = check_pid, fp: Fingerprint | None = None) -> ExperimentRecord:
    """
    Build the record of the experiment directory `path` (below `root`). Never raises on
    malformed content: every problem becomes an `Issue`. `fp` avoids scanning the directory again.

    @ai-generated
    """
    exp_id = path.relative_to(root).as_posix()
    if fp is None:
        fp = scan_fingerprint(path, root, pid_checker)
    mtime_key, pids = fp.key, fp.pids
    issues = list[Issue]()
    raw: dict[str, Any] = {}
    params: ParamsCapability = "none"
    exp_file = path / EXPERIMENT_FILE
    if not exp_file.exists():
        issues.append(
            Issue(Level.WARNING, "missing-experiment-json", "The experiment has no experiment.json.", "experiment", EXPERIMENT_FILE)
        )
    else:
        parsed, exc = _read_json(exp_file)
        if exc is not None:
            issues.append(
                Issue(
                    Level.ERROR,
                    "invalid-experiment-json",
                    "experiment.json is not valid JSON.",
                    "experiment",
                    EXPERIMENT_FILE,
                    exception_detail(exc),
                )
            )
        elif not isinstance(parsed, dict):
            params = "raw"
            issues.append(
                Issue(Level.ERROR, "invalid-experiment-json", "experiment.json is not a JSON object.", "experiment", EXPERIMENT_FILE)
            )
        else:
            raw = parsed
            missing = [k for k in REQUIRED_KEYS if k not in raw]
            params = "partial" if missing else "full"
            if missing:
                issues.append(
                    Issue(
                        Level.WARNING,
                        "missing-keys",
                        f"experiment.json has no {', '.join(missing)} (lightweight experiment).",
                        "experiment",
                        ",".join(missing),
                    )
                )
            stored = raw.get("logdir")
            if stored is not None and not _path_matches(stored, path):
                issues.append(
                    Issue(
                        Level.INFO,
                        "logdir-mismatch",
                        f"experiment.json points to {stored}; the actual location is used.",
                        "experiment",
                        EXPERIMENT_FILE,
                    )
                )

    n_steps = extract.n_steps(raw)
    none = PidState(None, "none")
    runs = sorted(
        _run_pool.map(lambda name: build_run(path / name, exp_id, n_steps, pids.get(name, none), fp.files[name]), list(fp.files)),
        key=_sort_key,
    )
    _flag_missing_tables(runs)
    if n_steps is None:
        run_steps = Counter(s for s in (extract.n_steps(r.raw) for r in runs) if s is not None)
        n_steps = run_steps.most_common(1)[0][0] if run_steps else None

    summary = ExperimentSummary(
        name=path.name,
        algo=extract.algo(raw) or next((a for a in (extract.algo(r.raw) for r in runs) if a is not None), None),
        env_name=extract.env_name(raw) or next((e for e in (extract.env_name(r.raw) for r in runs) if e is not None), None),
        test_env_name=extract.test_env_name(raw),
        n_steps=n_steps,
        created=extract.created(raw, path),
        loggers=extract.loggers(raw),
    )
    metrics = any(t.names for run in runs for t in run.tables.values())
    return ExperimentRecord(
        id=exp_id,
        path=path,
        raw=raw,
        summary=summary,
        runs=runs,
        issues=issues,
        capabilities=Capabilities(metrics=metrics, params=params),
        mtime_key=mtime_key,
    )


def _flag_missing_tables(runs: list[RunRecord]):
    """Add a `missing-table` issue to runs (with at least one table) lacking a table that others have. @ai-generated"""
    all_tables = sorted({name for run in runs for name in run.tables})
    for run in runs:
        if not run.tables:
            continue
        for name in all_tables:
            if name not in run.tables:
                run.issues.append(Issue(Level.WARNING, "missing-table", f"{run.dirname} has no {name} table.", run.scope, name))

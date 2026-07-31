"""Restore full ``Run`` metadata from an experiment configuration.

Some older experiment directories contain ``run.json`` files serialized as
``LightRun`` instances.  This script fills in the shared ``trainer``, ``env``,
and ``test_env`` fields from ``experiment.json``, corrects stale absolute
paths, and changes the discriminator to ``Run``.

Every JSON file that would be changed is first copied next to itself with a
``.pre-restore.json`` suffix. CSV files and all other log files are untouched.

Example:
    uv run python scripts/restore_light_runs.py logs/sequential-2-dqn-failed
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from marl.models.run import Run

RUN_FILENAME = "run.json"
EXPERIMENT_FILENAME = "experiment.json"
BACKUP_SUFFIX = ".pre-restore.json"
REQUIRED_EXPERIMENT_FIELDS = ("trainer", "env", "test_env")
REQUIRED_RUN_FIELDS = ("seed", "rundir", "n_steps")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "logdir",
        type=Path,
        help="experiment directory to repair",
    )
    parser.add_argument("--dry-run", action="store_true", help="validate and report changes without writing files")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def atomic_write(path: Path, content: bytes) -> None:
    """Atomically replace ``path`` after its backup has been written."""
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as temporary_file:
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def backup_before_replacing(path: Path) -> None:
    backup = path.with_name(f"{path.stem}{BACKUP_SUFFIX}")
    if backup.exists():
        raise FileExistsError(f"Refusing to overwrite existing backup: {backup}")
    atomic_write(backup, path.read_bytes())


def restored_run_data(original: dict[str, Any], experiment: dict[str, Any], rundir: Path) -> dict[str, Any]:
    missing = [field for field in REQUIRED_RUN_FIELDS if field not in original]
    if missing:
        raise ValueError(f"{rundir / RUN_FILENAME} is missing required run fields: {', '.join(missing)}")
    if original.get("class-name") not in ("LightRun", "Run"):
        raise ValueError(f"{rundir / RUN_FILENAME} has unexpected class-name: {original.get('class-name')!r}")

    repaired = copy.deepcopy(original)
    repaired["rundir"] = rundir.resolve().as_posix()
    repaired["trainer"] = copy.deepcopy(experiment["trainer"])
    repaired["env"] = copy.deepcopy(experiment["env"])
    repaired["test_env"] = copy.deepcopy(experiment["test_env"])
    repaired["class-name"] = "Run"
    repaired["name"] = "Run"

    # Run.from_dict uses destructive discriminator dispatch internally, so keep
    # the output dictionary intact for serialization after validation.
    restored = Run.from_dict(copy.deepcopy(repaired), exact_type=True)
    if restored.runpath != rundir.resolve():
        raise AssertionError(f"Restored rundir does not match {rundir}")
    return repaired


def json_bytes(data: dict[str, Any]) -> bytes:
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def main() -> int:
    args = parse_args()
    logdir = args.logdir.resolve()
    experiment_path = logdir / EXPERIMENT_FILENAME
    if not experiment_path.is_file():
        raise FileNotFoundError(f"Missing experiment configuration: {experiment_path}")

    experiment = load_json(experiment_path)
    missing = [field for field in REQUIRED_EXPERIMENT_FIELDS if field not in experiment]
    if missing:
        raise ValueError(f"{experiment_path} is missing: {', '.join(missing)}")
    if experiment.get("class-name") != "Experiment":
        raise ValueError(f"{experiment_path} is not a full Experiment configuration")

    repaired_experiment = copy.deepcopy(experiment)
    repaired_experiment["logdir"] = logdir.as_posix()

    run_paths = sorted(path / RUN_FILENAME for path in logdir.glob("run-*") if (path / RUN_FILENAME).is_file())
    if not run_paths:
        raise ValueError(f"No {RUN_FILENAME} files found below {logdir}")

    repaired_runs = [(path, restored_run_data(load_json(path), experiment, path.parent)) for path in run_paths]
    changes = sum(load_json(path) != data for path, data in repaired_runs)
    experiment_changed = experiment != repaired_experiment

    print(f"validated {len(repaired_runs)} run files in {logdir}")
    print(f"will restore {changes} run.json files; experiment.json path update: {experiment_changed}")
    if args.dry_run:
        print("dry run: no files changed")
        return 0

    # Retain original JSON before replacing it. No CSV or other log file is read,
    # removed, renamed, or rewritten.
    if experiment_changed:
        backup_before_replacing(experiment_path)
        atomic_write(experiment_path, json_bytes(repaired_experiment))
    for path, data in repaired_runs:
        if load_json(path) == data:
            continue
        backup_before_replacing(path)
        atomic_write(path, json_bytes(data))

    print("restoration complete; original JSON is retained as *.pre-restore.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

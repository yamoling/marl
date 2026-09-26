"""
Fixture log builder for MARL Studio tests.

`build_logs(root)` writes one experiment directory per scenario of the WP0 fixture table
(phases.md) and returns the launcher processes of the `running` scenario.
"""

import json
import os
import signal
import sqlite3
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import polars as pl
import pytest

FIXTURES = Path(__file__).parent / "fixtures"
INIT_SQL = Path(__file__).parents[2] / "src" / "marl" / "logging" / "sql_logger" / "init.sql"

N_STEPS = 10_000
TEST_INTERVAL = 1_000
N_TESTS = 2
SEEDS = (0, 1, 2)
SCENARIOS = (
    "healthy",
    "light",
    "unknown-class",
    "corrupt-run",
    "missing-table",
    "extra-table",
    "late-column",
    "no-experiment-json",
    "running",
    "jsonl",
    "sqlite",
)


def healthy_raw(expdir: Path) -> dict:
    raw = json.loads((FIXTURES / "healthy-experiment.json").read_text())
    raw["logdir"] = str(expdir)
    return raw


def run_raw(rundir: Path, seed: int, n_steps: int = N_STEPS) -> dict:
    return {
        "seed": seed,
        "rundir": str(rundir),
        "n_steps": n_steps,
        "test_interval": TEST_INTERVAL,
        "n_tests": N_TESTS,
        "loggers": ["csv"],
        "save_weights": False,
        "save_actions": True,
        "class-name": "Run",
        "name": "Run",
    }


def run_frames(seed: int, n_steps: int = N_STEPS) -> dict[str, pl.DataFrame]:
    """Deterministic test/train/training_data tables of one run. Each run starts at a different time."""
    rng = np.random.default_rng(seed)
    t0 = 1_700_000_000.0 + 3_600 * seed
    steps = np.repeat(np.arange(0, n_steps + 1, TEST_INTERVAL), N_TESTS)
    test = pl.DataFrame(
        {
            "gems_collected": rng.integers(0, 4, len(steps)),
            "exit_rate": rng.random(len(steps)),
            "episode_len": rng.integers(10, 78, len(steps)),
            "score-0": rng.normal(steps / n_steps, 0.3),
            "timestamp_sec": t0 + steps * 0.01 + rng.random(len(steps)) * 1e-3,
            "time_step": steps,
        }
    )
    train_steps = np.cumsum(rng.integers(1, 7, 2_900))
    train_steps = train_steps[train_steps <= n_steps]
    train = pl.DataFrame(
        {
            "episode_num": np.arange(len(train_steps)),
            "exit_rate": rng.random(len(train_steps)),
            "score-0": rng.normal(train_steps / n_steps, 0.5),
            "timestamp_sec": t0 + train_steps * 0.01,
            "time_step": train_steps,
        }
    )
    td_steps = np.arange(65, n_steps + 1, 5)
    training_data = pl.DataFrame(
        {
            "td-loss": rng.random(len(td_steps)),
            "grad_norm": rng.random(len(td_steps)) * 10,
            "epsilon": np.maximum(0.05, 1 - td_steps / 4_000),
            "timestamp_sec": t0 + td_steps * 0.01,
            "time_step": td_steps,
        }
    )
    return {"test": test, "train": train, "training_data": training_data}


def write_run(expdir: Path, seed: int, tables=("test", "train", "training_data"), fmt: str = "csv", run_json: bool = True) -> Path:
    rundir = expdir / f"run-{seed}"
    rundir.mkdir(parents=True)
    if run_json:
        (rundir / "run.json").write_text(json.dumps(run_raw(rundir, seed)))
    for name, df in run_frames(seed).items():
        if name not in tables:
            continue
        if fmt == "csv":
            df.write_csv(rundir / f"{name}.csv")
        elif fmt == "jsonl":
            df.write_ndjson(rundir / f"{name}.jsonl")
    return rundir


def write_experiment(expdir: Path, raw: dict | None):
    expdir.mkdir(parents=True)
    if raw is not None:
        (expdir / "experiment.json").write_text(json.dumps(raw))


def write_sqlite(expdir: Path, seeds=SEEDS):
    """The `test` tables of `healthy` in the sql_logger schema, one experiment-level database."""
    con = sqlite3.connect(expdir / "experiment.db")
    con.executescript(INIT_SQL.read_text())
    for seed in seeds:
        run_id = con.execute("INSERT INTO run (created_at, seed) VALUES (?, ?)", ("2026-01-01T00:00:00+00:00", seed)).lastrowid
        test = run_frames(seed)["test"]
        for i, row in enumerate(test.iter_rows(named=True)):
            stamp = (
                time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(row["timestamp_sec"])) + f".{int(row['timestamp_sec'] % 1 * 1e6):06d}+00:00"
            )
            test_id = con.execute(
                "INSERT INTO test (run, timestamp, time_step, seed) VALUES (?, ?, ?, ?)", (run_id, stamp, row["time_step"], i % N_TESTS)
            ).lastrowid
            con.executemany(
                "INSERT INTO test_metric (test, key, value) VALUES (?, ?, ?)",
                [(test_id, k, float(v)) for k, v in row.items() if k not in ("time_step", "timestamp_sec")],
            )
    con.commit()
    con.close()


FAKE_LAUNCHER = textwrap.dedent(
    """
    import subprocess, sys, time
    expdir, pid_file = sys.argv[1], sys.argv[2]
    child = subprocess.Popen(["sleep", "120"])
    with open(pid_file + ".tmp", "w") as f:
        f.write(str(child.pid))
    import os
    os.replace(pid_file + ".tmp", pid_file)
    child.wait()
    """
)


def start_fake_run(expdir: Path, rundir: Path, tmp: Path) -> subprocess.Popen:
    """Start `<tmp>/bin/start_run.py <expdir> <pid file>`, which spawns a sleeping child and writes its pid."""
    script = tmp / "bin" / "start_run.py"
    script.parent.mkdir(exist_ok=True)
    script.write_text(FAKE_LAUNCHER)
    pid_file = rundir / "pid"
    proc = subprocess.Popen([sys.executable, str(script), str(expdir), str(pid_file)], start_new_session=True)
    deadline = time.time() + 10
    while not pid_file.exists():
        if time.time() > deadline or proc.poll() is not None:
            stop_processes([proc])
            raise RuntimeError("The fake launcher did not write its pid file")
        time.sleep(0.01)
    return proc


def stop_processes(processes: list[subprocess.Popen]):
    for proc in processes:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=5)


def build_logs(root: Path, running: bool = True) -> list[subprocess.Popen]:
    """Write every scenario under `root`. Returns the processes to stop (see `stop_processes`)."""
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()

    exp = root / "healthy"
    write_experiment(exp, healthy_raw(exp))
    for seed in SEEDS:
        rundir = write_run(exp, seed)
        actions = rundir / "test" / str(TEST_INTERVAL) / "actions.json"
        actions.parent.mkdir(parents=True)
        actions.write_text("[[[0, 1]], [[1, 0]]]")

    exp = root / "light"
    light = {k: v for k, v in healthy_raw(exp).items() if k not in ("trainer", "env", "test_env")}
    write_experiment(exp, light)
    write_run(exp, 0)

    exp = root / "unknown-class"
    raw = healthy_raw(exp)
    raw["trainer"]["mixer"]["class-name"] = "QMixerV1"
    write_experiment(exp, raw)
    write_run(exp, 0)

    exp = root / "corrupt-run"
    write_experiment(exp, healthy_raw(exp))
    write_run(exp, 0)
    bad = write_run(exp, 1)
    (bad / "run.json").write_text(json.dumps(run_raw(bad, 1))[:40])

    exp = root / "missing-table"
    write_experiment(exp, healthy_raw(exp))
    for seed in SEEDS:
        write_run(exp, seed, tables=("train", "training_data") if seed == 2 else ("test", "train", "training_data"))

    exp = root / "extra-table"
    write_experiment(exp, healthy_raw(exp))
    for seed in (0, 1):
        rundir = write_run(exp, seed)
        rng = np.random.default_rng(seed)
        exited = rng.random(20) > 0.5
        pl.DataFrame(
            {
                "score-0": rng.random(20),
                "agent-0-exited": exited,
                "cooperative-trajectory": ~exited,
                "timestamp_sec": 1_700_000_000.0 + np.arange(20),
                "time_step": np.full(20, N_STEPS),
            }
        ).write_csv(rundir / "test-policy-on-test-envs.csv")

    exp = root / "late-column"
    write_experiment(exp, healthy_raw(exp))
    for seed in (0, 1):
        rundir = write_run(exp, seed, tables=("train", "training_data"))
        steps = np.arange(0, N_STEPS + 1, 20)
        late = np.where(steps >= 6_000, steps / N_STEPS, None)
        pl.DataFrame(
            {
                "score-0": steps / N_STEPS + seed,
                "late-metric": pl.Series(late.tolist(), dtype=pl.Float64),
                "timestamp_sec": 1_700_000_000.0 + steps,
                "time_step": steps,
            }
        ).write_csv(rundir / "test.csv")

    exp = root / "no-experiment-json"
    write_experiment(exp, None)
    for seed in (0, 1):
        write_run(exp, seed)

    exp = root / "running"
    write_experiment(exp, healthy_raw(exp))
    rundir = write_run(exp, 0, tables=("train",))
    processes = [start_fake_run(exp, rundir, root.parent)] if running else []

    exp = root / "jsonl"
    write_experiment(exp, healthy_raw(exp))
    for seed in SEEDS:
        write_run(exp, seed, fmt="jsonl")

    exp = root / "sqlite"
    write_experiment(exp, healthy_raw(exp))
    for seed in SEEDS:
        write_run(exp, seed, tables=())
    write_sqlite(exp)
    return processes


# ---------------------------------------------------------------- API clients

BASE_URL = "http://localhost:5000"


def make_client(root, **kwargs):
    """TestClient of a Studio app on `root`, with fast event intervals."""
    from fastapi.testclient import TestClient

    from studio.backend.app import create_app
    from studio.backend.services.events import EventsConfig

    kwargs.setdefault("events", EventsConfig(running_interval=0.05, scan_interval=0.2, ping_interval=0.3))
    return TestClient(create_app(root, **kwargs), base_url=BASE_URL)


@pytest.fixture(scope="module")
def api(fixture_logs):
    """Client on the shared read-only fixture logs."""
    return make_client(fixture_logs)


@pytest.fixture
def mut_api(make_logs):
    """Client on fresh fixture logs that the test may modify, and their root."""
    return make_client(make_logs), make_logs


@pytest.fixture
def popen(monkeypatch):
    """Replace the launcher's `subprocess.Popen`: the child is "still starting" after the early-failure window."""
    from unittest.mock import Mock

    from studio.backend.services import launcher

    process = Mock()
    process.wait.side_effect = subprocess.TimeoutExpired(cmd="start_run", timeout=2)
    spawn = Mock(return_value=process)
    monkeypatch.setattr(launcher, "_popen", spawn)
    return spawn


@pytest.fixture
def make_logs(tmp_path):
    """Fresh fixture logs (every scenario, including a live `running` process) for tests that mutate them."""
    processes = build_logs(tmp_path / "logs")
    yield (tmp_path / "logs").resolve()
    stop_processes(processes)


@pytest.fixture(scope="session")
def fixture_logs(tmp_path_factory):
    """Shared read-only fixture logs (every scenario), built once per session."""
    tmp = tmp_path_factory.mktemp("studio")
    processes = build_logs(tmp / "logs")
    yield (tmp / "logs").resolve()
    stop_processes(processes)

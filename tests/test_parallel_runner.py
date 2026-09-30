from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from marl.models.run import Run
from marl.runners.parallel_runner import _spawn_worker, _start_run


def test_spawn_worker_passes_only_run_directory():
    context = Mock()
    run = SimpleNamespace(rundir="logs/example/run-0")

    process = _spawn_worker(context, run, "cpu", quiet=True, render_tests=False, limit_torch_threads=None)

    assert process is context.Process.return_value
    process.start.assert_called_once()
    args = context.Process.call_args.kwargs["args"]
    assert args[0] == run.rundir
    assert run not in args


def test_worker_loads_run_from_directory(monkeypatch, tmp_path):
    run = SimpleNamespace(rundir=tmp_path.as_posix())
    loaded_paths = []
    simple_run = Mock(return_value="result")

    def load(path: Path):
        loaded_paths.append(path)
        return run

    monkeypatch.setattr(Run, "load", staticmethod(load))
    monkeypatch.setattr("marl.runners.parallel_runner.simple_run", simple_run)

    _start_run(tmp_path.as_posix(), "cpu", quiet=True, render_tests=False, limit_torch_threads=None)

    assert loaded_paths == [tmp_path]
    simple_run.assert_called_once()
    args = simple_run.call_args.args
    assert args[0] is run
    assert args[1:3] == (True, False)
    assert args[3].type == "cpu"

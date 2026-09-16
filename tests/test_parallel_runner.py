from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from marl.models.run import Run
from marl.runners.parallel_runner import _start_run, submit


def test_submit_passes_only_run_directory_to_worker():
    pool = Mock()
    handle = object()
    pool.apply_async.return_value = handle
    run = SimpleNamespace(rundir="logs/example/run-0")

    result = submit(
        pool,
        run,
        "cpu",
        quiet=True,
        render_tests=False,
        estimated_gpu_memory=0,
        gpu_strategy="scatter",
        disabled_gpus=(),
        limit_torch_threads=False,
    )

    assert result is handle
    worker_kwargs = pool.apply_async.call_args.kwargs["kwds"]
    assert worker_kwargs["rundir"] == run.rundir
    assert "run" not in worker_kwargs


def test_worker_loads_run_from_directory(monkeypatch, tmp_path):
    run = SimpleNamespace(rundir=tmp_path.as_posix())
    loaded_paths = []
    simple_run = Mock(return_value="result")

    def load(path: Path):
        loaded_paths.append(path)
        return run

    monkeypatch.setattr(Run, "load", staticmethod(load))
    monkeypatch.setattr("marl.runners.parallel_runner.simple_run", simple_run)

    result = _start_run(
        tmp_path.as_posix(),
        "cpu",
        quiet=True,
        render_tests=False,
        estimated_gpu_memory=0,
        auto_device_strategy="scatter",
        disabled_gpus=(),
        limit_torch_threads=None,
        device_affinity=None,
    )

    assert result == "result"
    assert loaded_paths == [tmp_path]
    simple_run.assert_called_once()
    args = simple_run.call_args.args
    assert args[0] is run
    assert args[1:3] == (True, False)
    assert args[3].type == "cpu"

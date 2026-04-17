import torch

from marl.runners import parallel_runner
from marl.utils.gpu import GPU


def test_with_safety_margin():
    assert parallel_runner._with_safety_margin(1000) == 1100
    assert parallel_runner._with_safety_margin(100) == 256


def test_device_index_cpu_and_cuda():
    assert parallel_runner._device_index(torch.device("cpu")) is None
    assert parallel_runner._device_index(torch.device("cuda:2")) == 2
    assert parallel_runner._device_index(torch.device("cuda")) == 0


def test_preplan_scatter_devices(monkeypatch):
    snapshot = [
        GPU(index=0, total_memory=10000, used_memory=1000, free_memory=9000, utilization=10),
        GPU(index=1, total_memory=10000, used_memory=2000, free_memory=8000, utilization=30),
    ]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(parallel_runner, "list_gpus", lambda disabled_devices: snapshot)

    plan = parallel_runner._preplan_scatter_devices(remaining_runs=3, required_memory_mb=3000, disabled_gpus=[])

    assert plan == [0, 1, 0]


def test_select_run_device_waits_for_fit(monkeypatch):
    selected_gpu = GPU(index=3, total_memory=10000, used_memory=1000, free_memory=9000, utilization=10)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(parallel_runner, "wait_for_fitting_gpu", lambda **kwargs: selected_gpu)

    dev = parallel_runner._select_run_device(
        device="auto",
        auto_device_strategy="group",
        estimated_gpu_memory=1024,
        disabled_gpus=[],
    )

    assert isinstance(dev, torch.device)
    assert dev.type == "cuda"
    assert dev.index == 3

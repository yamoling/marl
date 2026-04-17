import subprocess

import torch

from marl.utils import gpu


def test_list_gpus_respects_disabled_indices(monkeypatch):
    output = "0,10000,1000,9000,10\n1,12000,3000,9000,50\n"

    def mock_check_output(cmd, shell):
        assert "--query-gpu=index" in cmd
        return output.encode()

    monkeypatch.setattr(subprocess, "check_output", mock_check_output)

    devices = gpu.list_gpus(disabled_devices=[0])

    assert len(devices) == 1
    assert devices[0].index == 1
    assert devices[0].free_memory == 9000


def test_select_gpu_scatter_prefers_high_score(monkeypatch):
    g0 = gpu.GPU(index=0, total_memory=10000, used_memory=1000, free_memory=9000, utilization=10)
    g1 = gpu.GPU(index=1, total_memory=10000, used_memory=500, free_memory=9500, utilization=80)
    monkeypatch.setattr(gpu, "list_gpus", lambda disabled_devices=None: [g0, g1])

    selected = gpu.select_gpu("scatter", estimated_memory_MB=1000, disabled_devices=[])

    assert selected is not None
    assert selected.index == 0


def test_wait_for_fitting_gpu_times_out(monkeypatch):
    monkeypatch.setattr(gpu, "select_gpu", lambda *args, **kwargs: None)

    selected = gpu.wait_for_fitting_gpu(
        fit_strategy="group",
        estimated_memory_MB=1000,
        disabled_devices=[],
        timeout_s=0.02,
        poll_interval_s=0.005,
    )

    assert selected is None


def test_get_device_with_explicit_gpu_index():
    dev = gpu.get_device(1)

    assert isinstance(dev, torch.device)
    assert dev.type == "cuda"
    assert dev.index == 1

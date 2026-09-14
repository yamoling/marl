from collections.abc import Collection
from typing import TYPE_CHECKING, Literal

import torch

from marl.utils.gpu import DeviceLike, get_device

from .simple_runner import simple_run

if TYPE_CHECKING:
    from marl import Run


def _torch_thread_limit(limit_torch_threads: int | Literal["auto"] | None):
    if limit_torch_threads in (None, "auto"):
        return
    torch.set_num_threads(limit_torch_threads)
    torch.set_num_interop_threads(limit_torch_threads)


def sequential_run(
    runs: "Collection[Run]",
    device: DeviceLike = "auto",
    gpu_strategy: Literal["scatter", "group"] = "group",
    quiet: bool = False,
    render_tests: bool = False,
    disabled_gpus: Collection[int] = (),
    device_affinity: int | None = None,
    limit_torch_threads: int | None | Literal["auto"] = None,
):
    _torch_thread_limit(limit_torch_threads)
    for run in runs:
        d = get_device(device, gpu_strategy, disabled_devices=disabled_gpus, affinity=device_affinity)
        simple_run(run, quiet, render_tests, d)

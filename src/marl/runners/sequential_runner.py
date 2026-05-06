from typing import TYPE_CHECKING, Collection, Literal

import torch

from marl.utils.gpu import get_device

from .simple_runner import simple_run

if TYPE_CHECKING:
    from marl import Run


def sequential_run(
    runs: "Collection[Run]",
    device: int | torch.device | str | Literal["auto", "cpu"] = "auto",
    gpu_strategy: Literal["scatter", "group"] = "group",
    quiet: bool = False,
    render_tests: bool = False,
    disabled_gpus: Collection[int] = (),
):
    for run in runs:
        device = get_device(device, gpu_strategy, disabled_devices=disabled_gpus)
        simple_run(run, quiet, render_tests, device)

from typing import TYPE_CHECKING, Literal, Sequence

import torch

from marl.utils.gpu import get_device

from .simple_runner import SimpleRunner

if TYPE_CHECKING:
    from marl import Experiment, Run


class SequentialRunner:
    def __init__(self, exp: "Experiment"):
        self.exp = exp

    def start(
        self,
        runs: "Sequence[Run]",
        device: int | torch.device | str | Literal["auto", "cpu"] = "auto",
        auto_device_strategy: Literal["scatter", "group"] = "group",
        quiet: bool = False,
        n_tests: int = 1,
        render_tests: bool = False,
    ):

        device = get_device(device, auto_device_strategy)
        for run in runs:
            runner = SimpleRunner.from_experiment(self.exp, n_tests, quiet).to(device)
            runner.start(run, render_tests)

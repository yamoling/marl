from typing import TYPE_CHECKING, Literal

import torch

from marl.utils.gpu import get_device

from .simple_runner import SimpleRunner

if TYPE_CHECKING:
    from marl import Experiment


class SequentialRunner:
    def __init__(self, exp: "Experiment"):
        self.exp = exp

    def start(
        self,
        device: int | torch.device | str | Literal["auto", "cpu"] = "auto",
        auto_device_strategy: Literal["scatter", "group"] = "scatter",
        quiet: bool = False,
        n_tests: int = 1,
        render_tests: bool = False,
    ):

        device = get_device(device, auto_device_strategy)
        for run in self.exp.runs:
            runner = SimpleRunner.from_experiment(self.exp, n_tests, quiet).to(device)
            runner.start(run, render_tests)

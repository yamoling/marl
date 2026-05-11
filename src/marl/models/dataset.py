from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class Dataset:
    logdir: str
    ticks: list[float]
    label: str
    category: str
    mean: npt.NDArray[np.float32]
    min: npt.NDArray[np.float32]
    max: npt.NDArray[np.float32]
    std: npt.NDArray[np.float32]
    ci95: npt.NDArray[np.float32]

    @property
    def nice_label(self):
        """
        Unsluggify and capitalize the label.

        Examples:
        -------
        - exit_rate -> Exit rate
        - score-0 -> Score 0
        """
        return self.label.capitalize().replace("-", " ").replace("_", " ")


@dataclass
class ExperimentResults:
    logdir: str
    datasets: list[Dataset]
    qvalue_ds: list[Dataset]

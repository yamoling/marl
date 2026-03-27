from dataclasses import dataclass
import numpy as np


@dataclass
class DetailedAction:
    action: np.ndarray
    label: str
    details: np.ndarray

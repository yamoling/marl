from dataclasses import dataclass
from typing import Any, overload

import numpy as np
import numpy.typing as npt


@dataclass
class Action[T: Any]:
    """
    An Action is a wrapper around a numpy array representing the action to perform. It also allows to store additional keyword arguments that can be used to store details on the decision-making such as:
     - Logging data such as the qvalues, the action probabilities or the logits.
     - Options in the case of option-based RL algorithms.
     - Meta-agent actions in the case of hierarchical RL algorithms.
    """

    action: npt.NDArray[T]
    action_probabilities: npt.NDArray[np.float32] | None = None
    q_values: npt.NDArray[np.float32] | None = None
    options: list | None = None
    meta_actions: npt.ArrayLike | None = None

    def __init__(
        self,
        action: npt.NDArray[T],
        *,
        action_probabilities: npt.NDArray[np.float32] | None = None,
        q_values: npt.NDArray[np.float32] | None = None,
        options: list | None = None,
        meta_actions: npt.ArrayLike | None = None,
    ):
        self.action = action
        self.action_probabilities = action_probabilities
        self.q_values = q_values
        self.options = options
        self.meta_actions = meta_actions

    def __array__(self, dtype=None):
        if dtype is None:
            return self.action
        else:
            return self.action.astype(dtype)

    def __numpy_dtype__(self):
        return self.action.dtype

    @overload
    def __getitem__(self, item: int) -> np.ndarray:
        """Get the action at the given index."""

    @overload
    def __getitem__(self, item: str) -> np.ndarray:
        """Get the keyword argument with the given name."""

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.action[item]
        return getattr(self, item)

    @property
    def details(self):
        return {name: value for name, value in self.__dict__.items() if name != "action" and value is not None}

from dataclasses import dataclass
from typing import overload

import numpy as np
import numpy.typing as npt


@dataclass
class Action:
    """
    An Action is a wrapper around a numpy array representing the action to perform. It also allows to store additional keyword arguments that can be used to store details on the decision-making such as:
     - Logging data such as the qvalues, the action probabilities or the logits.
     - Options in the case of option-based RL algorithms.
     - Meta-agent actions in the case of hierarchical RL algorithms.
    """

    action: npt.ArrayLike
    options: npt.ArrayLike | None
    meta_actions: npt.ArrayLike | None

    def __init__(
        self,
        action: npt.ArrayLike,
        *,
        options: npt.ArrayLike | None = None,
        meta_actions: npt.ArrayLike | None = None,
        **kwargs: npt.ArrayLike | None,
    ):
        self.action = action
        self.options = options
        self.meta_actions = meta_actions
        for name, value in kwargs.items():
            setattr(self, name, value)

    def __array__(self, dtype=None):
        if isinstance(self.action, np.ndarray):
            if dtype is None:
                return self.action
            return self.action.astype(dtype)
        return np.array(self.action, dtype)

    def __numpy_dtype__(self):
        match self.action:
            case np.ndarray():
                return self.action.dtype
            case int():
                return np.int64
            case float():
                return np.float32
            case other:
                raise TypeError(f"Unsupported action type: {type(other)}")

    @overload
    def __getitem__(self, item: int) -> np.ndarray:
        """Get the action at the given index."""

    @overload
    def __getitem__(self, item: str) -> np.ndarray:
        """Get the keyword argument with the given name."""
        pass

    def __getitem__(self, item: int | str):
        if isinstance(item, int):
            if not isinstance(self.action, np.ndarray):
                self.action = np.array(self.action)
            return self.action[item]
        return getattr(self, item)

    def __setitem__(self, key: str, value: npt.ArrayLike):
        setattr(self, key, value)

    @property
    def details(self):
        return {key: value for key, value in self.__dict__.items() if key != "action" and value is not None}

from dataclasses import dataclass
import numpy as np
import numpy.typing as npt


@dataclass
class Message:
    data: npt.NDArray[np.float32]

    # def __init__(self, n_agents: int):
    #     self.n_agents = n_agents

    def __init__(self, data: npt.NDArray[np.float32]):
        self.dat = data

    @property
    def n_agents(self) -> int:
        """The number of agents in the message"""
        return self.data.shape[0]

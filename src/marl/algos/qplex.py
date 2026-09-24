from dataclasses import dataclass

from marl.nn import mixers

from .dqn import DQN


@dataclass
class QPlex(DQN[mixers.QPlex]):
    def __post_init__(self):
        super().__post_init__()
        assert isinstance(self.mixer, mixers.QPlex), "QPlex training requires a QPlex mixer"

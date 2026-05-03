import logging
import math
from dataclasses import KW_ONLY, dataclass
from typing import Literal, Sequence

from marlenv import MARLEnv

from marl.models.nn import NN, ActorCritic, QNetwork
from marl.nn.model_bank import actor_critics, qnetworks

from .config import Config


@dataclass
class NetworkConfig[T: NN](Config[T]):
    input_shape: tuple[int, ...]
    extras_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    _: KW_ONLY
    is_recurrent: bool = False
    mlp_sizes: Sequence[int] = (128, 128)
    hidden_activation: Literal["relu", "tanh", "sigmoid", "leaky-relu"] = "relu"

    @property
    def input_size(self):
        return math.prod(self.input_shape)

    @property
    def extras_size(self):
        return math.prod(self.extras_shape)


@dataclass
class QNetworkConfig(NetworkConfig[QNetwork]):
    _: KW_ONLY
    noisy: bool = False

    @property
    def n_actions(self):
        return self.output_shape[0]

    def make(self) -> QNetwork:
        match (self.is_recurrent, len(self.input_shape)):
            case (False, 1):
                return qnetworks.QMLP(
                    self.output_shape,
                    self.input_size,
                    self.extras_size,
                    self.mlp_sizes,
                    noisy=self.noisy,
                    hidden_activation=self.hidden_activation,
                )
            case (False, 3):
                assert len(self.input_shape) == 3
                return qnetworks.QCNN(
                    self.output_shape,
                    self.input_shape,
                    self.extras_size,
                    self.mlp_sizes,
                    self.hidden_activation,
                    noisy=self.noisy,
                )
            case (True, 1):
                if self.noisy:
                    logging.warning("noisy is not yet supported for recurrent Q-networks")
                return qnetworks.QRNN(
                    (self.input_size,),
                    self.extras_size,
                    self.n_actions,
                    self.mlp_sizes,
                    self.hidden_activation,
                    noisy=self.noisy,
                )
        raise NotImplementedError(f"QNetworks are not implemented for input shape {self.input_shape}")

    @staticmethod
    def from_env(
        env: MARLEnv,
        mlp_sizes: Sequence[int] = (128, 128),
        hidden_activation: Literal["relu", "tanh", "sigmoid"] = "relu",
        noisy: bool = False,
    ):
        if env.is_multi_objective:
            output_shape = (env.n_actions, env.n_objectives)
        else:
            output_shape = (env.n_actions,)
        return QNetworkConfig(
            env.observation_shape,
            env.extras_shape,
            output_shape,
            mlp_sizes=mlp_sizes,
            hidden_activation=hidden_activation,
            noisy=noisy,
        )


@dataclass
class ActorCriticConfig(NetworkConfig[ActorCritic]):
    is_discrete: bool

    @property
    def n_actions(self):
        return self.output_shape[0]

    def make(self) -> ActorCritic:
        match (len(self.input_shape), self.is_discrete):
            case (1, True):
                return actor_critics.SimpleActorCritic(
                    self.input_size,
                    self.extras_size,
                    self.n_actions,
                    self.mlp_sizes,
                    self.hidden_activation,
                )
            case (1, False):
                return actor_critics.MLPContinuousActorCritic(
                    self.input_size,
                    self.extras_size,
                    self.n_actions,
                    self.mlp_sizes,
                    self.hidden_activation,
                )
            case (3, True):
                assert len(self.output_shape) == 1, "Multi-objective is not yet supported"
                assert len(self.input_shape) == 3
                return actor_critics.CNN_ActorCritic(
                    self.input_shape,
                    self.extras_size,
                    self.n_actions,
                    self.hidden_activation,
                    self.mlp_sizes,
                )
            case (3, False):
                assert len(self.input_shape) == 3
                return actor_critics.CNNContinuousActorCritic(
                    self.input_shape,
                    self.extras_size,
                    self.n_actions,
                    self.mlp_sizes,
                    self.hidden_activation,
                )
            case _:
                raise NotImplementedError(f"Actor-Critics are not implemented for input shape {self.input_shape}")

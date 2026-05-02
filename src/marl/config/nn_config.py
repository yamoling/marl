import math
from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, Literal, Sequence, overload

from marlenv import MARLEnv

from marl.utils import Serializable

if TYPE_CHECKING:
    from marl.models.nn import ActorCritic, QNetwork


@dataclass
class NetworkConfig(Serializable):
    input_shape: tuple[int, ...]
    extras_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    _: KW_ONLY
    mlp_sizes: Sequence[int] = (128, 128)
    hidden_activation: Literal["relu", "tanh", "sigmoid"] = "relu"
    output_activation: Literal["relu", "tanh", "sigmoid"] | None = None
    noisy: bool = False

    @property
    def input_size(self):
        return math.prod(self.input_shape)

    @property
    def extras_size(self):
        return math.prod(self.extras_shape)

    @staticmethod
    def from_env(
        env: MARLEnv,
        mlp_sizes: Sequence[int] = (128, 128),
        output: Literal["action-space"] | tuple[int, ...] = "action-space",
        hidden_activation: Literal["relu", "tanh", "sigmoid"] = "relu",
        noisy: bool = False,
    ):
        if isinstance(output, tuple):
            output_shape = output
        else:
            match (output, env.is_multi_objective):
                case ("action-space", True):
                    output_shape = (env.n_actions, env.n_objectives)
                case ("action-space", False):
                    output_shape = (env.n_actions,)
                case (other, _):
                    raise ValueError(f"Unknown output type: {other}.")

        return NetworkConfig(
            env.observation_shape,
            env.extras_shape,
            output_shape,
            mlp_sizes=mlp_sizes,
            hidden_activation=hidden_activation,
            noisy=noisy,
        )

    @overload
    def build(self, kind: Literal["q-network"]) -> QNetwork:
        """Construct a Q-Network from the configuration."""

    @overload
    def build(self, kind: Literal["actor-critic"]) -> ActorCritic:
        """Construct an Actor-Critic from the configuration."""

    def build(self, kind: Literal["q-network", "actor-critic"]):
        match kind:
            case "q-network":
                return self.make_qnetwork()
            case "actor-critic":
                return self.make_actor_critic()
            case other:
                raise ValueError(f"Unknown kind: {other}. Expected 'q-network' or 'actor-critic'.")

    def make_qnetwork(self) -> QNetwork:
        from marl.nn.model_bank import qnetworks

        if len(self.input_shape) == 1:
            return qnetworks.QMLP(
                self.output_shape,
                self.input_size,
                self.extras_size,
                self.mlp_sizes,
                noisy=self.noisy,
                hidden_activation=self.hidden_activation,
            )
        if len(self.input_shape) == 3:
            return qnetworks.QCNN(self.output_shape, self.input_shape, self.extras_size, self.mlp_sizes, self.noisy)
        raise NotImplementedError(f"QNetworks are not implemented for input shape {self.input_shape}")

    def make_actor_critic(self) -> ActorCritic:
        raise NotImplementedError()

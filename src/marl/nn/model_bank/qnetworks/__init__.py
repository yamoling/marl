from marlenv import DiscreteMARLEnv

from marl.env import EnvConfig
from marl.models.nn import QNetwork

from .generic import QCNN, QCRNN, QMLP, QRNN
from .maven import MAVENQnetwork


def from_env(env: DiscreteMARLEnv | EnvConfig[DiscreteMARLEnv], recurrent: bool = False) -> QNetwork:
    match (len(env.observation_shape), recurrent):
        case (1, False):
            return QMLP.from_env(env)
        case (3, False):
            return QCNN.from_env(env)
        case (1, True):
            return QRNN.from_env(env)
        case (3, True):
            return QCRNN.from_env(env)
    raise NotImplementedError("Unsupported environment observation shape and recurrent combination. Create your own Q-network !")


__all__ = [
    "MAVENQnetwork",
    "QMLP",
    "QCNN",
    "QRNN",
    "QCRNN",
    "from_env",
]

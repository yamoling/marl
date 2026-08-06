from marlenv import DiscreteMARLEnv

from marl.env import EnvConfig
from marl.models.nn import QNetwork

from .maven import MAVENQnetwork
from .qnetworks import QCNN, QCRNN, QMLP, QRNN


def from_env(
    env: DiscreteMARLEnv | EnvConfig[DiscreteMARLEnv],
    recurrent: bool = False,
    independent: bool = False,
    duelling: bool = True,
    noisy: bool = False,
    **init_kwargs,
) -> QNetwork:
    registry: dict[tuple[int, bool], type[QNetwork]] = {
        (1, False): QMLP,
        (3, False): QCNN,
        (1, True): QRNN,
        (3, True): QCRNN,
    }
    config = (len(env.observation_shape), recurrent)
    network_class = registry.get(config)
    if network_class is not None:
        return network_class.from_env(env, duelling=duelling, noisy=noisy, independent=independent, **init_kwargs)
    err_msg = "\n".join([f" - Shape Len: {s}, Recurrent: {r}" for s, r in registry])
    raise NotImplementedError(f"Unsupported configuration: {config}.\nSupported combinations are:\n{err_msg}")


__all__ = [
    "QCNN",
    "QCRNN",
    "QMLP",
    "QRNN",
    "MAVENQnetwork",
    "from_env",
]

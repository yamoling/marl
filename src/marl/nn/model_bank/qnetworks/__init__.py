from marlenv import DiscreteMARLEnv

from marl.env import EnvConfig
from marl.models.nn import QNetwork

from .independent import ICRNN, IQCNN, IQMLP, IQRNN
from .maven import MAVENQnetwork
from .shared_parameters import QCNN, QCRNN, QMLP, QRNN


def from_env(
    env: DiscreteMARLEnv | EnvConfig[DiscreteMARLEnv],
    recurrent: bool = False,
    independent: bool = False,
    duelling: bool = True,
    noisy: bool = False,
) -> QNetwork:
    registry: dict[tuple[int, bool, bool], type[QNetwork]] = {
        (1, False, False): QMLP,
        (1, False, True): IQMLP,
        (3, False, False): QCNN,
        (3, False, True): IQCNN,
        (1, True, False): QRNN,
        (1, True, True): IQRNN,
        (3, True, False): QCRNN,
        (3, True, True): ICRNN,
    }
    config = (len(env.observation_shape), recurrent, independent)
    network_class = registry.get(config)
    if network_class is not None:
        return network_class.from_env(env, duelling=duelling, noisy=noisy)
    err_msg = "\n".join([f" - Shape Len: {s}, Recurrent: {r}, Independent: {i}" for s, r, i in registry.keys()])
    raise NotImplementedError(f"Unsupported configuration: {config}.\nSupported combinations are:\n{err_msg}")


__all__ = [
    "MAVENQnetwork",
    "IQCNN",
    "ICRNN",
    "IQMLP",
    "IQRNN",
    "QMLP",
    "QCNN",
    "QRNN",
    "QCRNN",
    "from_env",
]

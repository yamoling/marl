from marl.utils import make_registry

from . import dqn_trainer, nodes
from .dqn_trainer import DQNTrainer

register, from_dict = make_registry(DQNTrainer, [dqn_trainer])


__all__ = ["nodes", "DQNTrainer", "register", "from_dict"]

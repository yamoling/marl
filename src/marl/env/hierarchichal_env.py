import numpy as np
from marlenv import MARLEnv
from marlenv.wrappers import RLEnvWrapper
from marlenv.models.spaces import ActionSpace, DiscreteActionSpace, ContinuousActionSpace
from dataclasses import dataclass


@dataclass
class HierarchicalActionSpace(ActionSpace):
    meta_space: ContinuousActionSpace
    worker_space: DiscreteActionSpace

    def __init__(self, meta_space: ContinuousActionSpace, worker_space: DiscreteActionSpace):
        self.meta_space = meta_space
        self.worker_space = worker_space
        super().__init__(worker_space.n_agents, worker_space.individual_action_space, worker_space.action_names)

    def sample(self, mask: np.ndarray | None = None):
        meta_action = self.meta_space.sample()
        worker_action = self.worker_space.sample(mask)
        return meta_action, worker_action


class HierachicalEnv(RLEnvWrapper[tuple[np.ndarray, np.ndarray]]):
    def __init__(self, env: MARLEnv[tuple[np.ndarray, np.ndarray], DiscreteActionSpace], n_subgoals: int):
        new_extras_shape = (env.extra_shape[0] + n_subgoals,)
        super().__init__(
            env,
            extra_shape=new_extras_shape,
            action_space=HierarchicalActionSpace(ContinuousActionSpace(1, [-1.0], [1.0]), env.action_space),
        )

    def step(self, actions: tuple[np.ndarray, np.ndarray]):
        _meta_action, worker_action = actions
        return self.wrapped.step(worker_action)

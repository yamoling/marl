from copy import copy
from dataclasses import dataclass

import numpy as np
from marlenv import Observation


@dataclass(frozen=True)
class HavenSpec:
    """Immutable description of HAVEN's hierarchy and observation layout."""

    n_workers: int
    n_subgoals: int
    k: int
    n_meta_extras: int
    n_agent_extras: int
    use_previous_meta_action: bool = True

    def __post_init__(self):
        """Validate dimensions shared by acting, collection and replay. @ai-generated"""
        if min(self.k, self.n_workers, self.n_subgoals) <= 0:
            raise ValueError("k, n_workers and n_subgoals must be positive")
        if min(self.n_meta_extras, self.n_agent_extras) < 0:
            raise ValueError("Extras sizes must be nonnegative")

    @property
    def meta_extras_size(self) -> int:
        return self.n_meta_extras + (self.n_subgoals if self.use_previous_meta_action else 0)

    @property
    def worker_extras_size(self) -> int:
        return self.n_meta_extras + self.n_agent_extras + self.n_subgoals

    def validate_action(self, action) -> np.ndarray:
        """Return a private, validated vector of discrete macro actions. @ai-generated"""
        result = np.asarray(action)
        if (
            result.shape != (self.n_workers,)
            or not np.issubdtype(result.dtype, np.integer)
            or np.any((result < 0) | (result >= self.n_subgoals))
        ):
            raise ValueError("HAVEN requires one discrete macro action per worker")
        return result.copy()

    def encode_goal(self, action) -> np.ndarray:
        """Encode one discrete macro action per worker as one-hot worker goals. @ai-generated"""
        return np.eye(self.n_subgoals, dtype=np.float32)[self.validate_action(action)]

    def meta_observation(self, observation: Observation, previous_action=None) -> Observation:
        """Project a worker observation into the macro policy's input space. @ai-generated"""
        result = copy(observation)
        result.extras = observation.extras[:, : self.n_meta_extras].copy()
        if self.use_previous_meta_action:
            previous = np.zeros((self.n_workers, self.n_subgoals), dtype=np.float32)
            if previous_action is not None:
                previous = self.encode_goal(previous_action)
            result.extras = np.concatenate((result.extras, previous), axis=-1)
        result.available_actions = np.ones((self.n_workers, self.n_subgoals), dtype=bool)
        return result

    def worker_observation(self, observation: Observation, action) -> Observation:
        """Copy a worker observation and fill its reserved subgoal suffix. @ai-generated"""
        result = copy(observation)
        result.extras = observation.extras.copy()
        result.extras[:, -self.n_subgoals :] = self.encode_goal(action)
        return result

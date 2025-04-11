from dataclasses import dataclass
from typing import Any, Generic, Sequence, Optional
from typing_extensions import TypeVar

import numpy as np
import numpy.typing as npt

from .observation import Observation
from .state import State
from .step import Step


A = TypeVar("A", default=np.ndarray)


@dataclass
class Transition(Generic[A]):
    """Transition model"""

    obs: Observation
    state: State
    action: A
    reward: npt.NDArray[np.float32]
    done: bool
    info: dict[str, Any]
    next_obs: Observation
    next_state: State
    truncated: bool
    qvalues: npt.NDArray[np.float32]
    other: dict[str, Any]

    def __init__(
        self,
        obs: Observation,
        state: State,
        action: A,
        reward: npt.NDArray[np.float32] | float | Sequence[float],
        done: bool,
        info: dict[str, Any],
        next_obs: Observation,
        next_state: State,
        truncated: bool,
        qvalues: Optional[npt.NDArray[np.float32]] = None,
        **kwargs,
    ):
        self.obs = obs
        if not isinstance(action, np.ndarray):
            action = np.array(action)
        self.action = action
        match reward:
            case np.ndarray():
                self.reward = reward
            case float() | int():
                self.reward = np.array([reward], dtype=np.float32)
            case other:
                # We assume this is a sequence of some sort
                self.reward = np.array(other, dtype=np.float32)
        self.done = done
        self.info = info
        self.next_obs = next_obs
        self.truncated = truncated
        self.state = state
        self.next_state = next_state
        self.qvalues = qvalues
        self.other = kwargs

    @staticmethod
    def from_step(
        prev_obs: Observation,
        prev_state: State,
        actions: A,
        step: Step,
        qvalues: Optional[npt.NDArray[np.float32]] = None,
        **kwargs,
    ):
        if qvalues is None:
            return Transition(
                obs=prev_obs,
                state=prev_state,
                action=actions,
                reward=step.reward,
                done=step.done,
                info=step.info,
                next_obs=step.obs,
                next_state=step.state,
                truncated=step.truncated,
                **kwargs,
            )
        else:
            return Transition(
                obs=prev_obs,
                state=prev_state,
                action=actions,
                reward=step.reward,
                done=step.done,
                info=step.info,
                next_obs=step.obs,
                next_state=step.state,
                truncated=step.truncated,
                qvalues=qvalues,
                **kwargs,
            )

    @property
    def is_terminal(self) -> bool:
        """Whether the transition is the last one"""
        return self.done or self.truncated

    @property
    def n_agents(self) -> int:
        """
        The number of agents computed from the number of actions.

        Note: this fails if the action does not have a __len__ method.
        """
        return len(self.action)  # type: ignore

    @property
    def n_actions(self) -> int:
        return int(self.obs.available_actions.shape[-1])

    def single_agent(self, agent_id: int, keep_dim: bool = True) -> "Transition":
        """Return a transition for a single agent"""
        obs = self.obs.agent(agent_id, keep_dim)
        next_obs = self.next_obs.agent(agent_id, keep_dim)
        if keep_dim:
            action = self.action[agent_id : agent_id + 1]  # type: ignore
        else:
            action = self.action[agent_id]  # type: ignore
        return Transition(
            obs=obs,
            state=self.state,
            action=action,
            reward=self.reward[agent_id : agent_id + 1],
            done=self.done,
            info=self.info,
            next_obs=next_obs,
            next_state=self.next_state,
            truncated=self.truncated,
            **self.other,
        )

    def __getitem__(self, key: str):
        if key not in self.other:
            keys = self.other.keys()
            if len(keys) == 0:
                raise KeyError(f"{key} not found in transition: no key available in transition")
            keys = ", ".join(keys)
            raise KeyError(f"Key {key} not found in transition. The availables keys are: {keys}")
        return self.other[key]

    def __hash__(self) -> int:
        ho = hash(self.obs)
        ho_ = hash(self.next_obs)
        hs = hash(self.state)
        hs_ = hash(self.next_state)
        if isinstance(self.reward, np.ndarray):
            hr = hash(self.reward.tobytes())
        else:
            hr = self.reward
        if isinstance(self.action, np.ndarray):
            ha = hash(self.action.tobytes())
        else:
            ha = hash(self.action)
        return hash((ho, ha, hr, self.done, ho_, self.truncated, hs, hs_))

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Transition):
            return False
        if isinstance(self.action, np.ndarray):
            if not isinstance(other.action, np.ndarray):
                return False
            if not np.array_equal(self.action, other.action):
                return False
        elif self.action != other.action:
            return False
        return (
            self.done == other.done
            and self.truncated == other.truncated
            and np.array_equal(self.reward, other.reward)
            and self.obs == other.obs
            and self.state == other.state
            and self.next_obs == other.next_obs
            and self.state == other.state
        )

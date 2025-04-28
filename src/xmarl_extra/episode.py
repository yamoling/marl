from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable, Generic, Optional, Sequence, TypeVar, overload

import numpy as np
import numpy.typing as npt
import cv2

from .observation import Observation
from .state import State
from .step import Step
from .transition import Transition
from .env import MARLEnv
from marlenv.exceptions import EnvironmentMismatchException, ReplayMismatchException


A = TypeVar("A")


@dataclass
class Episode(Generic[A]):
    """Episode model made of observations, actions, rewards, ..."""

    all_observations: list[npt.NDArray[np.float32]]
    all_extras: list[npt.NDArray[np.float32]]
    actions: list[npt.NDArray]
    qvalues: list[npt.NDArray]
    rewards: list[npt.NDArray[np.float32]]
    all_available_actions: list[npt.NDArray[np.bool_]]
    all_states: list[npt.NDArray[np.float32]]
    all_states_extras: list[npt.NDArray[np.float32]]
    metrics: dict[str, float]
    qvalues_met: dict[str, float]
    episode_len: int
    other: dict[str, list[Any]]
    is_done: bool = False
    is_truncated: bool = False
    """Whether the episode did reach a terminal state (different from truncated)"""

    @staticmethod
    def new(obs: Observation, state: State, metrics: Optional[dict[str, float]] = None) -> "Episode":
        if metrics is None:
            metrics = {}
        return Episode(
            all_observations=[obs.data],
            all_extras=[obs.extras],
            all_states=[state.data],
            all_states_extras=[state.extras],
            all_available_actions=[obs.available_actions],
            actions=[],
            qvalues=[],
            rewards=[],
            metrics=metrics,
            qvalues_met={},
            episode_len=0,
            is_done=False,
            is_truncated=False,
            other={},
        )

    @staticmethod
    def from_transitions(transitions: Sequence[Transition[A]]) -> "Episode":
        """Create an episode from a list of transitions"""
        episode = Episode.new(transitions[0].obs, transitions[0].state)
        for transition in transitions:
            episode.add(transition)
        return episode

    def padded(self, target_len: int) -> "Episode":
        """Copy of the episode, padded with mock items to the target length"""
        if target_len == self.episode_len:
            return self
        if target_len < self.episode_len:
            raise ValueError(f"Cannot pad episode to a smaller size: {target_len} < {self.episode_len}")
        padding_size = target_len - self.episode_len
        obs = self.all_observations + [self.all_observations[0]] * padding_size
        extras = self.all_extras + [self.all_extras[0]] * padding_size
        actions = self.actions + [self.actions[0]] * padding_size
        qvalues = self.qvalues + [self.qvalues[0]] * padding_size
        rewards = self.rewards + [self.rewards[0]] * padding_size
        availables = self.all_available_actions + [self.all_available_actions[0]] * padding_size
        states = self.all_states + [self.all_states[0]] * padding_size
        states_extras = self.all_states_extras + [self.all_states_extras[0]] * padding_size
        other = {key: value + [value[0]] * padding_size for key, value in self.other.items()}
        return Episode(
            all_observations=obs,
            all_extras=extras,
            all_states=states,
            all_states_extras=states_extras,
            all_available_actions=availables,
            actions=actions,
            qvalues=qvalues,
            rewards=rewards,
            metrics=self.metrics,
            qvalues_met=self.qvalues_met,
            episode_len=self.episode_len,
            is_done=self.is_done,
            is_truncated=self.is_truncated,
            other=other,
        )

    def __getitem__(self, key: str):
        if key not in self.other:
            keys = self.other.keys()
            if len(keys) == 0:
                raise KeyError(f"Key {key} not found in episode: no key available in episode.")
            keys = ", ".join(keys)
            raise KeyError(f"Key {key} not found in episode. The availables keys are: {keys}")
        return self.other[key]

    @property
    def observation_shape(self):
        return self.all_observations[0].shape

    @property
    def extras_shape(self):
        return self.all_extras[0].shape

    @cached_property
    def states(self):
        """The states"""
        return self.all_states[:-1]

    @cached_property
    def states_extras(self):
        """The extra features of the states"""
        return self.all_states_extras[:-1]

    @cached_property
    def next_states(self):
        """The next states"""
        return self.all_states[1:]

    @cached_property
    def next_states_extras(self):
        """The next extra features of the states"""
        return self.all_states_extras[1:]

    @cached_property
    def mask(self):
        """Get the mask for the current episide (when padded)"""
        mask = np.ones_like(self.rewards, dtype=np.float32)
        mask[self.episode_len :] = 0
        return mask

    @cached_property
    def obs(self):
        """The observations"""
        return self.all_observations[:-1]

    @cached_property
    def next_obs(self):
        """The next observations"""
        return self.all_observations[1:]

    @cached_property
    def extras(self):
        """Get the extra features"""
        return self.all_extras[:-1]

    @cached_property
    def next_extras(self):
        """Get the next extra features"""
        return self.all_extras[1:]

    @cached_property
    def n_agents(self):
        """The number of agents in the episode"""
        return self.all_extras[0].shape[0]

    @cached_property
    def n_actions(self):
        """The number of actions"""
        return len(self.all_available_actions[0][0])

    @cached_property
    def available_actions(self):
        """The available actions"""
        return self.all_available_actions[:-1]

    @cached_property
    def next_available_actions(self):
        """The next available actions"""
        return self.all_available_actions[1:]
    
    @cached_property
    def is_multi_objective(self):
        """Is multi objective"""
        return self.rewards[0].shape[0] > 1

    @cached_property
    def dones(self):
        """The done flags for each transition"""
        dones = np.zeros_like(self.rewards, dtype=np.float32)
        if self.is_done:
            dones[self.episode_len - 1 :] = 1.0
        return dones

    @property
    def is_finished(self) -> bool:
        """Whether the episode is done or truncated"""
        return self.is_done or self.is_truncated

    def transitions(self):
        """The transitions that compose the episode"""
        for i in range(self.episode_len):
            yield Transition(
                obs=Observation(
                    data=self.all_observations[i],
                    available_actions=self.all_available_actions[i],
                    extras=self.all_extras[i],
                ),
                state=State(data=self.all_states[i], extras=self.all_states_extras[i]),
                action=self.actions[i],
                reward=self.rewards[i],
                done=bool(self.dones[i][0] if self.is_multi_objective else self.dones[i]),
                info={},
                next_obs=Observation(
                    data=self.all_observations[i + 1],
                    available_actions=self.all_available_actions[i + 1],
                    extras=self.all_extras[i + 1],
                ),
                next_state=State(data=self.all_states[i + 1], extras=self.all_states_extras[i + 1]),
                truncated=not self.is_done and i == self.episode_len - 1,
            )

    def replay(
        self,
        env: MARLEnv[A, Any],
        seed: Optional[int] = None,
        *,
        after_reset: Optional[Callable[[Observation, State, MARLEnv[A]], None]] = None,
        after_step: Optional[Callable[[int, Step, MARLEnv[A]], None]] = None,
    ):
        """
        Replay the episode in the environment (i.e. perform the actions) and assert that the outcomes match.
        If provided, the callback is called at each step with the index of the step and the step itself.

        Note: this will most likely fail in stochastic environments.
        """
        if env.n_actions != self.n_actions or env.n_agents != self.n_agents:
            raise EnvironmentMismatchException(env, self)
        if seed is not None:
            env.seed(seed)
        obs, state = env.reset()
        if after_reset is not None:
            after_reset(obs, state, env)
        if not np.array_equal(obs.data, self.obs[0]):
            raise ReplayMismatchException("observation", obs.data, self.obs[0], time_step=0)
        if not np.array_equal(state.data, self.states[0]):
            raise ReplayMismatchException("state", state.data, self.states[0], time_step=0)
        for i, action in enumerate(self.actions):
            step = env.step(action)  # type: ignore
            if not np.array_equal(step.obs.data, self.next_obs[i]):
                raise ReplayMismatchException("observation", step.obs.data, self.next_obs[i], time_step=i)
            if not np.array_equal(step.state.data, self.next_states[i]):
                raise ReplayMismatchException("state", step.state.data, self.next_states[i], time_step=i)
            if not np.array_equal(step.reward, self.rewards[i]):
                raise ReplayMismatchException("reward", step.reward, self.rewards[i], time_step=i)
            if after_step is not None:
                after_step(i, step, env)
        # plot qvalues

    def get_images(self, env: MARLEnv[A, Any], seed: Optional[int] = None) -> list[np.ndarray]:
        images = []

        def collect_image(*_, **__):
            images.append(env.get_image())

        self.replay(env, seed, after_reset=collect_image, after_step=collect_image)
        return images

    def render(self, env: MARLEnv[A, Any], seed: Optional[int] = None, fps: int = 5):
        def render_callback(*_, **__):
            env.render()
            cv2.waitKey(1000 // fps)

        self.replay(env, seed, after_reset=render_callback, after_step=render_callback)

    def __iter__(self):
        return self.transitions()

    def __len__(self):
        return self.episode_len

    @cached_property
    def score(self) -> list[float]:
        """The episode score (sum of all rewards across all objectives)"""
        score = []
        for key, value in self.metrics.items():
            if key.startswith("score-"):
                score.append(value)
        return score

    def compute_returns(self, discount: float = 1.0):
        """Compute the returns (discounted sum of future rewards) of the episode at each time step"""
        returns = np.zeros_like(self.rewards)
        returns[-1] = self.rewards[-1]
        for t in range(len(self.rewards) - 2, -1, -1):
            returns[t] = self.rewards[t] + discount * returns[t + 1]
        return returns

    @overload
    def add(self, transition: Transition[A], /): ...

    @overload
    def add(self, step: Step, action: A, /): ...

    def add(self, *data):
        match data:
            case (Transition() as transition,):
                self.add_data(
                    transition.next_obs,
                    transition.next_state,
                    transition.action,
                    transition.qvalues,
                    transition.reward,
                    transition.other,
                    transition.done,
                    transition.truncated,
                    transition.info,
                )
            case (Step() as step, action):
                self.add_data(
                    step.obs,
                    step.state,
                    action,
                    None,
                    step.reward,
                    {},
                    step.done,
                    step.truncated,
                    step.info,
                )
            case (Step() as step, action, qvalues):
                self.add_data(
                    step.obs,
                    step.state,
                    action,
                    qvalues,
                    step.reward,
                    {},
                    step.done,
                    step.truncated,
                    step.info,
                )
            case other:
                raise ValueError(f"Invalid arguments: {other}")

    def add_data(
        self,
        next_obs,
        next_state,
        action: A,
        qvalues: np.ndarray,
        reward: np.ndarray,
        others: dict[str, Any],
        done: bool,
        truncated: bool,
        info: dict[str, Any],
    ):
        """Add a transition to the episode"""
        is_terminal = done or truncated
        self.episode_len += 1
        self.all_observations.append(next_obs.data)
        self.all_extras.append(next_obs.extras)
        self.all_available_actions.append(next_obs.available_actions)
        self.all_states.append(next_state.data)
        self.all_states_extras.append(next_state.extras)
        match action:
            case np.ndarray() as action:
                self.actions.append(action)
            case other:
                self.actions.append(np.array(other))
        if qvalues is not None: self.qvalues.append(qvalues)
        self.rewards.append(reward)
        for key, value in others.items():
            current = self.other.get(key, [])
            current.append(value)
            self.other[key] = current

        if is_terminal:
            # Only set the truncated flag if the episode is not done (both could happen with a time limit)
            self.is_truncated = truncated
            self.is_done = done
            # Add metrics that can be plotted
            for key, value in info.items():
                if isinstance(value, bool):
                    value = int(value)
                self.metrics[key] = value
            self.metrics["episode_len"] = self.episode_len

            rewards = np.array(self.rewards)
            scores = np.sum(rewards, axis=0)
            for i, s in enumerate(scores):
                self.metrics[f"score-{i}"] = float(s)

            if qvalues is not None:
                qvalues = np.array(self.qvalues)
                avg_qvalues = np.average(qvalues, axis=0)
                for ag_n, ag in enumerate(avg_qvalues):
                    if reward.size > 1:
                        for qv_n, qv in enumerate(ag.squeeze()):
                            self.qvalues_met[f"agent{ag_n}-qvalue{qv_n}"] = float(qv)
                    else: self.qvalues_met[f"agent{ag_n}-qvalue"] = float(ag)

    # def add_data(
    #     self,
    #     new_obs: Observation,
    #     new_state: State,
    #     action: A,
    #     reward: np.ndarray,
    #     done: bool,
    #     truncated: bool,
    #     info: dict[str, Any],
    #     **kwargs,
    # ):
    #     """Add a new transition to the episode"""
    #     self.episode_len += 1
    #     self.all_observations.append(new_obs.data)
    #     self.all_extras.append(new_obs.extras)
    #     self.all_available_actions.append(new_obs.available_actions)
    #     self.all_states.append(new_state.data)
    #     self.all_states_extras.append(new_state.extras)
    #     match action:
    #         case np.ndarray() as action:
    #             self.actions.append(action)
    #         case other:
    #             self.actions.append(np.array(other))
    #     self.rewards.append(reward)
    #     for key, value in kwargs.items():
    #         current = self.other.get(key, [])
    #         current.append(value)
    #         self.other[key] = current

    #     if done:
    #         # Only set the truncated flag if the episode is not done (both could happen with a time limit)
    #         self.is_truncated = truncated
    #         self.is_done = done
    #         # Add metrics that can be plotted
    #         for key, value in info.items():
    #             if isinstance(value, bool):
    #                 value = int(value)
    #             self.metrics[key] = value
    #         self.metrics["episode_len"] = self.episode_len

    #         rewards = np.array(self.rewards)
    #         scores = np.sum(rewards, axis=0)
    #         for i, s in enumerate(scores):
    #             self.metrics[f"score-{i}"] = float(s)

    def add_metrics(self, metrics: dict[str, float]):
        """Add metrics to the episode"""
        self.metrics.update(metrics)

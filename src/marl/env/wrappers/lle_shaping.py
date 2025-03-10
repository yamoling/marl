from typing import Sequence
import numpy as np
from marlenv.wrappers import RLEnvWrapper
from marlenv import Observation, DiscreteSpace, DiscreteActionSpace
from dataclasses import dataclass
from lle import LLE, Action


@dataclass
class LLEShaping(RLEnvWrapper[Sequence[int] | np.ndarray, DiscreteActionSpace]):
    reward_for_blocking: float

    def __init__(self, env: LLE, reward_for_blocking: float):
        super().__init__(env)
        self.lle = env
        self.world = env.world
        self.laser_colours = {laser.pos: laser.agent_id for laser in self.world.lasers}
        self.reward_for_blocking = reward_for_blocking

    def additional_reward(self, actions):
        r = 0
        for agent, agent_pos, agent_action in zip(self.world.agents, self.world.agents_positions, actions):
            # We increase the reward if the agent
            # - is alive, i.e. the laser is off (otherwise, no reward at all)
            # - is on a laser tile of an other colour
            if agent.is_dead:
                return 0
            if agent_action == Action.STAY.value:
                continue
            if agent_pos not in self.laser_colours:
                continue
            if self.laser_colours[agent_pos] != agent.num:
                r += self.reward_for_blocking
        return r

    def step(self, actions):
        step = self.wrapped.step(actions)
        step.reward += self.additional_reward(actions)
        return step


@dataclass
class LLEShapeEachLaser(RLEnvWrapper[Sequence[int] | np.ndarray, DiscreteActionSpace]):
    extra_reward: float

    def __init__(self, env_lvl6: LLE, extra_reward: float, enable_reward: bool, multi_objective: bool = False):
        # Whether the rewards for reaching lines 4, 5, 6 and 7 can still be collected by each agent
        self.rewards = [
            [True] * env_lvl6.n_agents,
            [True] * env_lvl6.n_agents,
            [True] * env_lvl6.n_agents,
            [True] * env_lvl6.n_agents,
        ]
        self.vertical_rewards = [True, True]
        extras_shape = (env_lvl6.extra_shape[0] + env_lvl6.n_agents * 4,)
        if multi_objective:
            reward_space = DiscreteSpace(env_lvl6.reward_space.size + 1, env_lvl6.reward_space.labels + ["shaping"])
        else:
            reward_space = env_lvl6.reward_space
        super().__init__(env_lvl6, extra_shape=extras_shape, reward_space=reward_space)
        self.world = env_lvl6.world
        self.extra_reward = extra_reward
        self.enable_reward = enable_reward

    def add_extra_information(self, obs: Observation):
        extra = np.array(self.rewards).reshape(-1)
        # extra = np.concatenate([np.array(self.rewards).reshape(-1), self.vertical_rewards])
        extra = np.tile(extra, (self.n_agents, 1))
        obs.extras = np.concatenate([obs.extras, extra], axis=-1)
        return obs

    def additional_reward(self):
        r = 0
        for agent_num, (i, j) in enumerate(self.world.agents_positions):
            if agent_num == 2:
                if 1 <= i <= 2 and j <= 1:
                    if self.vertical_rewards[j]:
                        r += self.extra_reward
                        self.vertical_rewards[j] = False
            index = i - 4
            if index < 0 or index > 3:
                continue
            if self.rewards[index][agent_num]:
                # Only reward agents if they cross the laser on the left hand side of the map
                if index == 3 and j > 6:
                    continue
                r += self.extra_reward
                self.rewards[index][agent_num] = False
        return r

    def reset(self):
        if not hasattr(self, "enable_reward"):
            self.enable_reward = True

        self.rewards = [
            [True] * self.n_agents,
            [True] * self.n_agents,
            [True] * self.n_agents,
            [True] * self.n_agents,
        ]
        self.vertical_rewards = [True, True]
        obs, state = self.wrapped.reset()
        return self.add_extra_information(obs), state

    def step(self, actions):
        step = self.wrapped.step(actions)
        step.obs = self.add_extra_information(step.obs)
        extra_reward = self.additional_reward()
        if self.enable_reward:
            step.reward += extra_reward
        return step

import numpy as np
from rlenv.wrappers import RLEnvWrapper
from rlenv import Observation
from dataclasses import dataclass
from serde import serde
from lle import LLE, Action


@serde
@dataclass
class LLEShaping(RLEnvWrapper):
    reward_for_blocking: float

    def __init__(self, env: LLE, reward_for_blocking: float):
        super().__init__(env)
        self.lle = env
        self.world = env.world
        self.laser_colours = {pos: laser.agent_id for pos, laser in self.world.lasers}
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
        obs, reward, done, truncated, info = self.wrapped.step(actions)
        reward += self.additional_reward(actions)
        return obs, reward, done, truncated, info


@serde
@dataclass
class LLEShapeEachLaser(RLEnvWrapper):
    extra_reward: float

    def __init__(self, env_lvl6: LLE, extra_reward: float):
        # Whether the rewards for reaching lines 4, 5, 6 and 7 can still be collected by each agent
        self.rewards = [
            [True] * env_lvl6.n_agents,
            [True] * env_lvl6.n_agents,
            [True] * env_lvl6.n_agents,
            [True] * env_lvl6.n_agents,
        ]
        extras_shape = (env_lvl6.extra_feature_shape[0] + env_lvl6.n_agents * 4,)
        super().__init__(env_lvl6, extra_feature_shape=extras_shape)
        self.world = env_lvl6.world
        self.extra_reward = extra_reward

    def add_extra_information(self, obs: Observation):
        extra = np.array(self.rewards).reshape(-1)
        extra = np.tile(extra, (self.n_agents, 1))
        obs.extras = np.concatenate([obs.extras, extra], axis=-1)
        return obs

    def additional_reward(self):
        r = 0
        for agent_num, (i, j) in enumerate(self.world.agents_positions):
            i = i - 4
            if i < 0 or i > 3:
                continue
            if self.rewards[i][agent_num]:
                # Only reward agents if they cross the laser on the left hand side of the map
                if i == 3 and j > 6:
                    continue
                r += self.extra_reward
                self.rewards[i][agent_num] = False
        return r

    def reset(self):
        self.rewards = [
            [True] * self.n_agents,
            [True] * self.n_agents,
            [True] * self.n_agents,
            [True] * self.n_agents,
        ]
        obs = self.wrapped.reset()
        return self.add_extra_information(obs)

    def step(self, actions):
        obs, reward, done, truncated, info = self.wrapped.step(actions)
        obs = self.add_extra_information(obs)
        reward += self.additional_reward()
        return obs, reward, done, truncated, info

import numpy as np
from rlenv.wrappers import RLEnvWrapper
from rlenv import Observation, RLEnv
from dataclasses import dataclass
from serde import serde
from lle import Position, World


@dataclass
class DelayedReward:
    started: bool
    reward: float
    delay: int
    countdown: int

    def __init__(self, reward: float, delay: int):
        self.started = False
        self.reward = reward
        self.delay = delay
        self.countdown = delay

    def step(self):
        if not self.started or self.countdown < 0:
            return 0.0
        self.countdown -= 1
        if self.countdown == 0:
            return self.reward
        return 0.0

    def reset(self):
        self.countdown = self.delay + 1

    def agent_enter(self):
        self.started = True


@dataclass
class DelayedRewardHandler:
    rewards: list[dict[Position, DelayedReward]]
    extras_size: int

    def __init__(self, n_agents: int, positions: list[list[Position]], delay: int, reward: float):
        self.rewards = [{} for _ in range(n_agents)]
        for agent_id, agent_positions in enumerate(positions):
            for pos in agent_positions:
                self.rewards[agent_id][pos] = DelayedReward(reward, delay)
        self.extras_size = sum(len(p) for p in positions)

    def get_state(self):
        res = []
        for reward_dict in self.rewards:
            for _, reward in reward_dict.items():
                if reward.countdown < 0:
                    res.append(-1)
                else:
                    res.append(reward.countdown / (reward.delay + 1))
        return np.array(res, dtype=np.float32)

    def step(self, agents_positions: list[Position]):
        for reward_dict, pos in zip(self.rewards, agents_positions):
            delayed_reward = reward_dict.get(pos)
            if delayed_reward is not None:
                delayed_reward.agent_enter()

        total = 0.0
        for reward_dict in self.rewards:
            for _, reward in reward_dict.items():
                total += reward.step()
        return total

    def reset(self):
        for reward_dict in self.rewards:
            for _, reward in reward_dict.items():
                reward.reset()


@serde
@dataclass
class BShaping(RLEnvWrapper):
    """Bottleneck shaping"""

    def __init__(self, env: RLEnv, world: World, extra_reward: float, delay: int):
        # Assumes that
        # - the start positions are on top
        # - the exits are on the bottom
        # - the lasers are horizontal
        # - lasers do not cross
        reward_positions = [list[Position]() for _ in range(world.n_agents)]
        for (i, j), laser in world.lasers:
            for agent_id in range(world.n_agents):
                if laser.agent_id != agent_id:
                    reward_positions[agent_id].append((i + 1, j))
        self.delayed_rewards = DelayedRewardHandler(world.n_agents, reward_positions, delay, extra_reward)
        extras_shape = (env.extra_feature_shape[0] + self.delayed_rewards.extras_size,)
        super().__init__(env, extra_feature_shape=extras_shape)
        self.world = world

    def add_extra_information(self, obs: Observation):
        extra = self.delayed_rewards.get_state()
        extra = np.tile(extra, (self.n_agents, 1))
        obs.extras = np.concatenate([obs.extras, extra], axis=-1)
        return obs

    def reset(self):
        self.delayed_rewards.reset()
        obs = self.wrapped.reset()
        return self.add_extra_information(obs)

    def step(self, actions):
        obs, reward, done, truncated, info = self.wrapped.step(actions)
        reward += self.delayed_rewards.step(self.world.agents_positions)
        obs = self.add_extra_information(obs)
        return obs, reward, done, truncated, info

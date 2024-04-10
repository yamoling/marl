import numpy as np
from rlenv.wrappers import RLEnvWrapper
from rlenv import Observation, DiscreteSpace
from dataclasses import dataclass
from serde import serde
from lle import LLE, Action, Position, AgentId


@dataclass
class DelayedReward:
    reward: float
    delay: int
    consumed: bool
    current_delay: int

    def __init__(self, reward: float, delay: int):
        self.consumed = False
        self.reward = reward
        self.delay = delay
        self.current_delay = 0

    def step(self) -> float:
        if self.consumed:
            return 0
        self.current_delay += 1
        if self.current_delay <= self.delay:
            return 0
        self.consumed = True
        return self.reward

    def reset(self):
        self.consumed = False
        self.current_delay = 0


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
                res.append(not reward.consumed)
        return np.array(res, dtype=np.float32)

    def step(self, agents_positions: list[Position]):
        total = 0.0
        for reward_dict, pos in zip(self.rewards, agents_positions):
            delayed_reward = reward_dict.get(pos)
            if delayed_reward is not None:
                total += delayed_reward.step()
        return total

    def reset(self):
        for reward_dict in self.rewards:
            for _, reward in reward_dict.items():
                reward.reset()


@serde
@dataclass
class BShaping(RLEnvWrapper):
    extra_reward: float

    def __init__(self, env: LLE, extra_reward: float, delay: int):
        # Assumes that
        # - the start positions are on top
        # - the exits are on the bottom
        # - the lasers are horizontal
        # - lasers do not cross
        reward_positions = [list[Position]() for _ in range(env.n_agents)]
        for (i, j), laser in env.world.lasers:
            for agent_id in range(env.n_agents):
                if laser.agent_id != agent_id:
                    reward_positions[agent_id].append((i + 1, j))
        self.delayed_rewards = DelayedRewardHandler(env.n_agents, reward_positions, delay, extra_reward)

        extras_shape = (env.extra_feature_shape[0] + self.delayed_rewards.extras_size,)
        super().__init__(env, extra_feature_shape=extras_shape)
        self.world = env.world

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
        obs = self.add_extra_information(obs)
        reward += self.delayed_rewards.step(self.world.agents_positions)
        return obs, reward, done, truncated, info

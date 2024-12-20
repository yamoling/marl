import numpy as np
from marlenv.wrappers import RLEnvWrapper
from marlenv import Observation, MARLEnv
from dataclasses import dataclass
from lle import Position, World
from lle.tiles import Direction


@dataclass
class DelayedReward:
    started: bool
    reward: float
    delay: int
    countdown: int
    i_condition: int

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
        self.started = False

    def activate(self):
        self.started = True


@dataclass
class DelayedRewardHandler:
    rewards: list[dict[int, DelayedReward]]
    extras_size: int

    def __init__(self, n_agents: int, i_positions: list[set[int]], delay: int, reward: float):
        self.rewards = [{} for _ in range(n_agents)]
        for agent_id, agent_rows in enumerate(i_positions):
            for row in agent_rows:
                self.rewards[agent_id][row] = DelayedReward(reward, delay)
        self.extras_size = sum(len(p) for p in i_positions)

    def get_state(self):
        res = []
        for reward_dict in self.rewards:
            for _, reward in reward_dict.items():
                if reward.countdown < 0:
                    res.append(-1)
                else:
                    res.append(reward.countdown / (reward.delay + 1))
        return np.array(res, dtype=np.float32)

    def trigger(self, agents_positions: list[Position]):
        for reward_dict, (i, _) in zip(self.rewards, agents_positions):
            delayed_reward = reward_dict.get(i)
            if delayed_reward is not None:
                delayed_reward.activate()

        total = 0.0
        for reward_dict in self.rewards:
            for _, reward in reward_dict.items():
                total += reward.step()
        return total

    def reset(self):
        for reward_dict in self.rewards:
            for _, reward in reward_dict.items():
                reward.reset()


@dataclass
class BShaping(RLEnvWrapper):
    """Bottleneck shaping"""

    def __init__(self, env: MARLEnv, world: World, extra_reward: float, delay: int, reward_in_laser: bool):
        # Assumes that
        # - the start positions are on top
        # - the exits are on the bottom
        # - the lasers are horizontal
        # - lasers do not cross
        reward_positions = [set[int]() for _ in range(world.n_agents)]
        for (i, _), laser in world.lasers:
            if laser.direction not in [Direction.EAST, Direction.WEST]:
                continue
            for agent_id in range(world.n_agents):
                if reward_in_laser:
                    reward_positions[agent_id].add(i)
                reward_positions[agent_id].add(i + 1)
        self.delayed_rewards = DelayedRewardHandler(world.n_agents, reward_positions, delay, extra_reward)
        extras_shape = (env.extra_shape[0] + self.delayed_rewards.extras_size,)
        super().__init__(env, extra_shape=extras_shape)
        self.world = world

    def add_extra_information(self, obs: Observation):
        extra = self.delayed_rewards.get_state()
        extra = np.tile(extra, (self.n_agents, 1))
        obs.extras = np.concatenate([obs.extras, extra], axis=-1)
        return obs

    def reset(self):
        self.delayed_rewards.reset()
        obs, state = self.wrapped.reset()
        return self.add_extra_information(obs), state

    def step(self, actions):
        step = self.wrapped.step(actions)
        step.reward += self.delayed_rewards.trigger(self.world.agents_positions)
        step.obs = self.add_extra_information(step.obs)
        return step

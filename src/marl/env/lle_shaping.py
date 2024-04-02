from rlenv.wrappers import RLEnvWrapper
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

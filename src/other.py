from typing import Sequence
from lle import LLE, Action
from numpy import ndarray
import marl
from marlenv import MultiDiscreteSpace
from create_experiments import make_dqn
import random

from marlenv import RLEnvWrapper, MARLEnv


class PreventActions(RLEnvWrapper[MultiDiscreteSpace]):
    def __init__(self, env: MARLEnv[MultiDiscreteSpace] | LLE):
        super().__init__(env)
        match env:
            case LLE():
                lle = env
            case RLEnvWrapper():
                lle = env.unwrapped
                assert isinstance(lle, LLE)
            case MARLEnv():
                raise ValueError("This wrapper only works with LLE or RLEnvWrapper")
        self.world = lle.world
        self.forbidden = {
            (3, 0): Action.SOUTH,
            (3, 1): Action.SOUTH,
            (3, 3): Action.SOUTH,
            (3, 4): Action.SOUTH,
            (3, 5): Action.SOUTH,
        }

    def step(self, actions: ndarray | Sequence):
        if random.random() < 0.1:
            actions = self.action_space.sample()
        return super().step(actions)

    def available_actions(self):
        available = super().available_actions()
        pos = self.world.agents_positions[0]
        action = self.forbidden.get(pos)
        if action is not None:
            available[0][action.value] = False
        return available


def main():
    map_str = """
.  .  S0 .  . .
.  .  .  .  . .
.  .  .  .  . .
.  .  .  .  . .
.  .  .  .  . .
.  .  .  .  . .
.  .  .  .  . .
.  .  .  .  . .
.  .  X  .  . .
"""
    env = LLE.from_str(map_str).builder().time_limit(6 * 9 // 2).build()
    env = PreventActions(env)
    trainer = make_dqn(env, mixing="vdn")
    experiment = marl.Experiment.create(
        env,
        50_000,
        logdir="logs/1-agent-prevented",
        trainer=trainer,
        agent=trainer.make_agent(marl.policy.EpsilonGreedy.constant(0.1)),
    )
    for i in range(10):
        experiment.run(seed=i, n_tests=5)


if __name__ == "__main__":
    main()

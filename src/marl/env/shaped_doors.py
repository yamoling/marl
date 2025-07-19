import numpy as np
from collections import OrderedDict
from typing import Sequence
from lle import LLE
from marlenv import Builder, RLEnvWrapper, MultiDiscreteSpace
from dataclasses import dataclass


@dataclass
class ShapedDoors(RLEnvWrapper[MultiDiscreteSpace]):
    def __init__(self, delay: int):
        self.delay = delay
        self._key_pos = [(7, 3), (0, 7)]
        self._reward_countdown = OrderedDict.fromkeys(self._key_pos, delay)
        lle = LLE.from_file("maps/doors").obs_type("layered").state_type("state").build()
        self._world = lle.world
        env = Builder(lle).time_limit(lle.width * lle.height // 2).build()
        super().__init__(
            env,
            extra_shape=(len(self._key_pos) + env.extras_shape[0],),
            extra_meanings=env.extras_meanings + [f"checkpoint {i}" for i in range(len(self._key_pos))],
        )

    def reset(self):
        self._reward_countdown = OrderedDict.fromkeys(self._key_pos, self.delay)
        _, state = super().reset()
        return self.get_observation(), state

    def get_observation(self):
        obs = super().get_observation()
        extra = np.array([[r for r in self._reward_countdown.values()]], dtype=np.float32)
        if self.delay != 0:
            extra = (extra / self.delay).astype(np.float32)
        obs.add_extra(extra)
        return obs

    def step(self, actions: np.ndarray | Sequence):
        step = super().step(actions)
        agent_pos = self._world.agents_positions[0]
        if agent_pos in self._reward_countdown:
            countdown = self._reward_countdown[agent_pos]
            if countdown == 0:
                step.reward += 1.0
            self._reward_countdown[agent_pos] = max(-1, countdown - 1)
        step.obs = self.get_observation()
        if step.done:
            # Flush rewards for the final step
            for countdown in self._reward_countdown.values():
                if countdown >= 0 and countdown != self.delay:
                    step.reward += 1.0
        return step

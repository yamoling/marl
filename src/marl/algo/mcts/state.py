from marlenv import MARLEnv
from lle import WorldState, Action, World
from .base_state import BaseState


class MTCSLLEState(BaseState):
    def __init__(self, env: MARLEnv, world: World, state: WorldState, done: bool, truncated: bool):
        super().__init__()
        self.world = world
        self.env = env
        self.state = state
        self.done = done
        self.truncated = truncated
        # Convert to tuples such that they can be hashed
        self.possible_actions = [tuple(a) for a in self.world.available_joint_actions()]

    @property
    def is_terminal(self):
        return self.done or self.truncated

    def __hash__(self):
        return hash(self.state)

    def get_possible_actions(self):
        return self.possible_actions

    def take_action(self, action: list[Action]):
        self.world.set_state(self.state)
        _, reward, done, truncated, _ = self.env.step([a.value for a in action])
        return MTCSLLEState(self.env, self.world, self.world.get_state(), done, truncated), reward

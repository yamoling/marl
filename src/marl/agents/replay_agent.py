import logging

import numpy as np

from marl.models import Action, Agent


class ReplayAgent(Agent):
    def __init__(self, actions: np.ndarray | None, wrapped: Agent):
        super().__init__()
        self.actions = actions
        self.current_step = 0
        self.wrapped = wrapped
        self.mismatch = False
        """Whether there a mismatch has been encountered between the wrapped agent's actions and the replayed actions."""

    def choose_action(self, observation, *, with_details=False):
        wrapped_action = self.wrapped.choose_action(observation, with_details=with_details)
        if self.actions is None:
            return wrapped_action
        if self.current_step >= len(self.actions):
            raise IndexError("No more actions to replay.")
        replayed_action = self.actions[self.current_step]
        self.current_step += 1
        if not np.array_equal(wrapped_action.action, replayed_action):
            self.mismatch = True
            logging.warning(
                f"Wrapped agent chose a different action ({wrapped_action.action}) than the replayed action ({replayed_action})."
            )
            return Action(replayed_action)
        wrapped_action.action = replayed_action
        return wrapped_action

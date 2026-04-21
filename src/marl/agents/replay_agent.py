import logging

import numpy as np

from marl.models import Agent


class ReplayAgent[T](Agent[T]):
    def __init__(self, actions: np.ndarray | None, wrapped: Agent[T]):
        super().__init__()
        self.stored_actions = actions
        self.current_step = 0
        self.wrapped = wrapped
        self.mismatch = False
        """Whether there a mismatch has been encountered between the wrapped agent's actions and the replayed actions."""
        self.mismatch_details = []

    def choose_action(self, observation, *, with_details=False):
        online_action = self.wrapped.choose_action(observation, with_details=with_details)
        if self.stored_actions is None:
            return online_action
        if self.current_step < len(self.stored_actions):
            stored_action = self.stored_actions[self.current_step]
            self.current_step += 1
        else:
            msg = f"ReplayAgent has no more actions to replay at time step {self.current_step}, falling back to wrapped agent."
            logging.warning(msg)
            self.mismatch = True
            self.mismatch_details.append(msg)
            stored_action = online_action.action
        if not np.array_equal(online_action.action, stored_action):
            msg = f"Agent restored from disk chose action ({online_action.action})  which is different from the stored action ({stored_action}) at time step {self.current_step}."
            self.mismatch = True
            self.mismatch_details.append(msg)
            logging.warning(msg)
            online_action.action = stored_action
        return online_action

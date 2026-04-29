import logging

import numpy as np
import numpy.typing as npt

from marl.models import Action, Agent


class ReplayAgent(Agent[npt.ArrayLike]):
    def __init__(self):
        super().__init__()
        self.mismatch = False
        """Whether there a mismatch has been encountered between the wrapped agent's actions and the replayed actions."""
        self.mismatch_details = []

    @staticmethod
    def from_actions_only(actions: np.ndarray):
        return ReplayActionsOnlyAgent(actions)

    @staticmethod
    def from_agent_only(agent: Agent[npt.ArrayLike], weights_path: str):
        return SimpleReplayAgent(agent, weights_path)

    @staticmethod
    def from_agent_and_actions(agent: Agent[npt.ArrayLike], actions: np.ndarray, weights_path: str):
        agent.load(weights_path)
        return CombinedReplayAgent(actions, agent)


class CombinedReplayAgent(ReplayAgent):
    def __init__(self, actions: np.ndarray, wrapped: Agent[npt.ArrayLike]):
        super().__init__()
        self.actions_agent = ReplayActionsOnlyAgent(actions)
        self.saved_agent = wrapped

    def choose_action(self, observation, *, with_details=False):
        self.current_step += 1
        agent_action = self.saved_agent.choose_action(observation, with_details=with_details)
        saved_action = self.actions_agent.choose_action(observation, with_details=with_details)
        if not np.array_equal(saved_action.action, agent_action.action):
            msg = f"Agent restored from disk chose action ({agent_action.action})  which is different from the stored action ({saved_action.action}) at time step {self.current_step}."
            self.mismatch = True
            self.mismatch_details.append(msg)
            logging.warning(msg)
            agent_action.action = saved_action.action
        return agent_action


class ReplayActionsOnlyAgent(ReplayAgent):
    def __init__(self, actions: np.ndarray):
        super().__init__()
        self.stored_actions = actions
        self.current_step = -1

    def choose_action(self, observation, *, with_details=False):
        self.current_step += 1
        stored_action = self.stored_actions[self.current_step]
        return Action(stored_action)


class SimpleReplayAgent(ReplayAgent):
    def __init__(self, agent: Agent[npt.ArrayLike], weights_path: str):
        super().__init__()
        agent.load(weights_path)
        self.agent = agent

    def choose_action(self, observation, *, with_details=False):
        return self.agent.choose_action(observation, with_details=with_details)

from .deep_qwrapper import DeepQWrapper


class ForceActionWrapper(DeepQWrapper):
    def __init__(self, algo, force_actions: dict[int, int]):
        super().__init__(algo)
        self.force_actions = force_actions

    def choose_action(self, observation):
        actions = super().choose_action(observation)
        for agent_id, action in self.force_actions.items():
            actions[agent_id] = action
        return actions

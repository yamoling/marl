from marl.models.agent.bandit import CategoricalBandit, OneHotBandit


class UniformCategorical(CategoricalBandit):
    def choose_action(self, /, **kwargs):
        return self.space.sample()


class UniformOneHot(OneHotBandit):
    def __init__(self, n_actions: int):
        super().__init__(UniformCategorical(n_actions))

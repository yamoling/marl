from torch import Tensor, distributions

from marl.models import QNetwork
from marl.models.agent.bandit import OneHotBandit


class DeepBandit(OneHotBandit):
    def __init__(self, nn: QNetwork):
        self.nn = nn

    def choose_action(self, /, obs: Tensor | None = None, extras: Tensor | None = None, **kwargs):
        if obs is None or extras is None:
            raise ValueError("DeepBandit cannot choose an action without an `obs` and `extras` keyword arguments != None")
        expected_returns = self.nn.forward(obs, extras, **kwargs)
        dist = distributions.OneHotCategorical(probs=expected_returns)
        return dist.sample().numpy(force=True)

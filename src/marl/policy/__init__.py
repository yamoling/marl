from marl.utils.registry import make_registry

from . import qpolicies, random_policy
from .policy import Policy
from .qpolicies import ArgMax, EpsilonGreedy, SoftmaxPolicy
from .random_policy import RandomPolicy

register, from_dict = make_registry(Policy, [qpolicies, random_policy])


__all__ = ["Policy", "EpsilonGreedy", "SoftmaxPolicy", "ArgMax", "RandomPolicy", "register", "from_dict"]

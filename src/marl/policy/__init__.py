
from . import qpolicies, random_policy
from .policy import Policy
from .qpolicies import ArgMax, EpsilonGreedy, SoftmaxPolicy
from .random_policy import RandomPolicy



__all__ = ["Policy", "EpsilonGreedy", "SoftmaxPolicy", "ArgMax", "RandomPolicy", "register", "from_dict"]

from .policy import Policy
from .qpolicies import EpsilonGreedy, DecreasingEpsilonGreedy, SoftmaxPolicy, ArgMax
from .random_policy import RandomPolicy


from marl.utils.registry import make_registry
registry, from_summary = make_registry(Policy, [qpolicies, random_policy])

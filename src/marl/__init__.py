__version__ = "0.1.0"

from . import utils
from . import models
from . import logging
from . import nn
from . import intrinsic_reward
from .models import RLAlgo, Experiment

from . import training
from . import qlearning
from . import policy_gradient
from . import policy
from . import wrappers


register, load = utils.make_registry(RLAlgo, [qlearning, policy_gradient, utils.random_algo])
from .utils import seed

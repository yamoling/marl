__version__ = "0.1.0"

from . import utils
from . import models
from . import logging
from . import nn
from . import intrinsic_reward
from .models import RLAlgo, Runner, Experiment, Batch, Run


from . import qlearning
from . import policy_gradient
from . import policy
from . import wrappers


from .qlearning import DQN, RDQN, LinearVDN, RecurrentVDN, VanillaQLearning, ReplayTableQLearning
from .qlearning import DeepQBuilder 

from .utils import seed

register, from_summary = utils.make_registry(RLAlgo, [qlearning, policy_gradient, utils.random_algo])

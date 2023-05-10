__version__ = "0.1.0"

from . import models
from . import logging
from .models import RLAlgo, Runner, Experiment, Batch
from . import nn

from . import qlearning
from . import policy_gradient
from . import policy
from . import wrappers
from . import utils

from .qlearning import DQN, RDQN, LinearVDN, RecurrentVDN, VanillaQLearning, ReplayTableQLearning
from .qlearning import DeepQBuilder 

from .utils.registry import from_summary, register


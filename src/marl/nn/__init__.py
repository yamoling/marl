"""
NN module is where all the neural networks stand.
"""
from .interfaces import NN, LinearNN, RecurrentNN, ActorCriticNN, randomize
from .utils import make_cnn
from .icm_nn import ICM_NN
from . import model_bank

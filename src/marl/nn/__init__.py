"""
NN module is where all the neural networks stand.
"""
from . import loss_functions
from .interfaces import NN, LinearNN, RecurrentNN, ActorCriticNN
from .utils import make_cnn
from .icm_nn import ICM_NN
from . import model_bank


from marl.utils.registry import make_registry

register, from_summary = make_registry(NN, [model_bank])
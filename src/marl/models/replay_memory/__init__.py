from .replay_memory import ReplayMemory, TransitionMemory, EpisodeMemory
from .prioritized_memory import PrioritizedMemory
from .nstep_memory import NStepMemory

# from .builder import MemoryBuilder

from . import replay_memory
from . import prioritized_memory
from . import nstep_memory

from marl.utils.registry import make_registry
register, load = make_registry(ReplayMemory, [replay_memory, prioritized_memory, nstep_memory])

from .replay_memory import ReplayMemory, TransitionMemory, EpisodeMemory
from .prioritized_memory import PrioritizedMemory
from .slice_memory import TransitionSliceMemory
from .nstep_memory import NStepReturnMemory

from .builder import MemoryBuilder


from marl.utils.registry import make_registry
register, from_summary = make_registry(ReplayMemory, [replay_memory, prioritized_memory, slice_memory, nstep_memory])


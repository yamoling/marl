import marl
import lle
from lle import WorldState, LLE
from itertools import permutations


env = LLE.level(1, obs_type=lle.ObservationType.LAYERED_PADDED)

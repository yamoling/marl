from .mixer import Mixer
from .qmix import QMix
from .vdn import VDN


from marl.utils.registry import make_registry

register, load = make_registry(Mixer, [qmix, vdn])
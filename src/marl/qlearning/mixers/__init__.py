from .mixer import Mixer
from .qmix import QMix
from .vdn import VDN


from marl.utils.registry import make_registry

register, from_summary = make_registry(Mixer, [qmix_old, vdn])
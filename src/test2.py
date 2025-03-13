from marlenv.adapters import Overcooked
from lle import LLE


env = Overcooked.from_layout("bottleneck")

obs, state = env.reset()
print(obs.data.shape)

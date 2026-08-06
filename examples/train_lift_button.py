from marl import Experiment, algos
from marl.env import LLEConfig
from marl.nn.model_bank import qnetworks

MAP_PATH = "./lift.toml"
# MAP_PATH = 2


def short_run():
    # `lift.toml` has 3 layers, so `obs_type="layered"` produces a 4D observation
    # shape that qnetworks.from_env doesn't support (only 1D/3D). Use "flattened".
    env = LLEConfig(MAP_PATH, obs_type="flattened")
    trainer = algos.VDN(qnetworks.from_env(env))
    exp = Experiment.create(env, trainer, logdir="test")
    exp.run(test_interval=5_000)


if __name__ == "__main__":
    short_run()

from marl import Experiment, algos
from marl.env import LLEConfig
from marl.nn import mixers
from marl.nn.model_bank import qnetworks


def short_run():
    env = LLEConfig(6, obs_type="layered")
    rnd = algos.RND.from_env(env)
    trainer = algos.DQN(
        qnetworks.from_env(env, independent=True),
        memory_size=50_000,
        mixer=mixers.VDN.from_env(env),
        ir_module=rnd,
    )
    exp = Experiment.create(env, trainer, logdir="auto", n_steps=10_000)
    exp.run(test_interval=1000)


if __name__ == "__main__":
    short_run()

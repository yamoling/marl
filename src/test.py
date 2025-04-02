from marl.training import PPO
from marl.nn import model_bank
from marl import Experiment
from lle import LLE

if __name__ == "__main__":
    env = LLE.level(6).obs_type("flattened").pbrs().builder().agent_id().time_limit(78).build()
    # env = marlenv.make("CartPole-v1")
    # env = LLE.level(3).obs_type("flattened").builder().agent_id().time_limit(78).build()
    ac = model_bank.SimpleActorCritic.from_env(env)
    trainer = PPO(
        ac,
        # value_mixer=mixers.VDN(1),
        gamma=0.99,
        critic_c1=0.5,
        exploration_c2=0.01,
        n_epochs=20,
        minibatch_size=50,
        train_interval=200,
        lr_actor=3e-4,
        lr_critic=3e-4,
    )

    exp = Experiment.create(env, 2_000_000, trainer=trainer, logdir="logs/test2")
    exp.run(render_tests=True, n_tests=10)

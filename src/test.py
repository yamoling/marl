import marlenv
from marl.training import PPOTrainer
from marl.nn import model_bank, mixers
from marl import Experiment
from lle import LLE

if __name__ == "__main__":
    env = marlenv.make("CartPole-v1")
    # env = LLE.level(3).obs_type("flattened").builder().agent_id().time_limit(78).build()
    ac = model_bank.SimpleActorCritic.from_env(env)
    trainer = PPOTrainer(
        ac,
        value_mixer=mixers.VDN(1),
        gamma=0.99,
        gae_lambda=1.0,
        critic_c1=0.5,
        exploration_c2=0.01,
        n_epochs=80,
        minibatch_size=4000,
        train_interval=4000,
        lr_actor=3e-4,
        lr_critic=1e-3,
    )

    exp = Experiment.create(env, 100_000, trainer=trainer)
    exp.run(render_tests=True, n_tests=10)

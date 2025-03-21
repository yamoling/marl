import marlenv
from marl.training import PPOTrainer
from marl.nn import model_bank, mixers
from marl import Experiment

if __name__ == "__main__":
    env = marlenv.make("CartPole-v1")
    ac = model_bank.SimpleActorCritic.from_env(env)  # type: ignore
    trainer = PPOTrainer(
        ac,
        value_mixer=mixers.VDN(1),
        gamma=0.99,
        gae_lambda=0.95,
        critic_c1=0.5,
        exploration_c2=0,
        n_epochs=4,
        minibatch_size=5,
        train_interval=20,
        lr=3e-4,
    )

    exp = Experiment.create(env, 100_000, trainer=trainer)
    exp.run(render_tests=True, n_tests=10)

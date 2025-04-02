import marl
from lle import LLE


def main():
    env = LLE.level(3).builder().agent_id().time_limit(78).build()
    trainer = marl.training.PPO(
        marl.nn.model_bank.actor_critics.CNN_ActorCritic.from_env(env),
        value_mixer=marl.nn.mixers.VDN.from_env(env),
        gamma=0.95,
    )
    experiment = marl.Experiment.create(env, trainer=trainer, n_steps=1_000_000)
    experiment.run()


if __name__ == "__main__":
    main()

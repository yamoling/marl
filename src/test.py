import lle
import marl
from marl.nn.model_bank import options as options_nn
from marl.policy import EpsilonGreedy
from marl.training import OptionCritic


def main():
    env = lle.level(6).obs_type("layered").state_type("state").pbrs().builder().agent_id().time_limit(78).build()

    oc = options_nn.CNNOptionCritic.from_env(env, 4)
    trainer = OptionCritic(
        oc,
        env.n_agents,
        mixer=None,  # marl.nn.mixers.VDN.from_env(env),
        option_train_policy=EpsilonGreedy.linear(1.0, 0.05, 50_000),
    )

    logdir = f"logs/{env.name}-{trainer.name}"
    # logdir = "tests"
    exp = marl.Experiment.create(env, 1_000_000, trainer=trainer, test_interval=5000, logdir=logdir)
    exp.run(seeds=16, n_tests=5, fill_strategy="scatter", disabled_gpus=[1])


if __name__ == "__main__":
    main()

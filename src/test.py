import lle

import marl
from marl.nn import mixers
from marl.nn.model_bank import options as options_nn
from marl.policy import EpsilonGreedy
from marl.training import OptionCritic


def main():
    env = (
        lle.from_file("maps/four_rooms-2.toml")
        .obs_type("layered")
        .state_type("state")
        .builder()
        # .randomize_actions(1 / 5)
        .agent_id()
        .time_limit(1000)
        .build()
    )

    oc = options_nn.CNNOptionCritic.from_env(env, 4)
    trainer = OptionCritic(
        oc,
        env.n_agents,
        mixer=mixers.VDN(env.n_agents),
        option_train_policy=EpsilonGreedy.linear(1.0, 0.05, 50_000),
    )

    logdir = f"logs/{env.name}-{trainer.name}"
    # logdir = "tests"
    exp = marl.Experiment.create(env, 1_000_000, trainer=trainer, test_interval=5000, logdir=logdir)
    exp.run(seeds=16, n_tests=5)


if __name__ == "__main__":
    main()

import lle

import marl
from marl.nn import mixers
from marl.nn.model_bank import options as options_nn
from marl.policy import EpsilonGreedy
from marl.training.option_critic2 import OptionCritic


def main():
    env = (
        lle.from_file("maps/four_rooms_small.toml")
        .obs_type("layered")
        .state_type("state")
        .builder()
        # .randomize_actions(1 / 3)
        .agent_id()
        .time_limit(300)
        .build()
    )

    oc = options_nn.CNNOptionCritic.from_env(env, 4)
    trainer = OptionCritic(
        oc,
        env.n_agents,
        mixer=mixers.VDN(env.n_agents),
        option_train_policy=EpsilonGreedy.linear(1.0, 0.05, 50_000),
    )

    exp = marl.Experiment.create(env, 200_000, trainer=trainer, test_interval=2000)
    exp.run(seeds=10, n_tests=10, n_parallel=1)


if __name__ == "__main__":
    main()

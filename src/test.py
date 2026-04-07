import lle

import marl
from marl.nn import mixers
from marl.nn.model_bank import options as options_nn
from marl.policy import EpsilonGreedy
from marl.training.option_critic2 import OptionCriticTrainer


def main():
    N_OPTIONS = 4
    env = (
        lle.from_file("maps/four_rooms.toml")
        .obs_type("layered")
        .state_type("state")
        .builder()
        # .randomize_actions(1 / 3)
        .agent_id()
        .time_limit(1000)
        .build()
    )
    marl.seed(0, env)

    oc = options_nn.CNNOptionCritic.from_env(env, N_OPTIONS)
    trainer = OptionCriticTrainer(
        oc,
        env.n_agents,
        n_options=N_OPTIONS,
        mixer=mixers.VDN(env.n_agents),
        option_train_policy=EpsilonGreedy.linear(1.0, 0.05, 50_000),
    )

    exp = marl.Experiment.create(env, 200_000, trainer=trainer, test_interval=2000)
    exp.run(seeds=10, n_tests=10, n_parallel=3)


if __name__ == "__main__":
    main()

import lle

import marl
from marl.nn import mixers
from marl.nn.model_bank.options import CNNOptionCritic
from marl.policy import EpsilonGreedy
from marl.training import OptionCritic

WORLD_STR = '''
starts = [{ i_min = 0, j_min = 0 }]
n_agents = 1
world_string = """
@ @ @ @ @ @ @ @ @ @ @ @ @
@ . . . . . @ . . . . . @
@ . . . . . @ . . . . . @
@ . . . . . . . . . . . @
@ . . . . . @ . . . . . @
@ . . . . . @ . . . . . @
@ @ . @ @ @ @ . . . . . @
@ . . . . . @ @ @ X @ @ @
@ . . . . . @ . . . . . @
@ . . . . . @ . . . . . @
@ . . . . . . . . . . . @
@ . . . . . @ . . . . . @
@ @ @ @ @ @ @ @ @ @ @ @ @
"""
'''


def main():
    N_OPTIONS = 4
    env = (
        lle.from_str(WORLD_STR)
        .obs_type("layered")
        .state_type("state")
        .builder()
        .randomize_actions(1 / 3)
        .agent_id()
        .time_limit(1000)
        .build()
    )
    oc = CNNOptionCritic.from_env(env, N_OPTIONS)
    trainer = OptionCritic(
        oc,
        env.n_agents,
        mixer=mixers.VDN(env.n_agents),
        option_train_policy=EpsilonGreedy.linear(1.0, 0.05, 50_000),
    )

    exp = marl.Experiment.create(env, 200_000, trainer=trainer, test_interval=2000)
    exp.run(seeds=10, n_tests=10, n_parallel=3)


if __name__ == "__main__":
    main()

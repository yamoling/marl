from marl.env.wrappers.prevent_actions import PreventActions
from create_experiments import make_dqn
from marl import Experiment
from marlenv.wrappers import TimeLimit


if __name__ == "__main__":
    for width in range(1, 5):
        lle = TimeLimit(PreventActions(width), 10)
        dqn = make_dqn(lle, "vdn")
        Experiment.create(lle, n_steps=5_000, logdir=f"logs/prevent-{width}", trainer=dqn, test_interval=250)

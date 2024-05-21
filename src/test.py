def make_env():
    from lle import LLE, ObservationType

    env = LLE.level(6).obs_type(ObservationType.LAYERED).build()
    source = env.world.laser_sources[4, 0]
    env.world.set_laser_colour(source, 1)
    return env


def main():
    from marl import Experiment

    exp = Experiment.load("logs/shaping-test-lvl6")
    env = make_env()

    # if not exp.env.has_same_inouts(env):
    #    raise ValueError("The environment of the experiment and the test environment must have the same inputs and outputs")
    exp.test_on_other_env(env, "logs/vdn-shaped-test-modified-laser", 1, False, device="auto")


if __name__ == "__main__":
    main()

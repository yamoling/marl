"""
This script allows to test a trained RLAlgo on an other environment.
"""

from typing import Literal
import typed_argparse as tap


class Arguments(tap.TypedArgs):
    logdir: str = tap.arg(positional=True, help="The directory of the experiment with the trained algorithm")
    dest: str = tap.arg(positional=True, help="The directory to save the results of the test")
    env_logdir: str = tap.arg(positional=True, help="The directory of the environment to test the algorithm on")
    n_tests: int = tap.arg(default=5, help="Number of tests to run")
    device: Literal["auto", "cpu"] = tap.arg(default="auto")
    quiet: bool = tap.arg(default=False, help="Run the tests quietly")


def main(args: Arguments):
    from marl import Experiment

    exp = Experiment.load(args.logdir)
    env = Experiment.load(args.env_logdir).test_env

    if not exp.env.has_same_inouts(env):
        raise ValueError("The environment of the experiment and the test environment must have the same inputs and outputs")
    exp.test_on_other_env(env, args.dest, args.n_tests, args.quiet, device=args.device)


if __name__ == "__main__":
    tap.Parser(Arguments).bind(main).run()

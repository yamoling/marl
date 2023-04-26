import marl
import os
from argparse import ArgumentParser, Namespace

def set_run_arguments(parser: ArgumentParser):
    parser.add_argument("--logdir", type=str, help="The experiment directory", required=True)
    parser.add_argument("--seed", type=int, help="Random seed (torch, numpy, random)")
    parser.add_argument("--n_tests", type=int, default=5)
    parser.add_argument("--quiet", action="store_true", default=False)
    parser.add_argument("--loggers", type=str, choices=["csv", "tensorboard", "web"], default=["csv", "web"], nargs="*")


def set_experiment_arguments(parser: ArgumentParser):
    pass

def set_new_arguments(parser: ArgumentParser):
    p = parser.add_subparsers(title="New", required=True, dest="created_object")
    new_run_parser = p.add_parser("run", help="Create a new run")
    set_run_arguments(new_run_parser)
    new_experiment_parser = p.add_parser("experiment", help="Create a new experiment")

def set_resume_arguments(parser: ArgumentParser):
    parser.add_argument("rundirs", nargs='+', type=str, help="The run directories to resume")
    parser.add_argument("--n_tests", type=int, default=5)

def parse_args():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(title="RLEnv main entry point", required=True, dest="command")
    new_parser = subparsers.add_parser("new", help="Create a new run or experiment")
    set_new_arguments(new_parser)
    resume_parser = subparsers.add_parser("resume", help="Resume an existing run")
    set_resume_arguments(resume_parser)
    load_parser = subparsers.add_parser("load", help="Load an existing run or experiment")
    return parser.parse_args()

def new(args: Namespace):
    match args.created_object:
        case "run":
            experiment = marl.Experiment.load(args.logdir)
            runner = experiment.create_runner(*args.loggers, seed=args.seed, quiet=args.quiet)
            runner.train(n_tests=args.n_tests)
        case "experiment":
            raise NotImplementedError("Not implemented yet")

def resume(args: Namespace):
    def _resume_run(rundir: str, args: Namespace):
        logdir = os.path.dirname(rundir)
        experiment = marl.Experiment.load(logdir)
        runner = experiment.restore_runner(rundir)
        runner.train(n_tests=args.n_tests)
    
    for rundir in args.rundirs[:-1]:
        pid = os.fork()
        # Parent process: continue the loop
        if pid > 0:
            continue
        _resume_run(rundir, args)
        exit(0)
    rundir = args.rundirs[-1]
    _resume_run(rundir, args)

if __name__ == "__main__":
    args = parse_args()
    print(args)
    
    match args.command:
        case "new": new(args)
        case "resume": resume(args)
        case "load": print("TODO")
        case other: raise NotImplementedError("Command not implemented: " + other)
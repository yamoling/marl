import marl
import os
from argparse import ArgumentParser, Namespace
import argcomplete
from laser_env import StaticLaserEnv, ObservationType
import rlenv


class StatefulStaticLaserEnv(StaticLaserEnv):

    def __init__(self, env_file: str, obs_type: str | ObservationType = ObservationType.RELATIVE_POSITIONS):
        super().__init__(env_file, obs_type)
        self.state_observer = ObservationType.FLATTENED.get_observation_generator(self.world)
    
    def get_state(self):
        return self.state_observer.observe()[0]
    
    @property
    def state_shape(self):
        return self.state_observer.shape

rlenv.register(StatefulStaticLaserEnv)


def set_run_arguments(parser: ArgumentParser):
    parser.add_argument("logdir", type=str, help="The experiment directory")
    parser.add_argument("--n_runs", type=int, default=1, help="Number of runs to create")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (torch, numpy, random)")
    parser.add_argument("--n_tests", type=int, default=5)
    parser.add_argument("--quiet", action="store_true", default=False)
    parser.add_argument("--loggers", type=str, choices=["csv", "tensorboard", "web"], default=["csv", "web"], nargs="*")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda", "cuda:0", "cuda:1", "cuda:2"], default="auto")


def set_experiment_arguments(parser: ArgumentParser):
    pass

def set_new_arguments(parser: ArgumentParser):
    p = parser.add_subparsers(title="New", required=True, dest="created_object")
    new_run_parser = p.add_parser("run", help="Create a new run")
    set_run_arguments(new_run_parser)
    new_experiment_parser = p.add_parser("experiment", help="Create a new experiment")
    set_experiment_arguments(new_experiment_parser)

def set_resume_arguments(parser: ArgumentParser):
    parser.add_argument("rundirs", nargs='+', type=str, help="The run directories to resume")
    parser.add_argument("--n_tests", type=int, default=5)

def set_serve_arguments(parser: ArgumentParser):
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true", default=False)

def parse_args():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(title="RLEnv main entry point", required=True, dest="command")
    new_parser = subparsers.add_parser("new", help="Create a new run or experiment")
    set_new_arguments(new_parser)
    resume_parser = subparsers.add_parser("resume", help="Resume an existing run")
    set_resume_arguments(resume_parser)
    serve_parser = subparsers.add_parser("serve", help="Serve the web interface")
    set_serve_arguments(serve_parser)
    argcomplete.autocomplete(parser)
    return parser.parse_args()

def new(args: Namespace):
    match args.created_object:
        case "run":
            seed = args.seed
            for i in range(args.n_runs - 1):
                seed = args.seed + i
                if os.fork() == 0:
                    # Force child processes to be quiet
                    args.quiet = True
                    create_run(args, seed)
                    exit(0)
            create_run(args, seed)
        case "experiment":
            raise NotImplementedError("Not implemented yet")
        
def serve(args: Namespace):
    from ui.backend import run
    from marl.utils.env_pool import EnvPool
    import rlenv
    rlenv.register_wrapper(EnvPool)
    run(port=args.port, debug=args.debug)

def create_run(args: Namespace, seed):
    experiment = marl.Experiment.load(args.logdir)
    runner = experiment.create_runner(*args.loggers, seed=seed, quiet=args.quiet)
    runner.to(args.device)
    runner.train(n_tests=args.n_tests)

def resume(args: Namespace):
    def _resume_run(rundir: str, args: Namespace, quiet=True):
        if rundir.endswith("/"):
            rundir = rundir[:-1]
        logdir = os.path.dirname(rundir)
        experiment = marl.Experiment.load(logdir)
        runner = experiment.restore_runner(rundir)
        runner.train(n_tests=args.n_tests, quiet=quiet)
    
    for rundir in args.rundirs[:-1]:
        pid = os.fork()
        # Child process: resume the run
        if pid == 0:
            _resume_run(rundir, args)
            exit(0)
    rundir = args.rundirs[-1]
    _resume_run(rundir, args, quiet=False)

if __name__ == "__main__":
    args = parse_args()
    
    match args.command:
        case "new": new(args)
        case "resume": resume(args)
        case "load": print("TODO")
        case "serve": serve(args)
        case other: raise NotImplementedError("Command not implemented: " + other)
import os
from argparse import ArgumentParser, Namespace

import marl


def set_run_arguments(parser: ArgumentParser):
    parser.add_argument("logdir", type=str, help="The experiment directory")
    parser.add_argument("--n_runs", type=int, default=1, help="Number of runs to create")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_tests", type=int, default=5)
    parser.add_argument("--quiet", action="store_true", default=False)
    parser.add_argument("--loggers", type=str, choices=["csv", "tensorboard", "web"], default=["csv"], nargs="*")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda", "cuda:0", "cuda:1", "cuda:2"], default="auto")


def set_experiment_arguments(parser: ArgumentParser):
    pass


def set_new_arguments(parser: ArgumentParser):
    p = parser.add_subparsers(title="New", required=True, dest="created_object")
    new_run_parser = p.add_parser("run", help="Create a new run")
    set_run_arguments(new_run_parser)
    new_experiment_parser = p.add_parser("experiment", help="Create a new experiment")
    set_experiment_arguments(new_experiment_parser)


def set_serve_arguments(parser: ArgumentParser):
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true", default=False)


def parse_args():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(title="RLEnv main entry point", required=True, dest="command")
    new_parser = subparsers.add_parser("new", help="Create a new run or experiment")
    set_new_arguments(new_parser)
    serve_parser = subparsers.add_parser("serve", help="Serve the web interface")
    set_serve_arguments(serve_parser)
    return parser.parse_args()


def new(args: Namespace):
    match args.created_object:
        case "run":
            # Load the experiment from disk and start a child process for each run.
            # The run with seed=0 is spawned in the main process.
            for i in range(1, args.n_runs):
                seed = args.seed + i
                if os.fork() == 0:
                    import time

                    # Sleep for some time for each child process to allow GPUs to be allocated properly
                    time.sleep(i)
                    # Force child processes to be quiet
                    args.quiet = True
                    create_run(args, seed)
                    exit(0)
            create_run(args, args.seed)
        case "experiment":
            raise NotImplementedError("Not implemented yet")


def serve(args: Namespace):
    from ui.backend import run

    run(port=args.port, debug=args.debug)


def create_run(args: Namespace, seed: int):
    experiment = marl.Experiment.load(args.logdir)
    runner = experiment.create_runner(seed=seed)
    runner.to(args.device)
    runner.train(n_tests=args.n_tests)


if __name__ == "__main__":
    args = parse_args()
    match args.command:
        case "new":
            new(args)
        case "load":
            print("TODO")
        case "serve":
            serve(args)
        case other:
            raise NotImplementedError("Command not implemented: " + other)

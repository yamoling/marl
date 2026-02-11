import os
import typed_argparse as tap
from marl import exceptions
from marl import Experiment, Run


class ListArguments(tap.TypedArgs):
    logdir: str = tap.arg(default="logs", positional=False, help="The experiment directory")


class ShowArguments(tap.TypedArgs):
    logdir: str = tap.arg(positional=True, help="The experiment directory")


class KillArguments(tap.TypedArgs):
    logdir: str = tap.arg(positional=True, help="The experiment directory, or 'all' for all experiments")
    no_confirm: bool = tap.arg(positional=False, help="Do not ask for confirmation before killing runs", default=False)

    @property
    def experiments(self):

        if self.logdir == "all":
            return _list("logs")
        yield Experiment.load(self.logdir)


def print_status(exp: Experiment):
    runs = list(exp.runs)
    print(f"Experiment {exp.logdir} has {len(runs)} runs")
    if len(runs) == 0:
        print("No runs in experiment")
        return 0
    max_steps = exp.n_steps
    actives = 0
    for run in runs:
        pid = run.get_pid()
        if pid is not None:
            progress = run.get_progress(max_steps)
            print(f"\t[{progress * 100:6.2f} %] Run {run.rundir} is active with pid {pid}")
            actives += 1
    print(f"{actives}/{len(runs)} active runs")


def interrupt_runs(experiment: Experiment):
    exp: Experiment = experiment

    runs_to_cleanup = list[Run]()
    for run in exp.runs:
        try:
            run.kill()
        except exceptions.NotRunningExcception:
            pass
        except exceptions.RunProcessNotFound as e:
            print(f"PID {e.pid} of {e.rundir} not found")
            runs_to_cleanup.append(run)
    if len(runs_to_cleanup) > 0:
        resp = input(f"Cleanup {len(runs_to_cleanup)} run pid files ? [y/n] ")
        if resp.lower() == "y":
            for run in runs_to_cleanup:
                os.remove(run.pid_filename)
            print("Deleted.")
        else:
            print("Not deleted.")
    print("Done.")


def _list(root: str):
    for directory in os.listdir(root):
        directory = os.path.join(root, directory)
        try:
            yield Experiment.load(directory)
        except FileNotFoundError:
            continue
        except Exception:
            print(f"Unexpected error loading {directory}")


def list_active_runs(args: ListArguments):
    root = args.logdir
    for exp in _list(root):
        if exp.is_running:
            print_status(exp)


def kill_runs(args: KillArguments):
    for exp in args.experiments:
        print_status(exp)
        n_active = exp.n_active_runs()
        if n_active == 0:
            continue
        if not args.no_confirm:
            answer = input(f"Kill the {n_active} active runs ? [y/n] ")
            if answer.lower() != "y":
                print("Exiting without killing processes.")
                continue
        interrupt_runs(exp)


def show_status(args: ShowArguments):
    import marl

    exp = marl.Experiment.load(args.logdir)
    print_status(exp)


if __name__ == "__main__":
    tap.Parser(
        tap.SubParserGroup(
            tap.SubParser(
                "list",
                ListArguments,
                help="List all active runs",
            ),
            tap.SubParser(
                "kill",
                KillArguments,
                help="Send a SIGINT to all active runs",
            ),
            tap.SubParser(
                "show",
                ShowArguments,
                help="Show the status of a specific experiment",
            ),
        ),
    ).bind(
        list_active_runs,
        kill_runs,
        show_status,
    ).run()

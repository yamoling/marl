import os
import typed_argparse as tap


class ListArguments(tap.TypedArgs):
    logdir: str = tap.arg(default="logs", positional=False, help="The experiment directory")


class ShowArguments(tap.TypedArgs):
    logdir: str = tap.arg(positional=True, help="The experiment directory")


class KillArguments(tap.TypedArgs):
    logdir: str = tap.arg(positional=True, help="The experiment directory")


class Arguments(tap.TypedArgs):
    logdir: str = tap.arg(positional=True, help="The experiment directory")
    kill: bool = tap.arg(default=False, help="Whether to send a SIGINT to all active runs")


def print_status(experiment):
    from marl import Experiment

    exp: Experiment = experiment
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


def interrupt_runs(experiment):
    from marl.utils import exceptions
    from marl import Experiment, Run

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


def list_active_runs(args: ListArguments):
    import marl

    root = args.logdir
    for directory in os.listdir(root):
        directory = os.path.join(root, directory)
        try:
            exp = marl.Experiment.load(directory)
            if exp.is_running:
                print_status(exp)
        except FileNotFoundError:
            continue
        except Exception:
            print(f"Unexpected error loading {directory}")


def kill_runs(args: KillArguments):
    import marl

    exp = marl.Experiment.load(args.logdir)
    print_status(exp)
    n_active = exp.n_active_runs()
    if n_active == 0:
        print("No active runs to kill.")
        return

    answer = input(f"Kill the {n_active} active runs ? [y/n] ")
    if answer.lower() != "y":
        print("Exiting without killing processes.")
        return
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

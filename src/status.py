import typed_argparse as tap
import os
import signal
import marl


class ListArguments(tap.TypedArgs):
    logdir: str = tap.arg(default="logs", positional=False, help="The experiment directory")


class StatusArguments(tap.TypedArgs):
    logdir: str = tap.arg(positional=True, help="The experiment directory")


class KillArguments(tap.TypedArgs):
    logdir: str = tap.arg(positional=True, help="The experiment directory")


class Arguments(tap.TypedArgs):
    logdir: str = tap.arg(positional=True, help="The experiment directory")
    kill: bool = tap.arg(default=False, help="Whether to send a SIGINT to all active runs")


def print_status(experiment: marl.Experiment):
    if len(experiment.runs) == 0:
        print("No runs in experiment")
        return 0
    max_steps = experiment.n_steps
    actives = 0
    for run in experiment.runs:
        pid = run.get_pid()
        if pid is not None:
            progress = run.get_progress(max_steps)
            print(f"\t[{progress * 100:6.2f} %] Run {run.rundir} is active with pid {pid}")
            actives += 1
        else:
            print(f"Run {run.rundir} is inactive")
    print(f"{actives}/{len(experiment.runs)} active runs")


def interrupt_runs(exp: marl.Experiment):
    for run in exp.runs:
        pid = run.get_pid()
        if pid is not None:
            print(f"Sending SIGINT ({signal.SIGINT}) to {run.rundir} with pid {pid}")
            os.kill(pid, signal.SIGINT)
    print("Done.")


def list_active_runs(args: ListArguments):
    pass


def kill_runs(args: KillArguments):
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


def main(args: Arguments):
    # Only import here to make the script faster to load
    import marl

    def status(experiment: marl.Experiment) -> int:
        if len(experiment.runs) == 0:
            print("No runs in experiment")
            return 0
        max_steps = experiment.n_steps
        actives = 0
        for run in experiment.runs:
            pid = run.get_pid()
            if pid is not None:
                progress = run.get_progress(max_steps)
                print(f"\t[{progress * 100:6.2f} %] Run {run.rundir} is active with pid {pid}")
                actives += 1
            else:
                print(f"Run {run.rundir} is inactive")
        print(f"{actives}/{len(experiment.runs)} active runs")
        return actives

    def interrupt_runs(exp: marl.Experiment):
        for run in exp.runs:
            pid = run.get_pid()
            if pid is not None:
                print(f"Sending SIGINT ({signal.SIGINT}) to {run.rundir} with pid {pid}")
                os.kill(pid, signal.SIGINT)
        print("Done.")

    exp = marl.Experiment.load(args.logdir)
    n_active = status(exp)
    if not args.kill:
        return
    if n_active == 0:
        print("No active runs to kill.")
        return

    answer = input(f"Kill the {n_active} active runs ? [y/n] ")
    if answer.lower() != "y":
        print("Exiting without killing processes.")
        return
    interrupt_runs(exp)


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
                "status",
                StatusArguments,
                help="Give the status of a specific experiment",
            ),
        ),
    ).bind(
        run_foo_start,
        run_foo_stop,
        run_bar,
    ).run()
    tap.Parser(Arguments).bind(main).run()

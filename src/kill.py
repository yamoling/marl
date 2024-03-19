import typed_argparse as tap

import os
import signal
import marl


class Arguments(tap.TypedArgs):
    logdir: str = tap.arg(positional=True, help="The experiment directory")


def main(args: Arguments):
    exp = marl.Experiment.load(args.logdir)
    runs = exp.runs
    pids = [run.get_pid() for run in runs if run.get_pid() is not None]
    if len(pids) == 0:
        print("No active run.\nExiting.")
        return
    answer = input(f"Kill {len(pids)}/{len(runs)} active runs with pids {pids} ? [y/n] ")
    if answer.lower() != "y":
        print("Exiting without killing processes.")
        return
    for run in runs:
        pid = run.get_pid()
        if pid is not None:
            print(f"Sending SIGINT ({signal.SIGINT}) to {run.rundir} with pid {pid}")
            os.kill(pid, signal.SIGINT)
    print("Done.")


if __name__ == "__main__":
    tap.Parser(Arguments).bind(main).run()

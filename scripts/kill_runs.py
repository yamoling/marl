"""Kill all active runs belonging to an experiment."""

import argparse

import marl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_directory", help="Directory containing experiment.json")
    args = parser.parse_args()

    killed_pids, parent_ppid = marl.Experiment.load(args.experiment_directory).kill_runs()
    print(f"Killed {len(killed_pids)} process(es): {killed_pids or 'none'}.")
    if parent_ppid is None:
        print("Parent pool process: none.")
    else:
        print(f"Parent pool process: {parent_ppid}.")


if __name__ == "__main__":
    main()

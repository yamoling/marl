#! /usr/bin/python3
import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator


def find_event_file(log_dir) -> str:
    files = os.listdir(log_dir)
    candidates = [f for f in files if f.startswith("events.out.tfevents")]
    if len(candidates) > 1:
        raise ValueError("There is more than one event file in this directory ! Cannot tell which one to use !")
    if len(candidates) == 0:
        raise ValueError("The given directory does not contain a tensorboard events file.")
    return os.path.join(log_dir, candidates[0])


def parse_tensorboard(log_dirs: list[str], tags: list[str]=None) -> dict[str, pd.DataFrame]:
    """
    Parse tensorboard as a DataFrame. Each tag had its own dataframe.
    
    Each dataframe has one column by input log_dir.
    """
    dfs = {}
    for ld in log_dirs:
        print(f"Parsing {ld}")
        column_name = os.path.basename(ld)
        event_file = find_event_file(ld)
        ea = event_accumulator.EventAccumulator(event_file).Reload()
        for tag in ea.Tags()["scalars"]:
            # Only take test tags and tags that were asked by the user
            if tag.startswith("Test") and ((tags is None) or any([t in tag for t in tags])):
                if tag not in dfs:
                    index = [item.step for item in ea.Scalars(tag)]
                    dfs[tag] = pd.DataFrame(index=index)
                df = dfs[tag]
                df[column_name] = [item.value for item in ea.Scalars(tag)]
    return dfs


def plot_mean_std(df: pd.DataFrame, metric: str, output_dir: str|None=None, colour: str|None=None, clear_previous=True, label=None):
    if output_dir is None:
        output_dir=""
    metric = metric.replace("_", " ")
    desc = df.T.describe()
    mean = desc.loc["mean"]
    std = desc.loc["std"]
    if clear_previous:
        plt.clf()
    plt.xlabel("Episodes")
    plt.ylabel(metric)
    plt.plot(mean, label=label)
    plt.legend()
    plt.fill_between(mean.index, mean + std, mean - std, alpha=0.25)
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{metric.replace('/', '_')}.png")
    plt.savefig(filename)



def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--input_dirs", nargs="+")
    args.add_argument("--output_dir", type=str)
    args.add_argument("--tags", nargs="+")
    args.add_argument("--colour", type=str)
    return args.parse_args()



def main(args):
    print("Reading log files", end="")
    for d in args.input_dirs:
        print(f" {d}", end="")
    print("...")
    dfs = parse_tensorboard(args.input_dirs, args.tags)
    print("Plotting metrics")
    for metric, df in dfs.items():
        print(f"\t{metric}...")
        metric = metric.replace("/", "_")
        plot_mean_std(df, metric, args.output_dir, args.colour)

if __name__ == "__main__":
    arguments = parse_args()
    main(arguments)
    print("Done")

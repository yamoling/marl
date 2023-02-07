#! /usr/bin/python3
import pandas as pd
from analysis.tensorboard_mean_std import parse_tensorboard, plot_mean_std



def aggregate(logs: dict[str, list[str]], output_dir: str, metrics: list[str]=None):
    """Plot multiple results on the same chart."""
    if metrics is None:
        metrics = ["Test/avg_score", "Test/max_score", "Test/min_score", "Test/avg_episode_length"]
    dfs: list[tuple[str, dict[str, pd.DataFrame]]] = []
    labels = []
    for label, dirs in logs.items():
        labels.append(label)
        dfs.append((label, parse_tensorboard(dirs, metrics)))
    for metric in metrics:
        print(f"\t{metric}...")
        for i, (label, df) in enumerate(dfs):
            plot_mean_std(df[metric], metric, output_dir, clear_previous=(i==0), label=label)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--uniform", nargs="+")
    # parser.add_argument("--per", nargs="+")
    # args = parser.parse_args()
    logs = {'3-step return': ['logs/1675764590.3751836'], 'plain': ['logs/1675764590.4472265']}
    aggregate(logs, "aggregated", ["Test/avg_score", "Test/max_score"])

import marl
import matplotlib.pyplot as plt
import polars as pl

plt.rcParams.update(
    {
        "text.usetex": False,  # Set to True for more better rendering if you have LaTeX installed
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.family": "serif",
    }
)


def plot_manually():
    exp = marl.Experiment.load("logs/LLE-four_rooms_small-1.toml-OC")
    # 1) Gather the test metrics of each run
    run_dfs = [run.test_metrics for run in exp.runs]
    # 2) Average the performance **per run** every 5000 time steps
    run_dfs = [df.group_by("time_step").agg(pl.mean("*")) for df in run_dfs]
    # 3) Get the mean and confidence interval of the performance across runs
    df = pl.concat(run_dfs)
    columns = [col for col in df.columns if col not in ("time_step", "timestamp_sec")]
    print(df)
    df = (
        df.group_by("time_step")
        .agg(
            [pl.mean(col).alias(f"mean-{col}") for col in columns] + [pl.std(col).alias(f"std-{col}") for col in columns],
        )
        .sort("time_step")
    )
    # 4) Plot the results with 95% confidence intervals across runs
    n_runs = len(list(exp.runs))
    for col in columns:
        plt.plot(df["time_step"], df[f"mean-{col}"], label=col)
        ci95 = 1.96 * df[f"std-{col}"] / (n_runs**0.5)
        plt.fill_between(
            df["time_step"],
            df[f"mean-{col}"] - ci95,
            df[f"mean-{col}"] + ci95,
            alpha=0.2,
        )
        plt.xlabel("Time step")
        plt.ylabel(col)
        plt.show()


def plot_with_datasets():
    exp = marl.Experiment.load("logs/LLE-four_rooms_small-1.toml-OC")
    datasets = exp.get_experiment_results()
    metrics = set(dataset.label for dataset in datasets)
    for metric in metrics:
        for dataset in datasets:
            if dataset.label == metric:
                plt.plot(dataset.ticks, dataset.mean, label=f"{dataset.label} ({dataset.category})")
                plt.fill_between(
                    dataset.ticks,
                    dataset.mean - dataset.ci95,
                    dataset.mean + dataset.ci95,
                    alpha=0.2,
                )
        plt.xlabel("Time step")
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.show()


def main():
    # plot_manually()
    plot_with_datasets()


if __name__ == "__main__":
    main()

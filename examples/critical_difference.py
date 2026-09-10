import subprocess

import polars as pl
from critdd import Diagram

import marl


def retrieve_data(logdirs: list[str], time_steps: list[int]):
    experiments = [marl.Experiment.load(logdir) for logdir in logdirs]
    data = list[pl.DataFrame]()
    for exp in experiments:
        try:
            run_results = [
                run.test_metrics.with_columns(seed=pl.lit(run.seed))
                .filter(pl.col("time_step").is_in(time_steps))
                .group_by("time_step")
                .mean()
                for run in exp.runs
            ]
            raw_results = pl.concat(run_results).with_columns(logdir=pl.lit(exp.logdir))
            data.append(raw_results.select("seed", "exit_rate", "logdir", "time_step").collect())
        except pl.exceptions.NoDataError:
            print(f"No data for {exp.logdir}")
    return pl.concat(data)


def main(time_steps: list[int], logdirs: list[str], do_compile: bool = False):
    results = retrieve_data(logdirs=logdirs, time_steps=time_steps)
    df = results.pivot("logdir", index=("seed", "time_step"), values="exit_rate").drop("seed")

    for step in time_steps:
        output_file = f"plots/statistical-{step}.tex"
        sub_df = df.filter(time_step=step).drop("time_step")
        print(sub_df)
        diagram = Diagram(sub_df.to_numpy(), treatment_names=sub_df.columns, maximize_outcome=True)
        diagram.to_file(
            output_file,
            alpha=0.05,
            adjustment="holm",
            reverse_x=True,
            # ticklabel style={anchor=south, yshift=1.3*\pgfkeysvalueof{/pgfplots/major tick length}, font=\small},
            axis_options={
                "title": f"Mean exit rate at time step {step}",
                "ticklabel style": r"anchor=south, yshift=1*\pgfkeysvalueof{/pgfplots/major tick length}, font=\large",
                "title style": r"yshift=0.7\baselineskip, font=\Large",
            },
            tikzpicture_options={
                "treatment label/.style": r"font=\Large",
            },
            as_document=do_compile,
        )
        print(f"Created {output_file}")
        if do_compile:
            # Compile the latex
            subprocess.run(["pdflatex", output_file], check=True)


if __name__ == "__main__":
    import os

    logdirs = [f"logs/{d}" for d in os.listdir("logs") if d.startswith("VDN")]
    time_steps = [100_000, 400_000, 700_000, 1_000_000]
    main(time_steps, logdirs, do_compile=True)

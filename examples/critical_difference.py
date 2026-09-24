import subprocess
from pathlib import Path

import polars as pl
from critdd import Diagram

import marl


def retrieve_data(logdirs: list[str], time_steps: list[int]) -> pl.DataFrame:
    """Average test episodes within each run and requested step. @ai-edited"""
    data = []
    for logdir in logdirs:
        exp = marl.Experiment.load(logdir)
        for run in exp.runs:
            try:
                result = (
                    run.test_metrics.filter(pl.col("time_step").is_in(time_steps))
                    .group_by("time_step")
                    .agg(pl.col("exit_rate").mean())
                    .with_columns(seed=pl.lit(run.seed), logdir=pl.lit(str(exp.logdir)))
                    .select("seed", "exit_rate", "logdir", "time_step")
                    .collect()
                )
            except (pl.exceptions.NoDataError, pl.exceptions.ColumnNotFoundError):
                print(f"No test exit_rate data for {exp.logdir}, seed {run.seed}")
                continue
            if not result.is_empty():
                data.append(result)
    if not data:
        raise ValueError("No test exit_rate data found at the requested time steps")
    return pl.concat(data)


def main(time_steps: list[int], logdirs: list[str], do_compile: bool = False):
    """Generate one diagram per step using seeds shared by all treatments. @ai-edited"""
    results = retrieve_data(logdirs=logdirs, time_steps=time_steps)
    output_dir = Path("plots")
    for step in time_steps:
        step_results = results.filter(pl.col("time_step") == step)
        if step_results.is_empty():
            raise ValueError(f"No test exit_rate data at time step {step}")
        output_file = output_dir / f"statistical-{step}.tex"
        sub_df = step_results.pivot("logdir", index="seed", values="exit_rate").sort("seed").drop("seed").drop_nulls()
        if sub_df.width < 3 or sub_df.height < 2:
            raise ValueError(f"Need at least three treatments and two shared seeds at time step {step}")
        output_dir.mkdir(exist_ok=True)
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
            subprocess.run(["pdflatex", "-output-directory", str(output_dir), str(output_file)], check=True)


if __name__ == "__main__":
    logdirs = [str(file.parent) for file in Path("logs").glob("VDN*/experiment.json")]
    time_steps = [100_000, 400_000, 700_000, 1_000_000]
    main(time_steps, logdirs, do_compile=True)

---
name: rl-experiment-report
description: Generate a concise Markdown report for MARL experiment results using Experiment.get_results_datasets and related experiment metadata.
---

# RL Experiment Report

Use this skill when you need to turn a completed MARL experiment into a Markdown report for human review.

## Goal
Produce a report that is grounded in the experiment data and structured around:
- A short description of why the experiment was run
- The parameters and methodology
- The results
- A discussion with conclusions 
- Ideas of what to do next

## Workflow
1. Load the experiment from its log directory with `Experiment.load(logdir)`.
2. Inspect the experiment metadata needed for the report, including the environment, trainer, `n_steps`, logdir, test environment, and the run structure.
3. Use `Experiment.get_results(granularity=5000)` to extract the aggregated metrics for reporting as a `polars.LazyFrame`, or `Experiment.get_results_datasets(granularity=5000)` to retrieve the results as a list of `Dataset` objects.
4. If you need raw aggregated tables for comparison or a small summary, use `Experiment.get_results(granularity, aggregate_by)` or the narrower helpers such as `get_test_results()`.
5. Summarize the runs by seed, test interval, number of tests, aggregation mode, and any other parameters that materially affect interpretation.
6. Write the report in Markdown with clear section headings and concise, factual language.

## What To Include
The report should be concise.

### 1. Why this experiment was run
State the experimental question, hypothesis, or motivation. If the experiment metadata does not state it explicitly, infer only from the trainer, environment, and run configuration, and clearly label the statement as an inference.

### 2. Parameters and methodology
Include the most relevant setup details:
- Environment and test environment
- Trainer / algorithm name
- Number of steps
- Seeds or number of runs
- Number of tests per run
- Test interval
- Aggregation settings such as `granularity` and `aggregate_by`
- Any metric filters passed to `get_results_datasets`

Describe the methodology briefly, focusing on how results were aggregated rather than on implementation internals.

### 3. Results
Use the values returned by `get_results_datasets` as the main source of truth.
- Highlight the most important metrics. In the Laser Learning Environment, it is the `exit_rate`, then the `score`. You can discuss `training data` metrics such as loss, intrinsic rewards, etc. if relevant.
- Compare train, test, and training-data signals when relevant.
- Mention stability across runs when multiple seeds were used.
- Call out missing metrics or empty categories instead of inventing values.
- Prefer compact tables, bullet summaries, or short paragraphs over long prose.
- Generate relevant plots with matplotlib. Scripts to generate plots should be located in the `src/plots` folder. Plots should be carefully designed.

### 4. Discussion and next steps
Interpret the results and end with practical conclusions.
- Say whether the experiment appears successful, inconclusive, or negative.
- Note likely causes if performance is unstable or underwhelming.
- Propose the next action: another hyperparameter sweep, environment change, more seeds, different aggregation, or further debugging.

## Output Requirements
- Write the final report in Markdown.
- Use headings such as `## Summary`, `## Parameters`, `## Results`, and `## Discussion`.
- Keep the report factual and traceable to experiment data.
- Do not fabricate values that are not present in the experiment outputs.
- If metadata is incomplete or corrupted, provide a summary of what is missing and suggest steps to recover or validate the data

## Completion Check
Before finishing, verify that the report:
- Explains why the experiment ran
- Describes the setup and methodology
- Summarizes the results from the experiment outputs
- Ends with conclusions and next steps
- Uses Markdown formatting suitable for sharing with researchers

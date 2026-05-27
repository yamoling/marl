---
name: rl-experiment-runner
description: Run a MARL experiment from provided parameters, selecting GPUs with nvidia-smi and capping GPU scatter to at most four devices.
---

# RL Experiment Runner

Use this skill when you need to launch a MARL experiment from given parameters and make a sensible device choice before training. Unless specified otherwise, you should run the experiments on at least 8 random seeds for statistical significance.

## Goal
Run the experiment safely and reproducibly, then verify that the training job started with the intended configuration.

## Workflow
1. Identify the experiment inputs that are already known: environment, trainer, number of steps, seeds or run count, test interval, number of tests, save options, and any requested device constraints.
2. Decide how the experiment should be named.
   - Use `logdir="auto"` when the run is exploratory, there is no existing naming convention to preserve, or the trainer/environment pair should define the directory automatically.
   - Use a specific logdir name when the user wants a stable path, is comparing sweeps, or needs a human-chosen experiment label.
3. Inspect the available GPUs with `nvidia-smi` before launching the job.
   - Prefer GPUs with the most free memory and the least active workload.
   - Ignore GPUs that are clearly unavailable, heavily occupied, or intentionally disabled.
   - Never scatter the experiment across more than 4 GPUs, even if more are available.
4. Convert the GPU inspection into a concrete launch choice.
   - If no suitable GPU is available, do not run the experiment at all.
   - If only one GPU is suitable, prefer a grouped or sequential run.
   - If multiple GPUs are suitable, use `scatter` only when it will help and only across at most 4 GPUs.
   - Keep `disabled_gpus` consistent with the GPUs that should not be used.
5. Create a python file for the experiment with a unique name in the `src/experiments` folder with a relevant name. Launch the experiment through the project’s training entry point or by calling `Experiment.run(...)` with the chosen parameters.
6. Verify that the run started correctly.
   - Confirm the experiment directory exists.
   - Confirm the expected number of runs was created.
   - Confirm the selected device strategy matches the intended GPU layout.

## Parameter Handling
- Use a CSV logger

## GPU Selection Rules
- Always inspect GPUs first with `nvidia-smi`.
- Use the freshest available signal from memory and process load to choose devices.
- Use at most 4 GPUs for scattering.
- If the system is borderline or the available GPU set is ambiguous, choose fewer GPUs rather than more.
- Do not assume that the highest-numbered GPU is the best choice.

## Output Requirements
- Record the final launch decision in concise Markdown or plain text.
- State the chosen logdir, seed/run count, test settings, and device choice.
- Note which GPUs were selected or disabled.
- If the experiment could not be launched, explain the blocker and what additional information is needed.

## Completion Check
Before finishing, verify that:
- The experiment parameters match the user’s request.
- GPU selection was based on `nvidia-smi`.
- No more than 4 GPUs were used for scattering.
- The experiment name was chosen appropriately, either `auto` or a specific label.
- The run started and the expected output location is known.
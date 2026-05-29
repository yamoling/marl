# Experiment Runner Guide

Purpose
- Describe how to run experiments reproducibly with the repository's runner scripts, recommended flags, GPU selection, outputs, and best practices.

Quickstart
- Basic run (example):

```python
env = LLEConfig(6, obs_type="flattened")
trainer = algos.QMix(qnetworks.from_env(env), memory_size=50_000, mixer=mixers.QMix.from_env(env))
exp = Experiment(env, trainer, logdir="auto", n_steps=5_000)
exp.run(test_interval=500)
```


GPU selection patterns
- Manual:
```python
exp.run(seeds=10, test_interval=500, device="cuda:0")
# Or
exp.run(seeds=10, test_interval=500, device=0)
```

- Automatic:
```python
# Groups runs on the first available GPU, defaults to N_GPUs parallel jobs.
exp.run(seeds=10, test_interval=500, device="auto")
# Use all available GPUs, grouping as many as possible on the same device
exp.run(seeds=10, test_interval=500, device="auto", gpu_strategy="scatter")
# Scatter across gpus 0-3
exp.run(seeds=10, test_interval=500, device="auto", disabled_gpus=range(4, 8), gpu_strategy="scatter")
```


A run can also be started with the `start_run.py` script, which can start new runs from an experiment that has already been created.
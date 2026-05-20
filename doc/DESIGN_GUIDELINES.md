# Design Guidelines and Key Decisions

Purpose: capture conventions and architectural constraints the team must respect.

Core principles
- Persistence: Every experiment saves its own parameters in an `experiment.json` file. All experiments are located in the `logs/` directory.
- Reproducibility: each experiment can be re-run multiple times with different seeds. A `run` is a single execution of an experiment with a specific seed and is located in a subfolder of the experiment folder. Each run saves its own results in its dedicated folder.
- Modular algorithms: add new algorithms as modules under `src/marl/` and provide a thin wrapper.
- Code-first or config-first: exeperiments are always created from code. To run an experiment, create a small Python script that imports the relevant trainer and environment,, creates an `Experiment` object with the relevant parameters and saves it to the `logs/` folder.

Project conventions
- File layout: keep core algorithm code in `src/marl/` and UI tools in `src/ui/`.
- Naming: experiment identifiers use human-readable names.
- Logging: run results are logged in CSV format by default, but other loggers (e.g. TensorBoard, Weights & Biases) can be added as needed.
- Tests: there are not many tests because algorithms are difficult or long to test. However, core utilities should be tested when possible.
- Good defaults are important for ease of use, but all parameters should be configurable for reproducibility and experimentation.

Performance and compute
- GPU selection: experiments should always be run on the GPU, never on the CPU. Use `nvidia-smi` to inspect available GPUs and choose the best one(s) for the experiment. Never scatter experiments across more than 4 GPUs.
- Batch sizes and steps: keep defaults safe for most GPUs; document recommended settings in `README.md` or experiment configs.


Code quality
- Small focused PRs: change one thing at a time (e.g., algorithm, runner, or visualization).
- No backwards compatibility: config compatibility if not a concern. If a change breaks old configs, that’s fine since the CSV logs are the source of truth for results. 

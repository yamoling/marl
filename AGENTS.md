# Agents instructions

## Core Constraints

- **Commits:** You are NOT allowed to create commits on your own initiative, unless explicitly requested by the user.
- **Watermarking:** Every non-trivial function or method you write or edit in the main code of the repository must include an `@ai-generated` tag in its docstring/documentation unless the edit is minimal or the function is trivial (e.g.: constructors or one-liners). The user will verify and remove this tag later. Files in the `.agents/` folder do not need this waterkmarking.

## Contextual Imports

@readme.md

You MUST read the [readme.md](readme.md) before writing any code, if you haven't already.

## Project Description

MARL is a research-oriented Python repository for prototyping multi-agent reinforcement-learning algorithms. It combines trainable algorithms and neural-network model banks with serializable environment configurations, experiment/run orchestration, logging, and result visualisation.

- Main Python package: `src/marl/`.
- Environments are configured through `EnvConfig` subclasses (for example `LLEConfig`) and produce `MARLEnv` instances from `multi-agent-rlenv`; the Laser Learning Environment is provided through the `laser-learning-environment` dependency.
- Training algorithms live in `src/marl/algos/`; reusable neural-network components and model banks live in `src/marl/nn/`.
- An `Experiment` stores an environment/trainer specification and can contain multiple seeded `Run` instances. Results, checkpoints, actions, and logger output are stored under `logs/`.
- CSV is the default experiment logger. Use Polars lazy frames exposed by runs (`test_metrics`, `train_metrics`, and `training_data`) for local result analysis.
- Runnable examples are in `examples/`; tests are in `tests/`; project scripts are in `scripts/`; user-facing architecture and design documentation is in `doc/`.
- The repository is intended for algorithm development rather than as a stable external library. Preserve serialization compatibility and experiment provenance when making changes.

## Behaviour

### Prompt output

At the end of a prompt, you should not explicitly state that you have formatted the code or used the 60 seconds timeout since these are expected from you. Only report useful information related to the prompt or failing tests, if applicable.

## Watermark Examples

```python
def complex_function():
    """
    Executes core agent logic.

    @ai-generated
    """
    return True
```

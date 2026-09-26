# MARL

This repository contains a variety of Multi-Agent Reinforcement Learning (MARL) algorithms. Its purpose is to develop new algorithms and it is not intended to be a stable library.

`marl` is strongly typed and has high code quality standards. Any contribution to this repository is expected to exhibit a similar quality. `marl` comes with a web interface to visualise the results of your experiments (more info down below).

## Getting started

To install all the dependencies, run `uv sync`. If you are using an older GPU (e.g. the MLG GPU cluster) that only supports PyTorch <= 2.7.1, use the `legacy-gpu` extra.

```bash
$ uv sync                    # Default, installs latest PyTorch
$ uv sync --extra=legacy-gpu # Older GPUs (installs PyTorch < 2.8)
```

### Running an experiment

Check the examples in the `examples` folder. You can also have a look at more complex setups in `create_experiments.py` and run the experiment created directly with the `--run` option. The results of the experiment are stored in the `logs` folder.

```bash
$ python src/create_experiments.py --run
```

### Checking results

#### Logs

When creating your experiment, you can decide which logging method to use (csv, tensorboard, weights & biases, or neptune). All log files are stored in the `logs` folder.

For instance, to check your tensorboard logs, run

```bash
$ tensorboard --logdir logs
```

### EnvConfig

Describes an environment to instanciate. A generic Pickling implementation exists, but it is sensitive to version changes etc, and if you have a project under development, you should prefer to implement your own EnvConfig subclass that can seamlessly be serialised and deserialised.

#### MARL Studio

Studio reads experiments from `./logs` (relative to the backend process’s working directory) by default. From the repository root, install the Python dependencies with `uv sync`, then build the frontend and start the backend:

```bash
cd src/studio/frontend
npm ci
npm run build                 # Output: src/studio/frontend/dist/
cd ../../..
uv run python scripts/serve_studio.py  # http://127.0.0.1:5000
```

For frontend development, run these in separate terminals from the repository root:

```bash
uv run python scripts/serve_studio.py          # Backend: http://127.0.0.1:5000
cd src/studio/frontend && npm run dev           # Vite: http://localhost:5173 (proxies /api to the backend)
```

To point Studio at a different logs directory, pass its path as the first argument (e.g. `uv run python scripts/serve_studio.py /path/to/logs`). Studio can launch, rename and delete experiments; use a copy of logs when testing those actions.

The library search accepts free text (case-insensitive substring of experiment identity or parameter values) and parameter queries of the form `path OP value`. Operators are `=`, `!=`, `>`, `<`, `>=`, `<=` and `~` (case-insensitive substring). For example, `id=healthy`, `trainer.lr<1e-3`, `mixer=qmix`, or `name~vdn`. A leaf path such as `lr` also matches nested paths ending in `.lr`; quote values containing spaces (`name~"my experiment"`). Separate terms with spaces to AND them (`mixer=qmix lr<1e-3`). Numeric comparisons require numeric values; `=`/`!=` also work for strings and booleans.

Browser smoke tests run only against generated temporary fixture logs, never your `logs/` directory. After `uv sync` and `npm ci`, install Chromium once with `cd src/studio/frontend && npx playwright install chromium`, then run `npm run test:e2e` from that directory. The test command builds the frontend and starts its own fixture-only backend.

#### Legacy web UI (`src/ui`)

The old Vue UI remains available but is legacy. **With the Brave browser:** disable Brave Shields if it blocks the UI. To build and serve it:

```bash
cd src/ui
npm install
npm run build
cd ../..
python src/serve.py
```

For legacy UI development, run `cd src/ui && npm run dev` and `python src/serve.py` in separate terminals.

## Repository Architecture & Guidelines

This repository is aimed at prototyping but tries to follows good software engineering practices as much as possible.

## Serialization

To serialize and deserialize experiments to/from the disk, many classes are Python dataclasses. The `Serializable` class handles the heavy lifting, dynamically finding the appropriate class to instanciate.

**TODO**: complete this section

### Models (`src/marl/models/`)

The models module exposes:

- abstract classes that algorithms can work with (e.g. `Actor`, `Critic` or `QNetwork`);
- implementation of utility objects such as `Experiment`, `Run`, `Batch` or `ReplayMemory`.

The `models` module should absolutely not contain implementations of neural networks or algorithms.

### A few important classes and functions

- `Agent`: Abstract class that encapsulate the decision-making logic. It exposes the `choose_action()` method and is agnostic to the learning algorithm.
- `Trainer`: Abstract base class for learning algorithms that train agents. Trainers implement `update_step()` and `update_episode()` methods, expose trainable neural networks, and implement `make_agent()` to produce their corresponding agent.
- `Experiment` and `Run`: an `Experiment` is defined by a specific training algorithm and a specific environment and their related set of parameters. Each `Experiment` is stored in its dedicated folder. An `Experiment` can be run multiple times with different seeds, hence the `Run` class. Every `Run` has its own results stored in its dedicated folder.
- `simple_run`: this function contains the boilerplate regarding the training of an agent. It essentially takes a trainer and an environment as input and trains the agent over. It orchestrates the training/testing loop. Tests are seeded to be reproduceable and the agent's weights can be stored at test steps to be re-loaded afterwards.

### Neural Networks (`src/marl/nn/`)

This module contains neural network related classes and functions as well as a _model bank_. The _model bank_ contains a series of models that serve a specific purpose (e.g. a CNN Q-network, a MLP Q-network, etc). Mixing networks such as VDN, QMIX or QPLEX also have their own `src/marl/nn/mixers` module.

All classes inherit from the `NN` abstract class that enables each device management, randomization, and saving/loading.

### Web UIs

MARL Studio uses a Vue frontend in `src/studio/frontend/` and a FastAPI backend in `src/studio/backend/`. The older UI in `src/ui/` is legacy.

## Algorithm Organization

Each training algorithm has its own dedicated file in the `src/marl/training` module. This module also contains components that provide intrinsic rewards such as RandomNetworkDistillation.

| Algorithm            | Multi-Objective | Status      | Notes                                                                                 |
| -------------------- | :-------------: | ----------- | ------------------------------------------------------------------------------------- |
| Q-Learning (Tabular) |        ✗        | Working     | Classic tabular approach                                                              |
| DQN/IQL              |        ✓        | ✓           | Independent Q-learning (DQN with `mixer=None`)                                        |
| VDN                  |        ✓        | ✓           | Value Decomposition Network                                                           |
| QMIX                 |        ✓        | ✓           |                                                                                       |
| QPLEX                |        ?        | Almost      | Factorization architecture                                                            |
| QTRAN                |        ?        | Not tested  | Transitivity-aware factorization                                                      |
| QATTEN               |        ?        | Not tested  | Attention-based mixing                                                                |
| IPPO                 |        ?        | ✓           | MAPPO with `mixer=None`                                                               |
| MAPPO                |        ?        | ✓           | Multi-Agent PPO with centralized critic                                               |
| DDPG                 |        ✗        | ✗           | Continuous control                                                                    |
| Option-Critic        |        ✗        | ?           | Hierarchical RL                                                                       |
| RND                  |        ✓        | ✓           | Random Network Distillation                                                           |
| Social Influence     |        ✗        | ✓           | Causal influence intrinsic reward with a Model of Other Agents (on top of IPPO/MAPPO) |
| ICM                  |        ✓        | ✓           | Intrinsic Curiosity Module, agent-wise (on top of any trainer with an `ir_module`)    |
| HAVEN                |        ✗        | Unit-tested | Two-level value decomposition with advantage rewards                                  |
| MASER                |        ✗        | Unit-tested | Subgoals from the replay buffer with actionable-distance intrinsic rewards (on QMIX)  |
| REINFORCE            |        ✗        | ✓           | Policy gradient method                                                                |
| AlphaZero/MCTS       |        ✗        | ?           | Tree search-based                                                                     |

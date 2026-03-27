# MARL
This repository contains a variety of Multi-Agent Reinforcement Learning (MARL) algorithms. Its purpose is to develop new algorithms and it is not intended to be a stable library.

`marl` is strongly typed and has high code quality standards. Any contribution to this repository is expected to exhibit a similar quality. `marl` comes with a web interface to visualise the results of your experiments (more info down below).

## Getting started
To install all the dependencies, run `uv sync`. If you are using a GPU whose support has ended, use the `legacy-gpu` extra.

```bash
$ uv sync                    # Standard install
$ uv sync --extra legacy-gpu # Install for older GPUs
```

### Running an experiment
Setup your experiment accoring to the examples in `create_experiments.py` and run it directly with the `--run` option. The results of the experiment are stored in the `logs` folder.

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

#### Web UI
**With the Brave browser:** you have to deactivate the Brave shield.

You can also inspect your results with a dedicated web UI. You first have to build the sources, and then serve the files with the `serve.py` script.

```bash
$ cd src/ui
$ npm install   # or deno install or bun install
$ npm run build # Build the sources to src/ui/dist.
$ cd ../..      # Go back to the root of the project
$ python src/serve.py
```



To serve the files in development mode, you need two terminals.
```bash
$ cd src/ui && npm run dev  # In one terminal
$ python src/serve.py       # In an other terminal
```


## Repository Architecture & Guidelines
This repository is aimed at prototyping but tries to follows good software engineering practicies as much as possible.

### Models (`src/marl/models/`)
The models module exposes:
- abstract classes that algorithms can work with (e.g. `Actor`, `Critic` or `QNetwork`);
- implementation of utility objects such as `Experiment`, `Run`, `Batch` or `ReplayMemory`.

The `models` module should absolutely not contain implementations of neural networks or algorithms.

### A few important classes
- **Agent**: Abstract class that encapsulate the decision-making logic. It exposes the `choose_action()` method and is agnostic to the learning algorithm.
- **Trainer**: Abstract base class for learning algorithms that train agents. Trainers implement `update_step()` and `update_episode()` methods, expose trainable neural networks, and implement `make_agent()` to produce their corresponding agent.
- **Experiment** and **Run**: an `Experiment` is defined by a specific training algorithm and a specific environment and their related set of parameters. Each `Experiment` is stored in its dedicated folder. An `Experiment` can be run multiple times with different seeds, hence the `Run` class. Every `Run` has its own results stored in its dedicated folder.
- **Runner**: the runner orchestrates the training/testing loop. The runner manages the lifecycle of training runs with proper seeding and checkpointing such that test episodes can be replayed.

### Neural Networks (`src/marl/nn/`)
This module contains neural network related classes and functions as well as a *model bank*. The *model bank* contains a series of models that serve a specific purpose (e.g. a CNN Q-network, a MLP Q-network, etc). Mixing networks such as VDN, QMIX or QPLEX also have their own `src/marl/nn/mixers` module.

All classes inherit from the `NN` abstract class that enables each device management, randomization, and saving/loading.

## Algorithm Organization
Each training algorithm has its own dedicated file in the `src/marl/training` module. This module also contains components that provide intrinsic rewards such as RandomNetworkDistillation.

### Implemented Algorithms

| Algorithm | Multi-Objective | Status | Notes |
|-----------|:---:|---|---|
| Q-Learning (Tabular) | ✗ | Working | Classic tabular approach |
| DQN/IQL | ✓ | ✓ | Independent Q-learning (DQN with `mixer=None`) |
| VDN | ✓ | ✓ | Value Decomposition Network |
| QMIX | ✓ | ✓ | |
| QPLEX | ? | Almost | Factorization architecture |
| QTRAN | ? | Not tested | Transitivity-aware factorization |
| QATTEN | ? | Not tested | Attention-based mixing |
| IPPO | ? | Working | MAPPO with `mixer=None` |
| MAPPO | ? | Working | Multi-Agent PPO with centralized critic |
| DDPG | ✗ | ✗ | Continuous control |
| Option-Critic | ✗ | ? | Hierarchical RL |
| RND | ✓ | ? | Random Network Distillation |
| ICM | ✓ | ? | Intrinsic Curiosity Module |
| AlphaZero/MCTS | ✗ | ? | Tree search-based |




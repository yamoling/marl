# Local Advantage Networks

Implemented from Avalos et al., _Local Advantage Networks for Multi-Agent
Reinforcement Learning in Dec-POMDPs_, TMLR (October 2023),
[arXiv v3](https://arxiv.org/abs/2112.12458v3). The paper was read through Zotero
(item `GNUG9GJ5`; its catalog title is _Local Advantage Networks for Cooperative
Multi-Agent Reinforcement Learning_).

`marl.algos.LAN.from_env(config)` constructs a serializable trainer. Use
`agent_id=True, last_action=True` on the environment configuration to supply the
inputs used in the paper. Only scalar shared rewards are supported.

The local policy is a shared FC(64)/ReLU/GRU(64)/linear advantage network.
The centralized network embeds each agent's GRU output, observation, and extras
with a shared FC(128)/ReLU, sums the embeddings, concatenates the global state
(and any state extras), and applies two FC(128)/ReLU layers and a scalar output.
Gradients from the centralized value also reach the local history encoder.
The acting agent only contains the local network.

Training minimizes the masked mean squared TD error for each agent's
`V(s, history) + A(local_history, action)` proxy (equations 1 and 3).
Next actions are selected with online advantages among legal actions and
evaluated with target advantages and target value. Recurrent target histories
are unrolled from the initial observation. True terminals stop bootstrapping;
time limits do not. Padded transitions do not contribute to the loss, and dead
agents continue receiving the shared learning signal. The default advantages
are unconstrained. `mean_center=True` selects Appendix D's LAN-mean ablation.

Appendix B defaults: Adam at 0.0005, gamma 0.99, gradient norm clipping at 10,
5,000 replay episodes, batches of 32 episodes, two independently sampled
updates after each episode, hard target copies every 200 optimizer updates,
and epsilon decreasing linearly from 1 to 0.05 over 50,000 environment steps.
The appendix specifies the epsilon endpoints and duration; linear interpolation
follows the QMIX-style schedule referenced by the experimental protocol.

## Running on the experiment machine

Install this repository's dependencies with `uv sync` (or the documented
`legacy-gpu` extra), then install StarCraft II **4.6.2.69232** and the SMAC maps.
Set `SC2PATH` to that installation. The scripts request game version `4.6.2`;
verify that it resolves to build 69232. A newer SC2 version changes the benchmark.
The repository `SMACConfig` preserves the version when seeding and exposes
SMAC episode limits as truncations.

Run from the repository root:

```bash
# All 14 maps, ten seeds, two million steps per seed, evaluations every
# 10,000 steps over 32 greedy episodes. One seed runs at a time by default.
uv run python scripts/reproduce_lan.py --run

# Include the Appendix D ablation; choose a fresh prefix for a new study.
uv run python scripts/reproduce_lan.py --variants lan lan-mean --log-prefix lan-study --run

# Restrict maps/seeds or select a particular GPU on the other machine.
uv run python scripts/reproduce_lan.py --maps corridor --seeds 0 1 --device cuda:0 --run

# Omitting --run creates specifications only. Later, --load --run loads them.
# Existing seeds are skipped, including interrupted runs, to preserve results.
uv run python scripts/reproduce_lan.py --load --seeds 10 11 --run

uv run python scripts/summarize_lan.py logs/lan-paper-v3/lan --output reports/lan-win-rates.csv
```

The experiment skill's immutable specification/seed conventions are followed.
Record the repository revision, lockfile, SC2 build, GPU/PyTorch versions, and
command with the resulting data. `experiment.json` stores the environment,
network, and training configuration. CSV results, actions, and checkpoints
are retained by the repository runner. The summary exports the median and
first/third quartiles across completed seeds, with seed counts for each point.
The runner can finish an episode beyond the two-million-step budget; its final
extra evaluation point should be excluded when comparing fixed-budget scores.

These scripts reproduce LAN and LAN-mean on the main SMAC benchmark. They do
not recreate the authors' external baseline implementations, the no-fog IQL
study, or Appendix G's modified MPE environment. In particular, the repository's
QPlex trainer currently has an unimplemented target-action path, so it should
not be used as a claimed reproduction of the authors' QPLEX baseline.
No full training or reproduction script was run during implementation.

## Verification

```bash
uv run pytest -q tests/test_lan.py tests/test_smac_config.py
```

The LAN tests cover hand-computed targets, legal-action selection, terminal vs.
time-limit bootstrapping, recurrent execution, decentralized independence,
padding masks, joint gradients, update counts, configuration serialization,
and optimizer/target ownership after device transfer. CUDA tests run when a
GPU is available; otherwise pytest reports a skip. SMAC adapter tests use a
mock engine and do not launch StarCraft II.

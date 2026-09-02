# Training-loop optimizations

Candidate optimizations derived from the [profiling reports](../profiling/index.md), ordered by expected GPU impact.
Each file holds a technical description of one optimization and, once evaluated, the measured result and the decision.

Evaluation protocol: [`.agents/experiments/optimizations/bench.py`](../../.agents/experiments/optimizations/bench.py) reproduces the
profiling configuration (LLE level 6, layered observations, CNN + independent MLP, `torch.compile`) on the RTX 3060 and reports
steady-state steps/s (mean ± SD over three 2,000-step repetitions after a 256-step warm-up). The baseline is recorded in
[`baseline.json`](../../.agents/experiments/optimizations/results/baseline.json) (VDN 380.5, MAPPO 282.3, IPPO 294.8 steps/s). The very
first run after a cold start (fresh inductor cache, idle GPU clocks) is 3–6% slower than later runs (`baseline-cold.json`); it is not
used as a reference. Run-to-run noise on a warm GPU is about ±2%, so anything within that band is treated as "no effect" and discarded;
borderline gains are re-measured before a decision.

## Table of contents

| # | Optimization | Targets | Status |
|--:|---|---|---|
| 1 | [PPO metrics aggregated on device](01-ppo-metrics-on-device.md) | MAPPO, IPPO | discarded (no effect) |
| 2 | [PPO minibatches by device indexing](02-ppo-minibatch-device-indexing.md) | MAPPO, IPPO | pending |
| 3 | [Single-pass, pinned `TransitionBatch` packing](03-transition-batch-single-pass.md) | VDN, MAPPO, IPPO | pending |
| 4 | [Foreach soft target update](04-soft-update-foreach.md) | VDN (and all DQN variants) | kept: VDN +2.8% / +3.3% in two paired runs |
| 5 | [Fused Adam/AdamW on CUDA](05-fused-adam.md) | VDN, MAPPO, IPPO | kept: VDN +2.8%, MAPPO +6.9%, IPPO +4.1% |
| 6 | [GAE and returns without per-step kernels](06-ppo-gae-vectorization.md) | MAPPO, IPPO | pending |
| 7 | [Deferred DQN metric synchronization](07-dqn-deferred-sync.md) | VDN | pending |
| 8 | [Single pinned transfer for per-step action selection](08-action-selection-transfer.md) | all | pending |
| 9 | [Channels-last CNN](09-channels-last.md) | all (low priority) | pending |

## Summary of results

Filled in as optimizations are evaluated.

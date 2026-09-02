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

Because single-shot comparisons against a stored baseline proved unreliable (identical code differed by up to 13% between runs),
optimizations 5 onwards are evaluated with the paired A/B mode of the benchmark script (`--ab`), which alternates a git worktree of the
committed baseline and the working tree in the same session. Optimizations 4 and 5 used 3 alternations; from optimization 2 onwards the
default is 6 alternations per algorithm to reduce the noise further.

## Table of contents

| # | Optimization | Targets | Status |
|--:|---|---|---|
| 1 | [PPO metrics aggregated on device](01-ppo-metrics-on-device.md) | MAPPO, IPPO | discarded (no effect) |
| 2 | [PPO minibatches by device indexing](02-ppo-minibatch-device-indexing.md) | MAPPO, IPPO | kept: MAPPO +14.3%, IPPO +6.8% |
| 3 | [Single-pass `TransitionBatch` packing](03-transition-batch-single-pass.md) | VDN, MAPPO, IPPO | kept (unpinned): VDN +4.7%, PPO unchanged |
| 4 | [Foreach soft target update](04-soft-update-foreach.md) | VDN (and all DQN variants) | kept: VDN +2.8% / +3.3% in two paired runs |
| 5 | [Fused Adam/AdamW on CUDA](05-fused-adam.md) | VDN, MAPPO, IPPO | kept: VDN +2.8%, MAPPO +6.9%, IPPO +4.1% |
| 6 | [GAE and returns without per-step kernels](06-ppo-gae-vectorization.md) | MAPPO, IPPO | discarded (MAPPO -0.3%, IPPO +2.5%) |
| 7 | [Deferred DQN metric synchronization](07-dqn-deferred-sync.md) | VDN | discarded (no effect, VDN -0.9%) |
| 8 | [Single pinned transfer for per-step action selection](08-action-selection-transfer.md) | all | discarded (no effect) |
| 9 | [Channels-last CNN](09-channels-last.md) | all (low priority) | discarded (regression -1.4% to -2.3%) |

## Summary of results

Four optimizations were kept and committed, five were discarded. Steps/s changes are paired A/B means against the commit preceding
each optimization, so they compound (later baselines already include earlier gains).

| Optimization | Commit | VDN | MAPPO | IPPO |
|---|---|---:|---:|---:|
| 5. Fused Adam/AdamW on CUDA | `8e00cad0` | +2.8% | +6.9% | +4.1% |
| 4. Foreach soft target update | `79d2e585` | +3.0% | – | – |
| 2. PPO minibatches by device indexing | `58f7bb5f` | – | +14.3% | +6.8% |
| 3. Single-pass `TransitionBatch` packing | `609494a1` | +4.7% | – | – |

Reference throughput before and after the campaign (warm GPU, same benchmark): VDN 380 → ~405 steps/s, MAPPO 282 → ~355 steps/s,
IPPO 295 → ~355 steps/s, i.e. roughly +7% for VDN and +20–25% for the PPO variants. These end-to-end figures come from the baseline
columns of successive A/B runs and carry the same ±3–5% drift as the individual measurements.

Discarded: 1 (PPO metrics on device), 6 (GAE on host), 7 (deferred DQN sync), 8 (pinned per-step transfers) had no measurable effect;
9 (channels-last) regressed by 1–2%.

### Lessons

- The training update is GPU-bound for these small networks. Removing host-side synchronization (1, 7) or Python overhead does not
  increase throughput; only changes that remove GPU kernels or host-to-device copies do (2, 3, 4, 5).
- Single-run throughput comparisons are unreliable on this GPU (identical code varied by up to 13%). Use the paired A/B mode of the
  benchmark script with at least six alternations.
- Remaining large levers are structural: vectorized environments to batch action selection (still ~20% of host time), and compiling
  the whole PPO epoch (forward, backward, clipping, optimizer) as one graph to cut kernel launches further.

### Reproducing an evaluation

```bash
git worktree add /tmp/marl-baseline HEAD            # committed baseline
# ... edit the working tree ...
uv run python .agents/experiments/optimizations/bench.py --ab <label> --algos vdn mappo ippo
```

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

## Big picture: before and after

The campaign changed how data reaches the GPU and how parameters are updated. It changed no algorithm, no hyper-parameter, no logged
metric and no serialized format. Training is numerically equivalent to before, apart from the reassociation that fused optimizer
kernels introduce.

Measured end to end, `e71455c5` (before) against the current tree, paired A/B over six alternations:

| Algorithm | Before (steps/s) | After (steps/s) | Change |
|---|---:|---:|---:|
| VDN | 370.9 ± 4.1 | 405.1 ± 4.3 | +9.2% |
| MAPPO | 277.4 ± 2.7 | 345.9 ± 7.6 | +24.7% |
| IPPO | 285.0 ± 2.3 | 347.4 ± 6.8 | +21.9% |

All eighteen rounds were positive and the per-round spread was narrow, between +7.5% and +28.0%. Raw data is in
[`00-cumulative.json`](../../.agents/experiments/optimizations/results/00-cumulative.json).

### What changed in the hot path

**Batch construction.** A `TransitionBatch` used to materialize each field on its own, with a Python list comprehension, a fresh
`np.array`, a `torch.from_numpy` and its own transfer. The nine fields trainers always need are now filled into pre-allocated arrays
in a single pass over the transitions and transferred once each, lazily, after the target device is known. Nothing is copied to the
GPU twice.

**PPO minibatching.** This was the single largest win. Each of the twenty epochs per update used to rebuild a whole batch from the
Python transition objects, re-running the packing and the host-to-device copy for every field it touched. An update now packs once
and each epoch takes an index-select on tensors that are already resident on the GPU. `Batch.for_individual_learners` became
idempotent so a minibatch inherits its parent's agent-wise tensors instead of expanding them again.

**Optimizer.** Adam and AdamW now use the fused CUDA implementation. The optimizer is rebuilt inside `to()`, because trainers are
constructed on the CPU and moved afterwards, so the device is only known at that point. A step that dispatched several elementwise
kernels per parameter tensor now dispatches one or two in total.

**Target networks.** The soft update looped over parameters in Python and allocated two products and a sum per tensor. It is now one
`torch._foreach_lerp_` call, and the hard update one `torch._foreach_copy_`.

**Action selection.** Per-step observations are staged through reusable pinned host buffers and copied asynchronously, instead of
issuing separate pageable copies per field. This one is within measurement noise on its own and was adopted for the structure it
provides rather than for a proven gain.

### What this says about the machine

The profiling reports ranked host-side synchronization high, because `.item()` and `.cpu()` dominated the Python profiles. That
ranking was misleading. Those calls were expensive because the host was waiting for the GPU, not because synchronizing costs
anything by itself. Removing them changed nothing measurable. Every optimization that paid off removed either GPU kernels or
host-to-device copies, and the two that touched only synchronization or Python overhead produced nothing.

### What is left

Action selection still costs about a fifth of host time and still runs one environment at a time. Batching several environments into
one forward pass is the largest remaining lever, and it needs a vectorized runner rather than a local change. Beyond that, compiling
a whole PPO epoch as a single graph would cut the remaining launch overhead of the forward, backward, clipping and optimizer
sequence.

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
| 8 | [Single pinned transfer for per-step action selection](08-action-selection-transfer.md) | all | kept (within noise, adopted anyway) |
| 9 | [Channels-last CNN](09-channels-last.md) | all (low priority) | discarded (regression -1.4% to -2.3%) |

## Summary of results

Five optimizations were kept and committed, four were discarded. Steps/s changes are paired A/B means against the commit preceding
each optimization, so they compound (later baselines already include earlier gains).

| Optimization | Commit | VDN | MAPPO | IPPO |
|---|---|---:|---:|---:|
| 5. Fused Adam/AdamW on CUDA | `8e00cad0` | +2.8% | +6.9% | +4.1% |
| 4. Foreach soft target update | `79d2e585` | +3.0% | – | – |
| 2. PPO minibatches by device indexing | `58f7bb5f` | – | +14.3% | +6.8% |
| 3. Single-pass `TransitionBatch` packing | `609494a1` | +4.7% | – | – |
| 8. Pinned staging for action selection | `a17602d2` | +0.8% | +2.1% | +1.7% |

The cumulative effect measured directly against the pre-campaign commit is VDN +9.2%, MAPPO +24.7% and IPPO +21.9%. See
[Big picture: before and after](#big-picture-before-and-after) above. The per-optimization rows do not sum to it, because each was
measured against a different baseline and the same bottleneck can be attacked twice.

Optimization 8 was adopted despite being within measurement noise: it is self-contained, cannot be slower in principle, and
prepares the staging path for a vectorized runner.

Discarded: 1 (PPO metrics on device), 6 (GAE on host), 7 (deferred DQN sync) had no measurable effect; 9 (channels-last) regressed
by 1–2%.

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

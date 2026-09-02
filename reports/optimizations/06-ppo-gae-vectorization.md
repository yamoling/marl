# 6. GAE and returns without per-step kernels

**Targets:** MAPPO, IPPO. **Files:** `src/marl/models/batch/batch.py` (`compute_gae`, `compute_mc_returns`).

## Evidence

Both `compute_gae` and `compute_mc_returns` iterate over the 64 time steps of the PPO batch in Python and perform several tiny CUDA
operations per iteration (index read, multiply-add, index write), i.e. roughly 500 kernel launches plus Python overhead per PPO
update, on tensors of at most 4 elements. On CUDA this is launch-latency bound.

## Description

Two possible implementations, to be benchmarked:

1. **Host-side recursion.** Move `rewards`, `next_values`, `values` and `not_dones` to the CPU once (a single `.cpu()` on a stacked
   tensor), run the backward recursion with NumPy or CPU torch tensors, then copy the result back to the device once. PPO already
   synchronizes at this point, so the extra sync is nearly free.
2. **Closed-form vectorization.** With `f_t = gamma * lambda * not_done_t`, `A_t = delta_t + f_t * A_{t+1}` can be written as a
   matrix-vector product `A = M @ delta` where `M[t, s] = prod_{k=t}^{s-1} f_k` for `s >= t` (upper-triangular). Build `M` on device with
   `cumprod`/`cumsum` of logs and a triangular mask, or with a loop over `log2(T)` doubling steps. Same for the Monte Carlo returns.

Option 1 is simpler and exact; option 2 keeps everything on device. Both must handle the multi-objective reward shape
(`reward_size > 1`) and both `TransitionBatch` and `EpisodeBatch` layouts (time is dimension 0 in both cases).

## Validation

- `uv run pytest tests -q` must pass (see `tests/test_batch.py` for GAE/returns tests).
- Add a test comparing the new implementation against the existing recursive one on random data with `dones` present, both on
  1-d and multi-objective rewards.

## Results

Implemented option 1 (host-side recursion) in `compute_gae` and `compute_mc_returns`: when `self.device.type != "cpu"`, the
relevant tensors are detached and moved to the CPU with a single `.to("cpu")` each, the backward recursion runs on CPU
tensors exactly as before, and the result is copied back to `self.device` once. The CPU path (`self.device.type == "cpu"`)
is untouched.

Paired A/B benchmark, 6 rounds, `uv run python .agents/experiments/optimizations/bench.py --ab 06-gae --algos vdn mappo ippo`:

| Algorithm | Baseline (steps/s) | Candidate (steps/s) | Change (mean) | Per-round change |
|---|---:|---:|---:|---|
| vdn | 416.4 ± 2.6 | 419.0 ± 3.2 | +0.6% | +0.6%, +0.3%, +0.1%, +1.3%, +1.3%, +0.2% |
| mappo | 354.0 ± 5.7 | 353.1 ± 9.1 | -0.3% | +0.9%, -6.4%, -0.7%, +2.9%, +1.2%, +0.8% |
| ippo | 349.0 ± 9.2 | 357.7 ± 4.8 | +2.5% | +1.9%, +0.2%, +8.1%, +0.8%, +2.2%, +2.0% |

VDN (regression check, doesn't call `compute_gae`/`compute_mc_returns`) is flat as expected, +0.6% within run-to-run noise.
IPPO shows a consistent, mostly-positive gain (+2.5% mean, 5/6 rounds positive, one large +8.1% outlier), consistent with
the hypothesis that removing ~500 tiny CUDA kernel launches per PPO update helps on GPU. MAPPO, however, is flat-to-negative
(-0.3% mean, only 4/6 rounds positive, with one -6.4% outlier), i.e. it does not show the same benefit — plausibly because
MAPPO's mixer/critic forward pass dominates its per-update time, or because the CPU round-trip is small enough that overall
sync overhead in the batch's cadence hides the effect there. Since both PPO variants were targeted and only one (IPPO)
clears the ~2% bar with consistent sign, the effect is not consistent enough across both targets.

**Decision:** discarded (no consistent effect; MAPPO -0.3%, IPPO +2.5%). The change was reverted.
a later benchmark shows a cleaner win, but does not clear the bar for both MAPPO and IPPO in this run).

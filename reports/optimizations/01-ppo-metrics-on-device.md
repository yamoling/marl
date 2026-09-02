# 1. PPO metrics aggregated on device

**Targets:** MAPPO, IPPO. **Files:** `src/marl/algos/ppo.py` (`PPO.train`).

## Evidence

GPU Python profiles: tensor `.item()` (0.69–0.70 s) and `.cpu()` (0.32–0.35 s) account for **14.6–15.0%** of the MAPPO/IPPO
profile. Every one of the 20 epochs per update calls `.item()` five times (KL, norm, actor/critic/entropy loss, loss) and `.cpu().numpy()`
twice (full `ratio` and `entropy` arrays). Each call blocks the host until the CUDA stream drains, so the CPU cannot queue the next
epoch's kernels while the GPU is still busy.

## Description

Keep all per-epoch diagnostics as detached 0-d tensors on the training device and synchronize once per `train` call:

- Replace `log_lists[key].append(x.item())` with appending detached tensors (`x.detach()`), and for `ratio`/`entropy` append only the
  on-device `mean`, `max` and `min` (the logged quantities are `mean/max/min` anyway; keeping whole arrays serves no logger).
- Do not call `.item()` on the gradient norm; keep `norm.detach()`.
- After the epoch loop, `torch.stack` each list, compute `mean/max/min` on device, then transfer everything with a single `.cpu()`
  (e.g. stack all summary scalars into one tensor and call `.tolist()` once).
- KL early stopping (`early_stopping_kl`) is the only diagnostic that legitimately needs a host value inside the loop. Only call
  `.item()` on `approx_kl_div` when `self.early_stopping_kl is not None`.
- Preserve the exact set of logged keys (`ppoc/mean-*`, `ppoc/max-*`, `ppoc/min-*`) and their semantics: for `ratios` and `entropies`
  the `mean` is the mean over all elements of all epochs, and `max`/`min` are the global extrema; for the scalar keys they are the
  statistics over epochs. Mean of per-epoch means equals the global mean because every minibatch has the same number of elements.
  Note that `entropy` is multiplied by `masks` before logging, keep that.

## Validation

- `uv run pytest tests -q` must pass.
- Logged values before/after must match to float precision for a fixed seed on CPU (a quick ad-hoc check is enough).

## Results

Implemented as described (detached on-device scalars, per-epoch `mean/max/min` of `ratio`/`entropy`, one `.tolist()` after the
epoch loop). Logged keys and values matched the previous implementation to 1e-6 on a fixed seed for IPPO and MAPPO. All tests passed.

| Algorithm | Baseline (steps/s) | Optimized, run 1 | Optimized, run 2 | Change (run 2) |
|---|---:|---:|---:|---:|
| VDN (untouched) | 380.5 ± 0.6 | 379.1 ± 3.2 | – | -0.4% |
| MAPPO | 282.3 ± 2.7 | 286.4 ± 6.4 | 283.9 ± 1.6 | +0.6% |
| IPPO | 294.8 ± 0.9 | 306.3 ± 5.3 | 294.2 ± 4.7 | -0.2% |

A first comparison against a cold baseline suggested +4.6% (MAPPO) and +7.8% (IPPO), but the untouched VDN moved by +5.4% in the
same run, and a warm re-measurement of the baseline removed the gain. The PPO update is GPU-bound: the `.item()` calls were expensive in
the Python profile only because the host was *waiting* for the GPU, not because the synchronization itself cost anything. Removing them
lets the host run ahead, but the GPU stream is the bottleneck, so throughput is unchanged.

**Decision:** discarded (no measurable effect). The change was reverted.

**Consequence for the remaining candidates:** host-side synchronization and Python-overhead reductions (optimizations 7 and 8) are
unlikely to pay off unless they also reduce the GPU kernel count. Priority goes to changes that cut GPU work: fused optimizer (5), fewer
tiny kernels (4, 6), fewer H→D copies (2, 3), channels-last (9).

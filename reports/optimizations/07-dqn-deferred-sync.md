# 7. Deferred DQN metric synchronization

**Targets:** VDN (all DQN variants). **Files:** `src/marl/algos/dqn.py` (`DQN.train`, `DQN._prepare_batch`).

## Evidence

`DQN.train` calls `td_loss.item()` right after computing the loss and `clip_grad_norm_(...).item()` right after `backward()`. Each
call blocks the host until the CUDA stream is empty, so the optimizer step, target update and the following action selection
cannot be queued while the GPU is working. With 400 updates per 2,000 steps this is 800 full pipeline drains. The intrinsic-reward
statistics in `_prepare_batch` add three more `.item()` calls when an IR module is used.

## Description

Queue all GPU work for an update before reading any scalar back:

- Keep `td_loss.detach()` and the norm tensor, run `optimiser.step()`, then read both values with a single transfer at the end of
  `train` (e.g. `torch.stack([td_loss.detach(), norm]).tolist()`).
- In `_prepare_batch`, compute `ir.mean()`, `ir.min()`, `ir.max()` as a stacked tensor and convert once with `.tolist()`.
- Preserve the returned log keys and their float type.

Consider also doing the same in `DQN._update` ordering: run `target_updater.update` and `policy.update` before the read-back so the
target-network interpolation is queued behind the optimizer step without a sync in between (`SoftUpdate` has no host dependency).

## Validation

- `uv run pytest tests -q` must pass.
- Logged values are unchanged.

## Results

| Algorithm | Baseline (steps/s) | Candidate (steps/s) | Change (mean) | Per-round change |
|---|---:|---:|---:|---|
| vdn | 407.3 ± 5.3 | 403.8 ± 6.1 | -0.9% | +1.6%, -1.0%, +1.4%, -2.8%, +0.8%, -4.9% |
| mappo | 359.5 ± 4.8 | 352.6 ± 7.8 | -1.9% | -0.5%, -6.3%, -0.3%, -1.2%, +0.6%, -3.8% |
| ippo | 358.9 ± 3.9 | 357.0 ± 8.4 | -0.5% | -5.0%, -1.9%, -1.5%, +3.7%, -1.4%, +3.0% |

VDN (the target) shows no consistent gain: the mean change is -0.9% and only 3 of 6 rounds are positive, with the largest
single-round swing in either direction (-4.9%) exceeding any positive round. This suggests that, at this batch size and update
frequency, the per-update host syncs removed by this change (`td_loss.item()`, `grad_norm.item()`, three IR `.item()` calls) are
not the bottleneck — likely because `PrioritizedMemory.update` (called right after, inside `train`) already forces a sync via
`priorities.max().item()` and `priorities.cpu().tolist()`, so deferring the loss/grad-norm read-back mostly hides behind a sync
that still happens a few lines later. MAPPO/IPPO (regression checks, code path untouched by this change) show similarly noisy,
slightly negative deltas, consistent with run-to-run variance rather than an actual regression from the edit.

**Decision:** discard. The `src/marl/algos/dqn.py` edits were reverted; the working tree for that file matches the pre-change
content.

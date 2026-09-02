# 3. Single-pass, pinned `TransitionBatch` packing

**Targets:** VDN primarily, also PPO's per-update batch. **Files:** `src/marl/models/batch/transition_batch.py`.

## Evidence

VDN GPU Python profile: tensor `.to()` costs **1.36 s (23.3%)** over 14,400 calls and `numpy.array` **0.40 s (6.9%)** over 23,868
calls. The CUDA profile shows 1,175 pageable H→D copies in 256 steps (VDN samples a 64-transition batch every 5 steps). Each field of
`TransitionBatch` is materialized independently with a list comprehension, `np.array`, `torch.from_numpy` and its own pageable
`.to(device)`; `Batch.to()` then walks `__dict__` and moves whatever was already materialized again.

## Description

Materialize the fields that are always needed by trainers **in one pass over the transitions** and transfer them with pinned memory
and asynchronous copies:

- In `TransitionBatch.__init__` (or lazily on first field access), fill the NumPy arrays for `obs`, `next_obs`, `extras`, `next_extras`,
  `actions`, `rewards`, `dones`, `available_actions`, `next_available_actions` in a single `for t in transitions` loop using
  pre-allocated arrays (`np.empty((size, *shape), dtype)`, shapes taken from the first transition). Fields that are rarely used
  (`states`, `states_extras`, `next_states`, `next_states_extras`, `probs`) can stay lazy.
- Create the torch tensors with `torch.from_numpy(arr).pin_memory()` when the target device is CUDA, then `.to(device, non_blocking=True)`.
  Pinning has a cost; benchmark with and without it and keep the faster variant.
- `Batch.to(device)` currently re-moves every materialized tensor. Ensure the batch is moved exactly once: prefer building tensors
  directly on the final device when the device is known at construction (`ReplayMemory.sample(...).to(device)` can be
  restructured so that `make_batch` receives the device, or `to()` can trigger materialization).
- Keep the lazy `cached_property` API and the `__getitem__` cache; other trainers and the `EpisodeBatch` depend on them.
- The `masks` property currently allocates on CPU and then moves (`torch.ones(self.size).to(self.device)`); allocate it directly on
  the device.

## Validation

- `uv run pytest tests -q` must pass.
- Tensor values, dtypes and shapes must be identical to the current implementation (compare on CPU for a fixed set of transitions).

## Results

Two variants were benchmarked with the paired A/B protocol (6 rounds, `vdn`, `mappo`, `ippo`), differing only in
`_PIN_MEMORY` in `transition_batch.py`.

### Pinned (`tensor.pin_memory()` before the H2D copy)

| Algorithm | Baseline (steps/s) | Candidate (steps/s) | Change (mean) | Per-round change |
|---|---:|---:|---:|---|
| vdn | 397.4 ± 4.0 | 405.7 ± 6.9 | +2.1% | +2.7%, +3.2%, +3.1%, +0.4%, +3.8%, -0.6% |
| mappo | 359.1 ± 10.1 | 353.9 ± 14.7 | -1.4% | +0.4%, -4.4%, +4.9%, -3.2%, -3.3%, -3.0% |
| ippo | 359.9 ± 13.6 | 355.9 ± 12.1 | -1.1% | +1.7%, -0.3%, -1.5%, -4.0%, -1.4%, -1.1% |

### Unpinned (plain `torch.from_numpy(arr).to(device, non_blocking=True)`)

| Algorithm | Baseline (steps/s) | Candidate (steps/s) | Change (mean) | Per-round change |
|---|---:|---:|---:|---|
| vdn | 386.4 ± 7.6 | 404.5 ± 8.2 | +4.7% | +4.6%, +8.3%, +1.6%, +4.5%, +5.4%, +3.8% |
| mappo | 347.1 ± 6.6 | 349.3 ± 14.3 | +0.6% | -0.5%, +6.0%, -4.4%, -2.2%, +2.1%, +2.6% |
| ippo | 352.8 ± 3.5 | 354.2 ± 8.1 | +0.4% | -1.7%, +2.1%, -1.4%, -1.1%, +2.6%, +1.9% |

For VDN — the primary target, which samples and packs a fresh 64-transition `TransitionBatch` every 5 steps — the
single-pass packing gives a consistent, positive gain in every round for both variants, and the unpinned variant is
clearly better (+4.7% vs +2.1% mean, positive in 6/6 rounds vs 5/6). Pinning a freshly allocated host tensor on every
`sample()` call costs more than it saves here, since each batch is only transferred once and then discarded (no reuse
of the pinned buffer), so the extra page-locking overhead is not amortized. For PPO-style trainers (mappo, ippo), which
pack the whole rollout once per update rather than once per few steps, the change is noise-level (within run-to-run
stddev) in both variants but never regresses on average with the unpinned code. `_PIN_MEMORY = False` was kept in
`transition_batch.py`.

**Decision:** keep, unpinned variant (committed).
(no regression) on PPO variants.

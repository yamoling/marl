# 2. PPO minibatches by device indexing

**Targets:** MAPPO, IPPO. **Files:** `src/marl/models/batch/transition_batch.py`, `src/marl/models/batch/batch.py`,
possibly `src/marl/algos/ppo.py`.

## Evidence

The CUDA operator profile records **1,600–1,616 pageable host-to-device copies for only 256 environment steps** in PPO, and `.to()`
costs 0.57–0.77 s in the Python profile. The cause is `TransitionBatch.get_minibatch`, which builds a *new* `TransitionBatch` from the
Python `Transition` objects. Each of the 20 epochs therefore re-runs the list comprehension, `np.array`, `torch.from_numpy` and an
H→D copy for every accessed field (obs, extras, available_actions, actions, masks, ...), even though the parent batch already holds
those tensors on the GPU. It also recomputes `for_individual_learners()`, `masked_indices` and `n_items` per minibatch.

## Description

Make `TransitionBatch.get_minibatch(indices)` return a batch whose tensors are **views/index-selections of the parent's already
materialized device tensors**:

- Add a lightweight constructor path (e.g. a private classmethod `_from_tensors`) that receives the parent batch and an index tensor
  on the parent's device. For every cached field present in the parent (`__dict__` entries that are tensors, plus `_cache`), store
  `parent_field[indices]`. Fields not yet materialized in the parent should be materialized in the parent first (so that they are
  computed once, on the full batch) and then indexed.
- Keep `self.transitions = [parent.transitions[i] for i in indices]` so that code paths relying on lazy properties still work.
- Convert `indices` to a `torch.LongTensor` on the batch device once per epoch. `PPO.train` currently draws `indices` with
  `np.random.choice`; keep that RNG call so that experiment reproducibility is unchanged, and pass the same indices for
  `returns`/`advantages`/`old_log_probs` selection.
- `for_individual_learners()` is applied to the parent before the loop, so the minibatch inherits the already-repeated `rewards`,
  `dones` and `masks`; ensure the second `minibatch.for_individual_learners()` call in `PPO.train` is not applied twice (either skip
  it for the indexed minibatch or make the method idempotent by tracking a flag).
- Leave `EpisodeBatch.get_minibatch` unchanged.

## Validation

- `uv run pytest tests -q` must pass (see `tests/test_batch.py`).
- Add a unit test checking that `get_minibatch` on a device batch produces tensors equal to those of a fresh `TransitionBatch`
  built from the same transitions.

## Results

Implementation: `TransitionBatch.get_minibatch` now converts the indices to a device `LongTensor` and calls `_index_select`, which
creates a child batch whose materialized per-item tensors (cached properties in `__dict__` and the `__getitem__` cache) are
index-selections of the parent's device tensors. The child keeps the sliced `transitions` list so never-materialized fields remain
lazy. `Batch.for_individual_learners` is now idempotent through a `_individual_learners_applied` flag, so the second call in
`PPO.train` on the minibatch is a no-op. Three unit tests were added (equality with a freshly built batch, laziness of
unmaterialized fields, order-independence of `for_individual_learners` and minibatching). 470 tests pass.

Paired A/B measurement (six alternations of the committed baseline and the candidate):

| Algorithm | Baseline (steps/s) | Candidate (steps/s) | Change (mean) | Per-round change |
|---|---:|---:|---:|---|
| vdn | 399.9 ± 12.9 | 405.9 ± 14.6 | +1.5% | +7.1%, +0.4%, -0.3%, +2.4%, +0.7%, -1.2% |
| mappo | 314.4 ± 10.4 | 359.4 ± 19.7 | +14.3% | +20.8%, +15.6%, +14.4%, +18.0%, +8.9%, +8.1% |
| ippo | 333.7 ± 9.8 | 356.4 ± 21.5 | +6.8% | +11.2%, +8.4%, +9.6%, +4.8%, +3.3%, +3.0% |

All twelve PPO rounds are positive. The 20 epochs per update no longer rebuild ten fields from Python objects and copy them to the
GPU; they perform one gather per field on device instead. MAPPO benefits more than IPPO because it also uses `states` for the mixer.
VDN does not use minibatches and shows only noise.

**Decision:** keep (committed).

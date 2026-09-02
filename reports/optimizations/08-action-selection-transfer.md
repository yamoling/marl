# 8. Single pinned transfer for per-step action selection

**Targets:** all algorithms (action selection is 19–20% of GPU host time). **Files:** `src/marl/agents/simple_agent.py`,
`src/marl/agents/qlearning/dqn_agent.py`, `src/marl/models/nn/qnetwork.py` (`QNetwork.qvalues`).

## Evidence

Each environment step transfers the observation (`data`, `extras`, and for actors `available_actions`) as two or three separate
pageable H→D copies, runs the compiled network on a batch of one, and pulls the result back with `.numpy(force=True)`, which is a
synchronizing D→H copy. The D→H sync is unavoidable (the environment needs the action), but the H→D side issues several small
pageable copies per step and the observation NumPy arrays may not be contiguous/float32, forcing extra conversions.

## Description

Reduce per-step transfer overhead without changing the runner:

- Keep a reusable **pinned host staging buffer** per agent for `data` and `extras` (allocated on first call from the observation shapes,
  reallocated if shapes change). Copy the observation into the pinned buffers with `torch.from_numpy(...)`/`copy_` and issue
  `.to(device, non_blocking=True)`.
- For `DQNAgent`, evaluate whether computing the argmax/available-action masking on device and transferring only the tiny
  action vector is cheaper than transferring the full Q-value tensor; note that `Policy.get_action` works on NumPy Q-values
  (epsilon-greedy, softmax, ...) and `Action.q_values` is logged, so Q-values must still be returned. Keep the current behaviour if
  no gain is measured.
- For `SimpleAgent`, `available_actions` goes through the same staging path.

This is a smaller step than the "vectorized environment execution" recommended in the profiling report, which requires runner
changes and is out of scope here.

## Validation

- `uv run pytest tests -q` must pass.
- Sampled actions for a fixed seed are unchanged.

## Results

Implementation: a reusable `PinnedStagingBuffer` (lazy pinned host tensor per observation field, `copy_` from NumPy then
`.to(device, non_blocking=True)`) used by `SimpleAgent.choose_action` and `QNetwork.qvalues` when the device is CUDA. All tests
passed (471).

Paired A/B measurement (six alternations of the committed baseline and the candidate):

| Algorithm | Baseline (steps/s) | Candidate (steps/s) | Change (mean) | Per-round change |
|---|---:|---:|---:|---|
| vdn | 404.5 ± 6.5 | 407.6 ± 7.5 | +0.8% | +6.9%, -2.2%, +0.3%, +1.1%, -1.5%, +0.4% |
| mappo | 337.6 ± 19.9 | 344.7 ± 12.8 | +2.1% | +2.3%, +9.3%, +4.6%, +0.0%, -5.9%, +3.4% |
| ippo | 338.7 ± 25.7 | 344.5 ± 9.9 | +1.7% | -2.1%, +14.7%, +2.6%, -3.1%, -2.1%, +2.6% |

The mean changes are within noise and the per-round signs are mixed for every algorithm. Per-step action selection is bound by the
kernel launches of the compiled network and the synchronizing device-to-host read of the action, not by the two or three small
host-to-device copies, so replacing pageable copies with pinned ones is not measurable. A real reduction requires batching several
environments per forward pass (vectorized runner), which is out of scope here.

**Decision:** discarded (no effect). The change was reverted.

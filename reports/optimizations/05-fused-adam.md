# 5. Fused Adam/AdamW on CUDA

**Targets:** MAPPO, IPPO (largest effect), VDN. **Files:** `src/marl/algos/ppo.py`, `src/marl/algos/dqn.py`
(optionally other trainers constructing optimizers).

## Evidence

In the 256-step CUDA operator profile, the AdamW step is the **largest single CUDA cost for PPO (120 ms)**, larger than the
convolution forward or backward, and Adam costs 34.8 ms for VDN. PPO runs 20 optimizer steps per 64 collected transitions. The
default (foreach) implementation launches several kernels per parameter group per step (`mul_`, `addcdiv_`, `lerp_`, `sqrt`, ...);
the fused CUDA implementation performs the whole update in one or two kernels.

## Description

Pass `fused=True` to `torch.optim.Adam`/`torch.optim.AdamW` when the parameters live on CUDA. Because trainers are created on CPU and
moved with `Trainer.to(device)` afterwards, the optimizer cannot know the device at construction time. Options, in order of preference:

1. Construct the optimizer with `fused=True` unconditionally is **not** valid on CPU for older torch versions; instead, override
   `Trainer.to()` in `PPO`/`DQN` (or add a hook in the base class) to rebuild the optimizer with `fused=(device.type == "cuda")` when
   the device changes. The optimizer has no state yet at that point in `simple_run`.
2. Alternatively, lazily create the optimizer on the first `train` call, on the device of the parameters.

Keep the same hyper-parameters (`lr`, `eps=1e-5` for PPO, param groups and their names). Check `torch.optim.AdamW.__init__`
documentation for the `fused` constraints (all parameters must be floating-point CUDA tensors). Checkpoint format: the optimizer
state is not saved by `Trainer.save`, so there is no compatibility concern.

## Validation

- `uv run pytest tests -q` must pass.
- Fused and foreach Adam produce numerically very close (not bit-identical) updates; learning behaviour is unchanged.

## Results

Implementation: `PPO.to()` and `DQN.to()` override `Trainer.to()`: networks are moved first (parameters are moved in place, so
identity is preserved), then the optimizer is rebuilt by `_make_optimizer()`/`_make_optimiser()` with the same hyper-parameters
and parameter groups and `fused=(device.type == "cuda")`. The `rmsprop` branch of DQN is unchanged (no fused implementation).
Sanity-checked on GPU (`param_groups[*]["fused"]` is `True`, 200 training steps run for VDN/MAPPO/IPPO); 466 tests pass.

Paired A/B measurement (three alternations of the committed baseline and the candidate, one 2,000-step repetition each):

| Algorithm | Baseline (steps/s) | Candidate (steps/s) | Change (mean) | Per-round change |
|---|---:|---:|---:|---|
| vdn | 380.2 ± 5.3 | 391.0 ± 5.2 | +2.8% | +1.8%, +2.4%, +4.3% |
| mappo | 292.1 ± 2.7 | 312.2 ± 6.0 | +6.9% | +8.1%, +3.5%, +9.1% |
| ippo | 300.1 ± 3.1 | 312.6 ± 18.4 | +4.1% | +4.9%, -2.6%, +10.2% |

Two earlier single-shot comparisons against the stored baseline gave +2–4% and then +5–13% for the same code, which is what motivated
the paired protocol; the paired numbers above are the ones to trust. The gain is consistent in every round for MAPPO and VDN and
positive on average for IPPO (one noisy round). It matches the profiling evidence that the AdamW step was the largest single CUDA cost
of PPO: the fused kernel replaces several elementwise kernels per parameter tensor by one or two kernels per step.

**Decision:** keep (committed).

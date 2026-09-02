# 4. Foreach soft target update

**Targets:** VDN and every DQN-family trainer using `SoftUpdate`. **Files:** `src/marl/algos/qtarget_updater.py`.

## Evidence

`SoftUpdate.update` is **15.2%** of the CPU VDN Python profile (2.95 s). For each parameter tensor it allocates two products and a sum
and then copies the result into the target, i.e. four kernel launches and three temporaries per parameter, 400 times per 2,000 steps.
On CUDA the cost is kernel-launch bound (many small parameters in the CNN + independent MLP).

## Description

Replace the per-parameter arithmetic by a single fused, in-place interpolation:

```python
@torch.no_grad()
def update(self, time_step: int) -> dict[str, float]:
    torch._foreach_lerp_(self._target_params, self._parameters, self.tau)
    return {}
```

`target.lerp_(param, tau)` computes `target + tau * (param - target)`, which is mathematically `(1 - tau) * target + tau * param`.
If relying on the private `torch._foreach_lerp_` is undesirable, a per-tensor `target.lerp_(param, self.tau)` loop is the fallback;
benchmark both. Parameters and targets are all on the same device and dtype, which the foreach API requires.

Also revisit `HardUpdate.update`: `torch._foreach_copy_(targets, params)` avoids the Python loop.

## Validation

- `uv run pytest tests -q` must pass (see `tests/test_qtarget_updater.py`).
- Add/extend a test asserting `torch.allclose` between the old formula and the new one after several updates (tolerance 1e-6).

## Results

Two independent paired A/B sessions (`uv run python .agents/experiments/optimizations/bench.py --ab 04-soft-update --algos vdn mappo ippo`), each alternating the committed baseline and the working tree for 3 rounds:

**Run 1**

| Algorithm | Baseline (steps/s) | Candidate (steps/s) | Change (mean) | Per-round change |
|---|---:|---:|---:|---|
| vdn | 381.4 ± 7.2 | 392.0 ± 5.2 | +2.8% | -0.3%, +5.2%, +3.5% |
| mappo | 313.4 ± 26.8 | 300.1 ± 3.2 | -4.2% | -11.8%, +0.7%, -0.4% |
| ippo | 322.8 ± 14.3 | 312.2 ± 8.1 | -3.3% | -5.4%, -1.2%, -3.1% |

**Run 2**

| Algorithm | Baseline (steps/s) | Candidate (steps/s) | Change (mean) | Per-round change |
|---|---:|---:|---:|---|
| vdn | 385.6 ± 11.2 | 398.3 ± 8.2 | +3.3% | +8.3%, +3.7%, -1.8% |
| mappo | 312.9 ± 13.9 | 314.1 ± 5.5 | +0.4% | +6.7%, -0.7%, -4.4% |
| ippo | 320.9 ± 15.5 | 329.8 ± 5.6 | +2.8% | +8.4%, -3.3%, +3.7% |

VDN, the target of this optimization, is consistently faster in both sessions (+2.8% and +3.3% mean, both above the 2% threshold), even though individual rounds carry enough GPU-clock noise to occasionally show a small negative delta. MAPPO and IPPO — which never touch `SoftUpdate`/`HardUpdate` — showed an apparent regression in run 1 (-4.2%, -3.3%) that fully reversed in run 2 (+0.4%, +2.8%), confirming that swing was measurement noise rather than a real effect of this change, as expected since the optimized code path is not on their hot loop.

**Decision:** keep (committed).

# 9. Channels-last CNN (low priority)

**Targets:** all algorithms using the CNN model bank. **Files:** `src/marl/nn/model_bank/generic.py` (`CNN`), `src/marl/nn/utils.py`
(`make_cnn`).

## Evidence

The CUDA traces contain NCHW↔NHWC conversion kernels emitted by cuDNN/inductor around convolutions. The profiling report ranks their
cost below synchronization, transfers, `bmm` and optimizer work, hence the low priority.

## Description

Convert the CNN weights to `torch.channels_last` memory format (`self.cnn.to(memory_format=torch.channels_last)` after construction
and after `.to(device)`) and convert the input in `CNN.forward` with `obs.contiguous(memory_format=torch.channels_last)` before the
convolution stack. The flattened output is unaffected. Verify the result under `torch.compile` with `fullgraph=True`, which is how
the trainer runs.

## Validation

- `uv run pytest tests -q` must pass.
- Outputs identical to the NCHW path up to float tolerance.

## Results

`CNN.__post_init__` now converts the conv stack to `torch.channels_last` (`self.cnn.to(memory_format=torch.channels_last)`), `CNN.to`
is overridden to re-apply channels-last after a device move, and `CNN.forward` calls `obs.contiguous(memory_format=torch.channels_last)`
on the reshaped NCHW input before the conv stack. Correctness was verified on CUDA with a throwaway script: conv weights are
channels-last, the `torch.compile(fullgraph=True)` path runs without graph breaks, and its output is bit-identical (max abs diff `0.0`)
to the eager NCHW path. `uv run pytest tests -q` passes (471 passed).

Paired A/B benchmark, 6 rounds, `.agents/experiments/optimizations/bench.py --ab 09-channels-last --algos vdn mappo ippo`:

| Algorithm | Baseline (steps/s) | Candidate (steps/s) | Change (mean) | Per-round change |
|---|---:|---:|---:|---|
| vdn | 413.6 ± 5.0 | 408.0 ± 4.0 | -1.4% | -4.2%, +0.3%, -1.6%, +0.3%, -0.7%, -2.3% |
| mappo | 356.7 ± 7.3 | 348.4 ± 9.2 | -2.3% | -0.3%, -2.9%, -5.9%, -5.9%, +1.6%, -0.3% |
| ippo | 357.9 ± 10.8 | 350.7 ± 14.1 | -2.0% | +1.2%, -2.6%, -10.6%, +0.2%, -3.7%, +3.9% |

Channels-last is a net regression here, not a gain: all three algorithms lose 1.4-2.3% on average, and the per-round variance is
large enough (e.g. ippo swings from -10.6% to +3.9%) that the sign is not even consistently negative round to round. This matches
the report's own priority assessment: LLE level-6 layered observations produce small feature maps, so the memory-format conversion
overhead (`obs.contiguous(memory_format=torch.channels_last)` every forward pass, plus the extra bookkeeping channels-last adds to
`torch.compile`'s graph) outweighs whatever NCHW<->NHWC cuDNN/inductor kernels it was meant to eliminate.

**Decision:** discarded (net regression, -1.4% to -2.3% mean, no algorithm improved). The change was reverted.

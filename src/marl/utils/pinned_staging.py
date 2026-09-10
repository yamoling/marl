"""
Reusable pinned-memory staging buffer for fast host-to-device transfers.

Used on the per-step action-selection hot path (`SimpleAgent.choose_action`,
`QNetwork.qvalues`) to replace several small pageable H->D copies per step with a single
pinned, non-blocking copy per field.
"""

import numpy as np
import torch


class PinnedStagingBuffer:
    """
    Reusable pinned host staging buffer for a single observation field.

    The pinned buffer is allocated lazily from the shape/dtype of the first NumPy array it
    stages, and reallocated whenever the shape or dtype changes. Each call copies the array
    into the pinned buffer and issues a non-blocking host-to-device copy.

    The returned tensor must be consumed before the buffer is reused (i.e. before the next
    call to `to`), since the non-blocking copy from pinned memory is asynchronous: the buffer
    cannot be safely overwritten until the copy has completed.

    @ai-generated
    """

    def __init__(self):
        self._buffer: torch.Tensor | None = None
        self._np_dtype: np.dtype | None = None

    def to(self, array: np.ndarray, device: torch.device) -> torch.Tensor:
        """
        Stage `array` into the pinned buffer and return a tensor transferred to `device`.

        @ai-generated
        """
        if self._buffer is None or tuple(self._buffer.shape) != array.shape or self._np_dtype != array.dtype:
            self._buffer = torch.empty(array.shape, dtype=torch.from_numpy(array).dtype, pin_memory=True)
            self._np_dtype = array.dtype
        self._buffer.copy_(torch.from_numpy(array))
        return self._buffer.to(device, non_blocking=True)

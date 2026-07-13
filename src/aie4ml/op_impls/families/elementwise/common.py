from __future__ import annotations

from ...utils.precision import storage_bytes_for_spec


def elementwise_vec_size(lhs_precision, device) -> int:
    """Return the AIE-ML vector lane count for elementwise kernels.

    AIE-ML vector registers are 512-bit (64 bytes).  The lane count is
    the number of elements that fit in one full register:
      int8 → 64, int16/bfloat16 → 32, int32/float → 16.
    """
    lhs_bytes = storage_bytes_for_spec(lhs_precision)
    return int(device.vector_bytes) // max(1, int(lhs_bytes))

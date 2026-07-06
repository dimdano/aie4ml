from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from ...utils import ParallelismConfig, TensorView


@dataclass(frozen=True)
class LayerNormConfig:
    """Resolved configuration for a fully-integer LayerNorm.

    cols must be a power of two and a multiple of vec_size;
    outer extent must be exactly partitionable across cas_num kernels.
    """

    precision: Dict[str, Any]
    parallelism: ParallelismConfig
    rows: int
    cols: int
    vec_size: int
    gamma_shift: int
    out_shift: int
    eps_q0: int
    isqrt_nr_iters: int
    use_aie_invsqrt: bool
    rounding_mode: Optional[str]
    io_views: Dict[str, TensorView]
    io_route: Dict[str, Any]

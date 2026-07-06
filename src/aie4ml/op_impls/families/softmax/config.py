from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from ...utils import ParallelismConfig, TensorView


@dataclass(frozen=True)
class SoftmaxConfig:
    precision: Dict[str, Any]
    parallelism: ParallelismConfig
    param_sets: int
    vec_size: int
    inv_shift: int
    use_clb: bool
    io_views: Dict[str, TensorView]
    io_route: Dict[str, Any]
    hccs: Dict[str, Any]

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ...utils import ParallelismConfig, TensorView
from ..matmul.config import MatmulMicrotileConfig


@dataclass(frozen=True)
class Conv2dFlags:
    use_relu: bool
    emit_flattened: bool  # write the result as a Dense LHS row (the folded flatten_2d view)


@dataclass(frozen=True)
class Conv2dConfig:
    """Resolved Conv2D: the canonical attributes plus the padded frames its kernel reads/writes."""

    precision: Dict[str, Any]
    parallelism: ParallelismConfig
    microtiling: MatmulMicrotileConfig
    io_views: Dict[str, TensorView]
    io_route: Dict[str, Any]
    shift: int
    accumulator_tag: Optional[str]
    rounding_mode: Optional[str]
    spatial_blocks: int  # mmul row tiles per accumulator set (2 on AIE, 4 on AIE-ML/MLv2)
    kernel_shape: Tuple[int, int]
    strides: Tuple[int, int]
    dilations: Tuple[int, int]
    pads: Tuple[int, int, int, int]  # (top, left, bottom, right)
    groups: int
    alternating_horizontal: bool
    flags: Conv2dFlags

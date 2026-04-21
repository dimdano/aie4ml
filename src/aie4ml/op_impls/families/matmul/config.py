from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from ...utils import TensorView


@dataclass(frozen=True)
class MatmulParallelismConfig:
    cas_length: int
    cas_num: int


@dataclass(frozen=True)
class MatmulMicrotileConfig:
    """aie::mmul intrinsic microtile dimensions.

    This is the smallest granularity: data must be rearranged to match these
    shapes before the aie::mmul intrinsic is called .
    """

    microtile_m: int
    microtile_k: int
    microtile_n: int


@dataclass(frozen=True)
class DenseFlags:
    use_relu: bool
    transpose_lhs: bool
    use_bias: bool


@dataclass(frozen=True)
class MatmulFlags:
    transpose_lhs: bool
    transpose_rhs: bool


@dataclass(frozen=True)
class DenseConfig:
    precision: Dict[str, Any]
    parallelism: MatmulParallelismConfig
    microtiling: MatmulMicrotileConfig
    io_views: Dict[str, TensorView]
    io_route: Dict[str, Any]
    shift: int
    accumulator_tag: Optional[str]
    rounding_mode: Optional[str]
    flags: DenseFlags


@dataclass(frozen=True)
class MatmulConfig:
    precision: Dict[str, Any]
    parallelism: MatmulParallelismConfig
    microtiling: MatmulMicrotileConfig
    io_views: Dict[str, TensorView]
    io_route: Dict[str, Any]
    shift: int
    accumulator_tag: Optional[str]
    rounding_mode: Optional[str]
    flags: MatmulFlags

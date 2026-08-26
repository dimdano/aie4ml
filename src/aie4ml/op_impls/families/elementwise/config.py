from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ...utils import MicrotileShape, ParallelismConfig, TensorView


@dataclass(frozen=True)
class AddFlags:
    transpose_lhs: bool
    transpose_rhs: bool


@dataclass(frozen=True)
class AddConfig:
    precision: Dict[str, Any]
    parallelism: ParallelismConfig
    vec_size: int
    io_views: Dict[str, TensorView]
    io_route: Dict[str, Any]
    shift: int
    accumulator_tag: Optional[str]
    rounding_mode: Optional[str]
    preserved_staging: Optional[Tuple[Dict[str, Any], ...]] = None
    #: Inputs `preserved_staging` describes verbatim, so they hand over with no memtile.
    preserved_tensors: Tuple[str, ...] = ()
    flags: AddFlags = AddFlags(transpose_lhs=False, transpose_rhs=False)
    microtile: Optional[MicrotileShape] = None

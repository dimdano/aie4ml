from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ...utils import TensorView


@dataclass(frozen=True)
class ElementwiseParallelismConfig:
    cas_num: int
    cas_length: int = 1


@dataclass(frozen=True)
class AddConfig:
    precision: Dict[str, Any]
    parallelism: ElementwiseParallelismConfig
    vec_size: int
    io_views: Dict[str, TensorView]
    io_route: Dict[str, Any]
    shift: int
    accumulator_tag: Optional[str]
    rounding_mode: Optional[str]
    staging_contract: str = 'outer'
    preserved_staging: Optional[Tuple[Dict[str, Any], ...]] = None

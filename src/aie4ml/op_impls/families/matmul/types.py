from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MatmulParallelismConfig:
    """Matmul-family parallel execution parameters: cascade chain geometry.

    `cas_length` is the number of AIE cores along the cascade (columns).
    `cas_num` is the number of parallel chains (rows).
    """

    cas_length: int
    cas_num: int


@dataclass(frozen=True)
class MatmulTilingConfig:
    """Matmul-family kernel tile sizes `(M, K, N)` for aie::mmul intrinsics."""

    tile_m: int
    tile_k: int
    tile_n: int

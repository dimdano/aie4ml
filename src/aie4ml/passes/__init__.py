"""Optimizer passes specific to the AIE backend."""

from ..frontends.hls4ml.lower import LowerToAieIr
from .compact_buffer_rank import CompactBufferRank
from .fanout_legalize import LegalizeFanoutEntries
from .fold_apply_alpha import FoldApplyAlpha
from .fold_transpose import FoldTransposeViews
from .force_float_mode import ForceFloatMode
from .fuse_activation import FuseActivationCasts
from .memory_plan import BuildMemoryPlan, CollectMemoryEntries, MaterializeMemoryPlan
from .memtile_legalize import LegalizeMemtilePortLimits
from .pack import PackKernelArtifacts
from .placement import PlaceKernels
from .resolve import Resolve

__all__ = [
    'LowerToAieIr',
    'FuseActivationCasts',
    'FoldApplyAlpha',
    'ForceFloatMode',
    'FoldTransposeViews',
    'CompactBufferRank',
    'LegalizeFanoutEntries',
    'LegalizeMemtilePortLimits',
    'Resolve',
    'PackKernelArtifacts',
    'PlaceKernels',
    'CollectMemoryEntries',
    'MaterializeMemoryPlan',
    'BuildMemoryPlan',
]

"""Optimizer passes specific to the AIE backend."""

from .fold_transpose import FoldTransposeViews
from .fuse_activation import FuseActivationCasts
from .lower import LowerToAieIr
from .memory_plan import BuildMemoryPlan
from .pack import PackKernelArtifacts
from .placement import PlaceKernels
from .quant import IntegerQuantizer
from .resolve import Resolve

__all__ = [
    'LowerToAieIr',
    'IntegerQuantizer',
    'FuseActivationCasts',
    'FoldTransposeViews',
    'Resolve',
    'PackKernelArtifacts',
    'PlaceKernels',
    'BuildMemoryPlan',
]

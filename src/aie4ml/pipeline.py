"""Core AIE compiler pipeline definition."""

from .passes import (
    ClassifyTransportEntries,
    CollectMemoryEntries,
    CompactBufferRank,
    FoldApplyAlpha,
    FoldScale,
    FoldViewOps,
    ForceFloatMode,
    FuseActivationCasts,
    LegalizeFanoutEntries,
    LegalizeMemtilePortLimits,
    MaterializeMemoryPlan,
    PackKernelArtifacts,
    PlaceKernels,
    Resolve,
)

HLS4ML_FLOW_SPEC = (
    ('force_float', ForceFloatMode),
    ('fold_apply_alpha', FoldApplyAlpha),
    ('fold_scale', FoldScale),
    ('fuse', FuseActivationCasts),
    ('fold_views', FoldViewOps),
    ('resolve', Resolve),
    ('pack', PackKernelArtifacts),
    ('memory_collect', CollectMemoryEntries),
    ('fanout_legalize', LegalizeFanoutEntries),
    ('transport_classify', ClassifyTransportEntries),
    ('placement', PlaceKernels),
    ('memtile_legalize', LegalizeMemtilePortLimits),
    ('memory_plan', MaterializeMemoryPlan),
    ('compact_batch', CompactBufferRank),
)

DEFAULT_PIPELINE = tuple(pass_cls for _, pass_cls in HLS4ML_FLOW_SPEC)

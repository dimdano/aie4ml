# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""Backend context shared across AIE passes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .graph import AIEPipelineIR

CONTEXT_ATTR = '_aie_backend_context'


@dataclass
class TraitDefinition:
    """Describes an optional capability attached to IR nodes."""

    name: str
    dialects: Tuple[str, ...]
    fields: Tuple[str, ...] = ()
    description: str = ''

    def supports(self, dialect: str) -> bool:
        return not self.dialects or dialect in self.dialects


@dataclass
class TraitRegistry:
    """Central registry of trait definitions."""

    _traits: Dict[str, TraitDefinition] = field(default_factory=dict)

    def register(self, trait: TraitDefinition) -> None:
        self._traits[trait.name] = trait

    def get(self, name: str) -> TraitDefinition:
        try:
            return self._traits[name]
        except KeyError as exc:
            raise KeyError(f'Unknown trait "{name}".') from exc

    def supported_for(self, dialect: str) -> List[TraitDefinition]:
        return [trait for trait in self._traits.values() if trait.supports(dialect)]


@dataclass
class BackendPolicies:
    """Policies steering graph lowering and transformation stages."""

    fusion: Dict[str, Any] = field(default_factory=dict)
    decomposition: Dict[str, Any] = field(default_factory=dict)
    pack: Dict[str, Any] = field(default_factory=dict)
    cache: Dict[str, Any] = field(default_factory=dict)
    tensors_have_batch: bool = False


@dataclass
class DeviceSpec:
    """Model-level device specification published to passes."""

    platform: str
    generation: str
    columns: int
    rows: int
    column_start: int
    row_start: int
    plio_width_bits: int
    bank_mem_bytes: int
    max_mem_in_ports: int
    max_mem_out_ports: int
    dialect: str
    vector_bytes: int = 64
    # PL on-chip budget for the data mover preload buffers, as block geometry. The buffers are
    # bound to URAM or BRAM depending on PLMemory, so both pools are carried here; system
    # planning picks the matching one. Sourced from the catalog's "UltraRAM"/"BlockRAM" entries;
    # 0 when the device does not declare a pool (only hardware-target system planning uses these).
    uram_total_bytes: int = 0
    uram_block_bytes: int = 0
    uram_blocks: int = 0
    bram_block_bytes: int = 0
    bram_blocks: int = 0
    # Per-block geometry (one RAM primitive): Depth x WidthBits. A 512-bit data-mover word is
    # width-pinned to ceil(512/WidthBits) blocks and its depth rounds up to Depth. Defaults are
    # the Versal AIE-ML values (URAM288 = 4096x72, RAMB36 SDP = 512x72) when a catalog omits them.
    uram_depth: int = 4096
    uram_width_bits: int = 72
    bram_depth: int = 512
    bram_width_bits: int = 72

    @classmethod
    def from_config(cls, platform: str, cfg: Dict[str, Any]) -> 'DeviceSpec':
        def _require_int(source: Dict[str, Any], key: str) -> int:
            if key not in source:
                raise KeyError(f'AIEConfig missing "{key}".')
            return int(source[key])

        def _require_bank_mem_bytes(source: Dict[str, Any]) -> int:
            if 'BankMemBytes' not in source:
                raise KeyError('AIEConfig Memory missing "BankMemBytes".')
            return int(source['BankMemBytes'])

        uram = cfg.get('UltraRAM', {}) or {}
        bram = cfg.get('BlockRAM', {}) or {}

        return cls(
            platform=platform,
            generation=str(cfg['Generation']),
            columns=_require_int(cfg, 'Columns'),
            rows=_require_int(cfg, 'Rows'),
            column_start=_require_int(cfg, 'ColumnStart'),
            row_start=_require_int(cfg, 'RowStart'),
            plio_width_bits=_require_int(cfg, 'PLIOWidthBits'),
            bank_mem_bytes=_require_bank_mem_bytes(cfg['Memory']),
            max_mem_in_ports=_require_int(cfg, 'MaxMemTileInPorts'),
            max_mem_out_ports=_require_int(cfg, 'MaxMemTileOutPorts'),
            dialect=detect_dialect(str(cfg['Generation'])),
            vector_bytes=int(cfg.get('VectorBytes', 64)),
            uram_total_bytes=int(uram.get('TotalBytes', 0)),
            uram_block_bytes=int(uram.get('BlockBytes', 0)),
            uram_blocks=int(uram.get('Blocks', 0)),
            bram_block_bytes=int(bram.get('BlockBytes', 0)),
            bram_blocks=int(bram.get('Blocks', 0)),
            uram_depth=int(uram.get('Depth', 4096)),
            uram_width_bits=int(uram.get('WidthBits', 72)),
            bram_depth=int(bram.get('Depth', 512)),
            bram_width_bits=int(bram.get('WidthBits', 72)),
        )


@dataclass
class ProjectConfig:
    """Project-level config populated during lowering; consumed by writer, simulation, and build."""

    output_dir: Path
    project_name: str
    stamp: Optional[str]
    custom_sources: Dict[str, str]


def detect_dialect(generation: str) -> str:
    norm = (generation or '').upper()
    if any(token in norm for token in ('AIE-ML', 'AIE-MLV2', 'XDNA', 'AIE2')):
        return 'AIE2'
    return 'AIE'


@dataclass
class AIEBackendContext:
    """Container carrying IR graph, device spec, traits and policies."""

    device: DeviceSpec
    policies: BackendPolicies
    project_config: ProjectConfig
    aie_config: Dict[str, Any] = field(default_factory=dict)
    traits: TraitRegistry = field(default_factory=TraitRegistry)
    ir: AIEPipelineIR = field(default_factory=AIEPipelineIR)

    def reset_ir(self) -> None:
        self.ir.reset()


def ensure_backend_context(model, factory: Callable[[], AIEBackendContext]) -> AIEBackendContext:
    """Return the shared backend context, creating it if needed."""
    ctx = getattr(model, CONTEXT_ATTR, None)
    if ctx is None:
        ctx = factory()
        setattr(model, CONTEXT_ATTR, ctx)
    return ctx


def get_backend_context(model_or_ctx: Union[Any, AIEBackendContext]) -> AIEBackendContext:
    if isinstance(model_or_ctx, AIEBackendContext):
        return model_or_ctx
    ctx = getattr(model_or_ctx, CONTEXT_ATTR, None)
    if ctx is None:
        raise RuntimeError('AIE backend context missing. Run lowering before invoking downstream passes.')
    return ctx

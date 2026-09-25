# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""Backend context shared across AIE passes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from .graph import AIEPipelineIR

CONTEXT_ATTR = '_aie_backend_context'


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
    core_stream_inputs: int
    core_stream_outputs: int
    stream_switch_width_bits: int
    cascade_width_bits: int
    bank_mem_bytes: int
    max_mem_in_ports: int
    max_mem_out_ports: int
    dialect: str
    vector_bytes: int = 64
    has_memtile: bool = True
    bank_count: int = 4
    tile_mem_bytes: int = 65536
    cascade_layout: str = 'uniform_right'
    aie_compiler_target: str = 'platform'
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
        compiler_target = str(cfg.get('AIECompilerTarget', 'platform')).lower()
        if compiler_target not in ('part', 'platform'):
            raise ValueError(f'Unsupported AIECompilerTarget {compiler_target!r}; expected "part" or "platform".')
        bank_mem_bytes = _require_bank_mem_bytes(cfg['Memory'])
        bank_count = int(cfg.get('BankCount', 4))

        return cls(
            platform=platform,
            generation=str(cfg['Generation']),
            columns=_require_int(cfg, 'Columns'),
            rows=_require_int(cfg, 'Rows'),
            column_start=_require_int(cfg, 'ColumnStart'),
            row_start=_require_int(cfg, 'RowStart'),
            plio_width_bits=_require_int(cfg, 'PLIOWidthBits'),
            core_stream_inputs=_require_int(cfg, 'CoreStreamInputs'),
            core_stream_outputs=_require_int(cfg, 'CoreStreamOutputs'),
            stream_switch_width_bits=_require_int(cfg, 'StreamSwitchWidthBits'),
            cascade_width_bits=_require_int(cfg, 'CascadeWidthBits'),
            bank_mem_bytes=bank_mem_bytes,
            max_mem_in_ports=_require_int(cfg, 'MaxMemTileInPorts'),
            max_mem_out_ports=_require_int(cfg, 'MaxMemTileOutPorts'),
            dialect=detect_dialect(str(cfg['Generation'])),
            vector_bytes=int(cfg.get('VectorBytes', 64)),
            has_memtile=bool(cfg.get('HasMemTile', True)),
            bank_count=bank_count,
            tile_mem_bytes=int(cfg.get('TileMemBytes', bank_count * bank_mem_bytes)),
            cascade_layout=str(cfg.get('CascadeLayout', 'uniform_right')),
            aie_compiler_target=compiler_target,
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
    norm = (generation or '').strip().upper()
    if norm == 'AIE':
        return 'AIE'
    if norm in ('AIE-ML', 'AIE-MLV2'):
        return 'AIE2'
    raise ValueError(f'Unknown AIE generation {generation!r}; expected one of "AIE", "AIE-ML", or "AIE-MLV2".')


@dataclass
class AIEBackendContext:
    """Container carrying IR graph, device spec and policies."""

    device: DeviceSpec
    policies: BackendPolicies
    project_config: ProjectConfig
    aie_config: Dict[str, Any] = field(default_factory=dict)
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

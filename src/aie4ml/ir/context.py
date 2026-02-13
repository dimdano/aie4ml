# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""Backend context shared across AIE passes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

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

    name: str
    generation: str
    columns: int
    rows: int
    column_start: int
    row_start: int
    plio_width_bits: int
    weight_mem_bytes: int
    max_mem_in_ports: int
    max_mem_out_ports: int
    dialect: str

    @classmethod
    def from_config(cls, name: str, cfg: Dict[str, Any]) -> 'DeviceSpec':
        def _require_int(source: Dict[str, Any], key: str) -> int:
            if key not in source:
                raise KeyError(f'AIEConfig missing "{key}".')
            return int(source[key])

        return cls(
            name=name,
            generation=str(cfg['Generation']),
            columns=_require_int(cfg, 'Columns'),
            rows=_require_int(cfg, 'Rows'),
            column_start=_require_int(cfg, 'ColumnStart'),
            row_start=_require_int(cfg, 'RowStart'),
            plio_width_bits=_require_int(cfg, 'PLIOWidthBits'),
            weight_mem_bytes=_require_int(cfg['Memory'], 'WeightMemBytes'),
            max_mem_in_ports=_require_int(cfg, 'MaxMemTileInPorts'),
            max_mem_out_ports=_require_int(cfg, 'MaxMemTileOutPorts'),
            dialect=detect_dialect(str(cfg['Generation'])),
        )


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


def get_backend_context(model) -> AIEBackendContext:
    ctx = getattr(model, CONTEXT_ATTR, None)
    if ctx is None:
        raise RuntimeError('AIE backend context missing. Run lowering before invoking downstream passes.')
    return ctx

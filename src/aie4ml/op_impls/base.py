from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from ..ir.graph import ExecutionValue, OpImplInstance, OpNode
from .common_types import PORT_KIND_STREAM, PortMap


@dataclass(frozen=True)
class BufferLocation:
    """One transport-visible buffer's footprint-relative location.

    Only an op's input and output buffers -- what a direct edge connects, per port -- are listed;
    weights, stacks and cascade resources stay the op graph's own business. The op's graph pins every
    listed buffer where it says, ping and pong one per bank, so each copy must fit one bank.
    """

    port_group: str
    port: int
    rel_col: int
    rel_row: int
    banks: Tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.port_group or self.port < 0:
            raise ValueError('Buffer locations require a port group and non-negative port index.')
        if (
            not self.banks
            or len(self.banks) > 2
            or len(set(self.banks)) != len(self.banks)
            or any(bank not in range(4) for bank in self.banks)
        ):
            raise ValueError(f'Invalid ADF bank set {self.banks}.')


@dataclass(frozen=True)
class RowFlow:
    """How one row of an op's kernels hands activations on, from the device's memory reach.

    Data flows left to right, except along a cascade on a row whose cascade runs right to left (odd
    rows on AIE), where the chain is reversed. Each hand-over buffer lives in the neighbouring tile
    both kernels reach: the upstream one where the reading kernel reaches it, otherwise the reading
    kernel's own. `input_col` offsets the tile holding a kernel's input from that kernel's column;
    `output_col` offsets the tile holding a chain's output from its last kernel's column.
    """

    reversed: bool
    input_col: int
    output_col: int


def row_flow(alternating_horizontal: bool, row: int, cas_length: int) -> RowFlow:
    """The flow on absolute row `row`. On AIE (`alternating_horizontal`) odd-row cores reach their
    east neighbour's memory and even-row cores their west's; on AIE-ML every core reaches west."""
    reaches_east = bool(alternating_horizontal and int(row) % 2)
    if reaches_east and int(cas_length) > 1:
        return RowFlow(reversed=True, input_col=1, output_col=0)
    if reaches_east:
        return RowFlow(reversed=False, input_col=0, output_col=1)
    return RowFlow(reversed=False, input_col=-1, output_col=0)


@dataclass(frozen=True)
class LayoutConversion:
    """A kernel graph that re-lays one input before the op reads it.

    The op reads `target`, a value that exists only in the execution graph and that its ports bind,
    instead of `source`; `variant` builds the converting graph from `config`, and `name` is that
    graph's instance name. `shared_memory` requires the hand-over to the op to pass through a
    memory both kernels reach, never a DMA -- a constraint the converter sets when a copy would be
    illegal or is not accounted for, not merely preferred.
    """

    name: str
    source: str
    target: str
    variant: 'OpImplVariant'
    config: Any
    shared_memory: bool


@dataclass(frozen=True)
class OpImplFootprint:
    """Rectangular tile footprint required by an op implementation."""

    width: int
    height: int
    extras: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f'Footprint dimensions must be positive, got {self.width}x{self.height}.')


class OpImplVariant:
    """Self-contained compilation unit for one op variant.

    Each subclass owns the full lifecycle: selection (matches + plevel),
    configuration (resolve), verification (validate_config), and code
    generation (build_template_params, build_ports, footprint, pack, get_artifacts).
    """

    variant_id: ClassVar[str] = ''
    op_type: ClassVar[str] = ''
    graph_header: ClassVar[str] = ''
    graph_name: ClassVar[str] = ''
    param_template: ClassVar[str] = ''
    plevel: ClassVar[int] = 10  # higher value = higher selection priority
    kernel_transposes_microtile: ClassVar[bool] = False

    def matches(self, _node: OpNode, _device: Any) -> bool:
        raise NotImplementedError

    def resolve(self, _node: OpNode, _device: Any, _directives: Optional[Dict[str, Any]] = None) -> Any:
        raise NotImplementedError

    def validate_config(self, _node: OpNode, _config: Any, _device: Any) -> None:
        """Post-lowering attribute verifier. Override to enforce kernel ABI rules."""

    def build_template_params(self, _node: OpNode, config: Any, _placement: Dict[str, int]) -> Dict[str, Any]:
        return config

    def input_conversions(
        self, _node: OpNode, _config: Any, _sources: Dict[str, ExecutionValue]
    ) -> Tuple['LayoutConversion', ...]:
        """Inputs this op reads in a layout their source does not write, each converted by a kernel
        graph of its own that the layout legalization pass inserts ahead of the op.

        `sources` maps each tensor the op reads to its execution value -- what actually writes it
        (a kernel, a folded view, or the graph boundary), which decides the layout it arrives in.
        """
        return ()

    def buffer_locations(self, _node: OpNode, _config: Any, _anchor_row: int) -> Tuple[BufferLocation, ...]:
        """Return transport-visible buffers relative to an op anchor.

        Repeated group/port pairs describe multicast graph ports. An edge that must pass through
        shared memory needs both of its ports listed, at the same place: placement accepts only
        positions where they coincide.
        """
        return ()

    def output_staging_contract(self, _node: OpNode, _config: Any, _tensor_name: str) -> Optional[str]:
        return None

    def output_port_count(self, _node: OpNode, config: Any) -> Optional[int]:
        return int(config.parallelism.cas_num)

    def pack(self, inst: OpImplInstance) -> Dict[str, Any]:
        raise NotImplementedError

    def get_artifacts(self, inst: OpImplInstance) -> List[Dict[str, Any]]:
        return []

    def input_precision(self, config: Any, role: str) -> Any:
        return config.precision[role]

    def output_precision(self, config: Any) -> Any:
        return config.precision['output']

    def describe_output_staging(
        self, _node: OpNode, _config: Any, _tensor_name: str, _port: int, _buf_dims: Any = None
    ) -> Any:
        return None

    def describe_input_staging(
        self,
        _consumer: OpNode,
        _config: Any,
        _tensor_name: str,
        _port: int,
        _buf_dims: Any = None,
        _producer: Optional[OpNode] = None,
    ) -> Any:
        return None

    def footprint(self, node: OpNode, config: Any) -> OpImplFootprint:
        raise NotImplementedError

    def build_ports(self, _node: OpNode, _config: Any) -> PortMap:
        """Assemble the PortMap for this variant.  Must be overridden."""
        raise NotImplementedError

    def validate_ports(self, node: OpNode, ports: PortMap, device: Any) -> None:
        """Every kernel of an op sees at most one port per group, so the stream groups must fit
        the core's stream ports (two in/out on AIE, one in/out on AIE-ML)."""
        for direction, bindings, budget in (
            ('input', ports.inputs, int(device.core_stream_inputs)),
            ('output', ports.outputs, int(device.core_stream_outputs)),
        ):
            streams = [name for name, binding in bindings.items() if binding.kind == PORT_KIND_STREAM]
            if len(streams) > budget:
                raise ValueError(
                    f'{node.name}: {self.variant_id} needs {len(streams)} {direction} stream ports per kernel '
                    f'({", ".join(streams)}) but {device.platform} cores have {budget}.'
                )

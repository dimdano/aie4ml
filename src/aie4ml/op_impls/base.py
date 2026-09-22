from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from ..ir.graph import OpImplInstance, OpNode
from .common_types import PORT_KIND_STREAM, PortMap


@dataclass(frozen=True)
class BufferLocation:
    """One transport-visible buffer's footprint-relative location."""

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

    def buffer_locations(self, _node: OpNode, _config: Any, _anchor_row: int) -> Tuple[BufferLocation, ...]:
        """Return transport-visible buffers relative to an op anchor.

        Repeated group/port pairs describe multicast graph ports.
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

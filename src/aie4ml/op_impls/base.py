from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from ..ir.graph import OpImplInstance, OpNode
from .common_types import PortMap


@dataclass(frozen=True)
class OpImplFootprint:
    """Rectangular tile footprint required by an op implementation."""

    width: int
    height: int
    extras: Dict[str, Any] = field(default_factory=dict)


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

    def matches(self, _node: OpNode, _device: Any) -> bool:
        raise NotImplementedError

    def resolve(self, _node: OpNode, _device: Any, _directives: Optional[Dict[str, Any]] = None) -> Any:
        raise NotImplementedError

    def validate_config(self, _node: OpNode, _config: Any, _device: Any) -> None:
        """Post-lowering attribute verifier. Override to enforce kernel ABI rules."""

    def build_template_params(self, _node: OpNode, config: Any) -> Dict[str, Any]:
        return config

    def output_staging_contract(self, _node: OpNode, _config: Any, _tensor_name: str) -> Optional[str]:
        return None

    def output_port_count(self, _node: OpNode, config: Any) -> Optional[int]:
        return int(config.parallelism.cas_num)

    def microtiling_options(self, generation: str, query: Any) -> List[Tuple[int, int, int]]:
        raise NotImplementedError

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

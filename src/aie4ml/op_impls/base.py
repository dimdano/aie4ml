from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from ..ir.graph import OpNode, ResolvedAttributes
from .common_types import PortBinding, PortMap, to_plain


@dataclass(frozen=True)
class OpImplConfig:
    """Frozen per-node implementation config produced during execution lowering.

    Built by `OpImplVariant.build_config()` and stored in `ExecutionIR`.
    `parameters` holds variant-specific typed compile-time config.
    `ports` holds the generic op-implementation port contract.
    """

    variant_id: str
    param_template: str
    graph_header: str
    graph_name: str
    parameters: Any
    ports: PortMap
    io_route: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'variant_id': self.variant_id,
            'param_template': self.param_template,
            'graph_header': self.graph_header,
            'graph_name': self.graph_name,
            'parameters': to_plain(self.parameters),
        }


@dataclass(frozen=True)
class OpImplSelectionContext:
    """Inputs used to test and build an op implementation for a node."""

    node: OpNode
    attributes: ResolvedAttributes
    device_generation: str
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class OpImplPlacementContext:
    """Inputs used to compute an implementation's placement requirements."""

    node: OpNode
    metadata: Dict[str, Any]
    config: OpImplConfig


@dataclass(frozen=True)
class OpImplFootprint:
    """Rectangular tile footprint required by an op implementation.

    Returned by `OpImplVariant.footprint()` and consumed by placement.
    """

    width: int
    height: int


@dataclass
class OpImplVariant:
    """Reusable implementation descriptor for one op type.

    Registered in `OpImplRegistry` and selected at compile time.
    Subclasses define support checks, config construction, packing, and footprint logic.
    """

    variant_id: str
    op_type: str
    supported_generations: Tuple[str, ...] = field(default_factory=tuple)
    supported_precisions: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    supported_input_modes: Tuple[str, ...] = field(default_factory=tuple)
    supported_output_modes: Tuple[str, ...] = field(default_factory=tuple)

    def supports_generation(self, generation: str) -> bool:
        if not self.supported_generations:
            return True
        norm = (generation or '').upper()
        for token in self.supported_generations:
            if token.upper() in norm:
                return True
        return False

    def supports(self, context: OpImplSelectionContext) -> bool:
        if not self.supports_generation(context.device_generation):
            return False

        if self.supported_input_modes:
            for mode in context.attributes.io_route.get('inputs', {}).values():
                if isinstance(mode, str) and mode not in self.supported_input_modes:
                    return False

        if self.supported_output_modes:
            for mode in context.attributes.io_route.get('outputs', {}).values():
                if isinstance(mode, str) and mode not in self.supported_output_modes:
                    return False

        node_prec = _numeric_precisions(context.attributes)
        if self.supported_precisions:
            if not any(all(node_prec.get(k) == v for k, v in spec.items()) for spec in self.supported_precisions):
                return False

        return True

    def build_config(self, context: OpImplSelectionContext) -> OpImplConfig:
        raise NotImplementedError

    def tiling_options(self, generation: str, query: Any):
        raise NotImplementedError

    def pack(self, inst):
        raise NotImplementedError

    def get_artifacts(self, inst):
        return []

    def describe_output_staging(
        self,
        node: OpNode,
        config: 'OpImplConfig',
        tensor_name: str,
        port: int,
        buf_dims=None,
    ):
        return None

    def describe_input_staging(
        self,
        consumer: OpNode,
        config: 'OpImplConfig',
        tensor_name: str,
        port: int,
        buf_dims=None,
        producer: Optional[OpNode] = None,
    ):
        return None

    def footprint(self, context: OpImplPlacementContext) -> OpImplFootprint:
        raise NotImplementedError

    def _build_port_map(
        self,
        context: OpImplSelectionContext,
        input_port_count: int,
        output_port_count: int,
    ) -> PortMap:
        inputs: Dict[str, PortBinding] = {}
        outputs: Dict[str, PortBinding] = {}

        data_inputs = [tensor for tensor in context.node.inputs if not tensor.is_parameter]
        for index, tensor in enumerate(data_inputs):
            inputs[tensor.name] = PortBinding(group=f'in{index+1}', count=input_port_count)

        for index, tensor in enumerate(context.node.outputs):
            outputs[tensor.name] = PortBinding(group=f'out{index+1}', count=output_port_count)

        return PortMap(inputs=inputs, outputs=outputs)


def _numeric_precisions(attrs: ResolvedAttributes) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for kind, dtype in attrs.numeric.items():
        out[kind] = int(dtype.width)
    input_dtype = attrs.numeric.get('input')
    out['input_c_type'] = getattr(input_dtype, 'c_type', '') or ''
    return out

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Mapping, Optional, Tuple

from ..ir.graph import OpNode
from .common_types import PortBinding, PortMap


@dataclass(frozen=True)
class OpImplFootprint:
    """Rectangular tile footprint required by an op implementation."""

    width: int
    height: int
    extras: Dict[str, Any] = field(default_factory=dict)


class OpImplVariant:
    """Reusable implementation descriptor for one op type."""

    variant_id: ClassVar[str] = ''
    op_type: ClassVar[str] = ''
    graph_header: ClassVar[str] = ''
    graph_name: ClassVar[str] = ''
    param_template: ClassVar[str] = ''
    supported_generations: ClassVar[Tuple[str, ...]] = ()
    supported_precisions: ClassVar[Tuple[Dict[str, Any], ...]] = ()
    supported_input_modes: ClassVar[Tuple[str, ...]] = ()
    supported_output_modes: ClassVar[Tuple[str, ...]] = ()

    def supports_generation(self, generation: str) -> bool:
        if not self.supported_generations:
            return True
        norm = (generation or '').upper()
        for token in self.supported_generations:
            if token.upper() in norm:
                return True
        return False

    def supports_io_route(self, io_route: Dict[str, Any]) -> bool:
        if self.supported_input_modes:
            for mode in io_route.get('inputs', {}).values():
                if isinstance(mode, str) and mode not in self.supported_input_modes:
                    return False

        if self.supported_output_modes:
            for mode in io_route.get('outputs', {}).values():
                if isinstance(mode, str) and mode not in self.supported_output_modes:
                    return False

        return True

    def supports_precisions(self, precision: Dict[str, Any]) -> bool:
        if not self.supported_precisions:
            return True
        return any(all(precision.get(k) == v for k, v in spec.items()) for spec in self.supported_precisions)

    def build_template_params(self, node: OpNode, config: Any) -> Any:
        """Return the parameters object passed as ``P`` to the Jinja parameter template.

        Subclasses override this to attach derived shape fields.
        """
        return config

    def output_staging_contract(self, node: OpNode, config: Any, tensor_name: str) -> Optional[str]:
        return None

    def output_port_count(self, node: OpNode, config: Any) -> Optional[int]:
        """Number of ports that partition each output tensor of this variant.

        Default: config.parallelism.cas_num by hardware design convention.
        """
        return int(config.parallelism.cas_num)

    def validate_config(self, node: OpNode, config: Any, device) -> None:
        return None

    def microtiling_options(self, generation: str, query: Any):
        raise NotImplementedError

    def pack(self, inst):
        raise NotImplementedError

    def get_artifacts(self, inst):
        return []

    def input_precision(self, config: Any, role: str):
        return config.precision[role]

    def output_precision(self, config: Any):
        return config.precision['output']

    def describe_output_staging(
        self,
        node: OpNode,
        config: Any,
        tensor_name: str,
        port: int,
        buf_dims=None,
    ):
        return None

    def describe_input_staging(
        self,
        consumer: OpNode,
        config: Any,
        tensor_name: str,
        port: int,
        buf_dims=None,
        producer: Optional[OpNode] = None,
    ):
        return None

    def footprint(self, node: OpNode, config: Any) -> OpImplFootprint:
        raise NotImplementedError

    def build_ports(
        self,
        node: OpNode,
        input_port_count: int | Mapping[str, int],
        output_port_count: int | Mapping[str, int],
    ) -> PortMap:
        inputs: Dict[str, PortBinding] = {}
        outputs: Dict[str, PortBinding] = {}

        def _count(spec: int | Mapping[str, int], tensor_name: str) -> int:
            if isinstance(spec, Mapping):
                if tensor_name not in spec:
                    raise KeyError(f'Missing port count for tensor {tensor_name}.')
                return int(spec[tensor_name])
            return int(spec)

        data_inputs = [tensor for tensor in node.inputs if not tensor.is_parameter]
        for index, tensor in enumerate(data_inputs):
            inputs[tensor.name] = PortBinding(group=f'in{index+1}', count=_count(input_port_count, tensor.name))

        for index, tensor in enumerate(node.outputs):
            outputs[tensor.name] = PortBinding(group=f'out{index+1}', count=_count(output_port_count, tensor.name))

        return PortMap(inputs=inputs, outputs=outputs)

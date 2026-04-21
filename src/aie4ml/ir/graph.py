"""Intermediate representation for the aie4ml backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from ..aie_types import PrecisionIntent

if TYPE_CHECKING:  # pragma: no cover - runtime circular import guard
    from ..op_impls import OpImplVariant


@dataclass
class TensorVar:
    """Logical tensor value with producer/consumer connectivity.

    Parameter tensors (weights, biases) have data set and producer=None.
    Activation tensors have data=None and a producer node.
    """

    name: str
    shape: Tuple[int, ...]
    precision: Optional[PrecisionIntent] = None
    data: Optional[np.ndarray] = None
    producer: Optional['OpNode'] = None
    consumers: List['OpNode'] = field(default_factory=list)

    @property
    def is_parameter(self) -> bool:
        return self.data is not None


def _deep_copy(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _deep_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deep_copy(v) for v in value]
    return value


@dataclass
class TraitInstance:
    """Instance data for a trait that augments an IR node."""

    name: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OpNode:
    """
    Logical IR node capturing semantic computation only.
    """

    name: str
    op_type: str
    dialect: str

    inputs: List[TensorVar] = field(default_factory=list)
    outputs: List[TensorVar] = field(default_factory=list)

    artifacts: Dict[str, Any] = field(default_factory=dict)
    traits: Dict[str, TraitInstance] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    directives: Dict[str, Any] = field(default_factory=dict)

    def add_trait(self, trait: TraitInstance) -> None:
        self.traits[trait.name] = trait

    def trait_data(self, name: str) -> Dict[str, Any]:
        return self.traits.get(name, TraitInstance(name)).data


def input_role_map(node: OpNode) -> Dict[str, str]:
    roles = list(node.metadata.get('input_roles') or [])
    return {tensor.name: str(roles[index]) for index, tensor in enumerate(node.inputs) if index < len(roles)}


def input_role(node: OpNode, tensor_name: str) -> str:
    return input_role_map(node).get(tensor_name, '')


def input_tensor_for_role(node: OpNode, role: str) -> TensorVar:
    for tensor in node.inputs:
        if input_role(node, tensor.name) == role:
            return tensor
    raise ValueError(f'{node.name}: missing {role} tensor.')


@dataclass
class LogicalIR:
    """
    Tensor-centric logical IR graph.

    - tensors: all TensorVars in the graph
    - nodes:   ordered list of logical nodes
    """

    tensors: Dict[str, TensorVar] = field(default_factory=dict)
    nodes: List[OpNode] = field(default_factory=list)
    input_tensor_names: List[str] = field(default_factory=list)
    output_tensor_names: List[str] = field(default_factory=list)

    def add_tensor(self, tensor: TensorVar) -> None:
        if tensor.name in self.tensors:
            raise ValueError(f"Tensor '{tensor.name}' already exists.")
        self.tensors[tensor.name] = tensor

    def add_node(self, node: OpNode) -> None:
        self.nodes.append(node)

    def mark_graph_input(self, tensor_name: str) -> None:
        if tensor_name not in self.tensors:
            raise ValueError(f"Graph input tensor '{tensor_name}' does not exist.")
        if tensor_name not in self.input_tensor_names:
            self.input_tensor_names.append(tensor_name)

    def mark_graph_output(self, tensor_name: str) -> None:
        if tensor_name not in self.tensors:
            raise ValueError(f"Graph output tensor '{tensor_name}' does not exist.")
        if tensor_name not in self.output_tensor_names:
            self.output_tensor_names.append(tensor_name)

    def remove_node(self, node: OpNode, mode: str = 'bypass') -> None:
        if len(node.inputs) == 1 and len(node.outputs) == 1:
            if mode == 'bypass':
                self._bypass_node(node)
            elif mode == 'contract':
                in_tv = node.inputs[0]
                if len(in_tv.consumers) == 1:
                    self._contract_node(node)
                else:
                    raise ValueError(
                        f'Cannot contract node {node.name}: input tensor '
                        f'{in_tv.name} has {len(in_tv.consumers)} consumers.'
                    )
            else:
                raise ValueError(f'Unknown mode: {mode}')
        else:
            self._detach_node(node)

        if node in self.nodes:
            self.nodes.remove(node)
        node.inputs.clear()
        node.outputs.clear()

    def _bypass_node(self, node: OpNode):
        in_tv, out_tv = node.inputs[0], node.outputs[0]

        for consumer in list(out_tv.consumers):
            for i, inp in enumerate(consumer.inputs):
                if inp is out_tv:
                    consumer.inputs[i] = in_tv
            if consumer not in in_tv.consumers:
                in_tv.consumers.append(consumer)

        if node in in_tv.consumers:
            in_tv.consumers.remove(node)

        out_tv.producer = None
        out_tv.consumers.clear()
        self.tensors.pop(out_tv.name, None)

    def _contract_node(self, node: OpNode):
        in_tv, out_tv = node.inputs[0], node.outputs[0]
        producer = in_tv.producer

        if producer:
            for i, outp in enumerate(producer.outputs):
                if outp is in_tv:
                    producer.outputs[i] = out_tv
            out_tv.producer = producer
        else:
            out_tv.producer = None

        if node in in_tv.consumers:
            in_tv.consumers.remove(node)

        in_tv.producer = None
        if not in_tv.consumers:
            self.tensors.pop(in_tv.name, None)

    def _detach_node(self, node: OpNode):
        """Fallback: Just cut the node out without merging tensors."""
        for t in node.inputs:
            if node in t.consumers:
                t.consumers.remove(node)

        for t in node.outputs:
            if t.producer is node and t.consumers:
                raise ValueError(f'Cannot detach node {node.name}: output tensor {t.name} still has consumers.')
            if t.producer is node:
                t.producer = None
            if not t.consumers and t.producer is None:
                self.tensors.pop(t.name, None)

    def graph_inputs(self) -> List[TensorVar]:
        if not self.input_tensor_names:
            raise ValueError('LogicalIR.graph_inputs requires explicit input_tensor_names.')
        return [self.tensors[name] for name in self.input_tensor_names]

    def graph_outputs(self) -> List[TensorVar]:
        if not self.output_tensor_names:
            raise ValueError('LogicalIR.graph_outputs requires explicit output_tensor_names.')
        return [self.tensors[name] for name in self.output_tensor_names]

    def __iter__(self):
        return iter(self.nodes)

    def __len__(self) -> int:
        return len(self.nodes)


STAGING_CONTRACTS: frozenset = frozenset({'outer', 'inner'})
"""Compiler-wide vocabulary of valid 2D execution partition-axis contracts."""


@dataclass(frozen=True)
class TensorContract:
    """Resolved execution contract for a single tensor edge.

    Used by downstream resolvers to inherit compatible partitioning and staging.
    """

    contract: str  # 'outer' | 'inner'
    port_staging: Tuple[Dict[str, Any], ...]


@dataclass
class ExecutionEntry:
    """Materialized execution selection for a logical node."""

    node: OpNode
    variant: 'OpImplVariant'
    ports: Any
    io_route: Dict[str, Any]
    io_views: Dict[str, Any]
    config: Any
    graph_header: str
    graph_name: str
    param_template: str
    artifacts: Dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.node.name

    @property
    def op_type(self) -> str:
        return self.node.op_type


OpImplInstance = ExecutionEntry


@dataclass
class ExecutionIR:
    """Container for selected implementation instances derived from logical nodes."""

    instances: Dict[str, ExecutionEntry] = field(default_factory=dict)
    tensor_contracts: Dict[str, TensorContract] = field(default_factory=dict)

    def register(
        self,
        node: OpNode,
        variant: 'OpImplVariant',
        ports: Any,
        io_route: Dict[str, Any],
        io_views: Dict[str, Any],
        config: Any,
        graph_header: str,
        graph_name: str,
        param_template: str,
    ) -> ExecutionEntry:
        inst = ExecutionEntry(
            node=node,
            variant=variant,
            ports=ports,
            io_route=io_route,
            io_views=io_views,
            config=config,
            graph_header=graph_header,
            graph_name=graph_name,
            param_template=param_template,
        )
        self.instances[node.name] = inst
        return inst

    def get(self, name: str) -> Optional[ExecutionEntry]:
        return self.instances.get(name)

    def clear(self) -> None:
        self.instances.clear()
        self.tensor_contracts.clear()

    def prune(self, active_names: Iterable[str]) -> bool:
        keep = set(active_names)
        removed = False
        for name in list(self.instances.keys()):
            if name not in keep:
                del self.instances[name]
                removed = True
        return removed

    def __iter__(self):
        return iter(self.instances.values())


@dataclass
class PhysicalIR:
    """Physical layer of the IR capturing placement and routing."""

    placements: Dict[str, Dict[str, int]] = field(default_factory=dict)
    plan: Dict[str, Any] = field(default_factory=dict)

    def reset(self) -> None:
        self.placements.clear()
        self.plan.clear()

    def to_dict(self) -> Dict[str, Any]:
        return {
            'placements': _deep_copy(self.placements),
            'plan': _deep_copy(self.plan),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], *, require_plan_buffers: bool = False) -> 'PhysicalIR':
        if not isinstance(data, dict):
            raise RuntimeError('Missing or invalid physical IR section.')

        placements = data.get('placements')
        plan = data.get('plan')
        if not isinstance(placements, dict):
            raise RuntimeError('Physical IR is missing placements.')
        if not isinstance(plan, dict):
            raise RuntimeError('Physical IR is missing plan.')
        if require_plan_buffers and not isinstance(plan.get('buffers'), list):
            raise RuntimeError('Physical IR plan is missing buffers required for IO layout reconstruction.')

        return cls(
            placements=_deep_copy(placements),
            plan=_deep_copy(plan),
        )


@dataclass
class AIEPipelineIR:
    """Three-level IR bundle shared across backend passes."""

    logical: LogicalIR = field(default_factory=LogicalIR)
    execution: ExecutionIR = field(default_factory=ExecutionIR)
    physical: PhysicalIR = field(default_factory=PhysicalIR)

    def reset(self) -> None:
        self.logical = LogicalIR()
        self.execution = ExecutionIR()
        self.physical = PhysicalIR()

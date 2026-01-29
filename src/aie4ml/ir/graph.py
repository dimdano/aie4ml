"""Intermediate representation for the aie4ml backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

from ..aie_types import AIEDataType

if TYPE_CHECKING:  # pragma: no cover - runtime circular import guard
    from ..kernel_registry import KernelVariant


@dataclass
class TensorVar:
    """Logical tensor value with producer/consumer connectivity."""

    name: str
    shape: Tuple[int, ...]
    producer: Optional['OpNode'] = None
    consumers: List['OpNode'] = field(default_factory=list)


@dataclass
class ResolvedAttributes:
    """Bundle of fully resolved attributes consumed by downstream passes."""

    tiling: Dict[str, int] = field(default_factory=dict)
    slices: Dict[str, int] = field(default_factory=dict)
    numeric: Dict[str, AIEDataType] = field(default_factory=dict)
    parallelism: Dict[str, int] = field(default_factory=dict)
    placement: Optional[Dict[str, int]] = None
    pack: Dict[str, Any] = field(default_factory=dict)
    flags: Dict[str, Any] = field(default_factory=dict)
    scalars: Dict[str, Any] = field(default_factory=dict)
    staging: Dict[str, Any] = field(default_factory=dict)
    io_route: Dict[str, Any] = field(default_factory=dict)
    ports: Dict[str, Dict[Tuple[str, int], List[Dict[str, Any]]]] = field(
        default_factory=lambda: {'inputs': {}, 'outputs': {}}
    )

    def copy(self) -> 'ResolvedAttributes':
        placement = None if self.placement is None else dict(self.placement)
        return ResolvedAttributes(
            tiling=dict(self.tiling),
            slices=dict(self.slices),
            numeric=dict(self.numeric),
            parallelism=dict(self.parallelism),
            placement=placement,
            pack=dict(self.pack),
            flags=dict(self.flags),
            scalars=dict(self.scalars),
            staging={k: _deep_copy(v) for k, v in self.staging.items()},
            io_route={k: _deep_copy(v) for k, v in self.io_route.items()},
            ports=_copy_ports(self.ports),
        )

    def ensure_keys(self, keys: Iterable[str], namespace: str) -> None:
        for key in keys:
            if key not in self.__dict__ or self.__dict__[key] in (None, {}):
                raise RuntimeError(f'Missing required {namespace} attribute "{key}" for resolved IR node.')


def _deep_copy(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _deep_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deep_copy(v) for v in value]
    return value


def _copy_ports(
    ports: Dict[str, Dict[Tuple[str, int], List[Dict[str, Any]]]],
) -> Dict[str, Dict[Tuple[str, int], List[Dict[str, Any]]]]:
    copied: Dict[str, Dict[Tuple[str, int], List[Dict[str, Any]]]] = {'inputs': {}, 'outputs': {}}
    for direction in ('inputs', 'outputs'):
        for key, desc_list in ports.get(direction, {}).items():
            copied[direction][key] = [_deep_copy(desc) for desc in desc_list]
    return copied


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

    def add_trait(self, trait: TraitInstance) -> None:
        self.traits[trait.name] = trait

    def trait_data(self, name: str) -> Dict[str, Any]:
        return self.traits.get(name, TraitInstance(name)).data


@dataclass
class LogicalIR:
    """
    Tensor-centric logical IR graph.

    - tensors: all TensorVars in the graph
    - nodes:   ordered list of logical nodes
    """

    tensors: Dict[str, TensorVar] = field(default_factory=dict)
    nodes: List[OpNode] = field(default_factory=list)

    def add_tensor(self, tensor: TensorVar) -> None:
        if tensor.name in self.tensors:
            raise ValueError(f"Tensor '{tensor.name}' already exists.")
        self.tensors[tensor.name] = tensor

    def add_node(self, node: OpNode) -> None:
        self.nodes.append(node)

    def remove_node(self, node: OpNode) -> None:
        if len(node.inputs) > 1 or len(node.outputs) > 1:
            raise Exception('Cannot delete a node with multiple inputs/outputs')
        
        in_tv = node.inputs[0]
        out_tv = node.outputs[0]
        if in_tv.producer is None:
            raise ValueError('Cannot remove input node')
    
        dst_nodes = list(out_tv.consumers)
        for dst in dst_nodes:
            # Replace out_tv with in_tv in dst inputs
            for i, tv in enumerate(dst.inputs):
                if tv is out_tv:
                    dst.inputs[i] = in_tv        
            
            if dst not in in_tv.consumers:
                in_tv.consumers.append(dst)
                
        # Disconnect node from input tensor
        if node in in_tv.consumers:
            in_tv.consumers.remove(node)
            
        # Cleanup output tensor
        out_tv.consumers.clear()
        out_tv.producer = None
        self.tensors.pop(out_tv.name, None)
        
        # Remove node
        self.nodes.remove(node)


    # TODO graph inputs wrong
    def graph_inputs(self) -> List[TensorVar]:
        """Tensors with no producer."""
        return [t for t in self.tensors.values() if t.producer is None]

    def graph_outputs(self) -> List[TensorVar]:
        """Tensors with no consumers."""
        return [t for t in self.tensors.values() if not t.consumers]

    def __iter__(self):
        return iter(self.nodes)

    def __len__(self) -> int:
        return len(self.nodes)


@dataclass
class KernelInstance:
    """Materialized kernel selection for a logical node."""

    node: OpNode
    variant: 'KernelVariant'
    attributes: ResolvedAttributes
    config: Dict[str, Any]
    artifacts: Dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.node.name

    @property
    def op_type(self) -> str:
        return self.node.op_type


@dataclass
class KernelIR:
    """Container for kernel instances derived from logical nodes."""

    instances: Dict[str, KernelInstance] = field(default_factory=dict)

    def register(
        self,
        node: OpNode,
        variant: 'KernelVariant',
        attributes: ResolvedAttributes,
        config: Dict[str, Any],
    ) -> KernelInstance:
        inst = KernelInstance(node=node, variant=variant, attributes=attributes, config=config)
        self.instances[node.name] = inst
        return inst

    def get(self, name: str) -> Optional[KernelInstance]:
        return self.instances.get(name)

    def clear(self) -> None:
        self.instances.clear()

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


@dataclass
class AIEPipelineIR:
    """Three-level IR bundle shared across backend passes."""

    logical: LogicalIR = field(default_factory=LogicalIR)
    kernels: KernelIR = field(default_factory=KernelIR)
    physical: PhysicalIR = field(default_factory=PhysicalIR)

    def reset(self) -> None:
        self.logical = LogicalIR()
        self.kernels = KernelIR()
        self.physical = PhysicalIR()

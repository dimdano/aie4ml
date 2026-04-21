"""Public exports for the AIE intermediate representation."""

from .context import (
    AIEBackendContext,
    BackendPolicies,
    DeviceSpec,
    TraitDefinition,
    TraitRegistry,
    ensure_backend_context,
    get_backend_context,
)
from .graph import (
    AIEPipelineIR,
    ExecutionEntry,
    ExecutionIR,
    LogicalIR,
    OpImplInstance,
    OpNode,
    PhysicalIR,
    TensorVar,
    TraitInstance,
    input_role,
    input_role_map,
    input_tensor_for_role,
)

__all__ = [
    'AIEBackendContext',
    'AIEPipelineIR',
    'ExecutionEntry',
    'ExecutionIR',
    'input_role',
    'input_role_map',
    'input_tensor_for_role',
    'LogicalIR',
    'PhysicalIR',
    'OpImplInstance',
    'OpNode',
    'TensorVar',
    'BackendPolicies',
    'DeviceSpec',
    'TraitDefinition',
    'TraitInstance',
    'TraitRegistry',
    'ensure_backend_context',
    'get_backend_context',
]

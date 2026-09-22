from __future__ import annotations

from typing import Any, Dict

from ..ir import TraitDefinition
from ..ir.graph import TENSOR_LAYOUTS
from ..op_impls.common_types import PORT_KINDS


def register_default_traits(ctx) -> None:
    ctx.traits.register(
        TraitDefinition(
            name='fused_activation',
            dialects=(ctx.device.dialect,),
            fields=('activation',),
            description='Indicates that an activation has been fused into the producer op.',
        )
    )
    ctx.traits.register(
        TraitDefinition(
            name='io_view',
            dialects=(ctx.device.dialect,),
            fields=('inputs', 'outputs'),
            description='Per-tensor logical-to-physical view mapping for IO/staging.',
        )
    )


def normalize_directives(name: str, raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise TypeError(f'{name}: layer directives must be a dict.')

    directives: Dict[str, Any] = {}

    if 'placement' in raw:
        placement_cfg = raw['placement']
        if not isinstance(placement_cfg, dict):
            raise TypeError(f'{name}: placement override must be a dict.')
        placement: Dict[str, int] = {}
        if 'col' in placement_cfg:
            placement['col'] = int(placement_cfg['col'])
        if 'row' in placement_cfg:
            placement['row'] = int(placement_cfg['row'])
        if placement:
            directives['placement'] = placement

    if 'microtiling' in raw:
        tiling_cfg = raw['microtiling']
        if not isinstance(tiling_cfg, dict):
            raise TypeError(f'{name}: tiling override must be a dict.')
        tiling: Dict[str, int] = {}
        for key in ('microtile_m', 'microtile_k', 'microtile_n'):
            if key in tiling_cfg:
                tiling[key] = int(tiling_cfg[key])
        if tiling:
            directives['microtiling'] = tiling

    if 'parallelism' in raw:
        parallel_cfg = raw['parallelism']
        if not isinstance(parallel_cfg, dict):
            raise TypeError(f'{name}: parallelism override must be a dict.')
        parallelism: Dict[str, Any] = {}
        for key in ('cas_num', 'cas_length', 'parallel_factor'):
            if key in parallel_cfg:
                parallelism[key] = int(parallel_cfg[key])
        if 'contract' in parallel_cfg:
            parallelism['contract'] = str(parallel_cfg['contract'])
        if parallelism:
            directives['parallelism'] = parallelism

    if 'io_route' in raw:
        io_route = raw['io_route']
        if not isinstance(io_route, dict):
            raise TypeError(f'{name}: io_route override must be a dict.')
        directives['io_route'] = dict(io_route)

    if 'layout' in raw:
        layout = str(raw['layout'])
        if layout not in TENSOR_LAYOUTS:
            raise ValueError(f'{name}: unknown layout {layout!r}; expected one of {sorted(TENSOR_LAYOUTS)}.')
        directives['layout'] = layout

    if 'ports' in raw:
        ports = str(raw['ports'])
        if ports not in PORT_KINDS:
            raise ValueError(f'{name}: unknown ports {ports!r}; expected one of {sorted(PORT_KINDS)}.')
        directives['ports'] = ports

    if 'approximation' in raw:
        directives['approximation'] = str(raw['approximation'])

    if 'hccs' in raw:
        hccs_cfg = raw['hccs']
        if not isinstance(hccs_cfg, dict):
            raise TypeError(f'{name}: hccs override must be a dict.')
        directives['hccs'] = dict(hccs_cfg)

    return directives

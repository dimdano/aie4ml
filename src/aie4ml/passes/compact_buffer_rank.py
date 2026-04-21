# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from math import prod

from ..ir import get_backend_context
from .base import AIEPass

log = logging.getLogger(__name__)


class CompactBufferRank(AIEPass):
    def __init__(self):
        self.name = 'compact_buffer_rank'

    def transform(self, model_or_ctx) -> bool:
        ctx = get_backend_context(model_or_ctx)
        plan = ctx.ir.physical.plan
        buffers = plan['buffers']
        changed = False

        for buf in buffers:
            rank = len(buf['dimension'])
            if rank <= 2:
                continue

            desc_entries = [(endpoint['descriptor'], endpoint['source_type'] == 'plio') for endpoint in buf['writers']]
            desc_entries += [(endpoint['descriptor'], endpoint['target_type'] == 'plio') for endpoint in buf['readers']]
            descriptors = [desc for desc, _ in desc_entries]

            axis_pairs = self._axis_pairs(descriptors, rank)
            if axis_pairs is None:
                self._warn_skip(buf, 'feature/independent axes are not collapsible to a 2D contract')
                continue

            drop_axes = list(range(2, rank))
            if not drop_axes:
                continue

            if not self._is_legal_to_collapse(desc_entries, drop_axes):
                self._warn_skip(buf, 'descriptor prevent legal rank compaction')
                continue

            old_buf_dim = [int(x) for x in buf['dimension']]
            factor = int(prod(old_buf_dim[a] for a in drop_axes))

            outer_dims = {indep for _feat, indep in axis_pairs}
            if len(outer_dims) == 1:
                merge_axis = next(iter(outer_dims))
            elif factor == 1:
                merge_axis = 1
            else:
                self._warn_skip(buf, 'independent axes disagree across descriptors, so collapse is unsafe')
                continue

            buf['dimension'] = [old_buf_dim[0], old_buf_dim[1] * factor]
            changed = True

            for desc, is_graph_io in desc_entries:
                self._collapse_descriptor(desc, drop_axes, factor, merge_axis=merge_axis, is_graph_io=is_graph_io)

        return changed

    def _warn_skip(self, buf, reason: str) -> None:
        log.warning(
            '%s: skipping compact_buffer_rank because %s; x86 simulation may still work, but HW compilation may fail.',
            buf.get('name', '<unnamed buffer>'),
            reason,
        )

    def _axis_pairs(self, descriptors, rank: int):
        pairs = []
        for desc in descriptors:
            if 'inner_dimension' not in desc or 'outer_dimension' not in desc:
                return None

            inner_dim = int(desc['inner_dimension'])
            outer_dim = int(desc['outer_dimension'])

            if inner_dim >= rank or outer_dim >= rank or inner_dim == outer_dim:
                raise ValueError(
                    f'Invalid axes for collapse: inner_dim={inner_dim}, outer_dim={outer_dim}, rank={rank}'
                )

            if {inner_dim, outer_dim} != {0, 1}:
                return None

            pairs.append((inner_dim, outer_dim))

        return pairs

    def _is_legal_to_collapse(self, desc_entries, drop_axes) -> bool:
        for desc, is_graph_io in desc_entries:
            buf = [int(x) for x in desc['buffer_dimension']]

            for axis in drop_axes:
                if int(desc['offset'][axis]) != 0:
                    return False

                if not is_graph_io and int(desc['tiling_dimension'][axis]) != 1:
                    return False

                if 'boundary_dimension' in desc and int(desc['boundary_dimension'][axis]) != int(buf[axis]):
                    return False
                if 'io_boundary_dimension' in desc and int(desc['io_boundary_dimension'][axis]) != int(buf[axis]):
                    return False
                if 'io_tiling_dimension' in desc and int(desc['io_tiling_dimension'][axis]) != int(buf[axis]):
                    return False

            if 'tile_traversal' in desc:
                for step in desc['tile_traversal']:
                    d = int(step['dimension'])
                    if d in drop_axes:
                        s = int(step['stride'])
                        w = int(step['wrap'])
                        if s != 1 or w != int(buf[d]):
                            return False

        return True

    def _collapse_descriptor(self, desc, drop_axes, factor: int, merge_axis: int, is_graph_io: bool) -> None:
        vector_fields = (
            'buffer_dimension',
            'tiling_dimension',
            'offset',
            'boundary_dimension',
            'io_tiling_dimension',
            'io_boundary_dimension',
        )
        for key in vector_fields:
            if key in desc:
                vec = [int(x) for x in desc[key]]
                if key != 'tiling_dimension' or is_graph_io:
                    vec[merge_axis] = int(vec[merge_axis]) * int(factor)
                desc[key] = [vec[0], vec[1]]

        if 'tile_traversal' in desc:
            traversal = desc['tile_traversal']
            compact = []
            for step in traversal:
                d = int(step['dimension'])
                s = int(step['stride'])
                w = int(step['wrap'])
                if d in drop_axes:
                    continue
                if d == merge_axis:
                    w = int(w) * int(factor)
                compact.append({'dimension': d, 'stride': s, 'wrap': w})
            desc['tile_traversal'] = compact

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from ...ir import OpNode
from ...op_impls.utils.tensor_view import map_view_axis
from .model import Connection, EdgeEntry, Endpoint


class TransportCollector:
    """Expand the execution graph's connectivity, views included, into semantic transport entries.

    It reads the execution IR only: entries, the values they read and write, and the graph boundary.
    """

    def __init__(self, ctx):
        self.ctx = ctx
        self.execution = ctx.ir.execution

    def collect(self) -> List[EdgeEntry]:
        return self._group_edges(self._collect_connections())

    def _collect_connections(self) -> List[Connection]:
        producers: Dict[str, Tuple[OpNode, str]] = {}
        for inst in self.execution:
            for tname in inst.outputs:
                producers[tname] = (inst.node, inst.ports.outputs[tname].group)

        connections: List[Connection] = []
        seen_outputs: set[str] = set()
        graph_output_names = set(self.execution.graph_outputs)
        for name in graph_output_names:
            view = self.execution.values[name].view
            if view is not None:
                raise NotImplementedError(f'{name}: {view.kind}-backed graph outputs are not implemented.')

        for inst in self.execution:
            n = inst.node
            for item in inst.inputs:
                tname = item.tensor
                cg = inst.ports.inputs[tname].group
                concat_view = self._concat_view_for_tensor(tname)
                if concat_view is not None:
                    connections.extend(self._concat_connections(n, tname, cg, concat_view, producers))
                    seen_outputs.add(tname)
                    for view_item in concat_view.get('slices', []):
                        seen_outputs.add(str(view_item['input']))
                    continue
                slice_view = self._slice_view_for_tensor(tname)
                if slice_view is not None:
                    connections.append(self._slice_connection(n, tname, cg, slice_view, producers))
                    seen_outputs.add(tname)
                    seen_outputs.add(str(slice_view['source']))
                    continue
                if tname in producers:
                    p, pg = producers[tname]
                    connections.append(Connection(tname, Endpoint(p, tname, pg), Endpoint(n, tname, cg)))
                    seen_outputs.add(tname)
                else:
                    connections.append(Connection(tname, Endpoint(None, tname, 'graph_input'), Endpoint(n, tname, cg)))

        # graph outputs
        for inst in self.execution:
            for tname in inst.outputs:
                if tname not in graph_output_names and tname in seen_outputs:
                    continue
                pg = inst.ports.outputs[tname].group
                connections.append(Connection(tname, Endpoint(inst.node, tname, pg), None))

        return connections

    def _concat_view_for_tensor(self, tensor: str) -> Optional[Dict[str, Any]]:
        view = self.execution.values[tensor].view
        if view is None or view.kind != 'concat':
            return None
        data = dict(view.data)
        if data.get('output') != tensor:
            raise ValueError(f'{view.node}: concat_view output does not match tensor {tensor!r}.')
        return data

    def _slice_view_for_tensor(self, tensor: str) -> Optional[Dict[str, Any]]:
        view = self.execution.values[tensor].view
        if view is None or view.kind not in ('slice', 'split'):
            return None
        data = dict(view.data)
        matches = [item for item in data.get('slices', []) if item.get('output') == tensor]
        if len(matches) != 1:
            raise ValueError(f'{view.node}: slice_view does not define output tensor {tensor!r} exactly once.')
        return {
            'source': str(data['source']),
            'axis': int(data['axis']),
            'start': int(matches[0]['start']),
            'extent': int(matches[0]['extent']),
        }

    def _slice_connection(
        self,
        consumer: OpNode,
        slice_tensor: str,
        consumer_group: str,
        slice_view: Dict[str, Any],
        producers: Dict[str, Tuple[OpNode, str]],
    ) -> Connection:
        source_name = str(slice_view['source'])
        producer, producer_group = self._kernel_source(
            slice_tensor,
            source_name,
            producers,
            view_kind='slice',
        )
        ports, offset_base, buffer_dimension = self._slice_producer_ports(
            producer,
            source_name,
            int(slice_view['axis']),
            int(slice_view['start']),
            int(slice_view['extent']),
        )
        return Connection(
            slice_tensor,
            Endpoint(
                producer,
                source_name,
                producer_group,
                ports=ports,
                offset_base=offset_base,
                buffer_dimension=buffer_dimension,
            ),
            Endpoint(consumer, slice_tensor, consumer_group),
        )

    def _slice_producer_ports(
        self, producer: OpNode, source_tensor: str, axis: int, start: int, extent: int
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
        inst = self._kernel_inst(producer)
        view = inst.config.io_views[source_tensor]
        axis_dim = self._view_axis_to_buffer_dim(view, axis)
        total_ports = int(inst.ports.outputs[source_tensor].count)
        end = int(start) + int(extent)
        selected = []
        ranges = []
        first_desc = None
        for port in range(total_ports):
            desc = inst.variant.describe_output_staging(producer, inst.config, source_tensor, port, None)
            first_desc = desc if first_desc is None else first_desc
            port_start, port_end = self._descriptor_axis_range(desc, axis_dim)
            overlaps = port_start < end and port_end > start
            if overlaps and not (port_start >= start and port_end <= end):
                raise NotImplementedError(
                    f'{source_tensor}: slice range [{start}, {end}) crosses producer port {port} '
                    f'range [{port_start}, {port_end}); packed slice/relay is not implemented.'
                )
            if overlaps:
                selected.append(port)
                ranges.append((port_start, port_end))
        ordered_ranges = sorted(ranges)
        contiguous = all(left[1] == right[0] for left, right in zip(ordered_ranges, ordered_ranges[1:]))
        if not selected or ordered_ranges[0][0] != start or ordered_ranges[-1][1] != end or not contiguous:
            raise NotImplementedError(
                f'{source_tensor}: slice range [{start}, {end}) does not align exactly with producer ports.'
            )

        dims = list(first_desc['buffer_dimension'])
        dims[axis_dim] = int(extent)
        base = tuple(int(start) if dim == axis_dim else 0 for dim in range(len(dims)))
        return tuple(selected), base, tuple(dims)

    def _concat_connections(
        self,
        consumer: OpNode,
        concat_tensor: str,
        consumer_group: str,
        concat_view: Dict[str, Any],
        producers: Dict[str, Tuple[OpNode, str]],
    ) -> List[Connection]:
        ports_by_source = self._concat_consumer_ports(consumer, concat_tensor, concat_view)
        conns: List[Connection] = []
        for item in concat_view.get('slices', []):
            source_name = str(item['input'])
            ports = tuple(ports_by_source.get(source_name, ()))
            if not ports:
                continue
            offset_base = self._concat_consumer_offset_base(consumer, concat_tensor, concat_view, int(item['start']))
            producer, producer_group = self._kernel_source(
                concat_tensor,
                source_name,
                producers,
                view_kind='concat',
            )
            conns.append(
                Connection(
                    concat_tensor,
                    Endpoint(producer, source_name, producer_group),
                    Endpoint(
                        consumer,
                        concat_tensor,
                        consumer_group,
                        ports=ports,
                        offset_base=offset_base,
                    ),
                )
            )
        return conns

    def _concat_consumer_offset_base(
        self, consumer: OpNode, concat_tensor: str, concat_view: Dict[str, Any], start: int
    ) -> Tuple[int, ...]:
        inst = self._kernel_inst(consumer)
        view = inst.config.io_views[concat_tensor]
        axis_dim = self._view_axis_to_buffer_dim(view, int(concat_view['axis']))
        return tuple(int(start) if dim == axis_dim else 0 for dim in range(view.rank))

    def _concat_consumer_ports(
        self, consumer: OpNode, concat_tensor: str, concat_view: Dict[str, Any]
    ) -> Dict[str, List[int]]:
        inst = self._kernel_inst(consumer)
        if inst is None:
            raise RuntimeError(f'{concat_tensor}: concat consumer {consumer.name!r} is not resolved.')
        total_ports = int(inst.ports.inputs[concat_tensor].count)
        slices = [
            (str(item['input']), int(item['start']), int(item['start']) + int(item['extent']))
            for item in concat_view.get('slices', [])
        ]
        if not slices:
            raise ValueError(f'{concat_tensor}: concat_view has no input slices.')

        out: Dict[str, List[int]] = {name: [] for name, _, _ in slices}
        axis = int(concat_view['axis'])
        view = inst.config.io_views[concat_tensor]
        axis_dim = self._view_axis_to_buffer_dim(view, axis)
        for port in range(total_ports):
            desc = inst.variant.describe_input_staging(consumer, inst.config, concat_tensor, port, None, None)
            start, end = self._descriptor_axis_range(desc, axis_dim)

            owners = [name for name, lo, hi in slices if start >= lo and end <= hi]
            if len(owners) != 1:
                raise NotImplementedError(
                    f'{concat_tensor}: concat consumer port {port} axis {axis} range [{start}, {end}) does not '
                    'fit exactly inside one concat input slice; packed concat/relay is not implemented.'
                )
            out[owners[0]].append(int(port))
        return out

    @staticmethod
    def _descriptor_axis_range(desc: Dict[str, Any], dim: int) -> Tuple[int, int]:
        start = int(desc['offset'][dim])
        extent = int(desc['io_tiling_dimension'][dim])
        if extent <= 0:
            raise RuntimeError(f'invalid descriptor extent {extent} on dim{dim}.')
        return start, start + extent

    @staticmethod
    def _view_axis_to_buffer_dim(view, axis: int) -> int:
        rank = int(view.rank)
        normalized = int(axis)
        if normalized < 0:
            normalized += rank
        if normalized < 0 or normalized >= rank:
            raise ValueError(f'view axis {axis} is out of range for rank {rank}.')
        real_axis = map_view_axis(view, normalized)
        return int(view.buffer_order.index(int(real_axis)))

    # -------------------------------------------------------------------------
    # Group edges
    # -------------------------------------------------------------------------

    def _group_edges(self, connections: Iterable[Connection]) -> List[EdgeEntry]:
        grouped: Dict[Tuple[str, str, str, str], EdgeEntry] = {}

        for c in connections:
            consumer = c.consumer
            consumer_tensor = consumer.tensor if consumer is not None else c.logical_tensor
            consumer_group = consumer.group if consumer is not None else 'graph_output'
            key = (c.producer.tensor, c.producer.group, consumer_tensor, consumer_group)
            if key not in grouped:
                grouped[key] = EdgeEntry(
                    logical_tensor=c.logical_tensor,
                    producer=c.producer,
                    producer_port_count=self._producer_port_count(c.producer),
                )

            e = grouped[key]
            if e.producer != c.producer:
                raise RuntimeError(f'{c.logical_tensor}: inconsistent producer endpoint for grouped edge.')

            if c.consumer is None:
                e.graph_output = True
            else:
                e.consumers.append(c)
                if c.producer.node is None:
                    e.producer_port_count = max(
                        e.producer_port_count,
                        self._consumer_port_count(c.consumer),
                    )

        return list(grouped.values())

    def _kernel_source(
        self,
        logical_tensor: str,
        source_name: str,
        producers: Dict[str, Tuple[OpNode, str]],
        *,
        view_kind: str,
    ) -> Tuple[OpNode, str]:
        if source_name in producers:
            return producers[source_name]

        value = self.execution.values.get(source_name)
        if value is None:
            raise ValueError(f'{logical_tensor}: {view_kind} source tensor {source_name!r} does not exist.')
        if value.view is not None:
            raise NotImplementedError(
                f'{logical_tensor}: chained view transport through {value.view.kind} '
                f'{value.view.node!r} is not implemented.'
            )
        if value.producer is not None:
            raise RuntimeError(
                f'{logical_tensor}: {view_kind} source producer {value.producer!r} has no resolved ' 'execution output.'
            )
        if source_name not in self.execution.graph_inputs:
            raise RuntimeError(
                f'{logical_tensor}: {view_kind} source tensor {source_name!r} has no producer and is not a '
                'declared graph input.'
            )
        raise NotImplementedError(
            f'{logical_tensor}: {view_kind} input {source_name!r} is a graph input; '
            f'{view_kind}-backed graph-input legs are not implemented.'
        )

    def _kernel_inst(self, node):
        return self.ctx.ir.execution.get(node.name) if node else None

    def _producer_port_count(self, endpoint: Endpoint):
        if endpoint.node is None:
            return 1
        total = self._kernel_inst(endpoint.node).ports.outputs[endpoint.tensor].count
        return len(endpoint.selected_ports(total))

    def _consumer_port_count(self, endpoint: Endpoint):
        total = self._kernel_inst(endpoint.node).ports.inputs[endpoint.tensor].count
        return len(endpoint.selected_ports(total))

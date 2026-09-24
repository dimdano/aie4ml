"""Conv2D on the Dense mmul core: one kernel variant and everything it needs to configure it."""

from __future__ import annotations

from typing import Any, ClassVar, Dict

import numpy as np

from ....aie_types import FloatIntent
from ....ir.graph import STAGING_CONTRACTS, VIEW_FLATTEN_2D, OpImplInstance, OpNode, input_role, input_tensor_for_role
from ....passes.utils import sanitize_identifier
from ...base import BufferLocation, LayoutConversion, OpImplFootprint, OpImplVariant, row_flow
from ...common_types import PORT_KIND_BUFFER, PORT_KIND_STREAM, PortBinding, PortMap
from ...registry import register_variant
from ...utils import MicrotileShape, ParallelismConfig, TensorView, parse_directives, requested_port_kind
from ...utils.math import align_up
from ...utils.precision import (
    aie_rounding_token,
    resolve_accumulator_output_shift,
    resolve_bias_dtype,
    resolve_exact_storage_dtype,
    resolve_operand_precision,
    resolve_output_scale_shift,
)
from ..matmul.common import (
    MICROTILE_OPTIONS,
    describe_inner_output_staging,
    np_bias_dtype_for_spec,
    np_dtype_for_spec,
    quantize_to_int,
    select_generation_key,
)
from ..matmul.config import MatmulMicrotileConfig
from .common import (
    CHANNEL_BLOCK,
    describe_frame_staging,
    describe_logical_staging,
    frame_view,
    spatial_access_of,
)
from .config import Conv2dConfig, Conv2dFlags, FrameRetileConfig
from .frame_retile import FrameRetileOpImplVariant

STREAM_BAND_ROWS = 4
"""Output rows a streamed kernel computes per core call: the band it keeps in local memory.

Four rows keep the halo copy between bands (window - 1 rows) small relative to the work, while
the frame it holds stays a fraction of the whole image.
"""

_SPATIAL_BLOCKS = {'AIE': 2, 'AIE-ML': 4, 'AIE-MLV2': 4}
"""Register blocking measured best per generation: mmul row tiles per accumulator set.

Measured on AIE-ML (3x3, Cin 8 -> Cout 32): at OUT_W 16 the 4-tile blocking is 22% faster
(15.6 vs 20.1 cycles/pixel); at OUT_W 8 the two tie within 2%, so one constant serves both.
"""


@register_variant
class Conv2dOpImplVariant(OpImplVariant):
    """int8 Conv2D as an implicit GEMM over the Dense mmul core, on channel-blocked NHWC frames.

    Partitioning uses the Dense vocabulary on the frame: `cas_length` splits the reduction (input
    channel blocks) across a cascade chain, and `cas_num` splits either the output channel blocks
    ('inner') or the output rows ('outer', row bands that overlap by the window span).
    """

    variant_id = 'conv2d.b.r.v1'
    op_type = 'conv2d'
    graph_header = 'conv2d_graph.h'
    graph_name = 'conv2d_graph'
    param_template = 'conv2d'
    plevel = 10
    port_kind: ClassVar[str] = PORT_KIND_BUFFER
    # `placement` is the placement pass's, which serves every op; the rest the variant reads itself.
    # Anything else -- microtiling, layout -- is refused rather than ignored.
    supported_directives: ClassVar[frozenset] = frozenset(
        {'ports', 'io_route', 'input_contracts', 'parallelism', 'placement'}
    )

    def matches(self, node: OpNode, device) -> bool:
        lhs = input_tensor_for_role(node, 'lhs')
        rhs = input_tensor_for_role(node, 'rhs')
        out = node.outputs[0]
        if any(isinstance(t.precision, FloatIntent) for t in (lhs, rhs, out)):
            return False
        widths = (
            resolve_exact_storage_dtype(lhs.precision, namespace='lhs', layer_name=node.name).width,
            resolve_exact_storage_dtype(rhs.precision, namespace='rhs', layer_name=node.name).width,
            resolve_exact_storage_dtype(out.precision, namespace='output', layer_name=node.name).width,
        )
        return requested_port_kind(node) == self.port_kind and widths == (8, 8, 8)

    def resolve(self, node: OpNode, device, directives=None) -> Conv2dConfig:
        unsupported = sorted(set(directives or {}) - self.supported_directives)
        if unsupported:
            raise NotImplementedError(
                f'{node.name}: {self.variant_id} does not implement the directive(s) '
                f'{unsupported}; it supports {sorted(self.supported_directives)}.'
            )
        io_route, input_contracts, parallel_cfg = parse_directives(directives)
        lhs = input_tensor_for_role(node, 'lhs')
        rhs = input_tensor_for_role(node, 'rhs')
        out = node.outputs[0]
        access = spatial_access_of(node)
        # Kernel limits, as opposed to what the operation means (which the family verified).
        if int(lhs.shape[0]) != 1:
            raise NotImplementedError(f'{node.name}: {self.variant_id} runs one sample per call, got N={lhs.shape[0]}.')
        if access.dilations != (1, 1):
            raise NotImplementedError(
                f'{node.name}: {self.variant_id} does not implement dilations {access.dilations}.'
            )
        if (
            max(access.pads[0], access.pads[2]) >= access.kernel[0]
            or max(access.pads[1], access.pads[3]) >= access.kernel[1]
        ):
            raise NotImplementedError(
                f'{node.name}: {self.variant_id} pads {access.pads} must stay inside the kernel {access.kernel}.'
            )

        precision, accumulator_tag = resolve_operand_precision(node, device)
        precision['bias'] = resolve_bias_dtype(node, precision)
        generation = select_generation_key(device.generation)
        m, k, n = MICROTILE_OPTIONS[generation][('int8', 'int8')][0]
        microtiling = MatmulMicrotileConfig(microtile_m=m, microtile_k=k, microtile_n=n)
        spatial_blocks = _SPATIAL_BLOCKS[generation]

        view = node.trait_data('output_view')
        parallelism = self._resolve_parallelism(node, parallel_cfg, input_contracts, flatten=bool(view))
        block = spatial_blocks * m
        outer = parallelism.contract == 'outer'
        bands = parallelism.cas_num if outer else 1
        io_views = {
            lhs.name: frame_view(
                lhs, column_block=block, column_align=m, channel_slices=parallelism.cas_length, bands=bands
            ),
        }
        if view:
            if view['kind'] != VIEW_FLATTEN_2D:
                raise NotImplementedError(f'{node.name}: {self.variant_id} cannot write {view["kind"]}.')
            if int(rhs.shape[-1]) % CHANNEL_BLOCK:
                raise NotImplementedError(
                    f'{node.name}: a flattened conv needs output channels in whole {CHANNEL_BLOCK}-blocks, '
                    f'got {int(rhs.shape[-1])}.'
                )
            # The Dense LHS the consumer reads: 2*M rows, K in 2*microtile_k blocks.
            flat_k = int(out.shape[-1])
            padded = (align_up(int(out.shape[0]), 2 * m), align_up(flat_k, 2 * k))
            io_views[out.name] = TensorView(
                logical=tuple(int(x) for x in out.shape),
                full=padded,
                tile=padded,
                tile_raw=tuple(int(x) for x in out.shape),
                microtile=MicrotileShape(outer=m, inner=k),
            )
        else:
            io_views[out.name] = frame_view(
                out,
                column_block=block,
                column_align=m,
                channel_slices=1 if outer else parallelism.cas_num,
                bands=bands,
            )
            if outer and any(io_views[out.name].origin):
                raise NotImplementedError(
                    f'{node.name}: a band-split output cannot carry the zero border its consumer reads; '
                    'partition the channels instead, or let the consumer pad its own input.'
                )

        shift = resolve_accumulator_output_shift(lhs.precision, out.precision, rhs.precision)
        shift += resolve_output_scale_shift(node, is_float=False)
        fused_act = node.traits.get('fused_activation')
        use_relu = ((fused_act.data.get('activation') if fused_act else '') or '').lower() == 'relu'

        if self.port_kind == PORT_KIND_BUFFER and int(access.strides[1]) > 1:
            # A retiler kernel per row band feeds a strided conv; splitting its channels would need one
            # per channel slice (cas_length) or one frame for several chains (inner cas_num).
            banded = parallelism.contract == 'outer' and int(parallelism.cas_length) == 1
            if int(parallelism.cas_num) * int(parallelism.cas_length) > 1 and not banded:
                raise NotImplementedError(
                    f'{node.name}: {self.variant_id} splits a strided conv only into row bands '
                    f"(contract 'outer', cas_length 1); cas_num={parallelism.cas_num}, "
                    f'cas_length={parallelism.cas_length} with contract {parallelism.contract!r} is not implemented.'
                )

        return Conv2dConfig(
            precision=precision,
            parallelism=parallelism,
            microtiling=microtiling,
            io_views=io_views,
            io_route=io_route,
            shift=shift,
            accumulator_tag=accumulator_tag,
            rounding_mode=aie_rounding_token(precision['output']),
            spatial_blocks=spatial_blocks,
            kernel_shape=access.kernel,
            strides=access.strides,
            dilations=access.dilations,
            pads=access.pads,
            groups=int(node.metadata['groups']),
            alternating_horizontal=device.cascade_layout == 'alternating_horizontal',
            bank_mem_bytes=int(device.bank_mem_bytes),
            flags=Conv2dFlags(use_relu=use_relu, emit_flattened=bool(view)),
        )

    def _resolve_parallelism(self, node, parallel_cfg, input_contracts, *, flatten: bool) -> ParallelismConfig:
        """Tiles over the channel-block axis, in the Dense contract vocabulary."""
        contract = str(parallel_cfg.get('contract', 'inner'))
        if contract not in STAGING_CONTRACTS:
            raise ValueError(f'{node.name}: unknown parallelism contract {contract!r}.')
        lhs = input_tensor_for_role(node, 'lhs')
        access = spatial_access_of(node)
        in_blocks = align_up(int(lhs.shape[-1]), CHANNEL_BLOCK) // CHANNEL_BLOCK
        out_blocks = align_up(int(input_tensor_for_role(node, 'rhs').shape[-1]), CHANNEL_BLOCK) // CHANNEL_BLOCK
        reads_neighbour_rows = access.window[0] > 1 or access.pads[0] or access.pads[2]

        cas_length = int(parallel_cfg.get('cas_length', 1))
        producer = input_contracts.get(lhs.name)
        if producer is not None and producer.contract == 'outer':
            # The rows arrive already split into bands; a window that reaches past its own band
            # would need rows another band owns.
            if reads_neighbour_rows:
                raise NotImplementedError(
                    f'{node.name}: its input arrives in row bands, but its {access.kernel} window with pads '
                    f'{access.pads} reads rows the neighbouring band owns.'
                )
            if contract == 'inner' and 'contract' in parallel_cfg:
                raise ValueError(f'{node.name}: its input is banded, so it cannot be partitioned by channel.')
            if flatten:
                raise NotImplementedError(
                    f'{node.name}: its input arrives in row bands, but a flattened output is one row that '
                    'the consuming Dense reads whole.'
                )
            return ParallelismConfig(cas_num=len(producer.port_staging), cas_length=1, contract='outer')
        if producer is not None:
            # A frame is never re-staged on the way, so a chain reads exactly the slices its
            # producer wrote -- the same rule the Dense 'inner' contract follows.
            required = len(producer.port_staging)
            if 'cas_length' in parallel_cfg and cas_length != required:
                raise ValueError(
                    f'{node.name}: cas_length={cas_length} conflicts with the {required} ports its producer '
                    'writes; a spatial frame is handed over slice for slice.'
                )
            cas_length = required
        cas_num = int(parallel_cfg.get('cas_num', 1))
        if contract == 'outer':
            # Bands overlap by the window span, so a band reads rows its neighbours also read.
            # Only the graph boundary can serve that: the host clips each port's window against
            # the tensor and zero-fills the rest, while a producing kernel writes each row once.
            if lhs.producer is not None and reads_neighbour_rows:
                raise NotImplementedError(
                    f'{node.name}: a band-split input whose window reads neighbouring rows must come from the '
                    'graph boundary, because the bands overlap and a kernel writes every row exactly once.'
                )
            if flatten:
                raise NotImplementedError(f'{node.name}: a flattened output cannot be split into row bands.')
            out_rows = access.output_extent(int(lhs.shape[1]), int(lhs.shape[2]))[0]
            if cas_num < 1 or out_rows % cas_num:
                raise ValueError(
                    f'{node.name}: cas_num={cas_num} does not split {out_rows} output rows into equal bands.'
                )
            if cas_length < 1 or in_blocks % cas_length:
                raise ValueError(
                    f'{node.name}: cas_length={cas_length} does not split {in_blocks} '
                    f'{CHANNEL_BLOCK}-channel blocks evenly.'
                )
            return ParallelismConfig(cas_num=cas_num, cas_length=cas_length, contract=contract)
        if flatten and cas_num != 1:
            raise NotImplementedError(
                f'{node.name}: a flattened output interleaves the channel blocks of every pixel, so it cannot '
                f'be split across {cas_num} chains.'
            )
        for name, value, blocks in (('cas_length', cas_length, in_blocks), ('cas_num', cas_num, out_blocks)):
            if value < 1 or blocks % value:
                raise ValueError(
                    f'{node.name}: {name}={value} does not split {blocks} {CHANNEL_BLOCK}-channel blocks evenly.'
                )
        return ParallelismConfig(cas_num=cas_num, cas_length=cas_length, contract=contract)

    def validate_config(self, node: OpNode, config: Conv2dConfig, device) -> None:
        if config.shift < 0:
            raise ValueError(f'{node.name}: conv2d accumulator output shift must be non-negative, got {config.shift}.')
        params = self.build_template_params(node, config, {'row': 0, 'col': 0})
        # The kernel computes whole register tiles, so it reads past the last output pixel. A strided
        # frame holds its columns in `stride_w` polyphase classes and every class must reach that
        # far; at stride 1 there is one class, the whole row. Mirrors the kernel's static_assert.
        stride_w = int(config.strides[1])
        phase_cols = int(params['in_cols']) // stride_w
        read_span = (
            params['out_w_computed'] + (config.kernel_shape[1] - 1 + params['in_origin_c'] - config.pads[1]) // stride_w
        )
        if int(params['in_cols']) % stride_w or read_span > phase_cols:
            raise RuntimeError(
                f'{node.name}: the kernel reads {read_span} columns of each of {stride_w} column '
                f'class(es) but the padded frame holds {params["in_cols"]} columns.'
            )
        if self.port_kind == PORT_KIND_BUFFER:
            # Dense's bank schedule: one frame copy per bank (0 and 3) wherever the op contract puts it,
            # the weights in bank 2 of the kernel's tile, stack and bias in bank 1.
            bank = int(config.bank_mem_bytes)
            for what, size, splits in (
                ('input frame', params['in_bytes'], "row bands (contract 'outer') or `cas_length` over input channels"),
                ('output frame', params['out_bytes'], "row bands (contract 'outer') or `cas_num` over output channels"),
                (
                    'weights',
                    params['weight_count'],
                    '`cas_length` over input channels or `cas_num` over output '
                    'channels; row bands copy the weights to every tile',
                ),
            ):
                if int(size) > bank:
                    raise ValueError(
                        f"{node.name}: each tile's {what} is {size} B but one {device.platform} memory bank holds "
                        f'{bank} B. What shrinks it, where the shape splits evenly: {splits}.'
                    )
            return
        # The stream wrapper owns one frame each way, and its staging, in its own tile.
        tile_bytes = (
            params['in_bytes']
            + params['out_bytes']
            + params['weight_count']
            + 4 * params['bias_count']
            + self.staging_bytes(params)
        )
        if tile_bytes > int(device.tile_mem_bytes):
            raise ValueError(
                f'{node.name}: one tile needs {tile_bytes} B for its frames, weights and bias but a '
                f'{device.platform} tile has {device.tile_mem_bytes} B; split the layer with '
                '`parallelism: {cas_num: .., cas_length: ..}`.'
            )

    def staging_bytes(self, _params) -> int:
        """Tile memory the kernel holds beyond its frames. A buffer kernel holds none."""
        return 0

    def retiles_input(self, config: Conv2dConfig) -> bool:
        """Whether this conv reads a frame a retiler built.

        A strided window reads its columns grouped by residue, which neither source provides: the
        boundary carries the tensor in plain order, and a producing kernel writes whole register
        tiles, which span several groups. The stream variant refuses stride altogether.
        """
        return self.port_kind == PORT_KIND_BUFFER and int(config.strides[1]) > 1

    @staticmethod
    def retiled_frame(node) -> str:
        """The execution-only tensor a retiler writes and this conv reads in place of its input."""
        return f'{input_tensor_for_role(node, "lhs").name}__{node.name}_frame'

    def input_conversions(self, node, config: Conv2dConfig, sources):
        if not self.retiles_input(config):
            return ()
        lhs = input_tensor_for_role(node, 'lhs')
        source = sources[lhs.name]
        if source.view is not None:
            raise NotImplementedError(
                f'{node.name}: its strided input is the {source.view.kind} view {source.view.node!r}; a '
                'retiler reads only the boundary tensor or a whole frame a kernel wrote.'
            )
        view = config.io_views[lhs.name]
        from_boundary = source.producer is None
        bands = int(config.parallelism.cas_num) if config.parallelism.contract == 'outer' else 1
        if bands > 1 and not from_boundary:
            raise NotImplementedError(
                f'{node.name}: its strided input arrives from {source.producer} in row bands; a retiler reads a '
                "producer's frame whole."
            )
        frame = self.retiled_frame(node)
        retile = FrameRetileConfig.for_frame(
            config.precision['lhs'],
            view,
            int(config.strides[1]),
            source=lhs.name,
            target=frame,
            from_boundary=from_boundary,
            bands=bands,
            band_rows=self._band_input_rows(node, config),
            alternating_horizontal=config.alternating_horizontal,
            bank_mem_bytes=config.bank_mem_bytes,
        )
        # A performance constraint, not a functional one: a DMA copy of the finished frame would be
        # exact, but its latency, interval and descriptor limits are unmeasured, while the shared hand-
        # over is what every retiler figure was measured with. Until the copy is, the edge requires it.
        return (
            LayoutConversion(
                name=f'{node.name}_retile',
                source=lhs.name,
                target=frame,
                variant=FrameRetileOpImplVariant(),
                config=retile,
                shared_memory=True,
            ),
        )

    def buffer_locations(self, _node, config: Conv2dConfig, anchor_row):
        """The op contract Dense follows (`row_flow`), mirroring `place_graph`: chain `c` on row `c`, each
        kernel's input and each chain's output in banks 0 and 3 of the neighbouring tile both kernels of
        the hand-over reach."""
        cas_num, cas_length = int(config.parallelism.cas_num), int(config.parallelism.cas_length)
        outer = config.parallelism.contract == 'outer'
        locations = []
        for chain in range(cas_num):
            flow = row_flow(config.alternating_horizontal, int(anchor_row) + chain, cas_length)
            for pos in range(cas_length):
                col = cas_length - 1 - pos if flow.reversed else pos
                port = chain * cas_length + pos if outer else pos
                locations.append(BufferLocation('in1', port, col + flow.input_col, chain, (0, 3)))
            last = 0 if flow.reversed else cas_length - 1
            locations.append(BufferLocation('out1', chain, last + flow.output_col, chain, (0, 3)))
        return tuple(locations)

    def band_rows(self, node, config: Conv2dConfig) -> int:
        """Output rows one core call covers. A buffer kernel does the whole tile in one call; a
        stream kernel walks the image in bands, keeping only a window of it."""
        lhs = input_tensor_for_role(node, 'lhs')
        out_rows = spatial_access_of(node).output_extent(int(lhs.shape[1]), int(lhs.shape[2]))[0]
        if self.port_kind == PORT_KIND_BUFFER:
            return out_rows // int(config.parallelism.cas_num) if config.parallelism.contract == 'outer' else out_rows
        band = min(STREAM_BAND_ROWS, out_rows)
        while out_rows % band:
            band -= 1
        return band

    def build_template_params(self, node, config: Conv2dConfig, placement):
        lhs = input_tensor_for_role(node, 'lhs')
        out = node.outputs[0]
        in_view, out_view = config.io_views[lhs.name], config.io_views[out.name]
        _, in_rows, in_cols, in_channels = (int(x) for x in in_view.tile)
        kh, kw = config.kernel_shape
        _, in_h, in_w, cin = (int(x) for x in lhs.shape)
        outer = config.parallelism.contract == 'outer'
        cout = int(input_tensor_for_role(node, 'rhs').shape[-1])
        # 'inner' chains own a share of the output channels; 'outer' chains own rows and each
        # computes every channel.
        out_blocks = align_up(cout, CHANNEL_BLOCK) // CHANNEL_BLOCK
        if config.parallelism.contract == 'inner':
            out_blocks //= int(config.parallelism.cas_num)
        out_blocks_padded = out_blocks + out_blocks % 2
        out_h, out_w = spatial_access_of(node).output_extent(in_h, in_w)
        band = self.band_rows(node, config)
        streamed = self.port_kind == PORT_KIND_STREAM
        if outer:
            out_h //= int(config.parallelism.cas_num)
        # Who puts the zeros around the image: the kernel re-fills the border of a buffer another
        # kernel wrote, the host delivers it with the padded window at the boundary, the stream
        # wrapper keeps it in a frame it owns, and a retiler builds the whole frame.
        retile = self.retiles_input(config)
        fills_border = lhs.producer is not None and self.port_kind == PORT_KIND_BUFFER and not retile
        # A band's window starts mid-image, so the image no longer sits at the frame's origin.
        whole_image = not outer
        params = {field: getattr(config, field) for field in config.__dataclass_fields__}
        params.update(
            cin=cin,
            cout=cout,
            in_blocks=in_channels // CHANNEL_BLOCK,
            out_blocks=out_blocks,
            out_blocks_padded=out_blocks_padded,
            in_h=in_h if whole_image else in_rows,
            band_rows=band,
            bands=out_h // band,
            in_w=in_w,
            in_rows=in_rows,
            fills_border=fills_border,
            in_cols=in_cols,
            in_origin_r=int(in_view.origin[1]) if whole_image else 0,
            in_origin_c=int(in_view.origin[2]),
            out_h=out_h,
            out_w=out_w,
            out_w_computed=align_up(out_w, config.spatial_blocks * config.microtiling.microtile_m),
            in_bytes=int(np.prod(in_view.tile)),
            out_bytes=int(np.prod(out_view.tile)),
            weight_count=kh * kw * (in_channels // CHANNEL_BLOCK) * out_blocks_padded * CHANNEL_BLOCK**2,
            bias_count=out_blocks_padded * CHANNEL_BLOCK,
            buffer_locations=self.buffer_locations(node, config, int(placement['row'])),
            stream_io=self.port_kind == PORT_KIND_STREAM,
        )
        if streamed:
            # A streamed kernel holds one band, not the image: the rows its window reads and the
            # rows it writes. Everything else about the frame is unchanged.
            span_h = spatial_access_of(node).window[0]
            params.update(
                in_rows=band + span_h - 1,
                out_h=band,
                out_rows=band,
                out_origin_r=0,
                out_cols=int(out_view.tile[2]),
                out_origin_c=int(out_view.origin[2]),
                flat_k_padded=0,
                in_bytes=params['in_blocks'] * (band + span_h - 1) * in_cols * CHANNEL_BLOCK,
                out_bytes=params['out_blocks'] * band * int(out_view.tile[2]) * CHANNEL_BLOCK,
            )
            return params
        if config.flags.emit_flattened:
            params.update(out_rows=1, out_cols=1, out_origin_r=0, out_origin_c=0, flat_k_padded=int(out_view.full[-1]))
        else:
            params.update(
                out_rows=int(out_view.tile[1]),
                out_cols=int(out_view.tile[2]),
                out_origin_r=int(out_view.origin[1]),
                out_origin_c=int(out_view.origin[2]),
                flat_k_padded=0,
            )
        return params

    def footprint(self, _node, config) -> OpImplFootprint:
        return OpImplFootprint(width=int(config.parallelism.cas_length), height=int(config.parallelism.cas_num))

    def output_staging_contract(self, _node, config, _tensor_name):
        return str(config.parallelism.contract)

    def output_port_count(self, _node, config):
        return int(config.parallelism.cas_num)

    def _band_rows(self, node, config) -> int:
        """Output rows one chain owns, or 0 when the chains split channels instead."""
        if config.parallelism.contract != 'outer':
            return 0
        lhs = input_tensor_for_role(node, 'lhs')
        out_rows = spatial_access_of(node).output_extent(int(lhs.shape[1]), int(lhs.shape[2]))[0]
        return out_rows // int(config.parallelism.cas_num)

    def _band_input_rows(self, node, config) -> int:
        """Frame rows between the windows of neighbouring bands: each band's output rows, strided."""
        return self._band_rows(node, config) * int(config.strides[0])

    def describe_input_staging(self, node, config, tensor_name, port, _buf_dims=None, _producer=None):
        if self.port_kind == PORT_KIND_STREAM:
            return describe_logical_staging(config.io_views[tensor_name], 'read')
        # 'inner': the port is a channel slice every chain reads. 'outer': the port belongs to one
        # (band, channel slice) tile, so it selects both. A retiled frame is the same frame, its
        # columns grouped by residue, and the port reads it from the retiler.
        cas_length = int(config.parallelism.cas_length)
        outer = config.parallelism.contract == 'outer'
        band, channel_port = (int(port) // cas_length, int(port) % cas_length) if outer else (0, int(port))
        return describe_frame_staging(
            config.io_views[input_tensor_for_role(node, 'lhs').name],
            'read',
            channel_port,
            band=band,
            band_rows=self._band_input_rows(node, config),
            column_phases=int(config.strides[1]) if self.retiles_input(config) else 1,
        )

    def describe_output_staging(self, node, config, tensor_name, port, buf_dims=None):
        view = config.io_views[tensor_name]
        if self.port_kind == PORT_KIND_STREAM:
            return describe_logical_staging(view, 'write')
        if config.flags.emit_flattened:
            return describe_inner_output_staging(view, port, buf_dims)
        outer = config.parallelism.contract == 'outer'
        return describe_frame_staging(
            view,
            'write',
            0 if outer else int(port),
            band=int(port) if outer else 0,
            band_rows=self._band_rows(node, config),
        )

    def build_ports(self, node: OpNode, config: Conv2dConfig):
        cas_length = int(config.parallelism.cas_length)
        cas_num = int(config.parallelism.cas_num)
        lhs = input_tensor_for_role(node, 'lhs')
        # A retiled conv reads the frame its retiler writes, not the tensor itself.
        in_tensor = self.retiled_frame(node) if self.retiles_input(config) else lhs.name
        if config.parallelism.contract == 'outer':
            # Every tile reads its own band, so no port is shared.
            lhs_endpoints = tuple((f'kk[{tile}].in[0]',) for tile in range(cas_num * cas_length))
        else:
            # One input port per reduction column, multicast to the chain owning each output slice.
            lhs_endpoints = tuple(
                tuple(f'kk[{chain * cas_length + port}].in[0]' for chain in range(cas_num))
                for port in range(cas_length)
            )
        out_endpoints = tuple((f'kk[{chain * cas_length + cas_length - 1}].out[0]',) for chain in range(cas_num))
        return PortMap(
            inputs={in_tensor: PortBinding('in1', len(lhs_endpoints), self.port_kind, lhs_endpoints)},
            outputs={node.outputs[0].name: PortBinding('out1', cas_num, self.port_kind, out_endpoints)},
        )

    def pack(self, inst: OpImplInstance) -> Dict[str, Any]:
        """Weights per tile as Dense B tiles: [tap (ky, kx, cin block)][cout block][8 x 8].

        The compact `[kh, kw, Cin/groups, Cout]` tensor expands here into the dense form the mmul
        core consumes -- a grouped conv simply leaves the off-diagonal tiles zero -- and is then
        cut into the (chain, column) slice each tile owns.
        """
        p = inst.config
        weight = input_tensor_for_role(inst.node, 'rhs')
        lhs = input_tensor_for_role(inst.node, 'lhs')
        bias = next((t for t in inst.node.inputs if input_role(inst.node, t.name) == 'bias'), None)
        cas_num, cas_length = int(p.parallelism.cas_num), int(p.parallelism.cas_length)
        wi = weight.precision
        compact = np.asarray(
            quantize_to_int(
                weight.data,
                wi.frac,
                wi.width,
                signed=wi.signed,
                rounding_mode=wi.rounding,
                saturation_mode=wi.saturation,
            )
        )
        kh, kw, cin_g, cout = compact.shape
        groups = int(p.groups)
        cout_g = cout // groups
        in_channels = int(p.io_views[lhs.name].full[-1])
        blocks = align_up(cout, CHANNEL_BLOCK) // CHANNEL_BLOCK
        bands = p.parallelism.contract == 'outer'
        chain_blocks = blocks if bands else blocks // cas_num
        chain_blocks_padded = chain_blocks + chain_blocks % 2
        column_blocks = in_channels // CHANNEL_BLOCK // cas_length

        dense = np.zeros((kh, kw, in_channels, blocks * CHANNEL_BLOCK), dtype=np_dtype_for_spec(p.precision['rhs']))
        for group in range(groups):
            dense[:, :, group * cin_g : (group + 1) * cin_g, group * cout_g : (group + 1) * cout_g] = compact[
                :, :, :, group * cout_g : (group + 1) * cout_g
            ]
        # (kh, kw, cb, 8, nb, 8) -> tap-major (ky, kx, cb) x (nb) x (8 x 8), then cut per tile.
        tiles = dense.reshape(kh, kw, in_channels // CHANNEL_BLOCK, CHANNEL_BLOCK, blocks, CHANNEL_BLOCK).transpose(
            0, 1, 2, 4, 3, 5
        )
        packed_weights = np.zeros(
            (cas_num, cas_length, kh * kw * column_blocks * chain_blocks_padded * CHANNEL_BLOCK**2),
            dtype=tiles.dtype,
        )
        for chain in range(cas_num):
            block_base = 0 if bands else chain * chain_blocks
            for column in range(cas_length):
                tile = np.zeros(
                    (kh, kw, column_blocks, chain_blocks_padded, CHANNEL_BLOCK, CHANNEL_BLOCK), dtype=tiles.dtype
                )
                tile[:, :, :, :chain_blocks] = tiles[
                    :,
                    :,
                    column * column_blocks : (column + 1) * column_blocks,
                    block_base : block_base + chain_blocks,
                ]
                packed_weights[chain, column] = tile.reshape(-1)

        packed_bias = np.zeros(
            (cas_num, chain_blocks_padded * CHANNEL_BLOCK), dtype=np_bias_dtype_for_spec(p.precision['bias'])
        )
        if bias is not None:
            bi = bias.precision
            values = np.asarray(
                quantize_to_int(
                    bias.data,
                    lhs.precision.frac + wi.frac,
                    int(p.precision['bias'].width),
                    signed=bi.signed,
                    rounding_mode=bi.rounding,
                    saturation_mode=bi.saturation,
                )
            ).reshape(-1)
            for chain in range(cas_num):
                base = 0 if bands else chain * chain_blocks * CHANNEL_BLOCK
                chunk = values[base : base + chain_blocks * CHANNEL_BLOCK]
                packed_bias[chain, : chunk.size] = chunk
        return {'packed_weights': packed_weights, 'packed_bias': packed_bias}

    def get_artifacts(self, inst: OpImplInstance):
        inst_name = sanitize_identifier(inst.name)
        p = inst.config
        return [
            {
                'name': 'weights',
                'kind': '2d',
                'storage': 'rom',
                'array': inst.artifacts['packed_weights'],
                'dtype': p.precision['rhs'].c_type,
                'storage_dtype': p.precision['rhs'].storage_dtype,
                'filename': f'weights_{inst_name}.h',
                'port': 'wts',
            },
            {
                'name': 'bias',
                'kind': '1d',
                'storage': 'rom',
                'array': inst.artifacts['packed_bias'],
                'dtype': p.precision['bias'].c_type,
                'storage_dtype': p.precision['bias'].storage_dtype,
                'filename': f'bias_{inst_name}.h',
                'port': 'bias',
            },
        ]


@register_variant
class Conv2dStreamOpImplVariant(Conv2dOpImplVariant):
    """The same Conv2D on core streams: the frame crosses on the wire in linear row order and the
    kernel lands it in the blocked layout the compute core reads (`ports: stream`).

    It buys legality rather than speed: a frame of several channel blocks crosses the graph
    boundary in one port, which a DMA-fed frame cannot do. Measured on AIE1 (8x8, 3x3, one channel
    block): 1,279 cycles through buffer ports against 2,157 through streams, the difference being
    the wire and the landing loop.

    One tile only: a cascade would need its partial sums to share the core's two stream ports.
    """

    variant_id = 'conv2d.s.r.v1'
    port_kind = PORT_KIND_STREAM

    def resolve(self, node: OpNode, device, directives=None) -> Conv2dConfig:
        config = super().resolve(node, device, directives)
        if config.strides != (1, 1):
            raise NotImplementedError(
                f'{node.name}: {self.variant_id} does not implement strides {config.strides}; the '
                'band it keeps and the columns it places are both written for a dense window.'
            )
        if config.parallelism.cas_num != 1 or config.parallelism.cas_length != 1:
            raise NotImplementedError(f'{node.name}: {self.variant_id} does not implement partitioning yet.')
        if config.flags.emit_flattened:
            raise NotImplementedError(f'{node.name}: {self.variant_id} writes a frame, not a flattened row.')
        return config

    def validate_config(self, node: OpNode, config: Conv2dConfig, device) -> None:
        super().validate_config(node, config, device)
        params = self.build_template_params(node, config, {'row': 0, 'col': 0})
        lhs = input_tensor_for_role(node, 'lhs')
        out_rows = spatial_access_of(node).output_extent(int(lhs.shape[1]), int(lhs.shape[2]))[0]
        if int(params['bands']) * int(params['band_rows']) != out_rows:
            raise NotImplementedError(
                f'{node.name}: {out_rows} output rows do not divide into whole bands of {params["band_rows"]}.'
            )
        beat = 16  # bytes in one 128-bit stream access
        for name, elements in (
            ('input', int(np.prod(lhs.shape))),
            ('output', int(np.prod(node.outputs[0].shape))),
        ):
            if elements % beat:
                raise NotImplementedError(
                    f'{node.name}: its {name} is {elements} bytes, which is not a whole number of '
                    f'{beat}-byte stream beats.'
                )

    def buffer_locations(self, _node, _config, _anchor_row):
        return ()  # core streams: no buffer to place

    def staging_bytes(self, params) -> int:
        """A channel count that does not fill a block stages the band's bytes in a buffer: the
        wire reads into one before placing, and gathers into one before writing, because a stream
        access inside either loop stops it pipelining."""
        total = 0
        if int(params['cin']) % CHANNEL_BLOCK:
            total += int(params['in_rows']) * int(params['in_w']) * int(params['cin']) + 16
        if int(params['cout']) % CHANNEL_BLOCK:
            total += int(params['band_rows']) * int(params['out_w']) * int(params['cout']) + 16
        return total

    def footprint(self, _node, _config) -> OpImplFootprint:
        return OpImplFootprint(width=1, height=1)

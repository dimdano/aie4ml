"""Conv2D: channel-blocked NHWC frames on buffer ports, conv -> depthwise -> 1x1 -> flatten -> dense."""

from __future__ import annotations

import numpy as np
import pytest
from helpers import PART, TensorProto, assert_x86_matches_onnx, helper, lower, make_model, numpy_helper, qdq

AIE1_PART = 'xcvp2802-vsva5601-2MHP-e-S'
# 24 channels = 3 blocks, so one model covers a 3-chain split and the first/middle/last cascade;
# 12 classes because a memtile shard needs whole 32-bit words.
H, W, CIN, C1, C2, C3, CLASSES = 8, 8, 3, 24, 24, 8, 12
FRAC = 4


def _qparams(prefix: str, *, frac: int, elem_type: int = TensorProto.INT8) -> list:
    return [
        helper.make_tensor(f'{prefix}_scale', TensorProto.FLOAT, [], [float(2.0**-frac)]),
        helper.make_tensor(f'{prefix}_zp', elem_type, [], [0]),
    ]


def _conv(nodes, inits, x, out, name, cin, cout, k, *, pad, groups=1, relu, seed):
    """Conv(x, W, b) [-> Relu] -> Q -> DQ with int8 weights and an int32 bias in the accumulator scale."""
    rng = np.random.default_rng(seed)
    w = rng.integers(-6, 6, size=(cout, cin // groups, k, k), dtype=np.int8)
    b = rng.integers(-64, 64, size=(cout,), dtype=np.int32)
    inits += [
        numpy_helper.from_array(w, f'{name}_w_q'),
        numpy_helper.from_array(b, f'{name}_b_q'),
        *_qparams(f'{name}_w', frac=FRAC),
        *_qparams(f'{name}_b', frac=2 * FRAC, elem_type=TensorProto.INT32),
        *_qparams(f'{name}o', frac=FRAC),
    ]
    for tag in ('w', 'b'):
        nodes.append(
            helper.make_node(
                'DequantizeLinear', [f'{name}_{tag}_q', f'{name}_{tag}_scale', f'{name}_{tag}_zp'], [f'{name}_{tag}']
            )
        )
    nodes.append(
        helper.make_node(
            'Conv',
            [x, f'{name}_w', f'{name}_b'],
            [f'{name}_conv'],
            name=name,
            kernel_shape=[k, k],
            pads=[pad, pad, pad, pad],
            group=groups,
        )
    )
    pre_q = f'{name}_conv'
    if relu:
        nodes.append(helper.make_node('Relu', [pre_q], [f'{name}_relu'], name=f'{name}_relu'))
        pre_q = f'{name}_relu'
    qdq(nodes, pre_q, out, f'{name}o')


def _head(nodes, inits, src, rows, seed):
    """Flatten the ONNX NCHW view and classify it, so the Gemm rows follow the canonical order."""
    nodes.append(helper.make_node('Flatten', [src], ['flat'], axis=1, name='flat'))
    w = np.random.default_rng(seed).integers(-4, 4, size=(rows, CLASSES), dtype=np.int8)
    inits += [numpy_helper.from_array(w, 'fc_w_q'), *_qparams('fc_w', frac=FRAC), *_qparams('fco', frac=FRAC)]
    nodes.append(helper.make_node('DequantizeLinear', ['fc_w_q', 'fc_w_scale', 'fc_w_zp'], ['fc_w']))
    nodes.append(helper.make_node('Gemm', ['flat', 'fc_w'], ['fc_mm'], name='fc'))
    qdq(nodes, 'fc_mm', 'y', 'fco')


def _start(nodes, inits):
    inits += [*_qparams('x', frac=FRAC)]
    nodes.append(helper.make_node('DequantizeLinear', ['x_q', 'x_scale', 'x_zp'], ['x'], name='x_dq'))
    nodes.append(helper.make_node('Transpose', ['x'], ['x_nchw'], perm=[0, 3, 1, 2], name='to_nchw'))


def _model(name, nodes, inits):
    return make_model(
        name,
        nodes=nodes,
        inputs=[('x_q', TensorProto.INT8, [1, H, W, CIN])],
        outputs=[('y', TensorProto.FLOAT, [1, CLASSES])],
        initializers=inits,
    )


# c1 splits its 3 output channel blocks across 3 chains; c2 then reads those 3 slices along one
# cascade (first + middle + last), and its own output stays whole for the pointwise stage.
DIRECTIVES = {'c1': {'parallelism': {'cas_num': 3}}, 'c2': {'parallelism': {'cas_num': 1}}}


@pytest.fixture
def conv_model():
    """NHWC input viewed as ONNX NCHW, conv -> depthwise -> pointwise, then flatten -> Gemm."""
    nodes: list = []
    inits: list = []
    _start(nodes, inits)
    _conv(nodes, inits, 'x_nchw', 'a1', 'c1', CIN, C1, 3, pad=1, relu=True, seed=1)
    _conv(nodes, inits, 'a1', 'a2', 'c2', C1, C2, 3, pad=1, groups=C1, relu=True, seed=2)
    _conv(nodes, inits, 'a2', 'a3', 'c3', C2, C3, 1, pad=0, relu=False, seed=3)
    _head(nodes, inits, 'a3', H * W * C3, seed=4)
    return _model('conv2d_chain', nodes, inits)


def _valid_model(k: int):
    """One 'valid' conv (no padding): its output is narrower than the register block it computes."""
    nodes: list = []
    inits: list = []
    _start(nodes, inits)
    _conv(nodes, inits, 'x_nchw', 'a', 'c', CIN, C3, k, pad=0, relu=True, seed=7)
    _head(nodes, inits, 'a', (H - k + 1) ** 2 * C3, seed=8)
    return _model(f'conv2d_valid_k{k}', nodes, inits)


def _frame_model(channels_in=CIN, channels_out=C3, name='conv_frame'):
    """A same-padded conv straight from the graph input, writing its frame to the boundary.

    Nothing gathers row bands back together, so a banded chain ends here; a streamed conv also
    has to start here, because its frame must arrive with the border already in it.
    """
    nodes: list = []
    inits: list = []
    _start(nodes, inits)
    _conv(nodes, inits, 'x_nchw', 'a', 'b', channels_in, channels_out, 3, pad=1, relu=True, seed=21)
    nodes.append(helper.make_node('Transpose', ['a'], ['y'], perm=[0, 2, 3, 1], name='to_nhwc'))
    return make_model(
        name,
        nodes=nodes,
        inputs=[('x_q', TensorProto.INT8, [1, H, W, channels_in])],
        outputs=[('y', TensorProto.FLOAT, [1, H, W, channels_out])],
        initializers=inits,
    )


def _band_model():
    return _frame_model(name='conv_bands')


def _stream_model():
    """Several channel blocks, so the wire order and the blocked frame really differ -- and more
    than one block crosses the boundary, which a DMA-fed frame cannot do."""
    return _frame_model(channels_in=C1, channels_out=C2, name='conv_stream')


BAND_DIRECTIVES = {'b': {'parallelism': {'contract': 'outer', 'cas_num': 2}}}
STREAM_DIRECTIVES = {'b': {'ports': 'stream'}}


def _feed() -> np.ndarray:
    rng = np.random.default_rng(11)
    return rng.integers(-40, 40, size=(1, H, W, CIN), dtype=np.int8)


# --------------------------------------------------------------------------- #
# contract: layouts, frames, transport plan
# --------------------------------------------------------------------------- #


def test_conv_chain_lowers_to_blocked_frames(conv_model, tmp_path):
    ctx = lower(conv_model, tmp_path, part=AIE1_PART)
    execution = ctx.ir.execution
    plan = ctx.ir.physical.plan

    c1, c2, c3, fc = (execution.get(f'{n}_aie') for n in ('c1', 'c2', 'c3', 'fc'))
    assert {inst.variant.variant_id for inst in (c1, c2, c3)} == {'conv2d.b.r.v1'}
    assert fc.variant.variant_id == 'dense.b.r.v1'
    assert 'fused_activation' in c1.node.traits and 'bias' in c1.node.roles.values()
    assert c3.node.trait_data('output_view')['kind'] == 'flatten_2d' and c3.config.flags.emit_flattened
    assert (c2.config.kernel_shape, c2.config.pads, c2.config.groups) == ((3, 3), (1, 1, 1, 1), C2)

    # The graph input's frame: 3 channels padded to one 8-block, a 1-pixel zero border, and the
    # image at column 2 so every stored register tile stays aligned.
    x_view = c1.config.io_views[c1.node.inputs[0].name]
    assert x_view.full == (1, H + 2, 12, 8) and x_view.origin == (0, 1, 2, 0)
    # A tensor's padded frame follows from the tensor alone, so producer and consumer agree on it.
    for producer, consumer in ((c1, c2), (c2, c3)):
        tensor = producer.node.outputs[0].name
        assert producer.config.io_views[tensor] == consumer.config.io_views[tensor]
    assert c3.config.io_views[c2.node.outputs[0].name].origin == (0, 0, 0, 0)  # 1x1 needs no border

    # Every internal edge is a direct whole-frame copy; the flattened output feeds Dense's LHS.
    assert {(e['source'], e['target']) for e in plan['direct_edges']} == {
        ('ifm[0]', 'c1_aie.in1[0]'),
        ('c1_aie.out1[0]', 'c2_aie.in1[0]'),
        ('c2_aie.out1[0]', 'c3_aie.in1[0]'),
        ('c3_aie.out1[0]', 'fc_aie.in1[0]'),
        ('fc_aie.out1[0]', 'ofm[0]'),
    }
    assert plan['buffers'] == []

    graph_input = next(p['staging'] for p in plan['io_ports'] if p['direction'] == 'input')
    assert graph_input['storage_layout'] == 'linear'
    assert graph_input['tiling_dimension'] == [8, 12, H + 2, 1]  # the PLIO carries the whole frame
    assert graph_input['io_boundary_dimension'] == [CIN, W, H, 1]
    assert graph_input['logical_origin'] == [0, -2, -1, 0]  # the window opens on the border


@pytest.mark.parametrize('k', [3, 7])
def test_valid_conv_frame_covers_the_computed_width(k, tmp_path):
    """The kernel computes whole register tiles, so the frame must hold every column it reads."""
    ctx = lower(_valid_model(k), tmp_path, part=AIE1_PART)
    conv = ctx.ir.execution.get('c_aie')
    params = conv.variant.build_template_params(conv.node, conv.config, {'row': 0, 'col': 0})
    out_w = H - k + 1
    assert params['out_w'] == out_w
    assert params['out_w_computed'] == -(-out_w // 4) * 4  # 2 register tiles x M=2 on AIE1
    assert params['in_origin_c'] - conv.config.pads[1] + params['out_w_computed'] + k - 1 <= params['in_cols']


def test_conv_weights_pack_compact_groups_into_dense_tiles(conv_model, tmp_path):
    ctx = lower(conv_model, tmp_path, part=AIE1_PART)
    c2 = ctx.ir.execution.get('c2_aie')
    # The IR keeps the compact per-group form; the variant expands the groups when it packs.
    assert tuple(c2.node.inputs[1].shape) == (3, 3, 1, C2)
    packed = c2.artifacts['packed_weights']
    blocks = C2 // 8
    padded = blocks + blocks % 2  # the kernel walks output blocks in pairs
    assert packed.shape == (1, 1, 9 * (C1 // 8) * padded * 64)
    tiles = packed.reshape(9, C1 // 8, padded, 8, 8)  # (tap, cin block, cout block, 8, 8)
    # Depthwise: a tap's (cin, cout) plane is block-diagonal, and each diagonal holds the one
    # coefficient that channel has -- this is the compact-weight contract the packer expands.
    compact = np.asarray(c2.node.inputs[1].data).reshape(3, 3, 1, C2)
    for tap, (ky, kx) in enumerate([(ky, kx) for ky in range(3) for kx in range(3)]):
        for cb in range(C1 // 8):
            for nb in range(padded):
                block = tiles[tap, cb, nb]
                if cb != nb:
                    assert not block.any()
                else:
                    assert np.array_equal(np.diag(block), np.rint(compact[ky, kx, 0, cb * 8 : cb * 8 + 8] * 16))
                    assert np.count_nonzero(block) == np.count_nonzero(np.diag(block))


def test_conv_partitions_channel_blocks_across_tiles(conv_model, tmp_path):
    """cas_num splits the output channel blocks; the consumer reads back exactly those slices."""
    ctx = lower(conv_model, tmp_path, DIRECTIVES, part=AIE1_PART)
    c1, c2 = ctx.ir.execution.get('c1_aie'), ctx.ir.execution.get('c2_aie')
    assert (c1.config.parallelism.cas_num, c1.config.parallelism.cas_length) == (3, 1)
    assert (c2.config.parallelism.cas_num, c2.config.parallelism.cas_length) == (1, 3)
    assert c2.variant.footprint(c2.node, c2.config).width == 3  # a three-tile cascade

    # Each of c1's ports carries one channel block of the shared frame, and c2 reads them back.
    tensor = c1.node.outputs[0].name
    write = [c1.variant.describe_output_staging(c1.node, c1.config, tensor, port) for port in range(3)]
    read = [c2.variant.describe_input_staging(c2.node, c2.config, tensor, port) for port in range(3)]
    assert [d['tiling_dimension'][0] for d in write] == [8, 8, 8]
    assert [d['offset'][0] for d in write] == [0, 8, 16]
    assert all(w['tile_traversal'] == r['tile_traversal'] and w['offset'] == r['offset'] for w, r in zip(write, read))

    # Weights are cut per (chain, column) tile; the bias belongs to the tile that stores, one per
    # chain, and must survive the cut -- a zeroed bias here is the cascade-bias regression.
    assert c1.artifacts['packed_weights'].shape[:2] == (3, 1)
    assert c2.artifacts['packed_weights'].shape[:2] == (1, 3)
    assert c2.artifacts['packed_bias'].shape[0] == 1 and c2.artifacts['packed_bias'].any()
    assert all(row.any() for row in c1.artifacts['packed_bias'])
    plan = ctx.ir.physical.plan
    edges = {(e['source'], e['target']) for e in plan['direct_edges']}
    assert {('c1_aie.out1[2]', 'c2_aie.in1[2]'), ('c1_aie.out1[0]', 'c2_aie.in1[0]')} <= edges


def test_conv_rejects_partitions_it_cannot_cut(conv_model, tmp_path):
    with pytest.raises(ValueError, match='cas_num=5 does not split'):
        lower(conv_model, tmp_path, {'c1': {'parallelism': {'cas_num': 5}}}, part=AIE1_PART)
    # c1 feeds a 3x3 conv, so its output frame carries a border no band can own.
    with pytest.raises(NotImplementedError, match='band-split output'):
        lower(conv_model, tmp_path, {'c1': {'parallelism': {'contract': 'outer', 'cas_num': 2}}}, part=AIE1_PART)


def test_frame_refuses_consumers_that_read_different_windows(tmp_path):
    """One padded frame serves one window: a fanout whose branches pad differently needs a
    per-consumer view, which transport does not materialize, so it is refused up front."""
    nodes: list = []
    inits: list = []
    _start(nodes, inits)
    _conv(nodes, inits, 'x_nchw', 'a', 'c', CIN, C3, 3, pad=1, relu=True, seed=7)
    _conv(nodes, inits, 'a', 'b1', 'w3', C3, C3, 3, pad=1, relu=True, seed=8)  # needs a 1-pixel border
    _conv(nodes, inits, 'a', 'b2', 'w1', C3, C3, 1, pad=0, relu=True, seed=9)  # needs none
    _conv(nodes, inits, 'b1', 'b3', 'p1', C3, C3, 1, pad=0, relu=False, seed=10)
    _conv(nodes, inits, 'b2', 'b4', 'p2', C3, C3, 1, pad=0, relu=False, seed=11)
    nodes.append(helper.make_node('Add', ['b3', 'b4'], ['sum'], name='sum'))
    inits += _qparams('sumo', frac=FRAC)
    qdq(nodes, 'sum', 'a2', 'sumo')
    _conv(nodes, inits, 'a2', 'a3', 'p3', C3, C3, 1, pad=0, relu=False, seed=12)
    _head(nodes, inits, 'a3', H * W * C3, seed=13)
    with pytest.raises(NotImplementedError, match='read different windows'):
        lower(_model('conv_fanout', nodes, inits), tmp_path, part=AIE1_PART)


def test_outer_splits_rows_into_overlapping_bands(tmp_path):
    """Bands overlap by the window span, and a band's window may open before the image: the host
    clips it against the tensor and zero-fills the rest, so no kernel has to own that border."""
    ctx = lower(_band_model(), tmp_path, BAND_DIRECTIVES, part=AIE1_PART)
    conv = ctx.ir.execution.get('b_aie')
    assert conv.config.parallelism.contract == 'outer' and conv.config.parallelism.cas_num == 2

    view = conv.config.io_views[conv.node.inputs[0].name]
    assert view.full[1] == H + 2 and view.tile[1] == H // 2 + 2  # a band of 4 rows plus its halo
    ports = [conv.variant.describe_input_staging(conv.node, conv.config, conv.node.inputs[0].name, p) for p in (0, 1)]
    assert [d['offset'][2] for d in ports] == [0, 4]  # the second band starts 4 frame rows in
    assert [d['logical_origin'][2] for d in ports] == [-1, 3]  # band 0 opens on the top border
    assert all(d['tiling_dimension'][2] == H // 2 + 2 for d in ports)

    params = conv.variant.build_template_params(conv.node, conv.config, {'row': 0, 'col': 0})
    assert (params['out_h'], params['in_rows']) == (H // 2, H // 2 + 2)
    assert not params['fills_border']  # the host delivers each band's window, border included

    # Each band writes its own rows of the output, which the host reassembles.
    out_ports = [
        conv.variant.describe_output_staging(conv.node, conv.config, conv.node.outputs[0].name, p) for p in (0, 1)
    ]
    assert [d['logical_origin'][2] for d in out_ports] == [0, H // 2]
    assert all(d['tiling_dimension'][2] == H // 2 for d in out_ports)

    plan = ctx.ir.physical.plan
    edges = {(e['source'], e['target']) for e in plan['direct_edges']}
    assert {('ifm[0]', 'b_aie.in1[0]'), ('ifm[1]', 'b_aie.in1[1]')} <= edges
    assert {('b_aie.out1[0]', 'ofm[0]'), ('b_aie.out1[1]', 'ofm[1]')} <= edges


def test_stream_conv_carries_the_logical_tensor(tmp_path):
    """A stream carries a wire order, not a memory layout, and the wire is the tensor itself: no
    border, no padded channels, no computed-width tail. The kernel owns all of that."""
    ctx = lower(_stream_model(), tmp_path, STREAM_DIRECTIVES, part=AIE1_PART)
    conv = ctx.ir.execution.get('b_aie')
    assert conv.variant.variant_id == 'conv2d.s.r.v1'
    assert int(conv.config.io_views[conv.node.inputs[0].name].full[-1]) // 8 == 3  # three blocks
    assert {b.kind for b in (*conv.ports.inputs.values(), *conv.ports.outputs.values())} == {'stream'}

    logical = [C2, W, H, 1]  # buffer order: channels, columns, rows, batch
    for staging in (
        conv.variant.describe_input_staging(conv.node, conv.config, conv.node.inputs[0].name, 0),
        conv.variant.describe_output_staging(conv.node, conv.config, conv.node.outputs[0].name, 0),
    ):
        assert staging['storage_layout'] == 'linear'
        assert staging['tiling_dimension'] == staging['buffer_dimension'] == logical
        assert staging['logical_origin'] == [0, 0, 0, 0]  # the window is the tensor

    plan = ctx.ir.physical.plan
    assert plan['buffers'] == [] and plan['kernel_write_accesses'] == []
    assert ('ifm[0]', 'b_aie.in1[0]') in {(e['source'], e['target']) for e in plan['direct_edges']}


def test_conv_refuses_stride(tmp_path):
    nodes: list = []
    inits: list = []
    _start(nodes, inits)
    inits += [
        numpy_helper.from_array(np.zeros((C3, CIN, 3, 3), np.int8), 'w_q'),
        *_qparams('w', frac=FRAC),
        *_qparams('co', frac=FRAC),
    ]
    nodes.append(helper.make_node('DequantizeLinear', ['w_q', 'w_scale', 'w_zp'], ['w']))
    nodes.append(helper.make_node('Conv', ['x_nchw', 'w'], ['cv'], kernel_shape=[3, 3], strides=[2, 2], name='conv'))
    qdq(nodes, 'cv', 'a', 'co')
    _head(nodes, inits, 'a', 3 * 3 * C3, seed=9)
    with pytest.raises(NotImplementedError, match='strides'):
        lower(_model('conv_stride', nodes, inits), tmp_path, part=AIE1_PART)


# --------------------------------------------------------------------------- #
# numerics
# --------------------------------------------------------------------------- #


@pytest.mark.requires_vitis
@pytest.mark.parametrize('part', [AIE1_PART, PART], ids=['aie1', 'aie-ml'])
def test_stream_conv_matches_onnx(tmp_path, part):
    """The wire order and the blocked frame must agree, or the image lands scrambled."""
    feed = np.random.default_rng(12).integers(-40, 40, size=(1, H, W, C1), dtype=np.int8)
    assert_x86_matches_onnx(
        _stream_model(), {'x_q': feed}, STREAM_DIRECTIVES, tmp_path, batch=1, frac=FRAC, max_code_diff=1, part=part
    )


@pytest.mark.requires_vitis
def test_stream_conv_repeats_without_stale_state(tmp_path):
    """The band frame and the beat cursor outlive the call, so a second inference must not inherit
    the rows and the half beat the first one left behind."""
    feed = np.random.default_rng(12).integers(-40, 40, size=(1, H, W, C1), dtype=np.int8)
    assert_x86_matches_onnx(
        _stream_model(),
        {'x_q': feed},
        STREAM_DIRECTIVES,
        tmp_path,
        batch=1,
        frac=FRAC,
        max_code_diff=1,
        part=AIE1_PART,
        iterations=2,
    )


@pytest.mark.requires_vitis
@pytest.mark.parametrize('part', [AIE1_PART, PART], ids=['aie1', 'aie-ml'])
def test_outer_bands_match_onnx(tmp_path, part):
    """Same-padded conv in two row bands: the halo rows and the delivered top/bottom border are
    what this checks, so any mistake in the band windows shows up as wrong pixels."""
    assert_x86_matches_onnx(
        _band_model(), {'x_q': _feed()}, BAND_DIRECTIVES, tmp_path, batch=1, frac=FRAC, max_code_diff=1, part=part
    )


@pytest.mark.requires_vitis
@pytest.mark.parametrize('part', [AIE1_PART, PART], ids=['aie1', 'aie-ml'])
def test_conv_chain_matches_onnx(conv_model, tmp_path, part):
    """One compile covers the single-tile conv, a 3-chain output split, a three-tile cascade
    (first + middle + last, whose last tile owns the bias), a depthwise group and flatten -> Dense.
    The biases are non-zero and differ per channel, so a chain that dropped one would show up."""
    assert_x86_matches_onnx(
        conv_model, {'x_q': _feed()}, DIRECTIVES, tmp_path, batch=1, frac=FRAC, max_code_diff=1, part=part
    )

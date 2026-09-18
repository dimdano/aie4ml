from __future__ import annotations

import numpy as np
import pytest
from helpers import TensorProto, dq, helper, lower, make_model, numpy_helper, qdq, qparams

AIE1_PART = 'xcvp2802-vsva5601-2MHP-e-S'


def _input(nodes, initializers, name: str, rows: int):
    dq(nodes, f'{name}_q', name, name)
    initializers.extend(qparams(name))
    return (f'{name}_q', TensorProto.INT8, [rows, 16])


def _dense(nodes, initializers, source: str, name: str):
    weight_name = f'{name}_w'
    dq(nodes, f'{weight_name}_q', weight_name, weight_name)
    nodes.append(helper.make_node('MatMul', [source, weight_name], [f'{name}_mm'], name=name))
    qdq(nodes, f'{name}_mm', f'{name}_out', name)
    initializers.extend(
        [
            *qparams(weight_name),
            *qparams(name),
            numpy_helper.from_array(np.ones((16, 16), dtype=np.int8), f'{weight_name}_q'),
        ]
    )
    return f'{name}_out'


def _parallelism(cas_num: int):
    return {'parallelism': {'cas_num': cas_num, 'cas_length': 1, 'contract': 'outer'}}


def _split_model(*, graph_output: bool = False):
    nodes, initializers = [], []
    graph_input = _input(nodes, initializers, 'x', 8)
    root = _dense(nodes, initializers, 'x', 'root')
    split_sizes = numpy_helper.from_array(np.asarray([4, 4], dtype=np.int64), 'split_sizes')
    initializers.append(split_sizes)
    nodes.append(helper.make_node('Split', [root, 'split_sizes'], ['upper', 'lower'], name='split', axis=0))

    if graph_output:
        outputs = [('upper', TensorProto.FLOAT, [4, 16])]
    else:
        left = _dense(nodes, initializers, 'upper', 'left')
        right = _dense(nodes, initializers, 'lower', 'right')
        outputs = [(left, TensorProto.FLOAT, [4, 16]), (right, TensorProto.FLOAT, [4, 16])]

    return make_model(
        'aie1_split_views',
        nodes=nodes,
        inputs=[graph_input],
        outputs=outputs,
        initializers=initializers,
    )


def _chained_view_model():
    nodes, initializers = [], []
    graph_input = _input(nodes, initializers, 'x', 8)
    root = _dense(nodes, initializers, 'x', 'root')
    initializers.append(numpy_helper.from_array(np.asarray([4, 4], dtype=np.int64), 'split_sizes'))
    nodes.append(helper.make_node('Split', [root, 'split_sizes'], ['upper', 'lower'], name='split', axis=0))
    nodes.append(helper.make_node('Concat', ['upper', 'lower'], ['joined'], name='concat', axis=0))
    output = _dense(nodes, initializers, 'joined', 'tail')
    return make_model(
        'aie1_chained_views',
        nodes=nodes,
        inputs=[graph_input],
        outputs=[(output, TensorProto.FLOAT, [8, 16])],
        initializers=initializers,
    )


def _slice_model(start: int, end: int, *, source_is_graph_input: bool = False):
    nodes, initializers = [], []
    graph_input = _input(nodes, initializers, 'x', 8)
    source = 'x' if source_is_graph_input else _dense(nodes, initializers, 'x', 'root')
    for name, value in (('starts', [start]), ('ends', [end]), ('axes', [0])):
        initializers.append(numpy_helper.from_array(np.asarray(value, dtype=np.int64), name))
    nodes.append(helper.make_node('Slice', [source, 'starts', 'ends', 'axes'], ['slice_out'], name='slice'))
    output = _dense(nodes, initializers, 'slice_out', 'tail')
    return make_model(
        'aie1_slice_view',
        nodes=nodes,
        inputs=[graph_input],
        outputs=[(output, TensorProto.FLOAT, [end - start, 16])],
        initializers=initializers,
    )


def _concat_model(first_rows: int, second_rows: int, *, graph_output: bool = False):
    nodes, initializers = [], []
    first_input = _input(nodes, initializers, 'a', first_rows)
    second_input = _input(nodes, initializers, 'b', second_rows)
    first = _dense(nodes, initializers, 'a', 'first')
    second = _dense(nodes, initializers, 'b', 'second')
    nodes.append(helper.make_node('Concat', [first, second], ['joined'], name='concat', axis=0))

    if graph_output:
        outputs = [('joined', TensorProto.FLOAT, [first_rows + second_rows, 16])]
    else:
        output = _dense(nodes, initializers, 'joined', 'tail')
        outputs = [(output, TensorProto.FLOAT, [first_rows + second_rows, 16])]

    return make_model(
        'aie1_concat_view',
        nodes=nodes,
        inputs=[first_input, second_input],
        outputs=outputs,
        initializers=initializers,
    )


def test_port_aligned_split_maps_disjoint_producer_ports_directly(tmp_path):
    ctx = lower(
        _split_model(),
        tmp_path,
        part=AIE1_PART,
        batch=8,
        directives={'root': _parallelism(2), 'left': _parallelism(1), 'right': _parallelism(1)},
    )
    edges = {(edge['source'], edge['target']) for edge in ctx.ir.physical.plan['direct_edges']}

    assert ('root_aie.out1[0]', 'left_aie.in1[0]') in edges
    assert ('root_aie.out1[1]', 'right_aie.in1[0]') in edges
    assert ctx.ir.physical.plan['buffers'] == []


def test_slice_crossing_producer_port_requires_relay(tmp_path):
    with pytest.raises(
        NotImplementedError,
        match=r'slice range \[2, 6\) crosses producer port 0 range \[0, 4\); packed slice/relay',
    ):
        lower(
            _slice_model(2, 6),
            tmp_path,
            part=AIE1_PART,
            batch=8,
            directives={'root': _parallelism(2), 'tail': _parallelism(1)},
        )


def test_port_aligned_concat_maps_each_source_to_one_consumer_port(tmp_path):
    ctx = lower(
        _concat_model(4, 4),
        tmp_path,
        part=AIE1_PART,
        batch=8,
        directives={'first': _parallelism(1), 'second': _parallelism(1), 'tail': _parallelism(2)},
    )
    edges = {(edge['source'], edge['target']) for edge in ctx.ir.physical.plan['direct_edges']}

    assert ('first_aie.out1[0]', 'tail_aie.in1[0]') in edges
    assert ('second_aie.out1[0]', 'tail_aie.in1[1]') in edges
    assert ctx.ir.physical.plan['buffers'] == []


def test_concat_port_spanning_two_sources_requires_relay(tmp_path):
    with pytest.raises(
        NotImplementedError,
        match=r'concat consumer port 0.*range \[0, 4\).*packed concat/relay',
    ):
        lower(
            _concat_model(3, 5),
            tmp_path,
            part=AIE1_PART,
            batch=8,
            directives={'first': _parallelism(1), 'second': _parallelism(1), 'tail': _parallelism(2)},
        )


@pytest.mark.parametrize(
    ('model', 'message'),
    [
        (_split_model(graph_output=True), r'upper: split-backed graph outputs are not implemented'),
        (_concat_model(4, 4, graph_output=True), r'joined: concat-backed graph outputs are not implemented'),
    ],
    ids=['split', 'concat'],
)
def test_view_backed_graph_outputs_fail_explicitly(model, message, tmp_path):
    with pytest.raises(NotImplementedError, match=message):
        lower(model, tmp_path, part=AIE1_PART, batch=8)


def test_view_backed_graph_input_fails_explicitly(tmp_path):
    with pytest.raises(NotImplementedError, match=r'slice input .* is a graph input'):
        lower(
            _slice_model(0, 4, source_is_graph_input=True),
            tmp_path,
            part=AIE1_PART,
            batch=8,
            directives={'tail': _parallelism(1)},
        )


def test_chained_views_fail_explicitly(tmp_path):
    with pytest.raises(NotImplementedError, match=r'chained view transport through split'):
        lower(
            _chained_view_model(),
            tmp_path,
            part=AIE1_PART,
            batch=8,
            directives={'root': _parallelism(2), 'tail': _parallelism(2)},
        )

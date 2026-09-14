from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from aie4ml.device_catalog import resolve_device
from aie4ml.frontends.onnx import lower_onnx_model
from aie4ml.op_impls.common_types import PortBinding, to_plain
from aie4ml.op_impls.families.matmul.common import select_generation_key
from aie4ml.op_impls.utils.precision import infer_accumulator_tag
from aie4ml.passes import Resolve
from aie4ml.writer import AIEProjectEmitter
from helpers import TensorProto, helper, make_model, numpy_helper
from jinja2 import Environment, FileSystemLoader

AIE1_PART = 'xcvp2802-vsva5601-2MHP-e-S'


def _qparams(prefix: str, elem_type: int) -> list:
    return [
        helper.make_tensor(f'{prefix}_scale', TensorProto.FLOAT, [], [2.0**-4]),
        helper.make_tensor(f'{prefix}_zp', elem_type, [], [0]),
    ]


def _dense_model(lhs_type: int = TensorProto.INT8, rhs_type: int = TensorProto.INT8):
    np_type = {TensorProto.INT8: np.int8, TensorProto.INT16: np.int16}
    nodes = [
        helper.make_node('DequantizeLinear', ['x_q', 'x_scale', 'x_zp'], ['x'], name='x_dq'),
        helper.make_node('DequantizeLinear', ['w_q', 'w_scale', 'w_zp'], ['w'], name='w_dq'),
        helper.make_node('MatMul', ['x', 'w'], ['mm'], name='dense'),
        helper.make_node('QuantizeLinear', ['mm', 'y_scale', 'y_zp'], ['y_q'], name='y_q'),
        helper.make_node('DequantizeLinear', ['y_q', 'y_scale', 'y_zp'], ['y'], name='y_dq'),
    ]
    return make_model(
        'aie1_dense',
        nodes=nodes,
        inputs=[('x_q', lhs_type, [8, 16])],
        outputs=[('y', TensorProto.FLOAT, [8, 8])],
        initializers=[
            *_qparams('x', lhs_type),
            *_qparams('w', rhs_type),
            *_qparams('y', TensorProto.INT8),
            numpy_helper.from_array(np.ones((16, 8), dtype=np_type[rhs_type]), 'w_q'),
        ],
    )


def _resolve_dense(model, tmp_path, *, part=AIE1_PART, directives=None):
    layer_directives = {'dense': dict(directives)} if directives else {}
    ctx = lower_onnx_model(
        model,
        {
            'Part': part,
            'AIEConfig': {'BatchSize': 8, 'Iterations': 1},
            'LayerDirectives': layer_directives,
        },
        output_dir=tmp_path,
        project_name='aie1_dense',
    )
    Resolve().transform(ctx)
    return ctx, ctx.ir.execution.get('dense_aie').config


def test_aie1_catalog_capabilities_and_raw_part_target(tmp_path):
    device, _ = resolve_device(AIE1_PART, {})

    assert device.generation == 'AIE'
    assert device.columns == 59
    assert device.rows == 8
    assert device.has_memtile is False
    assert device.bank_count == 4
    assert device.bank_mem_bytes == 8 * 1024
    assert device.tile_mem_bytes == 32 * 1024
    assert device.vector_bytes == 32
    assert device.cascade_layout == 'alternating_horizontal'
    assert device.aie_compiler_target == 'part'

    template_root = Path(AIEProjectEmitter()._template_root) / 'firmware'
    env = Environment(loader=FileSystemLoader(str(template_root)), trim_blocks=True, lstrip_blocks=True)
    ctx = SimpleNamespace(
        device=device,
        project_config=SimpleNamespace(project_name='probe', stamp=None),
    )
    AIEProjectEmitter()._render_makefile(tmp_path, ctx, env, None)
    makefile = (tmp_path / 'Makefile').read_text()
    assert f'AIE_PART      ?= {AIE1_PART}' in makefile
    assert 'AIE_TARGET    := --part=$(AIE_PART)' in makefile
    assert '--platform=$(PLATFORM) $(APP_NAME).cpp' not in makefile


@pytest.mark.parametrize('generation', ['', 'AIE-ML3', 'not-a-device'])
def test_unknown_generation_is_rejected(generation):
    with pytest.raises(ValueError, match='Unknown AIE generation'):
        select_generation_key(generation)


def test_aie1_onnx_int8_dense_prefers_direct_compatible_microtile(tmp_path):
    _ctx, config = _resolve_dense(_dense_model(), tmp_path)

    got = (config.microtiling.microtile_m, config.microtiling.microtile_k, config.microtiling.microtile_n)
    assert got == (2, 8, 8)
    assert config.accumulator_tag == 'acc32'


@pytest.mark.parametrize('microtile', [(4, 8, 4), (1, 16, 8)])
def test_aie1_onnx_int8_dense_accepts_other_native_microtiles(tmp_path, microtile):
    m, k, n = microtile
    _ctx, config = _resolve_dense(
        _dense_model(),
        tmp_path,
        directives={'microtiling': {'microtile_m': m, 'microtile_k': k, 'microtile_n': n}},
    )

    assert (config.microtiling.microtile_m, config.microtiling.microtile_k, config.microtiling.microtile_n) == microtile
    assert config.accumulator_tag == 'acc32'


def test_aie1_onnx_int16_int8_dense_uses_compile_proven_shape_and_acc48(tmp_path):
    _ctx, config = _resolve_dense(_dense_model(TensorProto.INT16, TensorProto.INT8), tmp_path)

    assert (config.microtiling.microtile_m, config.microtiling.microtile_k, config.microtiling.microtile_n) == (4, 4, 4)
    assert config.accumulator_tag == 'acc48'


def test_aie1_onnx_int8_int16_dense_is_rejected_before_shift_resolution(tmp_path):
    with pytest.raises(RuntimeError, match=r'unsupported int8 x int16.*output shift'):
        _resolve_dense(_dense_model(TensorProto.INT8, TensorProto.INT16), tmp_path)


def test_aie1_acc80_vocabulary_covers_32_bit_integer_path():
    device, _ = resolve_device(AIE1_PART, {})

    from aie4ml.aie_types import AIEDataType

    assert infer_accumulator_tag(device, AIEDataType('int32'), AIEDataType('int32'), None) == 'acc80'


def test_aie1_onnx_dense_rejects_emulated_int8_microtile(tmp_path):
    with pytest.raises(ValueError, match=r'microtiling \(4, 16, 4\) not supported'):
        _resolve_dense(
            _dense_model(),
            tmp_path,
            directives={'microtiling': {'microtile_m': 4, 'microtile_k': 16, 'microtile_n': 4}},
        )


def test_aie1_onnx_dense_rejects_unenabled_int16_int16(tmp_path):
    with pytest.raises(ValueError, match=r'no dense variant matches.*generation=.AIE.'):
        _resolve_dense(_dense_model(TensorProto.INT16, TensorProto.INT16), tmp_path)


@pytest.mark.parametrize(
    ('part', 'expected_microtile'),
    [('xilinx_vek280_base_202520_1', (4, 8, 8)), ('vek385_base', (8, 8, 8))],
)
def test_existing_ml_generation_default_resolution_is_unchanged(tmp_path, part, expected_microtile):
    _ctx, config = _resolve_dense(_dense_model(), tmp_path, part=part)
    got = (config.microtiling.microtile_m, config.microtiling.microtile_k, config.microtiling.microtile_n)
    assert got == expected_microtile


def test_port_binding_default_is_an_explicit_buffer_in_serialization():
    implicit = PortBinding(group='in1', count=2)

    assert implicit == PortBinding(group='in1', count=2, kind='buffer')
    assert to_plain(implicit) == {'group': 'in1', 'count': 2, 'kind': 'buffer'}

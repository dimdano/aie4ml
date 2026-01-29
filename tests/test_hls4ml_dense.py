import os
from pathlib import Path

import pytest


def _require_vitis():
    if 'XILINX_VITIS' not in os.environ:
        pytest.skip('AMD Vitis not found (XILINX_VITIS not set)')


def _imports():
    np = pytest.importorskip('numpy')
    hls4ml = pytest.importorskip('hls4ml')
    keras = pytest.importorskip('keras')
    qkeras = pytest.importorskip('qkeras')
    return np, hls4ml, keras, qkeras


def _par_summary(layers):
    return '_'.join(f"{k}_c{v['cas_num']}x{v['cas_length']}" for k, v in layers.items())


def _build_qkeras_mlp(qkeras, in_features, hidden1, hidden2, out_features, bits):
    from keras.models import Sequential
    from qkeras import QActivation, QDense, quantized_bits, quantized_relu

    INT_BITS = 2

    q_in = quantized_bits(bits, INT_BITS)
    q_w = quantized_bits(bits, 0, alpha=1)
    q_b = quantized_bits(bits, 0, alpha=1)

    model = Sequential(
        [
            QActivation(q_in, name='input_quant', input_shape=(in_features,)),
            QDense(hidden1, name='dense0', kernel_quantizer=q_w, bias_quantizer=q_b, bias_initializer='random_uniform'),
            QActivation(quantized_relu(bits, 0), name='act0'),
            QDense(hidden2, name='dense1', kernel_quantizer=q_w, bias_quantizer=q_b, bias_initializer='random_uniform'),
            QActivation(quantized_relu(bits, 0), name='act1'),
            QDense(
                out_features, name='dense2', kernel_quantizer=q_w, bias_quantizer=q_b, bias_initializer='random_uniform'
            ),
        ]
    )
    model.compile(optimizer='adam', loss='mse')
    return model


def _make_cfg(hls4ml, model, layer_parallelism):
    cfg = hls4ml.utils.config_from_keras_model(model, granularity='name')
    cfg.setdefault('LayerName', {})

    for lname, params in layer_parallelism.items():
        cfg['LayerName'].setdefault(lname, {})
        cfg['LayerName'][lname]['cas_num'] = int(params['cas_num'])
        cfg['LayerName'][lname]['cas_length'] = int(params['cas_length'])

    return cfg


def _make_aie_model(tmp_path, cfg_dict):
    np, hls4ml, _keras, qkeras = _imports()

    qmodel = _build_qkeras_mlp(
        qkeras,
        in_features=384,
        hidden1=256,
        hidden2=200,
        out_features=10,
        bits=cfg_dict['bits'],
    )

    cfg = _make_cfg(hls4ml, qmodel, cfg_dict['layers'])

    outdir = tmp_path / (f"aie_mlp_b{cfg_dict['bits']}_bs{cfg_dict['batch']}_" f"{_par_summary(cfg_dict['layers'])}")

    aie_model = hls4ml.converters.convert_from_keras_model(
        qmodel,
        hls_config=cfg,
        output_dir=str(outdir),
        backend='aie',
        project_name='proj_aie',
        batch_size=cfg_dict['batch'],
        iterations=5,
    )

    return qmodel, aie_model


CFG_LIST = [
    {
        'bits': 8,
        'batch': 1,
        'layers': {
            'dense0': {'cas_num': 2, 'cas_length': 4},
            'dense1': {'cas_num': 1, 'cas_length': 4},
            'dense2': {'cas_num': 1, 'cas_length': 1},
        },
    },
    {
        'bits': 8,
        'batch': 10,
        'layers': {
            'dense0': {'cas_num': 2, 'cas_length': 4},
            'dense1': {'cas_num': 2, 'cas_length': 2},
            'dense2': {'cas_num': 1, 'cas_length': 2},
        },
    },
]


@pytest.mark.aie_ir
@pytest.mark.parametrize('cfg', CFG_LIST)
def test_aie_backend_conversion_only(tmp_path, cfg):
    """
    Conversion + lowering test.
    No Vitis required.
    """
    qmodel, aie_model = _make_aie_model(tmp_path, cfg)
    assert aie_model is not None


@pytest.mark.aie_ir
@pytest.mark.requires_vitis
@pytest.mark.parametrize('cfg', CFG_LIST)
def test_aie_compile_x86_sim(tmp_path: Path, cfg: dict):
    """
    AIE backend + Vitis compile + x86 functional simulation.
    """
    _require_vitis()

    np, _hls4ml, _keras, _qkeras = _imports()
    qmodel, aie_model = _make_aie_model(tmp_path, cfg)

    aie_model.compile()

    x = np.random.random((cfg['batch'], 384)).astype('float32')

    y_ref = qmodel.predict(x, verbose=0)
    y_aie = aie_model.predict(x, simulator='x86')

    y_aie = y_aie[: cfg['batch']]

    assert y_ref.shape == y_aie.shape
    np.testing.assert_allclose(y_ref, y_aie, rtol=1e-5, atol=1e-5)


# Future TODO
#
# @pytest.mark.aie_ir
# @pytest.mark.requires_vitis
# @pytest.mark.slow
# def test_aie_build_and_hw_sim(...):
#     aie_model.build()
#

"""Integration tests for the float compute path (bfloat16 and float32)."""

import os

import numpy as np
import pytest


def _require_vitis():
    if 'XILINX_VITIS' not in os.environ:
        pytest.skip('AMD Vitis not found (XILINX_VITIS not set)')


def _imports():
    hls4ml = pytest.importorskip('hls4ml')
    keras = pytest.importorskip('keras')
    qkeras = pytest.importorskip('qkeras')
    return hls4ml, keras, qkeras


def _build_models(keras, qkeras, n_in, n_h, n_out, bits=16, seed=0):
    """Return (qkeras_model, keras_reference) sharing the same weights."""
    import tensorflow as tf

    tf.random.set_seed(seed)
    np.random.seed(seed)

    from keras import Input, Model
    from keras.initializers import RandomNormal
    from keras.layers import Dense, ReLU
    from qkeras import QActivation, QDense, quantized_bits, quantized_relu

    init = RandomNormal(mean=0.0, stddev=0.05, seed=seed)
    q_w = quantized_bits(bits, 2, alpha=1)
    q_b = quantized_bits(bits, 2, alpha=1)

    x_in = Input(shape=(n_in,), name='input_layer')
    x = QActivation(quantized_bits(bits, 2, alpha=1), name='input_quant')(x_in)
    x = QDense(n_h, name='dense0', kernel_quantizer=q_w, bias_quantizer=q_b, bias_initializer=init, use_bias=True)(x)
    x = QActivation(quantized_relu(bits, 2), name='act0')(x)
    x = QDense(n_out, name='dense1', kernel_quantizer=q_w, bias_quantizer=q_b, bias_initializer=init, use_bias=True)(x)
    x = QActivation(quantized_bits(bits, 2, alpha=1), name='output_quant')(x)
    qmodel = Model(inputs=x_in, outputs=x)

    # Plain Keras reference — this is the right baseline for ForceFloatMode
    xr = Input(shape=(n_in,), name='input_layer')
    xr_h = Dense(n_h, name='dense0', use_bias=True)(xr)
    xr_h = ReLU()(xr_h)
    xr_h = Dense(n_out, name='dense1', use_bias=True)(xr_h)
    kmodel = Model(inputs=xr, outputs=xr_h)

    kmodel.get_layer('dense0').set_weights(qmodel.get_layer('dense0').get_weights())
    kmodel.get_layer('dense1').set_weights(qmodel.get_layer('dense1').get_weights())

    return qmodel, kmodel


def _bf16_like(x):
    arr = np.asarray(x, dtype=np.float32)
    return ((arr.view(np.uint32) >> 16) << 16).view(np.float32)


def _float_reference(kmodel, x, compute_dtype):
    if compute_dtype != 'bfloat16':
        return kmodel.predict(x, verbose=0)

    x0 = _bf16_like(x)

    w0, b0 = kmodel.get_layer('dense0').get_weights()
    w1, b1 = kmodel.get_layer('dense1').get_weights()

    h = _bf16_like(x0) @ _bf16_like(w0) + np.asarray(b0, dtype=np.float32)
    h = np.maximum(h, 0.0).astype(np.float32, copy=False)
    y = _bf16_like(h) @ _bf16_like(w1) + np.asarray(b1, dtype=np.float32)
    return np.asarray(y, dtype=np.float32)


N_IN, N_H, N_OUT = 32, 32, 16
BATCH = 4

_DTYPES = pytest.mark.parametrize(
    'compute_dtype,rtol,atol',
    [
        ('bfloat16', 1e-2, 5e-2),
        ('float', 1e-3, 1e-3),
    ],
)


@pytest.mark.aie_ir
@_DTYPES
def test_float_conversion(tmp_path, compute_dtype, rtol, atol):
    """Conversion + lowering only — no Vitis required."""
    hls4ml, keras, qkeras = _imports()
    qmodel, _ = _build_models(keras, qkeras, N_IN, N_H, N_OUT)

    cfg = hls4ml.utils.config_from_keras_model(qmodel, granularity='name')
    aie_model = hls4ml.converters.convert_from_keras_model(
        qmodel,
        hls_config=cfg,
        output_dir=str(tmp_path / f'proj_{compute_dtype}'),
        backend='aie',
        project_name=f'proj_{compute_dtype}',
        batch_size=BATCH,
        iterations=3,
        compute_dtype=compute_dtype,
    )
    assert aie_model is not None


@pytest.mark.aie_ir
@pytest.mark.requires_vitis
@_DTYPES
def test_float_x86sim(tmp_path, compute_dtype, rtol, atol):
    """Compile + x86 simulation; output must match the Keras float32 reference."""
    _require_vitis()
    hls4ml, keras, qkeras = _imports()
    qmodel, kmodel = _build_models(keras, qkeras, N_IN, N_H, N_OUT)

    cfg = hls4ml.utils.config_from_keras_model(qmodel, granularity='name')
    aie_model = hls4ml.converters.convert_from_keras_model(
        qmodel,
        hls_config=cfg,
        output_dir=str(tmp_path / f'proj_{compute_dtype}'),
        backend='aie',
        project_name=f'proj_{compute_dtype}',
        batch_size=BATCH,
        iterations=3,
        compute_dtype=compute_dtype,
    )
    aie_model.compile()

    rng = np.random.default_rng(42)
    x = rng.uniform(-2.0, 2.0, size=(BATCH, N_IN)).astype(np.float32)

    y_ref = _float_reference(kmodel, x, compute_dtype)
    y_aie = aie_model.predict(x, simulator='x86')[:BATCH]

    assert y_ref.shape == y_aie.shape
    np.testing.assert_allclose(y_aie, y_ref, rtol=rtol, atol=atol)

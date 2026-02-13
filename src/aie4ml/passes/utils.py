"""Shared helpers for AIE backend passes."""

from __future__ import annotations

from typing import Any

from ..ir import OpNode, TraitInstance


def sanitize_identifier(name: str, prefix: str = 'id') -> str:
    """Return a C/C++ friendly identifier derived from ``name``."""

    if not name:
        return prefix

    filtered = ''.join(ch if (ch.isalnum() or ch == '_') else '_' for ch in str(name))
    filtered = filtered.lstrip('_') or '_'
    if filtered[0].isdigit():
        filtered = f'{prefix}_{filtered}'
    return filtered


def lookup_layer(model: Any, name: str):
    """Return the hls4ml layer instance with the given name, or None if missing."""
    try:
        return model.get_layer(name)
    except AttributeError:
        for layer in model.get_layers():
            if getattr(layer, 'name', None) == name:
                return layer
    return None


def attach_default_io_view(node: OpNode) -> None:
    data = {'inputs': {}, 'outputs': {}}
    for tensor in node.inputs:
        rank = len([int(x) for x in tensor.shape])
        data['inputs'][tensor.name] = {
            'layout': 'channels_last',
            'feature_axis': rank - 1,
            'independent_axes': list(range(rank - 1)),
            'buffer_order': list(reversed(range(rank))),
        }

    for tensor in node.outputs:
        rank = len([int(x) for x in tensor.shape])
        data['outputs'][tensor.name] = {
            'layout': 'channels_last',
            'feature_axis': rank - 1,
            'independent_axes': list(range(rank - 1)),
            'buffer_order': list(reversed(range(rank))),
        }

    node.add_trait(TraitInstance('io_view', data))


def assert_true_pointwise(layer) -> None:
    if layer.class_name == 'Conv1D':
        filt_w = layer.get_attr('filt_width')
        stride_w = layer.get_attr('stride_width')
        pad_l = layer.get_attr('pad_left')
        pad_r = layer.get_attr('pad_right')
        in_w = layer.get_attr('in_width')
        out_w = layer.get_attr('out_width')
        if not (filt_w == 1 and stride_w == 1 and pad_l == 0 and pad_r == 0 and in_w == out_w):
            raise ValueError(f'{layer.name}: PointwiseConv1D is not true pointwise.')
        return

    filt_h = layer.get_attr('filt_height')
    filt_w = layer.get_attr('filt_width')
    stride_h = layer.get_attr('stride_height')
    stride_w = layer.get_attr('stride_width')
    pad_t = layer.get_attr('pad_top')
    pad_b = layer.get_attr('pad_bottom')
    pad_l = layer.get_attr('pad_left')
    pad_r = layer.get_attr('pad_right')
    in_h = layer.get_attr('in_height')
    in_w = layer.get_attr('in_width')
    out_h = layer.get_attr('out_height')
    out_w = layer.get_attr('out_width')
    if not (
        (filt_h, filt_w) == (1, 1)
        and (stride_h, stride_w) == (1, 1)
        and (pad_t, pad_b, pad_l, pad_r) == (0, 0, 0, 0)
        and (in_h, in_w) == (out_h, out_w)
    ):
        raise ValueError(f'{layer.name}: PointwiseConv2D is not true pointwise.')


def is_pointwise_dense(layer) -> bool:
    if layer.class_name not in ('Conv1D', 'Conv2D'):
        return False
    assert_true_pointwise(layer)
    return True

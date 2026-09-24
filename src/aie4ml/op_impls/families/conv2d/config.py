from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ...utils import ParallelismConfig, TensorView
from ...utils.math import align_up
from ..matmul.config import MatmulMicrotileConfig

TRANSFER_ALIGN_BYTES = 16
"""A kernel buffer is whole 16-byte units -- the compiler rounds any other size up -- so an inference
transfer into one is too, the tensor followed by zeros that are not part of it."""


@dataclass(frozen=True)
class Conv2dFlags:
    use_relu: bool
    emit_flattened: bool  # write the result as a Dense LHS row (the folded flatten_2d view)


@dataclass(frozen=True)
class Conv2dConfig:
    """Resolved Conv2D: the canonical attributes plus the padded frames its kernel reads/writes."""

    precision: Dict[str, Any]
    parallelism: ParallelismConfig
    microtiling: MatmulMicrotileConfig
    io_views: Dict[str, TensorView]
    io_route: Dict[str, Any]
    shift: int
    accumulator_tag: Optional[str]
    rounding_mode: Optional[str]
    spatial_blocks: int  # mmul row tiles per accumulator set (2 on AIE, 4 on AIE-ML/MLv2)
    kernel_shape: Tuple[int, int]
    strides: Tuple[int, int]
    dilations: Tuple[int, int]
    pads: Tuple[int, int, int, int]  # (top, left, bottom, right)
    groups: int
    alternating_horizontal: bool
    bank_mem_bytes: int  # one memory bank: what a bank-pinned buffer copy, and the weights, must fit
    flags: Conv2dFlags


@dataclass(frozen=True)
class RetileWindow:
    """The rows one retiler kernel reads and where they land in the frame it builds.

    `rows` source rows starting at `first_row` of the tensor (for a boundary source: a linear row slice)
    fill the frame from row `origin_row`; the rest of the frame is the zero border the kernel clears.
    `transfer_bytes` is one inference of that source in its buffer, whole 16-byte units from the boundary.
    """

    first_row: int
    rows: int
    origin_row: int
    transfer_bytes: int


@dataclass(frozen=True)
class FrameRetileConfig:
    """A retiler: the source it reads and the column-grouped frames it builds, derived once.

    Both layouts are the conv's input frame view -- the tensor's padded frame, which producer and
    consumer derive alike -- read either as the tensor itself (`from_boundary`: the boundary carries it
    linearly) or as the frame a producing kernel wrote, columns in plain order. One kernel per window
    builds one band of that frame: the whole frame, or one of the conv's row bands. The kernels address
    the image in the source with the byte steps below.
    """

    precision: Any  # the element type, as the conv's lhs precision
    source: str  # the value it reads
    target: str  # the frame it writes, which only the execution graph knows
    frame_view: TensorView  # one band's frame: the whole frame when there is one window
    column_phases: int
    band_rows: int  # frame rows between the first rows of neighbouring bands; 0 for one window
    windows: Tuple[RetileWindow, ...]
    from_boundary: bool
    alternating_horizontal: bool  # AIE: odd-row cores reach their east neighbour's memory, not the west's
    bank_mem_bytes: int
    channels: int  # channels each source pixel moves
    source_base: int
    source_pixel: int
    source_row: int
    source_block: int

    @classmethod
    def for_frame(
        cls,
        precision,
        view: TensorView,
        column_phases: int,
        *,
        source: str,
        target: str,
        from_boundary: bool,
        bands: int,
        band_rows: int,
        alternating_horizontal: bool,
        bank_mem_bytes: int,
    ):
        _, height, width, channels = (int(x) for x in view.logical)
        _, rows, cols, padded_channels = (int(x) for x in view.tile)
        top = int(view.origin[1])
        if from_boundary:
            steps = dict(
                channels=channels, source_base=0, source_pixel=channels, source_row=width * channels, source_block=8
            )
            windows = []
            for band in range(int(bands)):
                # Band `band` covers frame rows from band * band_rows: those rows of the tensor that exist.
                start = band * int(band_rows) - top
                first, last = max(0, start), min(height, start + rows)
                transfer = align_up((last - first) * width * channels, TRANSFER_ALIGN_BYTES)
                windows.append(RetileWindow(first, last - first, first - start, transfer))
        else:
            if int(bands) != 1:
                raise NotImplementedError("a retiler reads a producer's frame whole, not in row bands.")
            steps = dict(
                channels=padded_channels,
                source_base=(top * cols + int(view.origin[2])) * 8,
                source_pixel=8,
                source_row=cols * 8,
                source_block=rows * cols * 8,
            )
            windows = [RetileWindow(0, height, top, int(np.prod(view.tile)))]
        return cls(
            precision=precision,
            source=source,
            target=target,
            frame_view=view,
            column_phases=int(column_phases),
            band_rows=int(band_rows),
            windows=tuple(windows),
            from_boundary=from_boundary,
            alternating_horizontal=bool(alternating_horizontal),
            bank_mem_bytes=int(bank_mem_bytes),
            **steps,
        )

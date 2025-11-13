"""Registration helper for the hls4ml AIE backend plugin."""

from __future__ import annotations

from typing import Callable

from hls4ml.backends.backend import Backend

from .aie_backend import AIEBackend
from .writer import AIEWriter


def register(
    register_backend: Callable[[str, type[Backend]], None],
    register_writer: Callable[[str, type], None],
) -> None:
    """Register the AIE backend and writer with hls4ml."""

    register_writer('AIE', AIEWriter)
    register_backend('AIE', AIEBackend)

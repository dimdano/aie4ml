from __future__ import annotations

import os

import pytest


def pytest_runtest_setup(item: pytest.Item) -> None:
    if 'requires_vitis' in item.keywords and 'XILINX_VITIS' not in os.environ:
        pytest.skip('needs AMD Vitis (XILINX_VITIS is not set)')

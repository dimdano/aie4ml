# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""Helpers for loading the AIE device catalog."""

from __future__ import annotations

import json
import os
import re
import shutil
import warnings
from pathlib import Path
from typing import Any, Dict

from .ir.context import DeviceSpec

_DEVICE_CATALOG: Dict[str, Any] | None = None


def load_device_catalog() -> Dict[str, Any]:
    """Return the cached device catalog loaded from aie_devices.json."""
    global _DEVICE_CATALOG
    if _DEVICE_CATALOG is None:
        catalog_path = Path(__file__).with_name('aie_devices.json')
        if catalog_path.exists():
            with open(catalog_path, 'r') as handle:
                _DEVICE_CATALOG = json.load(handle)
        else:
            _DEVICE_CATALOG = {}
    return _DEVICE_CATALOG


_RELEASE_SUFFIX = re.compile(r'_\d{6}_\d+$')


def lookup_device(part_name: str) -> Dict[str, Any]:
    """Return the catalog entry for a Vitis platform or raw device part."""

    catalog = load_device_catalog()
    name = str(part_name)
    for key in (name, name.lower(), _RELEASE_SUFFIX.sub('', name.lower())):
        entry = catalog.get(key)
        if entry:
            return entry
    return {}


PART_HELP = (
    'Pass a known Vitis platform name or raw AIE device part. Platform release suffixes must match '
    'the installed Vitis version. Known targets: {boards}.'
)


def known_boards() -> str:
    return ', '.join(sorted(load_device_catalog())) or '<none>'


def installed_platforms() -> list[str]:
    """Platform names in the local Vitis install, or [] when Vitis is not reachable.

    Mirrors the generated Makefile, which takes VITIS_HOME as dirname(dirname(which vitis)).
    """

    vitis = shutil.which('vitis')
    root = Path(vitis).parent.parent if vitis else Path(os.environ.get('XILINX_VITIS', ''))
    base = root / 'base_platforms'
    return sorted(entry.name for entry in base.iterdir() if entry.is_dir()) if base.is_dir() else []


def resolve_device(part_name: Any, aie_cfg: Dict[str, Any]) -> tuple[DeviceSpec, Dict[str, Any]]:
    entry = lookup_device(part_name)
    if not entry and 'Columns' not in aie_cfg:
        raise ValueError(f'Unknown part "{part_name}". {PART_HELP.format(boards=known_boards())}')
    merged = dict(entry)
    merged.update(aie_cfg)
    merged.setdefault('Generation', entry.get('Generation', ''))

    installed = installed_platforms()
    if merged.get('AIECompilerTarget', 'platform') == 'platform' and installed and str(part_name) not in installed:
        warnings.warn(
            f'Part "{part_name}" is not in this Vitis install, so the generated Makefile will '
            f'point at a missing .xpfm. Installed: {", ".join(installed)}.',
            stacklevel=2,
        )

    return DeviceSpec.from_config(str(part_name), merged), merged

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict


@dataclass(frozen=True)
class PortBinding:
    """Binds one tensor to an ADF port group and replication count.

    `group` is the ADF port array name (for example `in1` or `out2`).
    `count` is the number of physical ports in that group.
    """

    group: str
    count: int


@dataclass(frozen=True)
class PortMap:
    """Port contract for one op implementation instance."""

    inputs: Dict[str, PortBinding]
    outputs: Dict[str, PortBinding]


def to_plain(value: Any):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {k: to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    if hasattr(value, '__dataclass_fields__'):
        return {k: to_plain(getattr(value, k)) for k in value.__dataclass_fields__}
    return value

# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union


class RoundingMode(Enum):
    TRN = 'TRN'
    RND_MIN_INF = 'RND_MIN_INF'
    TRN_ZERO = 'TRN_ZERO'
    RND_ZERO = 'RND_ZERO'
    RND_INF = 'RND_INF'
    RND_CONV = 'RND_CONV'
    RND = 'RND'


class SaturationMode(Enum):
    WRAP = 'WRAP'
    SAT = 'SAT'
    SAT_ZERO = 'SAT_ZERO'
    SAT_SYM = 'SAT_SYM'


@dataclass(frozen=True)
class QuantIntent:
    width: int
    frac: int
    signed: bool
    rounding: RoundingMode
    saturation: SaturationMode


class FloatFormat(Enum):
    BF16 = 'bfloat16'
    FP32 = 'float32'


@dataclass(frozen=True)
class FloatIntent:
    width: int
    format: FloatFormat


PrecisionIntent = Union[QuantIntent, FloatIntent]


@dataclass(frozen=True)
class AIEDataType:
    width: int
    signed: bool
    frac: int = 0
    rounding: RoundingMode = RoundingMode.RND
    saturation: SaturationMode = SaturationMode.SAT
    c_type: Optional[str] = None

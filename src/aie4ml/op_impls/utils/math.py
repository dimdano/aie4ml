from __future__ import annotations


def align_up(value: int, multiple: int) -> int:
    """Return the smallest multiple of ``multiple`` that is >= ``value``."""
    if multiple <= 0:
        return max(0, int(value))
    return ((int(value) + int(multiple) - 1) // int(multiple)) * int(multiple)


def ceildiv(numer: int, denom: int) -> int:
    """Integer ceil division with explicit non-zero denominator validation."""
    denom = int(denom)
    if denom <= 0:
        raise ValueError(f'ceildiv requires denom > 0, got {denom}.')
    numer = int(numer)
    return (numer + denom - 1) // denom

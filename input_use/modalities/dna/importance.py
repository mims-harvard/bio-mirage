"""Locates the variant in a BioReason query: the positions where the reference and variant sequences
differ, and the smallest window that contains the difference, including for insertions and
deletions.
"""
from __future__ import annotations

from typing import List


def variant_positions(reference: str, variant: str) -> List[int]:
    """0-based positions where reference and variant differ (the variant SNV(s))."""
    n = min(len(reference), len(variant))
    return [i for i in range(n) if reference[i] != variant[i]]


def variant_span(reference: str, variant: str):
    """(start, ref_end, var_end): the smallest window that brackets the difference, indel-safe."""
    if reference == variant:
        mid = len(reference) // 2
        return mid, mid, mid
    i = 0
    while i < min(len(reference), len(variant)) and reference[i] == variant[i]:
        i += 1
    j = 0
    while (j < min(len(reference), len(variant)) - i
           and reference[len(reference) - 1 - j] == variant[len(variant) - 1 - j]):
        j += 1
    return i, len(reference) - j, len(variant) - j

"""Edits protein sequences: alanine substitution at chosen positions, whole-sequence shuffling that
keeps amino acid composition, deletion or retention of residue spans, reversal and terminal masking.
`make_conditions` builds the wild-type, functional residue substitution, control residue
substitution and shuffled sequences for one protein.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List

from input_use.modalities.protein.importance import SalientControl

AA = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass
class Condition:
    name: str
    sequence: str
    mutations: List[Dict] = field(default_factory=list)  # [{position, wt_aa, mut_aa}]
    note: str = ""


def alanine_substitute(sequence: str, positions: List[int],
                       target: str = "A", alt_for_target: str = "G") -> Condition:
    """Substitute each 1-based position to `target` (alanine); positions already == target go to
    `alt_for_target` (glycine) so the edit is always real. Returns the perturbed Condition.
    """
    s = list(sequence)
    muts = []
    for p in positions:
        if p < 1 or p > len(s):
            continue
        wt = s[p - 1]
        mut = alt_for_target if wt == target else target
        if wt == mut:
            continue
        s[p - 1] = mut
        muts.append({"position": p, "wt_aa": wt, "mut_aa": mut})
    return Condition(name="", sequence="".join(s), mutations=muts)


def scramble(sequence: str, seed: int = 0) -> Condition:
    """Composition-preserving shuffle of the whole sequence (destroys all positional/sequence
    information while keeping amino-acid composition identical).
    """
    rng = random.Random(seed)
    chars = list(sequence)
    rng.shuffle(chars)
    scr = "".join(chars)
    return Condition(name="scramble", sequence=scr,
                     note="whole-sequence shuffle (composition preserved)")


def frac_positions(positions: List[int], frac: float) -> List[int]:
    """First ceil(frac * len) positions (sorted) - for the salient/control dose-response."""
    import math
    k = max(1, min(len(positions), math.ceil(frac * len(positions))))
    return sorted(positions)[:k]


def window_positions(centers: List[int], k: int, seq_len: int) -> List[int]:
    """Union of 1-based positions within +/-k of any center (a +/-k window around each functional
    residue), clipped to [1, seq_len].
    """
    pos = set()
    for c in centers:
        pos.update(p for p in range(c - k, c + k + 1) if 1 <= p <= seq_len)
    return sorted(pos)


def delete_positions(sequence: str, positions) -> str:
    """Sequence with the given 1-based positions removed (used for the window-delete variant)."""
    drop = set(positions)
    return "".join(ch for i, ch in enumerate(sequence, start=1) if i not in drop)


def keep_span(sequence: str, start: int, end: int) -> str:
    """Subsequence at 1-based inclusive [start, end] (domain-only truncation)."""
    return sequence[start - 1:end]


def delete_span(sequence: str, start: int, end: int) -> str:
    """Sequence with the 1-based inclusive [start, end] block removed (remainder-only)."""
    return sequence[:start - 1] + sequence[end:]


def reverse_seq(sequence: str) -> str:
    """Whole-sequence reversal (preserves composition and the multiset of local content, flips N->C
    directionality).
    """
    return sequence[::-1]


def mask_terminal(sequence: str, k: int, where: str = "N", fill: str = "X") -> str:
    """Replace a k-residue block with the unknown token `X` (ESM2 <unk> amino acid), keeping length.
    where: 'N' (first k), 'C' (last k), or 'M' (centered).
    """
    L = len(sequence)
    k = max(0, min(k, L))
    if where == "N":
        return fill * k + sequence[k:]
    if where == "C":
        return sequence[:L - k] + fill * k
    mid = (L - k) // 2                      # centered block
    return sequence[:mid] + fill * k + sequence[mid + k:]


def random_nondomain_span(seq_len: int, length: int, exclude: set, seed: int = 0):
    """A contiguous 1-based [start, end] block of `length` residues whose positions avoid `exclude`
    (domain + functional + terminal positions) - the matched-size control for domain deletion.
    Returns None if no such block exists (e.g. the domain spans most of the protein).
    """
    if length <= 0 or length > seq_len:
        return None
    starts = [s for s in range(1, seq_len - length + 2)
              if all((p not in exclude) for p in range(s, s + length))]
    if not starts:
        return None
    rng = random.Random(seed + 31)
    s = rng.choice(starts)
    return (s, s + length - 1)


def make_conditions(sc: SalientControl, seed: int = 0,
                    include_scramble: bool = True) -> Dict[str, Condition]:
    """Standard counterfactual condition set for one protein from its S+/S- sets."""
    conds: Dict[str, Condition] = {}
    conds["wt"] = Condition(name="wt", sequence=sc.sequence)

    # Use the matched salient positions (those with an AA-matched control) so mut_salient and
    # mut_control perturb identical residue counts + AA compositions - the fair comparison.
    c = alanine_substitute(sc.sequence, sc.matched_salient_positions())
    c.name = "mut_salient"
    c.note = f"alanine-scan {len(c.mutations)} AA-matched UniProt functional-site residues"
    conds["mut_salient"] = c

    c = alanine_substitute(sc.sequence, sc.control.positions)
    c.name = "mut_control"
    c.note = f"alanine-scan {len(c.mutations)} AA-matched non-functional residues"
    conds["mut_control"] = c

    if include_scramble:
        conds["scramble"] = scramble(sc.sequence, seed=seed)
    return conds

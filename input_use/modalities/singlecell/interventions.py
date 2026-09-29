"""Edits of C2S-Scale gene sentences (space-separated gene names ranked by expression).

`make_conditions` builds, for one cell, the intact sentence, the fixed average cell sentence
(`no_modality`), gene order shuffling, replacement of the visible genes by expressed genes ranked
below them, and removal of a fraction of the strongest DEGs, the weakest DEGs, non-DEGs with similar
expression, or random genes. The DEG removal conditions are described in the RQ1 appendix "Removing
differentially expressed genes from cell sentences" (Figure 4, right).
"""
from __future__ import annotations

import math
import random
from typing import List, Optional

import numpy as np


def remove_genes(cell_sentence: str, genes_to_remove: List[str]) -> str:
    drop = set(genes_to_remove)
    return " ".join(g for g in cell_sentence.split(" ") if g and g not in drop)


def scramble_rank_sentence(cell_sentence: str, rng: random.Random) -> str:
    """Stress test 1a (C.SCRAMBLE_RANK): permute the order of the same genes already in the cell
    sentence -- every gene's position is randomized, but the visible gene set is untouched.
    """
    genes = [g for g in cell_sentence.split(" ") if g]
    rng.shuffle(genes)
    return " ".join(genes)


def scramble_geneset_sentence(cell_sentence: str, remainder_genes: List[str],
                                rng: random.Random) -> str:
    """Stress test 1b (C.SCRAMBLE_GENESET): replace the visible gene set with a different, same-size
    random sample drawn from `remainder_genes` (this cell's own real, expressed genes ranked beyond
    the visible window -- never zero-count filler, see prepare.py) -- the complement of
    scramble_rank_sentence: order is left faithful (the sample is re-sorted into remainder_genes'
    own true rank order) while gene identity is randomized.
    """
    genes = [g for g in cell_sentence.split(" ") if g]
    k = min(len(genes), len(remainder_genes))
    if k == 0:
        return ""
    substitute = set(rng.sample(remainder_genes, k))
    return " ".join(g for g in remainder_genes if g in substitute)  # restore true rank order


def scramble_rank_expression(expr: np.ndarray, top_idx: np.ndarray, rng: random.Random) -> np.ndarray:
    """Stress test 1a (C.SCRAMBLE_RANK), Pipeline B: permute expression values among the same top_idx
    gene positions (this cell's top-N by raw expression, computed by the caller to match Pipeline
    A's visible window) -- genes outside top_idx are untouched, so the same gene set remains the
    highest- expressed set after the permutation; only their relative order changes.
    """
    out = expr.copy()
    vals = out[top_idx].copy()
    perm = list(range(len(top_idx)))
    rng.shuffle(perm)
    out[top_idx] = vals[perm]
    return out


def scramble_geneset_expression(expr: np.ndarray, top_idx: np.ndarray,
                                  rng: random.Random) -> np.ndarray:
    """Stress test 1b (C.SCRAMBLE_GENESET), Pipeline B: transplant the top_idx genes' own value profile
    (sorted, descending) onto a randomly chosen same-size set of different gene positions (drawn
    from every index not in top_idx), and zero out the vacated original top_idx positions.
    """
    out = expr.copy()
    top_set = set(top_idx.tolist())
    remaining = [j for j in range(len(expr)) if j not in top_set]
    k = min(len(top_idx), len(remaining))
    substitute_idx = rng.sample(remaining, k)
    vals_sorted = np.sort(out[top_idx])[::-1][:k]
    out[top_idx] = 0.0
    out[np.array(substitute_idx, dtype=int)] = vals_sorted
    return out


def dropout_expression(expr: np.ndarray, gene_idx: List[int]) -> np.ndarray:
    """Stress test 2, Pipeline B: zero out the given gene positions -- the array-level equivalent of
    remove_genes for the text pipeline.
    """
    out = expr.copy()
    out[gene_idx] = 0.0
    return out


def make_conditions(cell_sentence: str, salient: List[str], control: List[str],
                    generic_sentence: str = "", rng: Optional[random.Random] = None,
                    remainder_genes: Optional[List[str]] = None, battery: str = "both",
                    deg_dropout_mode: str = "fraction") -> dict:
    """wt / no_modality (always) + stress_test_1 (scramble_rank / scramble_geneset) and/or
    stress_test_2 (top_deg_dropout_k{K} (S+) / matched_nondeg_dropout_k{K} (S-)), per `battery`
    ("stress_test_1", "stress_test_2", "both", or "none" -- wt/no_modality only, no stress-test
    additions at all; e.g.
    """
    from input_use.core import conditions as C
    assert battery in ("stress_test_1", "stress_test_2", "both", "none"), battery
    if rng is None:
        rng = random.Random()
    conds = {
        C.WT: cell_sentence,
        C.NO_MODALITY: generic_sentence,
    }
    if battery in ("stress_test_1", "both"):
        conds[C.SCRAMBLE_RANK] = scramble_rank_sentence(cell_sentence, rng)
        if remainder_genes:
            conds[C.SCRAMBLE_GENESET] = scramble_geneset_sentence(cell_sentence, remainder_genes, rng)
    if battery in ("stress_test_2", "both"):
        if deg_dropout_mode == "fraction":
            # n = ceil(frac * |S+|) -- ceil so the smallest dose still removes at least one DEG from
            # a cell with few of them (floor would silently make 10% a no-op wherever |S+| < 10,
            # i.e.
            for f in C.DEG_DROPOUT_FRACTIONS:
                n = min(len(salient), math.ceil(f * len(salient)))
                conds[C.top_deg_dropout_pct(f)] = remove_genes(cell_sentence, salient[:n])
                conds[C.matched_nondeg_dropout_pct(f)] = remove_genes(cell_sentence, control[:n])
                # `salient` is sorted by descending salience, so the last n are the weakest DEGs. At
                # f=1.0 this is the same set as top, so that duplicate is skipped.
                if f < 1.0:
                    conds[C.bottom_deg_dropout_pct(f)] = remove_genes(cell_sentence, salient[-n:])
            # Random arm drawn after every deterministic condition above.
            sentence_genes = [g for g in cell_sentence.split(" ") if g]
            for f in C.DEG_DROPOUT_FRACTIONS:
                n = min(len(salient), math.ceil(f * len(salient)))
                n = min(n, len(sentence_genes))
                conds[C.random_dropout_pct(f)] = remove_genes(
                    cell_sentence, rng.sample(sentence_genes, n))
        else:
            for k in C.DEG_DROPOUT_KS:
                conds[C.top_deg_dropout_k(k)] = remove_genes(cell_sentence, salient[:k])
                conds[C.matched_nondeg_dropout_k(k)] = remove_genes(cell_sentence, control[:k])
    return conds

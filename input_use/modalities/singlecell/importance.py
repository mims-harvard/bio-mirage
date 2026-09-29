"""Finds DEGs for single-cell inputs and selects non-DEGs with similar expression.

`compute_markers` runs a one-vs-rest Wilcoxon test per annotated cell type and keeps genes with log2
fold change of at least 1.0 and adjusted p-value below 0.05, scored by log2 fold change times -log10
adjusted p-value. `salient_and_control_genes` returns the DEGs in a cell's gene sentence and, for
each one, the non-DEG of the same sentence with the closest expression value.
"""
from __future__ import annotations

import math
from typing import Dict, List


def compute_markers(adata, groupby: str = "cell_type", n_top: int = None,
                    method: str = "wilcoxon", min_log2fc: float = 1.0,
                    max_padj: float = 0.05) -> Dict[str, Dict[str, float]]:
    """Per-cell-type DEGs (uppercase symbol -> salience score) via scanpy `rank_genes_groups`
    (one-vs-rest) on the log-norm expression.
    """
    import scanpy as sc
    a = adata.copy()
    sc.tl.rank_genes_groups(a, groupby, method=method, n_genes=n_top)
    res = a.uns["rank_genes_groups"]
    names, logfc, padj = res["names"], res["logfoldchanges"], res["pvals_adj"]
    markers = {}
    for grp in names.dtype.names:
        scores = {}
        for i in range(len(names[grp])):
            lfc = float(logfc[grp][i])
            q = float(padj[grp][i])
            if not (lfc >= min_log2fc and q < max_padj):
                continue
            gene = str(names[grp][i]).upper()
            scores[gene] = lfc * -math.log10(max(q, 1e-300))   # guard log10(0)
        markers[grp] = scores
    return markers


def salient_and_control_genes(visible_genes: List[str], cell_type: str,
                              markers: Dict[str, Dict[str, float]], expr_by_gene: Dict[str, float],
                              k: int = 25) -> Dict:
    """S+ = the cell-sentence genes that are DEGs of this cell's type (compute_markers already applied
    the log2FC + fdr thresholds), ranked by salience score. S- = matched non-DEG genes from the same
    sentence, nearest-expression to each S+, same count.
    """
    import bisect

    tmark = markers.get(cell_type, {})
    candidates = [g for g in visible_genes if g in tmark]
    candidates.sort(key=lambda g: tmark[g], reverse=True)
    s_plus = candidates if (k is None or k <= 0) else candidates[:k]

    nonmark = [g for g in visible_genes if g not in tmark]
    nonmark.sort(key=lambda g: expr_by_gene.get(g, 0.0))
    axis = [expr_by_gene.get(g, 0.0) for g in nonmark]
    used = [False] * len(nonmark)

    control = []
    for g in s_plus:
        target = expr_by_gene.get(g, 0.0)
        p = bisect.bisect_left(axis, target)
        # walk outward from the insertion point for the nearest not-yet-used control gene
        lo, hi, best = p - 1, p, None
        while lo >= 0 or hi < len(nonmark):
            cl = abs(axis[lo] - target) if lo >= 0 else float("inf")
            ch = abs(axis[hi] - target) if hi < len(nonmark) else float("inf")
            if cl == float("inf") and ch == float("inf"):
                break
            if cl <= ch:
                if not used[lo]:
                    best = lo; break
                lo -= 1
            else:
                if not used[hi]:
                    best = hi; break
                hi += 1
        if best is None:
            break
        used[best] = True
        control.append(nonmark[best])

    m = min(len(s_plus), len(control))
    return {"salient": s_plus[:m], "control": control[:m]}

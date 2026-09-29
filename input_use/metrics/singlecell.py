"""Cell type matching for single cell scoring: exact string match, and a Cell Ontology match that also
accepts an ancestor of the annotated type.
"""
from __future__ import annotations

from input_use.metrics.cell_ontology import cl_id_for, cl_coarse_match, resolve_free_text_to_cl


def fine_match(pred: str, gt: str) -> bool:
    return pred == gt


def coarse_match(pred: str, gt: str):
    """Cell Ontology match: True if `pred` equals `gt` or resolves to `gt`'s term or one of its
    `is_a` ancestors. False if `pred` resolves to no term, to a descendant of `gt`, or to an
    unrelated term. None if `gt` resolves to no Cell Ontology term.
    """
    # Resolve the ground truth against the full CL name/synonym index too, not just the 34 hardcoded
    # immune-atlas labels.
    gt_cl = cl_id_for(gt) or resolve_free_text_to_cl(gt)
    if gt_cl is None:
        return None
    if pred == gt:
        return True
    pred_cl = cl_id_for(pred) or resolve_free_text_to_cl(pred)
    if pred_cl is None:
        return False
    return cl_coarse_match(pred_cl, gt_cl)

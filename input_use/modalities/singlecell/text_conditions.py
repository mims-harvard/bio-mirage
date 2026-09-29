"""Metadata text conditions for single-cell prompts, with prompt fragments for C2S-Scale.

The conditions keep the intact cell sentence and state the tissue, sex, developmental stage and
donor of the cell, then remove the donor, stage and sex in turn. `text_full_no_modality` gives the
full metadata with the fixed average cell sentence. `text_positive_control` states the annotated
cell type in the prompt and carries a single `cell_type` field in place of the metadata fields.
"""
from __future__ import annotations

from typing import Dict, Optional

FIELD_ORDER = ["tissue", "sex", "development_stage", "donor_id"]


def make_text_field_sets(tissue: str, sex: str, development_stage: str, donor_id: str,
                         cell_type: str) -> Dict[str, dict]:
    """condition_name -> ordered dict of kept fields for every text-reliance condition except plain
    `wt` and text_full_no_modality (which reuses TEXT_FULL's field set -- see make_text_conditions):
    the 4 field-ablation levels (subset of FIELD_ORDER, always in FIELD_ORDER order; values are the
    per-cell obs strings from the cross-tissue immune atlas) plus text_positive_control (a single
    {"cell_type": ...} dict, this cell's ground-truth label -- a different shape on purpose, see
    module docstring).
    """
    from input_use.core import conditions as C
    full = {"tissue": tissue, "sex": sex, "development_stage": development_stage, "donor_id": donor_id}
    return {
        C.TEXT_FULL: dict(full),
        C.TEXT_DROP_DONOR: {k: full[k] for k in ("tissue", "sex", "development_stage")},
        C.TEXT_DROP_DONOR_STAGE: {k: full[k] for k in ("tissue", "sex")},
        C.TEXT_DROP_DONOR_STAGE_SEX: {"tissue": full["tissue"]},
        C.TEXT_POSITIVE_CONTROL: {"cell_type": cell_type},
    }


def make_text_conditions(cell_sentence: str, generic_sentence: str, tissue: str, sex: str,
                         development_stage: str, donor_id: str, cell_type: str) -> Dict[str, dict]:
    """Full text-reliance battery: the 4 field-ablation levels + text_positive_control at the wt cell
    sentence, plus the text_full_no_modality diagnostic corner. Returns condition_name ->
    {"cell_sentence": <wt|generic>, "text_fields": <field dict, see make_text_field_sets>}.
    """
    from input_use.core import conditions as C
    field_sets = make_text_field_sets(tissue, sex, development_stage, donor_id, cell_type)
    conds = {}
    for name in (C.TEXT_FULL, C.TEXT_DROP_DONOR, C.TEXT_DROP_DONOR_STAGE, C.TEXT_DROP_DONOR_STAGE_SEX,
                C.TEXT_POSITIVE_CONTROL):
        conds[name] = {"cell_sentence": cell_sentence, "text_fields": field_sets[name]}
    conds[C.TEXT_FULL_NO_MODALITY] = {"cell_sentence": generic_sentence,
                                      "text_fields": field_sets[C.TEXT_FULL]}
    return conds


def c2s_metadata_clause(text_fields: Optional[dict]) -> str:
    """C2S prompt metadata clause fragment, e.g. ' obtained from lung tissue of a female donor (sixth
    decade stage), donor A29' -- empty string (no leading space either) if text_fields is
    empty/None, so the stock C2S template wording is reproduced exactly at the zero-metadata level.
    """
    if not text_fields or "cell_type" in text_fields:
        return ""
    clause = f"obtained from {text_fields['tissue']} tissue"
    if "sex" in text_fields:
        clause += f" of a {text_fields['sex']} donor"
        if "development_stage" in text_fields:
            clause += f" ({text_fields['development_stage']})"
    if "donor_id" in text_fields:
        clause += f", donor {text_fields['donor_id']}"
    return " " + clause


def c2s_positive_control_clause(text_fields: Optional[dict]) -> str:
    """The clause that states the ground-truth cell type in the prompt, for the
    `text_positive_control` condition. Returns "" unless text_fields has the {"cell_type": ...}
    shape of that condition, else the sentence naming the cell type, prefixed with a newline.
    """
    if not text_fields or "cell_type" not in text_fields:
        return ""
    return f"\nA board-certified pathologist confirmed this cell is a {text_fields['cell_type']}."

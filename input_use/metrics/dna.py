"""Disease prediction scoring for BioReason, reimplemented from the released BioReason training script
so that scores match the checkpoints' own evaluation.

A generation is mapped onto the closed vocabulary of gold disease labels (longest label first) and
compared with the gold label. The module also holds the metrics for query pairs (pair accuracy,
response table, directional precision), the rate at which answers follow a donor variant, and A/B
choice parsing.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, Optional


def normalize_label_text(text) -> str:
    """`train_dna_qwen.py:204 _normalize_label_text`, verbatim."""
    if text is None:
        return ""
    n = text.strip().lower().replace("\n", " ")
    n = re.sub(r"[^\w\s\-'/]", " ", n)
    n = re.sub(r"\s+", " ", n)
    return n.strip()


_NORM_LABELS_CACHE: dict = {}


def _normalized_labels(labels: Iterable[str]):
    """Labels normalized and sorted longest-first, memoized on the vocabulary."""
    key = tuple(labels)
    hit = _NORM_LABELS_CACHE.get(key)
    if hit is None:
        hit = sorted(((normalize_label_text(l), l) for l in key if normalize_label_text(l)),
                     key=lambda x: len(x[0]), reverse=True)
        _NORM_LABELS_CACHE[key] = hit
    return hit


def extract_predicted_label(generation: str, labels: Iterable[str]):
    """`train_dna_qwen.py:214 _extract_predicted_label`, verbatim."""
    full = generation or ""
    seg = full
    if "<|im_start|>assistant" in seg:
        seg = seg.rsplit("<|im_start|>assistant", 1)[-1]
    if "Answer:" in seg:
        seg = seg.rsplit("Answer:", 1)[-1]
    if "<|im_end|>" in seg:
        seg = seg.split("<|im_end|>", 1)[0]
    seg = seg.strip()
    na, nf = normalize_label_text(seg), normalize_label_text(full)
    norm = _normalized_labels(labels)
    for n, raw in norm:
        if n in na:
            return raw, seg
    for n, raw in norm:
        if n in nf:
            return raw, seg
    return None, seg


def label_of(generation: str, labels: Iterable[str]) -> Optional[str]:
    """The mapped label alone. This is the canonical prediction: every comparison below is equality
    between two of these, so scoring is exact once the mapping is done.
    """
    return extract_predicted_label(generation, labels)[0]


def canonical_gold(gold: str) -> str:
    """Gold truncated at the first ';' and normalized, as the repo does at `train_dna_qwen.py:883`."""
    return normalize_label_text((gold or "").split(";")[0].strip())


def bioposttrain_correct(generation: str, gold: str, labels: Iterable[str]) -> bool:
    """The repo's `is_exact_match` (`train_dna_qwen.py:888`)."""
    pred = label_of(generation, labels)
    return pred is not None and normalize_label_text(pred) == canonical_gold(gold)


def gold_named_in(gold: str, generation: str) -> bool:
    """Did the generation mention the gold label anywhere, even if it answered something else?"""
    g = canonical_gold(gold)
    return bool(g) and g in normalize_label_text(generation or "")


# --- pair metrics -------------------------------------------------------------------------------
# `predicted` maps example_id -> mapped label (or None).

def pair_accuracy(pairs, predicted: Dict[str, Optional[str]]) -> dict:
    """Fraction of pairs where both members are answered correctly."""
    both = one = neither = missing = 0
    for p in pairs:
        pa, pb = predicted.get(p["a"]), predicted.get(p["b"])
        if pa is None or pb is None:
            missing += 1
            continue
        hits = ((normalize_label_text(pa) == canonical_gold(p["answer_a"]))
                + (normalize_label_text(pb) == canonical_gold(p["answer_b"])))
        both += hits == 2
        one += hits == 1
        neither += hits == 0
    n = both + one + neither
    return {"n_pairs": n, "n_missing": missing, "both_correct": both, "one_correct": one,
            "neither_correct": neither, "pair_accuracy": round(both / n, 4) if n else 0.0}


def pair_response_table(pairs, predicted: Dict[str, Optional[str]]) -> dict:
    """The should-change x did-change 2x2, pooled over both strata, as Youden's J."""
    tab = {"change_when_should": 0, "no_change_when_should": 0,
           "change_when_should_not": 0, "no_change_when_should_not": 0, "n_missing": 0}
    for p in pairs:
        pa, pb = predicted.get(p["a"]), predicted.get(p["b"])
        if pa is None or pb is None:
            tab["n_missing"] += 1
            continue
        did = normalize_label_text(pa) != normalize_label_text(pb)
        should = bool(p["should_change"])
        tab[("" if did else "no_") + "change_when_should" + ("" if should else "_not")] += 1
    pos = tab["change_when_should"] + tab["no_change_when_should"]
    neg = tab["change_when_should_not"] + tab["no_change_when_should_not"]
    # An empty margin means J is undefined, not bad.
    sens = tab["change_when_should"] / pos if pos else None
    spec = tab["no_change_when_should_not"] / neg if neg else None
    j = None if (sens is None or spec is None) else round(sens + spec - 1.0, 4)
    return {**tab, "n_should_change": pos, "n_should_not_change": neg,
            "sensitivity": None if sens is None else round(sens, 4),
            "specificity": None if spec is None else round(spec, 4),
            "youden_j": j}


def directional_precision(pairs, predicted: Dict[str, Optional[str]]) -> dict:
    """Of the should-change pairs where the answer did change, how often did it change correctly?"""
    changed = correct = 0
    for p in pairs:
        if not p.get("should_change"):
            continue
        pa, pb = predicted.get(p["a"]), predicted.get(p["b"])
        if pa is None or pb is None:
            continue
        if normalize_label_text(pa) == normalize_label_text(pb):
            continue                                   # no change; direction undefined
        changed += 1
        correct += (normalize_label_text(pa) == canonical_gold(p["answer_a"])
                    and normalize_label_text(pb) == canonical_gold(p["answer_b"]))
    return {"n_changed": changed, "n_correct_direction": correct,
            "directional_precision": round(correct / changed, 4) if changed else None,
            "chance": 0.5}


def genomic_utilization(pairs, predicted_by_arm: Dict[str, Dict[str, Optional[str]]],
                        content_control: str = "scramble", floor: str = "no_dna") -> dict:
    """Decompose selective genomic utilization into responsiveness and content."""
    out = {}
    for arm, pred in predicted_by_arm.items():
        rt = pair_response_table(pairs, pred)
        dp = directional_precision(pairs, pred)
        out[arm] = {"u_genome": rt["youden_j"], "sensitivity": rt["sensitivity"],
                    "specificity": rt["specificity"], **dp}
    wt, ctl = out.get("wt", {}).get("u_genome"), out.get(content_control, {}).get("u_genome")
    res = {"per_arm": out, "content_control": content_control, "floor": floor}
    if wt is not None and ctl is not None:
        res["u_genome_wt"] = wt
        res["u_genome_content_control"] = ctl
        res["content_attributable"] = round(wt - ctl, 4)
        res["fraction_surviving_scramble"] = round(ctl / wt, 4) if wt else None
    return res


def donor_following_rate(predicted: Dict[str, Optional[str]], donor_answers, ground_truth) -> dict:
    """On the donor-swap arm: did the answer move to the donor's label, stay on the original, or
    neither? Donors are drawn with a different answer by construction, so the three are disjoint.
    """
    follows_donor = follows_original = neither = 0
    for eid, pred in predicted.items():
        d, g = donor_answers.get(eid), ground_truth.get(eid)
        if d is None or g is None or pred is None:
            continue
        np_ = normalize_label_text(pred)
        if np_ == canonical_gold(d):
            follows_donor += 1
        elif np_ == canonical_gold(g):
            follows_original += 1
        else:
            neither += 1
    n = follows_donor + follows_original + neither
    return {"n": n, "follows_donor": follows_donor, "follows_original": follows_original,
            "neither": neither,
            "follows_donor_rate": round(follows_donor / n, 4) if n else 0.0,
            "follows_original_rate": round(follows_original / n, 4) if n else 0.0}


# --- two-choice parsing -------------------------------------------------------------------------

_CHOICE_ANCHORED = re.compile(r"(?im)^\s*(?:answer\s*)?[:\-]?\s*\(?([AB])\)?\s*[.):]?\s*$")
_CHOICE_LABELLED = re.compile(r"(?i)\banswer\s*(?:is)?\s*[:\-]?\s*\(?([AB])\)?\b")
_CHOICE_LOOSE = re.compile(r"(?<![A-Za-z])\(?([AB])\)?[.):]")


def parse_choice_letter(raw: str, options=None) -> Optional[str]:
    """Pull an A/B choice out of a generation, or None if the model never committed to one."""
    t = (raw or "").replace("</think>", " ").replace("<think>", " ")
    for rx in (_CHOICE_LABELLED, _CHOICE_ANCHORED, _CHOICE_LOOSE):
        m = list(rx.finditer(t))
        if m:
            return m[-1].group(1).upper()
    if options and len(options) == 2:
        nt = normalize_label_text(t)
        hits = [L for L, opt in zip("AB", options) if opt and normalize_label_text(opt) in nt]
        if len(hits) == 1:
            return hits[0]
    return None

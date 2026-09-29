#!/usr/bin/env python
"""Scores a run directory's records.jsonl (or its records.jsonl.N shards) and writes metrics.json
beside it.

Dispatches on the records' modality. For BioReason-Pro GO term prediction it computes weighted F_max
with the CAFA evaluator and the change from `wt` per condition, or scores the protein pair
conditions for evidence conflicts. For single cell cell type annotation it computes exact and Cell
Ontology matches and candidate likelihood accuracy. For DNA disease prediction it computes accuracy
and answer changes per condition.

    python -m input_use.core.score --rundir <run> [--go_obo <go-basic.obo>] [--go_ia <IA file>]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from input_use.core import config as cfg
from input_use.core.records import read_jsonl
from input_use.core import conditions as C
from input_use.metrics.protein import go_jaccard
from input_use.metrics.dna import gold_named_in, label_of, normalize_label_text, canonical_gold
# Note: `cafa_fmax` (needs the heavy `cafaeval` dep) is imported lazily inside score_protein so the
# ProTrek retrieval path can be scored in the ProTrek env without cafaeval installed.

try:
    from scipy.stats import wilcoxon
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

DEFAULT_OBO = str(cfg.GO_OBO)
DEFAULT_IA = str(cfg.GO_IA)
# `swap` is scored separately (A-embedding vs B-text), not as a wt-referenced delta/F_max condition.
PROTEIN_CONDS = ["mut_salient", "mut_control", "shuffle_esm3_and_gogpt", "shuffle_esm3_only",
                 "no_name", "no_symbolic", "no_modality",
                 C.SHUFFLE_ESM3_GOGPT, C.INTERPRO_ONLY, C.GOGPT_ONLY,
                 C.INTERPRO_ONLY_WT_ESM3, C.GOGPT_ONLY_WT_ESM3]
# Length bands for the stratified read-out. A uniform delta across these is the evidence that
# shuffle-invariance is a property of the model rather than of some easy subpopulation.
LEN_BANDS = [(0, 200), (200, 350), (350, 550), (550, 1025)]
SC_CONDS = [C.SCRAMBLE_RANK, C.SCRAMBLE_GENESET, "no_modality"]
# Graded dropout dose series (k = number of top marker/control genes removed outright).
SC_DOSE_KS = C.DEG_DROPOUT_KS
SC_GRADED_CONDS = [C.top_deg_dropout_k(k) for k in C.DEG_DROPOUT_KS] + \
                  [C.matched_nondeg_dropout_k(k) for k in C.DEG_DROPOUT_KS]
SC_CONDS = SC_CONDS + SC_GRADED_CONDS
# Text-reliance battery (metadata-text ablation; see modalities/singlecell/text_conditions.py).
SC_TEXT_CONDS = list(C.TEXT_RELIANCE_CONDS)
SC_CONDS = SC_CONDS + SC_TEXT_CONDS
# Recall@k condition set (viz/plot_recall_at_topk.py): stress test 1 + the salient (S+) arm only of
# the dropout dose series + no_modality + the text-reliance battery.
SC_RECALL_CONDS = [C.SCRAMBLE_RANK, C.SCRAMBLE_GENESET, "no_modality"] + \
                  [C.top_deg_dropout_k(k) for k in SC_DOSE_KS] + SC_TEXT_CONDS
# Recall@k values computed for the categorical conditions above (viz/plot_recall_at_topk.py
# generates one pair of figures per k).
SC_RECALL_TOPKS = [1, 5, 10]
DNA_CONDS = ["scramble", "no_pathway", "no_gene", "no_textkey", "no_modality"]


def _protein_strata(by_ex, dag):
    """Per-protein stratum labels for the stratified delta read-out."""
    strata = {}
    for ex, conds in by_ex.items():
        wt = conds.get("wt")
        if not wt:
            continue
        L = wt.get("payload", {}).get("length") or len(wt["input"].get("sequence", "")) or 0
        band = next((f"{lo}-{hi} aa" for lo, hi in LEN_BANDS if lo <= L < hi), f">={LEN_BANDS[-1][1]} aa")
        n_gt = len(wt["ground_truth"].get("go_terms", []))
        strata[ex] = {
            "length": band,
            "interpro": "InterPro hit" if (wt["input"].get("interpro") or "").strip() else "no InterPro hit",
            "n_gt": ("1-3 GO terms" if n_gt <= 3 else "4-10 GO terms" if n_gt <= 10 else ">10 GO terms"),
        }
    return strata


def score_protein(by_ex, obo_path, ia_path=None, n_boot=2000):
    delta = {c: [] for c in PROTEIN_CONDS}
    retention = {}                          # cond -> per-protein 1.0 if the GO set is unchanged vs WT
    esm3_chg = defaultdict(list)            # cond -> per-protein % aa changed vs WT in the ESM3 input
    gogpt_chg = defaultdict(list)           # cond -> per-protein % aa changed vs WT in the GO-GPT seq
    # input
    gogpt_off = set()                       # conditions where the GO-GPT/text channel is dropped
    preds_by_cond = defaultdict(dict)       # cond -> {protein -> set(GO terms)}  (regex-extracted)
    gt_by_protein = {}                      # protein -> set(GO terms)  (shared WT ground truth)
    for ex, conds in by_ex.items():
        if "wt" not in conds:
            continue
        gw = set(conds["wt"]["output"]["parsed"]["go_terms"])
        gt_by_protein[ex] = set(conds["wt"]["ground_truth"].get("go_terms", []))
        preds_by_cond["wt"][ex] = gw
        wt_seq = conds["wt"]["input"].get("sequence", "")
        for cond, rec in conds.items():     # % amino acids changed vs WT in each model input (same length
        # only)
            inp = rec.get("input", {})
            eseq = inp.get("sequence", "")                       # ESM3 embedding input
            if wt_seq and eseq and len(eseq) == len(wt_seq):
                esm3_chg[cond].append(100 * sum(a != b for a, b in zip(wt_seq, eseq)) / len(wt_seq))
            if (inp.get("channels") or {}).get("gogpt") == "none":
                gogpt_off.add(cond)
            else:
                sseq = inp.get("symbolic_sequence", eseq)        # GO-GPT/InterPro sequence input
                if wt_seq and sseq and len(sseq) == len(wt_seq):
                    gogpt_chg[cond].append(100 * sum(a != b for a, b in zip(wt_seq, sseq)) / len(wt_seq))
        # Per-seed shuffle repeats (`shuffle_esm3_only_s1`, ...) are scored alongside the base arms.
        for cond in list(conds):
            if cond == "wt" or cond == "swap":
                continue
            if cond not in PROTEIN_CONDS and "_s" not in cond:
                continue
            gc = set(conds[cond]["output"]["parsed"]["go_terms"])
            preds_by_cond[cond][ex] = gc
            delta.setdefault(cond, []).append(1 - go_jaccard(gw, gc))
            # Exact set-match retention - the metric the paper's Experiment 1 defines, P(y_shuffle
            # == y_original).
            retention.setdefault(cond, []).append(1.0 if gc == gw else 0.0)
    out = {"n_proteins": len(gt_by_protein), "delta_vs_wt": {}, "retention_rate": {},
           "go_fmax": {}, "go_fmax_by_aspect": {},
           "go_fmax_weighted": {}, "go_fmax_weighted_by_aspect": {}}
    for cond, v in delta.items():
        out["delta_vs_wt"][cond] = float(np.mean(v)) if v else None
    from input_use.core.stats import bootstrap_ci as _bci
    for cond, v in retention.items():
        if v:
            out["retention_rate"][cond] = {"rate": float(np.mean(v)), "n": len(v),
                                           "ci": list(_bci(v))}

    # ---- F_max via the official CAFA-evaluator (cafaeval) - identical engine + config to
    # BioReason-Pro's evals/cafa_evals.py (norm=cafa, prop=max, score=1.0 @ th=0.99, per-aspect
    # mf/BP/cc averaged, unweighted `f` + IA-weighted `f_w`).
    from input_use.metrics.cafa import cafa_fmax
    fmax, ia_used = cafa_fmax(preds_by_cond, gt_by_protein, obo_path, ia_path)
    out["go_fmax_weighted_available"] = ia_used
    if not ia_used:
        # The weighted score is the reported metric, so a missing IA file silently changes what the
        # run means rather than merely omitting a column. Say so where it will be seen.
        print(f"WARNING: no IA file at {ia_path!r} - GO F_max will be UNWEIGHTED, which is not the "
              f"reported metric. Generate one with `python -m input_use.metrics.make_ia --out <path>` "
              f"and re-score with `--go_ia <path>`.", flush=True)
    out["scoring_engine"] = (f"cafaeval (norm=cafa, prop=max, th_step=0.99, per-aspect mean; "
                             f"IA-weighted={'on' if ia_used else 'off'}) - matches BioReason-Pro "
                             f"evals/cafa_evals.py")
    for cond, d in fmax.items():
        out["go_fmax"][cond] = d["f"]
        out["go_fmax_by_aspect"][cond] = d["by_aspect"]
        out["go_fmax_weighted"][cond] = d["f_w"]
        out["go_fmax_weighted_by_aspect"][cond] = d["by_aspect_w"]

    # ---- Protein-level bootstrap CIs on F1, paired across conditions ---- F_max above is a point
    # estimate (BioReason-Pro reports it that way).
    from input_use.metrics.cafa import cafa_f1_bootstrap, load_ia, verify_against_cafaeval
    # Bootstrap the metric that is reported.
    ia_w = load_ia(ia_path) if ia_used else None
    out["go_bootstrap_weighted"] = ia_w is not None
    ver = verify_against_cafaeval(preds_by_cond, gt_by_protein, obo_path, ref=fmax, ia=ia_w)
    out["fast_f1_matches_cafaeval"] = ver["_ok"]
    out["fast_f1_max_abs_diff"] = ver["_max_abs_diff"]
    out["fast_f1_max_delta_diff"] = ver.get("_max_delta_diff")
    if ver["_ok"]:
        out["go_f1_bootstrap"] = cafa_f1_bootstrap(preds_by_cond, gt_by_protein, obo_path,
                                                   n_boot=n_boot, ia=ia_w)
    else:
        out["go_f1_bootstrap"] = None
        print(f"[score] WARNING: fast-F1 reimplementation disagrees with cafaeval by "
              f"{ver['_max_abs_diff']:.2e}; suppressing bootstrap CIs.")

    # ---- Stratified deltas: is the invariance uniform, or an artefact of an easy subset? ----
    from input_use.metrics import go_dag as _gd
    strata = _protein_strata(by_ex, _gd.load(obo_path))
    strat_out = {}
    for axis in ("length", "interpro", "n_gt"):
        groups = defaultdict(lambda: defaultdict(list))
        for ex, conds in by_ex.items():
            lab = strata.get(ex, {}).get(axis)
            if not lab or "wt" not in conds:
                continue
            gw = set(conds["wt"]["output"]["parsed"]["go_terms"])
            for cond in conds:
                if cond in ("wt", "swap"):
                    continue
                gc = set(conds[cond]["output"]["parsed"]["go_terms"])
                groups[lab][cond].append(1 - go_jaccard(gw, gc))
        strat_out[axis] = {
            lab: {"n": len(next(iter(cd.values()))) if cd else 0,
                  "delta_vs_wt": {c: {"mean": float(np.mean(v)), "ci": list(_bci(v))}
                                  for c, v in cd.items() if v}}
            for lab, cd in sorted(groups.items())}
    out["stratified"] = strat_out

    out["esm3_pct_changed"] = {c: round(float(np.mean(v)), 1) for c, v in esm3_chg.items() if v}
    out["gogpt_pct_changed"] = {c: "dropped" for c in gogpt_off}
    out["gogpt_pct_changed"].update({c: round(float(np.mean(v)), 1) for c, v in gogpt_chg.items() if v})
    # Grounding gap: the salient-vs-control contrast, present only in the functional-site profile
    # (the proteome shuffle battery has no salient/control arms).
    from input_use.core.stats import perm_test_paired_gt
    s, c = np.array(delta.get("mut_salient") or []), np.array(delta.get("mut_control") or [])
    if len(s) == len(c) and len(s):
        out["grounding_gap"] = float((s - c).mean())
        out["grounding_gap_ci"] = list(_bci(list(s - c)))
        out["p_salient_gt_control"] = perm_test_paired_gt(list(s), list(c))
        out["p_test"] = "one-sided paired sign-flip permutation (core/stats.py), 10000 perms"
    # M0 = F_max(no_modality)/F_max(wt) - fraction of F_max surviving with the protein modality
    # removed.
    if out["go_fmax"].get("wt"):
        out["M0_no_modality"] = (out["go_fmax"].get("no_modality") or 0) / out["go_fmax"]["wt"]
    if ia_used and out["go_fmax_weighted"].get("wt"):
        out["M0_no_modality_weighted"] = ((out["go_fmax_weighted"].get("no_modality") or 0)
                                          / out["go_fmax_weighted"]["wt"])
    # % of residues changed by the alanine scan, averaged over proteins (for the mut conditions)
    pct = {}
    for cm in ("mut_salient", "mut_control"):
        vals = [conds[cm].get("intervention", {}).get("pct_residues_changed")
                for conds in by_ex.values() if cm in conds]
        vals = [v for v in vals if v is not None]
        pct[cm] = round(float(np.mean(vals)), 2) if vals else None
    out["pct_residues_changed"] = pct
    # swap: ESM3 embedding of A + symbolic text of B - does the prediction follow A (embedding) or B
    # (text)?
    from input_use.core.stats import bootstrap_ci
    jA, jB, follows = [], [], []
    for ex, conds in by_ex.items():
        if "swap" not in conds:
            continue
        c = conds["swap"]
        pred = set(c["output"]["parsed"]["go_terms"])
        gA = set(c["ground_truth"].get("go_terms", []))          # A = ESM3-embedding source
        gB = set(c["ground_truth"].get("go_terms_text", []))     # B = symbolic-text source
        a_sim, b_sim = go_jaccard(pred, gA), go_jaccard(pred, gB)
        jA.append(a_sim); jB.append(b_sim)
        follows.append(1.0 if b_sim > a_sim else 0.0)
    if follows:
        out["swap"] = {"n": len(follows),
                       "mean_jaccard_to_embedding_A": float(np.mean(jA)),
                       "mean_jaccard_to_text_B": float(np.mean(jB)),
                       "follows_text_rate": float(np.mean(follows)),
                       "follows_text_rate_CI": bootstrap_ci(follows)}
    return out


def score_protein_pairs(by_ex, obo_path):
    """Scores the protein pair conditions (`aligned_*`, `conflict_*` in input_use.core.conditions)
    used for evidence conflicts. The pairs are built by input_use/modalities/protein/pairs.py.
    """
    from input_use.core.stats import bootstrap_ci
    from input_use.metrics import go_dag

    dag = go_dag.load(obo_path)

    def satisfies(pred, spec):
        if not all(dag.any_under(pred, r) for r in spec.get("require", [])):
            return False
        return not any(dag.any_under(pred, r) for r in spec.get("forbid", []))

    def _fam():
        return {"override": [], "retain": [], "other": []}

    per_cat = defaultdict(lambda: {"n_pairs": 0, "n_gated": 0, "override": [], "retain": [],
                                   "other": [], "aligned_a": [], "aligned_b": [],
                                   "families": defaultdict(_fam)})
    for _pair_id, conds in by_ex.items():
        if not all(c in conds for c in (C.PAIR_ALIGNED_A, C.PAIR_ALIGNED_B)):
            continue
        gt = conds[C.PAIR_ALIGNED_A]["ground_truth"]
        cat = gt.get("category", "?")
        spec = {"a": gt["spec_a"], "b": gt["spec_b"]}
        d = per_cat[cat]
        d["n_pairs"] += 1
        pa = set(conds[C.PAIR_ALIGNED_A]["output"]["parsed"]["go_terms"])
        pb = set(conds[C.PAIR_ALIGNED_B]["output"]["parsed"]["go_terms"])
        ok_a, ok_b = satisfies(pa, spec["a"]), satisfies(pb, spec["b"])
        d["aligned_a"].append(1.0 if ok_a else 0.0)
        d["aligned_b"].append(1.0 if ok_b else 0.0)
        if not (ok_a and ok_b):
            continue                      # fails the validation gate - excluded from the conflict stats
        d["n_gated"] += 1
        # Every conflict arm is scored the same way: does the output satisfy the spec of the protein
        # whose embedding it holds, or of the protein supplying the conflicting text channel(s)?
        for fam, fam_conds in C.PAIR_CONFLICT_FAMILIES.items():
            for cond in fam_conds:
                if cond not in conds:
                    continue
                own_key, other_key = C.PAIR_CONFLICT_LAYOUT[cond]
                pred = set(conds[cond]["output"]["parsed"]["go_terms"])
                prot_ok = satisfies(pred, spec[own_key])
                ctx_ok = satisfies(pred, spec[other_key])
                rec = (1.0 if (prot_ok and not ctx_ok) else 0.0,
                       1.0 if (ctx_ok and not prot_ok) else 0.0,
                       1.0 if (prot_ok == ctx_ok) else 0.0)
                d["families"][fam]["retain"].append(rec[0])
                d["families"][fam]["override"].append(rec[1])
                d["families"][fam]["other"].append(rec[2])
                if cond in (C.PAIR_CONFLICT_A, C.PAIR_CONFLICT_B):   # primary family
                    d["retain"].append(rec[0]); d["override"].append(rec[1]); d["other"].append(rec[2])

    out = {"n_pairs": sum(v["n_pairs"] for v in per_cat.values()), "by_category": {}}
    for cat, d in sorted(per_cat.items()):
        def mci(v):
            return {"rate": float(np.mean(v)), "ci": list(bootstrap_ci(v)), "n": len(v)} if v else None
        out["by_category"][cat] = {
            "n_pairs": d["n_pairs"], "n_passing_gate": d["n_gated"],
            "gate_pass_rate": d["n_gated"] / d["n_pairs"] if d["n_pairs"] else None,
            "aligned_a_correct": mci(d["aligned_a"]), "aligned_b_correct": mci(d["aligned_b"]),
            "context_override_rate": mci(d["override"]),
            "protein_retention_rate": mci(d["retain"]),
            "other_rate": mci(d["other"]),
            "reportable": d["n_gated"] >= 30,
            # Which channel drove it: both-swapped vs InterPro-text-only vs GO-GPT-only.
            "by_conflict_family": {
                fam: {"context_override_rate": mci(v["override"]),
                      "protein_retention_rate": mci(v["retain"]),
                      "other_rate": mci(v["other"])}
                for fam, v in d["families"].items() if v["override"]},
        }
    return out


# ProTrek perturbation conditions, grouped. Each should degrade retrieval if the model truly uses
# the destroyed information; a perturbation that does not degrade exposes a shortcut.
DOSE_TAGS = ["p33", "p67", "p100"]
SEQ_GROUPS = {
    "dose_salient": [f"mut_salient_{t}" for t in DOSE_TAGS],
    "dose_control": [f"mut_control_{t}" for t in DOSE_TAGS],
    # higher-dose function-centered perturbations: +/-k window around each functional residue
    "window": ["mutwin_salient_w3", "mutwin_control_w3", "mutwin_salient_w6", "mutwin_control_w6",
               "delwin_salient_w3", "delwin_control_w3"],
    "domain": ["keep_domain", "del_domain", "del_random"],
    "order": ["scramble", "reverse", "mask_nterm", "mask_cterm", "mask_middle"],
}
WIN_PAIRS = [("mutwin_salient_w3", "mutwin_control_w3", "alanine_pm3"),
             ("mutwin_salient_w6", "mutwin_control_w6", "alanine_pm6"),
             ("delwin_salient_w3", "delwin_control_w3", "delete_pm3")]
TEXT_CONDS = ["text_nameonly", "text_namefunc", "text_generic"]


def _macro_f1_and_auprc(rows):
    """rows: list of (gt, pred, candidate_scores) tuples for one condition -- pred and candidate_scores
    are the raw (un-debiased) argmax/scores (same source as Recall@k; see
    models/lm_candidate_scoring.py for why raw, not the debiased parsed.cell_type/candidate_scores,
    is the scoring input here).
    """
    if not rows:
        return None, None
    from sklearn.metrics import f1_score, average_precision_score
    y_true = [r[0] for r in rows]
    y_pred = [r[1] for r in rows]
    f1 = float(f1_score(y_true, y_pred, average="macro", labels=sorted(set(y_true) | set(y_pred)),
                        zero_division=0))
    scored = [r for r in rows if r[2]]
    auprc = None
    if scored:
        classes = sorted({c for _, _, cs in scored for c in cs})
        aps = []
        for c in classes:
            y_bin = [1 if gt == c else 0 for gt, _, _ in scored]
            if not any(y_bin):
                continue
            aps.append(average_precision_score(y_bin, [cs.get(c, 0.0) for _, _, cs in scored]))
        auprc = float(np.mean(aps)) if aps else None
    return f1, auprc


def score_singlecell(by_ex):
    from input_use.metrics.singlecell import fine_match, coarse_match
    from input_use.core.stats import bootstrap_ci
    from input_use.models.lm_candidate_scoring import argmax_candidate
    acc = defaultdict(list)                       # primary: fine (exact-string-match) accuracy (debiased
    # argmax)
    acc_coarse = defaultdict(list)                 # CL-ontology-grounded coarse accuracy (see metrics/
                                                    # singlecell.py); only cells where the ground truth resolves
                                                    # to a CL id are counted (None = skipped, not 0)
    cls_rows = defaultdict(list)                  # cond -> [(gt, raw argmax pred, candidate_scores_raw)] for
    # F1/AUPRC Recall@k (k in SC_RECALL_TOPKS) from raw
    # (un-debiased) candidate log-likelihoods: for each k, take
    # the k highest-scoring candidates and ask (a) does the
    # annotated cell type appear among them ["correct"] and (b)
    # does wt's own top-1 prediction appear among them ["same as
    # wt"] -- k=1 reduces exactly to the original single-argmax
    # comparison.
    recall_correct_raw = {k: defaultdict(list) for k in SC_RECALL_TOPKS}   # k -> cond -> [1/0]
    recall_same_raw = {k: defaultdict(list) for k in SC_RECALL_TOPKS}      # k -> cond -> [1/0]
    for ex, conds in by_ex.items():
        if "wt" not in conds:
            continue
        wt = conds["wt"]["output"]["parsed"]["cell_type"]
        gt = conds["wt"]["ground_truth"].get("cell_type", "")
        acc["wt"].append(1.0 if fine_match(wt, gt) else 0.0)
        wt_coarse = coarse_match(wt, gt)
        if wt_coarse is not None:
            acc_coarse["wt"].append(1.0 if wt_coarse else 0.0)
        for cond in SC_CONDS:
            if cond in conds:
                pc = conds[cond]["output"]["parsed"]["cell_type"]
                acc[cond].append(1.0 if fine_match(pc, gt) else 0.0)
                cond_coarse = coarse_match(pc, gt)
                if cond_coarse is not None:
                    acc_coarse[cond].append(1.0 if cond_coarse else 0.0)
                cs = conds[cond]["output"]["parsed"].get("candidate_scores_raw")
                if cs:
                    cls_rows[cond].append((gt, argmax_candidate(cs), cs))
        wt_raw_scores = conds["wt"]["output"]["parsed"].get("candidate_scores_raw")
        if wt_raw_scores:
            wt_raw_pred = argmax_candidate(wt_raw_scores)
            cls_rows["wt"].append((gt, wt_raw_pred, wt_raw_scores))
            for k in SC_RECALL_TOPKS:
                wt_topk = sorted(wt_raw_scores, key=wt_raw_scores.get, reverse=True)[:k]
                recall_correct_raw[k]["wt"].append(
                    1.0 if any(fine_match(c, gt) for c in wt_topk) else 0.0)
                recall_same_raw[k]["wt"].append(1.0)
                for cond in SC_RECALL_CONDS:
                    if cond in conds:
                        cs = conds[cond]["output"]["parsed"].get("candidate_scores_raw")
                        if not cs:
                            continue
                        cond_topk = sorted(cs, key=cs.get, reverse=True)[:k]
                        recall_correct_raw[k][cond].append(
                            1.0 if any(fine_match(c, gt) for c in cond_topk) else 0.0)
                        recall_same_raw[k][cond].append(
                            1.0 if any(fine_match(c, wt_raw_pred) or
                                      c.strip().lower() == wt_raw_pred.strip().lower()
                                      for c in cond_topk) else 0.0)
    out = {"n_cells": len(by_ex), "celltype_accuracy_fine": {}, "celltype_accuracy_fine_CI": {},
           "celltype_accuracy_coarse": {}, "celltype_accuracy_coarse_CI": {},
           "f1_macro": {}, "auprc_macro": {},
           "recall_at_topk_raw": {}, "recall_at_topk_raw_CI": {},
           "recall_same_as_wt_topk_raw": {}, "recall_same_as_wt_topk_raw_CI": {}}
    for cond, v in acc.items():
        out["celltype_accuracy_fine"][cond] = float(np.mean(v)) if v else None
        out["celltype_accuracy_fine_CI"][cond] = bootstrap_ci(v)
    # celltype_accuracy_coarse stays empty (not just per-condition None) on a dataset with no CL ids
    # (e.g.
    for cond, v in acc_coarse.items():
        out["celltype_accuracy_coarse"][cond] = float(np.mean(v)) if v else None
        out["celltype_accuracy_coarse_CI"][cond] = bootstrap_ci(v)
    for cond, rows in cls_rows.items():
        out["f1_macro"][cond], out["auprc_macro"][cond] = _macro_f1_and_auprc(rows)
    for k in SC_RECALL_TOPKS:
        ks = str(k)
        out["recall_at_topk_raw"][ks], out["recall_at_topk_raw_CI"][ks] = {}, {}
        out["recall_same_as_wt_topk_raw"][ks], out["recall_same_as_wt_topk_raw_CI"][ks] = {}, {}
        for cond, v in recall_correct_raw[k].items():
            out["recall_at_topk_raw"][ks][cond] = float(np.mean(v)) if v else None
            out["recall_at_topk_raw_CI"][ks][cond] = bootstrap_ci(v)
        for cond, v in recall_same_raw[k].items():
            out["recall_same_as_wt_topk_raw"][ks][cond] = float(np.mean(v)) if v else None
            out["recall_same_as_wt_topk_raw_CI"][ks][cond] = bootstrap_ci(v)
    # Dose-response: accuracy (salient vs control dropout) at each k of the dropout series.
    out["celltype_accuracy_fine_by_k"] = {}
    for k in SC_DOSE_KS:
        sk_name, ck_name = C.top_deg_dropout_k(k), C.matched_nondeg_dropout_k(k)
        out["celltype_accuracy_fine_by_k"][str(k)] = {
            "salient": float(np.mean(acc[sk_name])) if acc.get(sk_name) else None,
            "control": float(np.mean(acc[ck_name])) if acc.get(ck_name) else None}
    if out["celltype_accuracy_fine"].get("wt"):
        # None (not "or 0") when no_modality was not run at all
        nm_acc = out["celltype_accuracy_fine"].get("no_modality")
        out["M0_no_modality"] = (nm_acc / out["celltype_accuracy_fine"]["wt"]) if nm_acc is not None else None
        # Text-reliance summary: does the full metadata text clause alone recover accuracy when the
        # cell sentence/embedding carries no cell-specific signal (no_modality)?
        tfnm_acc = out["celltype_accuracy_fine"].get(C.TEXT_FULL_NO_MODALITY)
        out["M0_text_full_no_modality"] = (
            (tfnm_acc / out["celltype_accuracy_fine"]["wt"]) if tfnm_acc is not None else None)
    return out


def score_dna(by_ex):
    # Predictions are re-derived from output.raw at score time (so re-scoring an existing run needs
    # no GPU), using the released DNA checkpoints' own metric: map the generation onto the closed vocabulary of
    # gold labels, longest label first, then compare labels for equality (metrics/dna.py).
    labels = sorted({c["ground_truth"].get("answer", "")
                     for conds in by_ex.values() for c in conds.values()} - {""})
    flips = {c: [] for c in DNA_CONDS}
    acc, reached = defaultdict(list), defaultdict(list)
    for ex, conds in by_ex.items():
        if "wt" not in conds:
            continue
        gt = conds["wt"]["ground_truth"].get("answer", "")
        wt = label_of(conds["wt"]["output"]["raw"], labels)
        acc["wt"].append(1.0 if wt is not None and normalize_label_text(wt) == canonical_gold(gt) else 0.0)
        reached["wt"].append(1.0 if gold_named_in(gt, conds["wt"]["output"]["raw"]) else 0.0)
        for cond in DNA_CONDS:
            if cond in conds:
                raw = conds[cond]["output"]["raw"]
                pc = label_of(raw, labels)
                flips[cond].append(0.0 if pc == wt else 1.0)
                acc[cond].append(1.0 if pc is not None and normalize_label_text(pc) == canonical_gold(gt) else 0.0)
                reached[cond].append(1.0 if gold_named_in(gt, raw) else 0.0)
    from input_use.core.stats import bootstrap_ci
    out = {"n_variants": len(by_ex), "answer_change_vs_wt": {}, "disease_accuracy": {},
           "disease_accuracy_CI": {}, "reached_accuracy": {}}
    for cond, v in flips.items():
        out["answer_change_vs_wt"][cond] = float(np.mean(v)) if v else None
    for cond, v in acc.items():
        out["disease_accuracy"][cond] = float(np.mean(v)) if v else None
        out["disease_accuracy_CI"][cond] = bootstrap_ci(v)
    for cond, v in reached.items():
        out["reached_accuracy"][cond] = float(np.mean(v)) if v else None
    # No SNV grounding gap for KEGG (the disease label is locus-determined, not base-determined).
    if out["disease_accuracy"].get("wt"):
        out["M0_no_modality"] = (out["disease_accuracy"].get("no_modality") or 0) / out["disease_accuracy"]["wt"]
    if out["reached_accuracy"].get("wt"):
        out["M0_reached"] = (out["reached_accuracy"].get("no_modality") or 0) / out["reached_accuracy"]["wt"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rundir", required=True)
    ap.add_argument("--go_obo", default=DEFAULT_OBO)
    ap.add_argument("--go_ia", default=DEFAULT_IA,
                    help="Information-Accretion file for the CAFA weighted F_max (cafaeval `f_w`) - the "
                         "REPORTED GO metric here, and the one BioReason-Pro headlines. Defaults to the file "
                         "generated by `input_use.metrics.make_ia` from our GOA snapshot. If it is absent, "
                         "scoring still runs but reports unweighted F_max and warns loudly.")
    a = ap.parse_args()
    # A sharded run writes records.jsonl.0, .1, ... (one file per array element, since concurrent
    # appends to a single file interleave). Read whichever layout is present.
    rundir = Path(a.rundir)
    # Match `records.jsonl` and `records.jsonl.<N>` only - a bare glob would also pick up
    parts = [p for p in sorted(rundir.glob("records.jsonl*"), key=lambda q: q.name)
             if p.name == "records.jsonl" or p.name.split(".")[-1].isdigit()]
    if not parts:
        raise SystemExit(f"no records.jsonl[.N] found in {rundir}")
    recs = [r for p in parts for r in read_jsonl(p)]
    if len(parts) > 1:
        print(f"[score] merged {len(parts)} shards -> {len(recs)} records")
    modality, model = recs[0]["modality"], recs[0]["model"]
    by_ex = defaultdict(dict)
    for r in recs:
        by_ex[r["example_id"]][r["condition"]] = r
    # Experiment 2 runs carry the matched-pair conditions instead of a wt-referenced ladder.
    is_pairs = any(r["condition"] in C.PAIR_CONDS for r in recs[:50])
    m = (score_protein_pairs(by_ex, a.go_obo) if modality == "protein" and is_pairs
         else score_protein(by_ex, a.go_obo, a.go_ia) if modality == "protein"
         else score_dna(by_ex) if modality == "dna"
         else score_singlecell(by_ex))
    m["modality"], m["model"] = modality, model
    json.dump(m, open(Path(a.rundir) / "metrics.json", "w"), indent=2)
    print(f"-> {a.rundir}/metrics.json")


if __name__ == "__main__":
    main()

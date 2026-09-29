#!/usr/bin/env python
"""Measures how the genes referenced in C2S-Scale 27B rationales change when DEGs are removed from the
cell sentence.

Compares removal of 50% and 100% of the annotated type's strongest DEGs with removal of the same
number of non-DEGs with similar expression, on five atlases. Per cell it counts the referenced
genes, their overlap with the rationale for the unmodified sentence (Jaccard distance), ribosomal
and DEG hits against a uniform draw from the sentence, and whether the predicted cell type changed
(resolved to the Cell Ontology). Intervals come from a paired cell bootstrap. Writes --out (JSON), a
per-cell JSONL and a markdown summary beside it. Supports the RQ3 C2S-Scale rationale figure and
appendix.

    python -m input_use.core.analyze_rationale_removal --out outputs/analysis/rationale_removal.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from collections import Counter, defaultdict

import numpy as np

from input_use.core import paths as RD

_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-\.]*")
DATASETS = ["immune1", "immune2", "immune3", "pancreas", "lung"]
DOSES = [("50", "top_deg_dropout_p50", "matched_nondeg_dropout_p50"),
         ("100", "top_deg_dropout_p100", "matched_nondeg_dropout_p100")]
CONDS = ["wt"] + [c for _, t, m in DOSES for c in (t, m)]

# Atlas types whose label is study shorthand (pancreas, lung) or does not self-resolve (immune2's
# CD8aa(I) thymocyte). Each maps to the Cell Ontology ids a predicted label may resolve to.
MANUAL_TYPE_TERMS = {
    "pancreas": {
        "acinar": ["CL:0002064"], "alpha": ["CL:0000171"], "beta": ["CL:0000169"],
        "delta": ["CL:0000173"], "gamma": ["CL:0002275", "CL:0000696"], "epsilon": ["CL:0005019"],
        "ductal": ["CL:0002079"], "endothelial": ["CL:0000115"], "macrophage": ["CL:0000235"],
        "mast": ["CL:0000097"], "schwann": ["CL:0002573"],
        "activated_stellate": ["CL:0002410"], "quiescent_stellate": ["CL:0002410"]},
    "lung": {
        "Basal": ["CL:0000646"], "Differentiating.Basal": ["CL:0000646"],
        "Transitioning.Basal": ["CL:0000646"], "Proliferating.Basal": ["CL:0000646", "CL:4033082"],
        "Ciliated": ["CL:0000064", "CL:0005012", "CL:0000067"],
        "Secretory": ["CL:0000151", "CL:1000272", "CL:0000158", "CL:0000160"],
        "Suprabasal": ["CL:7770004"]},
    "immune2": {"CD8aa(I) thymocyte": ["CL:0000893"]},
}


def cited_genes(text, vocab):
    out, seen = [], set()
    for m in _TOKEN.finditer(re.sub(r"<[^>]*>", " ", text or "")):
        g = m.group(0).upper().rstrip(".")
        if g in vocab and g not in seen:
            seen.add(g)
            out.append(g)
    return out


def label_of(text):
    t = re.sub(r"<[^>]*>", " ", (text or "").strip()).strip()
    t = re.split(r"\bRationale\s*:", t)[0].split("\n")[0]
    return re.sub(r"^\s*Cell type\s*:", "", t, flags=re.I).strip().rstrip(". ").strip()


def sentence_genes(prompt):
    body = prompt.split("Cell sentence:", 1)[-1].split("\n")[0]
    # symbols such as AC002059.1 contain a dot; only the sentence's terminal period is stripped
    return [x.upper().rstrip(".") for x in body.split() if x.rstrip(".")]


def jd(a, b):
    u = len(a | b)
    return 1.0 - len(a & b) / u if u else 0.0


class Boot:
    """Paired cell bootstrap: one weight matrix per group, reused for every statistic."""

    def __init__(self, n, b, rng):
        self.n = n
        self.w = rng.multinomial(n, np.full(n, 1.0 / n), size=b).astype(float)  # (b, n)

    def ratio(self, num, den):
        num, den = np.asarray(num, float), np.asarray(den, float)
        pt = num.sum() / max(den.sum(), 1e-12)
        bs = (self.w @ num) / np.maximum(self.w @ den, 1e-12)
        return pt, bs

    def mean(self, x):
        x = np.asarray(x, float)
        return x.mean(), (self.w @ x) / self.n


def ci(bs):
    return [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def pack(pt, bs):
    return {"value": round(float(pt), 4), "ci95": ci(bs)}


def load_dataset(root, ds, model_tag):
    d = os.path.join(root, ds)
    meta = json.load(open(os.path.join(d, "meta.json")))
    deg_by_type = {k: set(v) for k, v in meta["deg_by_type"].items()}
    vocab = set().union(*deg_by_type.values())
    sent, salient, control, gt = {}, {}, {}, {}
    for r in map(json.loads, open(os.path.join(d, "examples_rationale.jsonl"))):
        cell = r["example_id"].split("__")[0]
        g = sentence_genes(r["payload"]["prompt"])
        sent[(cell, r["condition"])] = g
        vocab |= set(g)
        salient[cell] = [x.upper() for x in r["payload"]["salient"]]
        control[cell] = [x.upper() for x in r["payload"]["control"]]
        gt[cell] = r["ground_truth"]["cell_type"]
    recs = {}
    for f in glob.glob(os.path.join(d, f"records_rationale_{model_tag}_s*.jsonl")):
        for r in map(json.loads, open(f)):
            key = (r["example_id"].split("__")[0], r["condition"])
            assert key not in recs, f"duplicate record {key}"
            recs[key] = r["output"].get("raw") or ""
    cells = sorted(gt)
    for c in cells:
        for cond in CONDS:
            assert (c, cond) in recs, f"missing generation {ds} {c} {cond}"
    return meta, deg_by_type, vocab, sent, salient, control, gt, recs, cells


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rationales_root", default=RD.C2S_RATIONALES,
                    help="directory with one rationale run per atlas")
    ap.add_argument("--obo", default="input_use/data/cl-full.obo")
    ap.add_argument("--ribo", default="input_use/data/hgnc_ribosomal_proteins.json")
    ap.add_argument("--model_tag", default="27b")
    ap.add_argument("--n_boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from input_use.metrics.ontology_ner import OntologyNER
    ner = OntologyNER(a.obo)
    ribo = set(json.load(open(a.ribo))["symbols"])

    def resolve(label):
        return {m.term for m in ner.extract(label or "", min_ic=0.0) if m.kind == "cell_type"}

    def same_type(t1, t2):
        """Unchanged prediction: shared term, or one label's term is an ancestor of the other's."""
        if not t1 or not t2:
            return False
        anc1 = set().union(*(ner.ancestors(t) for t in t1))
        anc2 = set().union(*(ner.ancestors(t) for t in t2))
        return bool((t1 & anc2) or (t2 & anc1))

    per_cell = []          # one dict per (dataset, cell)
    diagnostics = {}
    for ds in DATASETS:
        meta, deg_by_type, vocab, sent, salient, control, gt, recs, cells = \
            load_dataset(a.rationales_root, ds, a.model_tag)
        # ontology ids accepted for each atlas type
        type_terms = {}
        for t in deg_by_type:
            ids = MANUAL_TYPE_TERMS.get(ds, {}).get(t)
            if ids is None:
                ids = [m.term for m in ner.extract(t, min_ic=0.0)
                       if m.kind == "cell_type" and ner.names.get(m.term) == t]
            type_terms[t] = set(ids)
        unmapped = [t for t, v in type_terms.items() if not v]
        assert not unmapped, f"{ds}: atlas types without an ontology id: {unmapped}"
        term_to_types = defaultdict(set)
        for t, ids in type_terms.items():
            for i in ids:
                term_to_types[i].add(t)

        def types_of(terms, tolerant=False):
            """Atlas types a resolved label names. strict: the label's term is an accepted id of the
            type. tolerant: the label's term is an ancestor of (or equal to) an accepted id.
            """
            if not terms:
                return set()
            if not tolerant:
                return set().union(*(term_to_types.get(t, set()) for t in terms))
            out = set()
            for t, ids in type_terms.items():
                if any(terms & ner.ancestors(i) for i in ids):
                    out.add(t)
            return out

        n_removed_ok = Counter()
        unterminated = Counter()
        for c in cells:
            D = deg_by_type[gt[c]]
            S = {k: set(sent[(c, k)]) for k in CONDS}
            raw = {k: recs[(c, k)] for k in CONDS}
            G = {k: cited_genes(raw[k], vocab) for k in CONDS}
            Gs = {k: set(v) for k, v in G.items()}
            lab = {k: label_of(raw[k]) for k in CONDS}
            terms = {k: resolve(lab[k]) for k in CONDS}
            for k in CONDS:
                txt = re.sub(r"<[^>]*>", "", raw[k]).strip()
                unterminated[k] += (not txt) or txt[-1] not in ".!?"
            row = {"dataset": ds, "cell": c, "gt": gt[c], "n_salient": len(salient[c]),
                   "label": lab, "cond": {}}
            gt_terms = resolve(gt[c]) if ds not in ("pancreas", "lung") else type_terms[gt[c]]
            for k in CONDS:
                n = len(salient[c]) if k.endswith("p100") else \
                    min(len(salient[c]), math.ceil(0.5 * len(salient[c])))
                R = S["wt"] - S[k]
                if k.startswith("top"):
                    n_removed_ok[k] += R == set(salient[c][:n])
                elif k.startswith("matched"):
                    n_removed_ok[k] += R == set(control[c][:n])
                g, gw = Gs[k], Gs["wt"]
                first5 = G[k][:5]
                remaining = D & S[k]
                row["cond"][k] = {
                    "n_ref": len(g), "n_ref5": len(first5),
                    "ref_ribo": sum(1 for x in g if x in ribo),
                    "ref_in_sent": sum(1 for x in g if x in S[k]),
                    "n_removed": len(R),
                    "jd": jd(gw, g),
                    "jd_retained": jd(gw - R, g - R),
                    "wt_ref_removed": len(gw & R),
                    "retained": len(gw & g),
                    "new": len(g - gw),
                    "rem_deg_hits": len(g & remaining),
                    "rem_deg_hits5": sum(1 for x in first5 if x in remaining),
                    "deg_any_hits": len(g & D),
                    "deg_any_hits5": sum(1 for x in first5 if x in D),
                    "rem_deg_chance": len(g) * (len(remaining) / len(S[k]) if S[k] else 0.0),
                    "rem_deg_chance5": len(first5) * (len(remaining) / len(S[k]) if S[k] else 0.0),
                    "n_remaining_deg": len(remaining),
                    "removed_ref": len(g & R),
                    "removed_ref_given_wt": len(g & gw & R),
                    "removed_ribo": len(R & ribo),
                    "removed_ref_ribo": len(g & R & ribo),
                    "correct": bool(gt_terms and (gt_terms & terms[k])),
                    "changed_vs_wt": (k != "wt") and not same_type(terms["wt"], terms[k]),
                    "label_str_changed": (k != "wt") and lab[k] != lab["wt"],
                    "resolved": bool(terms[k]),
                }
            # secondary test: newly predicted type under top-DEG removal, strict and tolerant
            for dose, top, matched in DOSES:
                for mode in ("strict", "tolerant"):
                    Tn = types_of(terms[top], tolerant=(mode == "tolerant"))
                    Tn -= types_of(terms["wt"], tolerant=(mode == "tolerant"))  # never the wt type
                    eligible = row["cond"][top]["changed_vs_wt"] and bool(Tn)
                    Dn = set().union(*(deg_by_type[t] for t in Tn)) if Tn else set()
                    ent = {"eligible": eligible, "n_new_types": len(Tn), "n_new_deg": len(Dn)}
                    if eligible:
                        for arm, k in (("top", top), ("matched", matched), ("wt", "wt")):
                            g, first5 = Gs[k], G[k][:5]
                            ent[arm] = {
                                "n_ref": len(g), "n_ref5": len(first5),
                                "hits": len(g & Dn),
                                "hits5": sum(1 for x in first5 if x in Dn),
                                "hits_in_sent": len(g & Dn & S[k]),
                                "chance": len(g) * (len(Dn & S[k]) / len(S[k]) if S[k] else 0.0),
                                "chance5": len(first5) * (len(Dn & S[k]) / len(S[k]) if S[k] else 0.0),
                                "n_new_deg_in_sent": len(Dn & S[k]),
                            }
                    row.setdefault("newtype", {})[f"{dose}:{mode}"] = ent
            per_cell.append(row)
        diagnostics[ds] = {
            "n_cells": len(cells), "vocab_size": len(vocab),
            "removed_set_equals_stored_prefix": dict(n_removed_ok),
            "unterminated_rate": {k: round(v / len(cells), 4) for k, v in unterminated.items()},
            "deg_definition": meta["deg_definition"],
            "type_terms_manual": ds in MANUAL_TYPE_TERMS,
        }
        print(f"[rr] {ds}: {len(cells)} cells, vocab {len(vocab)}, "
              f"removed-set check {dict(n_removed_ok)}", flush=True)

    # ------------------------------------------------------------------ statistics -------------
    rng = np.random.default_rng(a.seed)
    groups = {ds: [r for r in per_cell if r["dataset"] == ds] for ds in DATASETS}
    groups["pooled"] = list(per_cell)
    results = {}
    for gname, rows in groups.items():
        if not rows:
            continue
        n = len(rows)
        B = Boot(n, a.n_boot, rng)
        col = lambda k, f: np.array([r["cond"][k][f] for r in rows], float)
        out = {"n_cells": n, "conditions": {}, "paired": {}, "newtype": {}}
        for k in CONDS:
            e = {}
            e["accuracy"] = pack(*B.mean(col(k, "correct")))
            e["type_changed_vs_wt"] = pack(*B.mean(col(k, "changed_vs_wt")))
            e["label_string_changed_vs_wt"] = pack(*B.mean(col(k, "label_str_changed")))
            e["resolved_rate"] = round(float(col(k, "resolved").mean()), 4)
            e["mean_referenced"] = round(float(col(k, "n_ref").mean()), 2)
            e["mean_removed"] = round(float(col(k, "n_removed").mean()), 2)
            e["ribosomal_share"] = pack(*B.ratio(col(k, "ref_ribo"), col(k, "n_ref")))
            e["in_sentence_share"] = pack(*B.ratio(col(k, "ref_in_sent"), col(k, "n_ref")))
            e["set_change_jd"] = pack(*B.mean(col(k, "jd")))
            e["set_change_jd_retained"] = pack(*B.mean(col(k, "jd_retained")))
            e["wt_referenced_removed_share"] = pack(*B.ratio(col(k, "wt_ref_removed"),
                                                             col("wt", "n_ref")))
            e["retained_share_of_wt"] = pack(*B.ratio(col(k, "retained"), col("wt", "n_ref")))
            e["new_share_of_ref"] = pack(*B.ratio(col(k, "new"), col(k, "n_ref")))
            e["remaining_deg_precision"] = pack(*B.ratio(col(k, "rem_deg_hits"), col(k, "n_ref")))
            e["remaining_deg_chance"] = pack(*B.ratio(col(k, "rem_deg_chance"), col(k, "n_ref")))
            e["remaining_deg_precision5"] = pack(*B.ratio(col(k, "rem_deg_hits5"),
                                                          col(k, "n_ref5")))
            e["remaining_deg_chance5"] = pack(*B.ratio(col(k, "rem_deg_chance5"),
                                                       col(k, "n_ref5")))
            e["mean_remaining_deg"] = round(float(col(k, "n_remaining_deg").mean()), 2)
            # DEGs of the annotated type named whether or not they are in the input; minus the
            # in-input share this is the share of referenced genes that are DEGs absent from it
            e["deg_share_any"] = pack(*B.ratio(col(k, "deg_any_hits"), col(k, "n_ref")))
            e["deg_share_any5"] = pack(*B.ratio(col(k, "deg_any_hits5"), col(k, "n_ref5")))
            e["mean_deg_named"] = round(float(col(k, "deg_any_hits").mean()), 2)
            if k != "wt":
                e["removed_still_referenced"] = pack(*B.ratio(col(k, "removed_ref"),
                                                              col(k, "n_removed")))
                e["removed_referenced_on_wt"] = pack(*B.ratio(col(k, "wt_ref_removed"),
                                                              col(k, "n_removed")))
                e["removed_still_referenced_given_wt"] = pack(
                    *B.ratio(col(k, "removed_ref_given_wt"), col(k, "wt_ref_removed")))
                e["removed_share_of_referenced"] = pack(*B.ratio(col(k, "removed_ref"),
                                                                 col(k, "n_ref")))
                e["removed_ribosomal_share"] = pack(*B.ratio(col(k, "removed_ribo"),
                                                             col(k, "n_removed")))
                e["removed_still_referenced_ribosomal_share"] = pack(
                    *B.ratio(col(k, "removed_ref_ribo"), col(k, "removed_ref")))
            out["conditions"][k] = e

        for dose, top, mt in DOSES:
            p = {}
            for name, kind, num, den in [
                    ("set_change_jd", "mean", "jd", None),
                    ("set_change_jd_retained", "mean", "jd_retained", None),
                    ("remaining_deg_precision", "ratio", "rem_deg_hits", "n_ref"),
                    ("remaining_deg_precision5", "ratio", "rem_deg_hits5", "n_ref5"),
                    ("removed_still_referenced", "ratio", "removed_ref", "n_removed"),
                    ("removed_still_referenced_given_wt", "ratio", "removed_ref_given_wt",
                     "wt_ref_removed"),
                    ("ribosomal_share", "ratio", "ref_ribo", "n_ref"),
                    ("type_changed_vs_wt", "mean", "changed_vs_wt", None)]:
                if kind == "mean":
                    pt_t, bs_t = B.mean(col(top, num))
                    pt_m, bs_m = B.mean(col(mt, num))
                else:
                    pt_t, bs_t = B.ratio(col(top, num), col(top, den))
                    pt_m, bs_m = B.ratio(col(mt, num), col(mt, den))
                d = {"top": round(float(pt_t), 4), "matched": round(float(pt_m), 4),
                     "diff": round(float(pt_t - pt_m), 4), "diff_ci95": ci(bs_t - bs_m)}
                if kind == "mean" and name.startswith("set_change"):
                    x, y = col(top, num), col(mt, num)
                    d["frac_cells_top_gt_matched"] = round(float((x > y).mean()), 4)
                    d["frac_cells_top_lt_matched"] = round(float((x < y).mean()), 4)
                p[name] = d
            out["paired"][dose] = p

            # secondary: newly predicted type
            for mode in ("strict", "tolerant"):
                key = f"{dose}:{mode}"
                sub = [r for r in rows if r["newtype"][key]["eligible"]]
                n_changed = sum(1 for r in rows if r["cond"][top]["changed_vs_wt"])
                ent = {"n_type_changed": n_changed, "n_eligible": len(sub),
                       "n_eligible_by_dataset": dict(Counter(r["dataset"] for r in sub))}
                if len(sub) >= 5:
                    Bs = Boot(len(sub), a.n_boot, rng)
                    v = lambda arm, f: np.array([r["newtype"][key][arm][f] for r in sub], float)
                    ent["mean_new_deg"] = round(float(np.mean(
                        [r["newtype"][key]["n_new_deg"] for r in sub])), 1)
                    for arm in ("top", "matched", "wt"):
                        ent[arm] = {
                            "new_deg_share": pack(*Bs.ratio(v(arm, "hits"), v(arm, "n_ref"))),
                            "new_deg_share5": pack(*Bs.ratio(v(arm, "hits5"), v(arm, "n_ref5"))),
                            "new_deg_share_in_sent": pack(*Bs.ratio(v(arm, "hits_in_sent"),
                                                                    v(arm, "n_ref"))),
                            "chance": pack(*Bs.ratio(v(arm, "chance"), v(arm, "n_ref"))),
                            "chance5": pack(*Bs.ratio(v(arm, "chance5"), v(arm, "n_ref5"))),
                            "mean_new_deg_in_sent": round(float(v(arm, "n_new_deg_in_sent").mean()), 1),
                        }
                    pd = {}
                    for name, num, den in [("new_deg_share", "hits", "n_ref"),
                                           ("new_deg_share5", "hits5", "n_ref5"),
                                           ("new_deg_share_in_sent", "hits_in_sent", "n_ref"),
                                           ("chance", "chance", "n_ref")]:
                        pt_t, bs_t = Bs.ratio(v("top", num), v("top", den))
                        pt_m, bs_m = Bs.ratio(v("matched", num), v("matched", den))
                        pd[name] = {"diff": round(float(pt_t - pt_m), 4),
                                    "diff_ci95": ci(bs_t - bs_m)}
                    # enrichment over each arm's own chance, top vs matched
                    pt_t, bs_t = Bs.ratio(v("top", "hits") - v("top", "chance"), v("top", "n_ref"))
                    pt_m, bs_m = Bs.ratio(v("matched", "hits") - v("matched", "chance"),
                                          v("matched", "n_ref"))
                    pd["new_deg_share_minus_chance"] = {
                        "top": round(float(pt_t), 4), "matched": round(float(pt_m), 4),
                        "diff": round(float(pt_t - pt_m), 4), "diff_ci95": ci(bs_t - bs_m)}
                    ent["paired"] = pd
                out["newtype"][key] = ent
        results[gname] = out
        print(f"[rr] {gname}: n={n}", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"model_tag": a.model_tag, "seed": a.seed, "n_boot": a.n_boot,
               "conditions": CONDS, "diagnostics": diagnostics, "results": results},
              open(a.out, "w"), indent=1)
    cells_path = a.out.replace(".json", "_cells.jsonl")
    with open(cells_path, "w") as f:
        for r in per_cell:
            f.write(json.dumps(r) + "\n")
    print(f"[rr] wrote {a.out} and {cells_path}")

    # ---------------------------------------------------------------- readable summary ---------
    def fmt(e):
        return f"{100 * e['value']:.2f} [{100 * e['ci95'][0]:.2f},{100 * e['ci95'][1]:.2f}]"

    lines = ["# Rationale content under DEG removal vs matched non-DEG removal (C2S-Scale 27B)", ""]
    for gname in DATASETS + ["pooled"]:
        R = results[gname]
        lines.append(f"## {gname}  (n cells = {R['n_cells']})")
        lines.append("| dose | metric | top-DEG | matched non-DEG | diff (top-matched) [95% CI] |")
        lines.append("|---|---|---|---|---|")
        for dose, top, mt in DOSES:
            for name in ["set_change_jd", "set_change_jd_retained", "remaining_deg_precision",
                         "remaining_deg_precision5", "removed_still_referenced",
                         "removed_still_referenced_given_wt", "ribosomal_share",
                         "type_changed_vs_wt"]:
                d = R["paired"][dose][name]
                lines.append(f"| {dose}% | {name} | {100 * d['top']:.2f} | {100 * d['matched']:.2f} "
                             f"| {100 * d['diff']:+.2f} [{100 * d['diff_ci95'][0]:+.2f},"
                             f"{100 * d['diff_ci95'][1]:+.2f}] |")
            ct, cm = R["conditions"][top], R["conditions"][mt]
            lines.append(f"| {dose}% | remaining_deg_chance | {100 * ct['remaining_deg_chance']['value']:.2f} "
                         f"| {100 * cm['remaining_deg_chance']['value']:.2f} | |")
            lines.append(f"| {dose}% | wt_referenced_removed_share | "
                         f"{100 * ct['wt_referenced_removed_share']['value']:.2f} | "
                         f"{100 * cm['wt_referenced_removed_share']['value']:.2f} | |")
            lines.append(f"| {dose}% | removed_referenced_on_wt | "
                         f"{100 * ct['removed_referenced_on_wt']['value']:.2f} | "
                         f"{100 * cm['removed_referenced_on_wt']['value']:.2f} | |")
            lines.append(f"| {dose}% | accuracy | {ct['accuracy']['value']:.3f} | "
                         f"{cm['accuracy']['value']:.3f} | |")
            lines.append(f"| {dose}% | removed_ribosomal_share | "
                         f"{100 * ct['removed_ribosomal_share']['value']:.2f} | "
                         f"{100 * cm['removed_ribosomal_share']['value']:.2f} | |")
            lines.append(f"| {dose}% | removed_still_referenced_ribosomal_share | "
                         f"{100 * ct['removed_still_referenced_ribosomal_share']['value']:.2f} | "
                         f"{100 * cm['removed_still_referenced_ribosomal_share']['value']:.2f} | |")
            d = R["paired"][dose]["set_change_jd"]
            lines.append(f"| {dose}% | frac cells JD top>matched / top<matched | "
                         f"{100 * d['frac_cells_top_gt_matched']:.1f} | "
                         f"{100 * d['frac_cells_top_lt_matched']:.1f} | |")
        w = R["conditions"]["wt"]
        lines.append(f"| wt | remaining_deg_precision (= Table precision) | {fmt(w['remaining_deg_precision'])} "
                     f"| chance {100 * w['remaining_deg_chance']['value']:.2f} | |")
        lines.append("")
        lines.append("| dose | resolution | n type-changed | n eligible | new-type DEG share: top | matched | wt "
                     "| chance top | chance matched | diff top-matched [CI] | (share-chance) diff [CI] |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for dose, _, _ in DOSES:
            for mode in ("strict", "tolerant"):
                e = R["newtype"][f"{dose}:{mode}"]
                if "top" in e:
                    pd = e["paired"]
                    lines.append(
                        f"| {dose}% | {mode} | {e['n_type_changed']} | {e['n_eligible']} "
                        f"| {fmt(e['top']['new_deg_share'])} | {fmt(e['matched']['new_deg_share'])} "
                        f"| {fmt(e['wt']['new_deg_share'])} | {100 * e['top']['chance']['value']:.2f} "
                        f"| {100 * e['matched']['chance']['value']:.2f} "
                        f"| {100 * pd['new_deg_share']['diff']:+.2f} [{100 * pd['new_deg_share']['diff_ci95'][0]:+.2f},"
                        f"{100 * pd['new_deg_share']['diff_ci95'][1]:+.2f}] "
                        f"| {100 * pd['new_deg_share_minus_chance']['diff']:+.2f} "
                        f"[{100 * pd['new_deg_share_minus_chance']['diff_ci95'][0]:+.2f},"
                        f"{100 * pd['new_deg_share_minus_chance']['diff_ci95'][1]:+.2f}] |")
                else:
                    lines.append(f"| {dose}% | {mode} | {e['n_type_changed']} | {e['n_eligible']} | too few | | | | | | |")
        lines.append("")
    md = a.out.replace(".json", "_summary.md")
    open(md, "w").write("\n".join(lines))
    print(f"[rr] wrote {md}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Tests whether the genes cited in C2S-Scale 27B rationales are chosen by their position in the cell
sentence or by their identity.

Compares the unmodified cell sentence with the same genes in a random order (`scramble_rank`).
Citation rates are computed per position decile and gene class (ribosomal or not, DEG of the
annotated type or not) with a paired cell bootstrap, and a logistic regression fits citation on
decile, ribosomal status and DEG status. Reads the rationale records under --rationales_root and writes
rationale_order.json, rationale_order_summary.md and a per-position CSV to --out_dir. Supports
the RQ3 appendix on C2S-Scale rationales.

    python -m input_use.core.analyze_rationale_order --out_dir outputs/analysis/rationale_order
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import time
from collections import Counter

import numpy as np

from input_use.core import paths as RD
from input_use.core.analyze_rationale_removal import (DATASETS, Boot, ci, cited_genes,
                                                       label_of, load_dataset)

CONDS = ["wt", "scramble_rank"]
N_DEC = 10
CLASSES = ["ribo_deg", "ribo_nondeg", "nonribo_deg", "nonribo_nondeg"]
CLS_IDX = {c: i for i, c in enumerate(CLASSES)}
# marginals over classes: which class columns are summed
MARGINALS = {"all": [0, 1, 2, 3], "ribo": [0, 1], "nonribo": [2, 3], "deg": [0, 2], "nondeg": [1, 3]}


def spearman(x, y):
    from scipy.stats import spearmanr
    if len(x) < 3:
        return float("nan")
    r = spearmanr(x, y).statistic
    return float(r) if np.isfinite(r) else float("nan")


def pack(pt, bs):
    return {"value": round(float(pt), 5), "ci95": ci(bs)}


class Boot2(Boot):
    """Boot plus a ratio of two pooled ratios, e.g. P(cited | decile 1) / P(cited | decile 10)."""

    def ratio_of_ratios(self, n1, d1, n2, d2):
        p1, b1 = self.ratio(n1, d1)
        p2, b2 = self.ratio(n2, d2)
        return p1 / max(p2, 1e-12), b1 / np.maximum(b2, 1e-12)


# ------------------------------------------------------------------------------------------------
def per_cell_rows(root, model_tag, ribo, slots_writer=None):
    rows = []
    for ds in DATASETS:
        meta, deg_by_type, vocab, sent, salient, control, gt, recs, cells = \
            load_dataset(root, ds, model_tag)
        n_dup = 0
        for c in cells:
            D = deg_by_type[gt[c]]
            row = {"dataset": ds, "cell": c, "gt": gt[c], "cond": {}}
            for k in CONDS:
                S = sent[(c, k)]
                n = len(S)
                Sset = set(S)
                n_dup += n - len(Sset)
                raw = recs[(c, k)]
                G = cited_genes(raw, vocab)
                idx = {g: i for i, g in enumerate(G)}
                tot = np.zeros((N_DEC, 4))
                cit = np.zeros((N_DEC, 4))
                ranks_cited, order_cited = [], []
                seen = set()
                for i, g in enumerate(S):
                    if g in seen:          # a repeated symbol keeps its first slot only
                        continue
                    seen.add(g)
                    dec = min(N_DEC - 1, (i * N_DEC) // n)
                    j = CLS_IDX[("ribo_" if g in ribo else "nonribo_") + ("deg" if g in D else "nondeg")]
                    tot[dec, j] += 1
                    hit = g in idx
                    cit[dec, j] += hit
                    if hit:
                        ranks_cited.append(i + 1)
                        order_cited.append(idx[g])
                    if slots_writer is not None:
                        slots_writer.writerow([ds, c, k, g, i + 1, n, dec, int(hit),
                                               idx.get(g, -1), int(g in D), int(g in ribo)])
                first5 = G[:5]
                rel = D & Sset
                row["cond"][k] = {
                    "tot": tot, "cit": cit, "n_genes": n,
                    "n_ref": len(G), "n_ref5": len(first5),
                    "ref_in_sent": sum(1 for g in G if g in Sset),
                    "ref_ribo": sum(1 for g in G if g in ribo),
                    "ref_deg": sum(1 for g in G if g in rel),
                    "ref5_deg": sum(1 for g in first5 if g in rel),
                    "ref5_ribo": sum(1 for g in first5 if g in ribo),
                    "chance_deg": len(G) * len(rel) / n,
                    "chance_deg5": len(first5) * len(rel) / n,
                    "chance_ribo": len(G) * len(Sset & ribo) / n,
                    "rho": spearman(order_cited, ranks_cited),
                    "mean_rank_frac": (float(np.mean(ranks_cited)) - 1) / max(n - 1, 1)
                    if ranks_cited else float("nan"),
                    "label": label_of(raw),
                }
            rows.append(row)
        print(f"[pos] {ds}: {len(cells)} cells, {n_dup} repeated symbols across both conditions",
              flush=True)
    return rows


# ------------------------------------------------------------------------------------------------
def logistic_or(rows_c, cond, atlas_idx, B_w, n_boot_reg, seed):
    """cited ~ 1 + decile + ribosomal + DEG + atlas dummies, on cell-aggregated slots, refitted on
    weighted cell resamples. Returns odds ratios with percentile intervals.
    """
    from sklearn.linear_model import LogisticRegression
    X, y, w, cell_of = [], [], [], []
    for ci_, r in enumerate(rows_c):
        e = r["cond"][cond]
        a = atlas_idx[r["dataset"]]
        for dec in range(N_DEC):
            for j, cls in enumerate(CLASSES):
                t, s = e["tot"][dec, j], e["cit"][dec, j]
                if t == 0:
                    continue
                feat = [dec, int(cls.startswith("ribo")), int(cls.endswith("_deg"))] + \
                       [int(a == q) for q in range(1, len(DATASETS))]
                for yy, ww in ((1, s), (0, t - s)):
                    if ww > 0:
                        X.append(feat); y.append(yy); w.append(ww); cell_of.append(ci_)
    X, y, w, cell_of = np.array(X, float), np.array(y), np.array(w, float), np.array(cell_of)
    names = ["decile", "ribosomal", "deg"] + DATASETS[1:]

    def fit(sw):
        m = LogisticRegression(penalty=None, max_iter=5000, tol=1e-8)
        m.fit(X, y, sample_weight=sw)
        return m.coef_[0]

    t0 = time.time()
    b0 = fit(w)
    bs = np.array([fit(w * B_w[b][cell_of]) for b in range(n_boot_reg)])
    print(f"[pos] {cond}: regression on {len(y)} aggregated rows, {n_boot_reg} refits in "
          f"{time.time() - t0:.0f} s", flush=True)
    out = {"n_rows_aggregated": int(len(y)), "n_slots": int(w.sum()), "n_boot": n_boot_reg,
           "covariates": names, "odds_ratio": {}}
    for i, nm in enumerate(names):
        out["odds_ratio"][nm] = {"value": round(float(np.exp(b0[i])), 5),
                                 "ci95": ci(np.exp(bs[:, i]))}
    return out


# ------------------------------------------------------------------------------------------------
def stats_for(rows_c, n_boot, n_boot_reg, seed, do_reg):
    n = len(rows_c)
    rng = np.random.default_rng(seed)
    B = Boot2(n, n_boot, rng)
    TOT = {k: np.stack([r["cond"][k]["tot"] for r in rows_c]) for k in CONDS}   # (n, 10, 4)
    CIT = {k: np.stack([r["cond"][k]["cit"] for r in rows_c]) for k in CONDS}
    col = lambda k, f: np.array([r["cond"][k][f] for r in rows_c], float)
    out = {"n_cells": n, "n_slots": {k: int(TOT[k].sum()) for k in CONDS}, "conditions": {}}
    for k in CONDS:
        e = {}
        e["mean_referenced"] = round(float(col(k, "n_ref").mean()), 2)
        e["in_sentence_share"] = pack(*B.ratio(col(k, "ref_in_sent"), col(k, "n_ref")))
        e["ribosomal_share"] = pack(*B.ratio(col(k, "ref_ribo"), col(k, "n_ref")))
        e["ribosomal_chance"] = pack(*B.ratio(col(k, "chance_ribo"), col(k, "n_ref")))
        e["ribosomal_share5"] = pack(*B.ratio(col(k, "ref5_ribo"), col(k, "n_ref5")))
        e["deg_precision"] = pack(*B.ratio(col(k, "ref_deg"), col(k, "n_ref")))
        e["deg_chance"] = pack(*B.ratio(col(k, "chance_deg"), col(k, "n_ref")))
        e["deg_precision5"] = pack(*B.ratio(col(k, "ref5_deg"), col(k, "n_ref5")))
        e["deg_chance5"] = pack(*B.ratio(col(k, "chance_deg5"), col(k, "n_ref5")))
        e["deg_precision_over_chance"] = pack(*B.ratio_of_ratios(
            col(k, "ref_deg"), col(k, "n_ref"), col(k, "chance_deg"), col(k, "n_ref")))
        e["deg_precision5_over_chance"] = pack(*B.ratio_of_ratios(
            col(k, "ref5_deg"), col(k, "n_ref5"), col(k, "chance_deg5"), col(k, "n_ref5")))
        rho = col(k, "rho")
        e["spearman_citation_order_vs_rank"] = {
            "mean": round(float(np.nanmean(rho)), 4), "n_cells": int(np.isfinite(rho).sum()),
            "frac_positive": round(float(np.nanmean(rho > 0)), 4)}
        mrf = col(k, "mean_rank_frac")
        e["mean_rank_fraction_of_cited"] = round(float(np.nanmean(mrf)), 4)
        # input composition of the sentence (share of slots), for reference
        tot_cls = TOT[k].sum(axis=(0, 1))
        e["input_class_share"] = {c: round(float(tot_cls[j] / tot_cls.sum()), 5)
                                  for j, c in enumerate(CLASSES)}
        # P(cited | decile, class) and marginals
        curves = {}
        for name, cols in list(MARGINALS.items()) + [(c, [j]) for j, c in enumerate(CLASSES)]:
            cur = []
            for dec in range(N_DEC):
                num = CIT[k][:, dec, cols].sum(axis=1)
                den = TOT[k][:, dec, cols].sum(axis=1)
                if den.sum() == 0:
                    cur.append(None)
                    continue
                d = pack(*B.ratio(num, den))
                d["n"] = int(den.sum())
                cur.append(d)
            curves[name] = cur
        e["p_cited_by_decile"] = curves
        # overall P(cited | class), and the position ratio decile 1 / decile 10
        pc = {}
        for name, cols in list(MARGINALS.items()) + [(c, [j]) for j, c in enumerate(CLASSES)]:
            num = CIT[k][:, :, cols].sum(axis=(1, 2))
            den = TOT[k][:, :, cols].sum(axis=(1, 2))
            d = pack(*B.ratio(num, den))
            d["n"] = int(den.sum())
            pc[name] = d
        e["p_cited"] = pc
        e["position_ratio_dec1_over_dec10"] = {}
        for name, cols in MARGINALS.items():
            n1, d1 = CIT[k][:, 0, cols].sum(1), TOT[k][:, 0, cols].sum(1)
            n2, d2 = CIT[k][:, N_DEC - 1, cols].sum(1), TOT[k][:, N_DEC - 1, cols].sum(1)
            e["position_ratio_dec1_over_dec10"][name] = pack(*B.ratio_of_ratios(n1, d1, n2, d2))
        # identity contrasts, crude (position-balanced on scramble_rank) and per decile
        contrasts = {
            "ribo_vs_nonribo": ([0, 1], [2, 3]),
            "deg_vs_nondeg_within_nonribo": ([2], [3]),
            "deg_vs_nondeg_within_ribo": ([0], [1]),
            "deg_vs_nondeg_all": ([0, 2], [1, 3]),
        }
        e["rate_ratio"] = {}
        for name, (ca, cb) in contrasts.items():
            n1, d1 = CIT[k][:, :, ca].sum((1, 2)), TOT[k][:, :, ca].sum((1, 2))
            n2, d2 = CIT[k][:, :, cb].sum((1, 2)), TOT[k][:, :, cb].sum((1, 2))
            d = pack(*B.ratio_of_ratios(n1, d1, n2, d2))
            d["by_decile"] = []
            for dec in range(N_DEC):
                n1, d1 = CIT[k][:, dec, ca].sum(1), TOT[k][:, dec, ca].sum(1)
                n2, d2 = CIT[k][:, dec, cb].sum(1), TOT[k][:, dec, cb].sum(1)
                if d1.sum() == 0 or d2.sum() == 0 or n2.sum() == 0:
                    d["by_decile"].append(None)
                else:
                    d["by_decile"].append(pack(*B.ratio_of_ratios(n1, d1, n2, d2)))
            e["rate_ratio"][name] = d
        out["conditions"][k] = e
    # paired wt - scramble differences on the cited-side shares
    out["paired_wt_minus_scramble"] = {}
    for name, num, den in [("ribosomal_share", "ref_ribo", "n_ref"),
                           ("deg_precision", "ref_deg", "n_ref"),
                           ("deg_precision5", "ref5_deg", "n_ref5"),
                           ("in_sentence_share", "ref_in_sent", "n_ref")]:
        p1, b1 = B.ratio(col("wt", num), col("wt", den))
        p2, b2 = B.ratio(col("scramble_rank", num), col("scramble_rank", den))
        out["paired_wt_minus_scramble"][name] = {"wt": round(float(p1), 5),
                                                 "scramble_rank": round(float(p2), 5),
                                                 "diff": round(float(p1 - p2), 5),
                                                 "diff_ci95": ci(b1 - b2)}
    if do_reg:
        atlas_idx = {ds: i for i, ds in enumerate(DATASETS)}
        out["logistic"] = {k: logistic_or(rows_c, k, atlas_idx, B.w[:n_boot_reg], n_boot_reg, seed)
                           for k in CONDS}
    return out


# ------------------------------------------------------------------------------------------------
def fmt(d, scale=100.0, nd=1):
    if d is None:
        return "-"
    lo, hi = d["ci95"]
    return f"{scale * d['value']:.{nd}f} [{scale * lo:.{nd}f}, {scale * hi:.{nd}f}]"


def summary_md(res, args):
    L = ["# C2S-Scale 27B rationale: input position vs gene identity", "",
         f"Conditions: `wt` (descending-expression order) and `scramble_rank` (same gene set, "
         f"order permuted). Model tag {args.model_tag}; bootstrap over cells, "
         f"{args.n_boot} draws (rates), {args.n_boot_reg} refits (regression), seed {args.seed}.",
         "Unit: gene slot (cell, condition, gene). Decile = tenth of the cell sentence by input rank.",
         ""]
    groups = DATASETS + ["pooled"]
    L += ["## Referenced-gene composition (%, pooled over referenced symbols)", "",
          "| group | cond | n cells | mean referenced | in sentence | ribosomal share | ribosomal "
          "input share | DEG precision | DEG chance | DEG precision@5 | chance@5 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for g in groups:
        R = res[g]
        for k in CONDS:
            e = R["conditions"][k]
            L.append(f"| {g} | {k} | {R['n_cells']} | {e['mean_referenced']} | "
                     f"{fmt(e['in_sentence_share'])} | {fmt(e['ribosomal_share'])} | "
                     f"{fmt(e['ribosomal_chance'])} | {fmt(e['deg_precision'])} | "
                     f"{fmt(e['deg_chance'])} | {fmt(e['deg_precision5'])} | {fmt(e['deg_chance5'])} |")
    L += ["", "## Q1. P(cited | input decile) (%), all genes", "",
          "| group | cond | " + " | ".join(f"d{i + 1}" for i in range(N_DEC)) + " | d1/d10 |",
          "|---|---|" + "---|" * (N_DEC + 1)]
    for g in groups:
        for k in CONDS:
            e = res[g]["conditions"][k]
            cur = e["p_cited_by_decile"]["all"]
            L.append(f"| {g} | {k} | " + " | ".join(f"{100 * d['value']:.1f}" for d in cur) +
                     f" | {fmt(e['position_ratio_dec1_over_dec10']['all'], 1.0, 2)} |")
    L += ["", "### P(cited | decile, class) (%), pooled", ""]
    for k in CONDS:
        L += [f"**{k}**", "", "| class | n slots | " + " | ".join(f"d{i + 1}" for i in range(N_DEC)) + " |",
              "|---|---|" + "---|" * N_DEC]
        e = res["pooled"]["conditions"][k]
        for c in CLASSES + ["ribo", "nonribo"]:
            cur = e["p_cited_by_decile"][c]
            L.append(f"| {c} | {e['p_cited'][c]['n']} | " +
                     " | ".join("-" if d is None else f"{100 * d['value']:.1f}" for d in cur) + " |")
        L.append("")
    if "logistic" in res["pooled"]:
        L += ["### Logistic regression, pooled: cited ~ decile + ribosomal + DEG + atlas", "",
              "| cond | OR per decile | OR ribosomal | OR DEG | n slots |", "|---|---|---|---|---|"]
        for k in CONDS:
            lg = res["pooled"]["logistic"][k]
            o = lg["odds_ratio"]
            L.append(f"| {k} | {fmt(o['decile'], 1.0, 3)} | {fmt(o['ribosomal'], 1.0, 2)} | "
                     f"{fmt(o['deg'], 1.0, 2)} | {lg['n_slots']} |")
        L.append("")
    L += ["## Q2 / Q3. Rate ratios P(cited | A) / P(cited | B), crude (position-balanced on "
          "scramble_rank)", "",
          "| group | cond | ribosomal / non-ribosomal | DEG / non-DEG within non-ribosomal | "
          "DEG / non-DEG within ribosomal | DEG / non-DEG, all |", "|---|---|---|---|---|---|"]
    for g in groups:
        for k in CONDS:
            rr = res[g]["conditions"][k]["rate_ratio"]
            L.append(f"| {g} | {k} | {fmt(rr['ribo_vs_nonribo'], 1.0, 2)} | "
                     f"{fmt(rr['deg_vs_nondeg_within_nonribo'], 1.0, 2)} | "
                     f"{fmt(rr['deg_vs_nondeg_within_ribo'], 1.0, 2)} | "
                     f"{fmt(rr['deg_vs_nondeg_all'], 1.0, 2)} |")
    L += ["", "### P(cited | class) (%), overall", "",
          "| group | cond | " + " | ".join(CLASSES) + " |", "|---|---|" + "---|" * len(CLASSES)]
    for g in groups:
        for k in CONDS:
            pc = res[g]["conditions"][k]["p_cited"]
            L.append(f"| {g} | {k} | " + " | ".join(fmt(pc[c], 100, 2) for c in CLASSES) + " |")
    L += ["", "## Citation order vs input rank (Spearman, mean over cells) and mean rank fraction "
          "of cited genes", "", "| group | cond | mean rho | n cells | frac rho>0 | mean rank frac |",
          "|---|---|---|---|---|---|"]
    for g in groups:
        for k in CONDS:
            e = res[g]["conditions"][k]
            s = e["spearman_citation_order_vs_rank"]
            L.append(f"| {g} | {k} | {s['mean']:.3f} | {s['n_cells']} | {s['frac_positive']:.3f} | "
                     f"{e['mean_rank_fraction_of_cited']:.3f} |")
    L += ["", "## Paired wt - scramble_rank (percentage points)", "",
          "| group | quantity | wt | scramble_rank | diff [95% CI] |", "|---|---|---|---|---|"]
    for g in groups:
        for name, d in res[g]["paired_wt_minus_scramble"].items():
            lo, hi = d["diff_ci95"]
            L.append(f"| {g} | {name} | {100 * d['wt']:.2f} | {100 * d['scramble_rank']:.2f} | "
                     f"{100 * d['diff']:.2f} [{100 * lo:.2f}, {100 * hi:.2f}] |")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rationales_root", default=RD.C2S_RATIONALES,
                    help="directory with one rationale run per atlas")
    ap.add_argument("--ribo", default="input_use/data/hgnc_ribosomal_proteins.json")
    ap.add_argument("--model_tag", default="27b")
    ap.add_argument("--n_boot", type=int, default=4000)
    ap.add_argument("--n_boot_reg", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_regression", action="store_true")
    ap.add_argument("--no_slots", action="store_true", help="skip the per-slot csv.gz")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    ribo = set(json.load(open(a.ribo))["symbols"])

    slots_path = os.path.join(a.out_dir, "rationale_order_slots.csv.gz")
    if a.no_slots:
        rows = per_cell_rows(a.rationales_root, a.model_tag, ribo)
    else:
        with gzip.open(slots_path + ".tmp", "wt", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["dataset", "cell", "condition", "gene", "rank", "n_genes", "decile",
                        "cited", "cite_index", "deg", "ribosomal"])
            rows = per_cell_rows(a.rationales_root, a.model_tag, ribo, w)
        os.replace(slots_path + ".tmp", slots_path)
        print(f"[pos] wrote {slots_path}", flush=True)

    res = {}
    for g in DATASETS + ["pooled"]:
        sub = rows if g == "pooled" else [r for r in rows if r["dataset"] == g]
        t0 = time.time()
        res[g] = stats_for(sub, a.n_boot, a.n_boot_reg, a.seed,
                           do_reg=(not a.no_regression) and g == "pooled")
        print(f"[pos] {g}: stats in {time.time() - t0:.0f} s", flush=True)
        for k in CONDS:
            e = res[g]["conditions"][k]
            print(f"      {k:14s} ribo share {100 * e['ribosomal_share']['value']:.1f} "
                  f"(input {100 * e['ribosomal_chance']['value']:.1f})  DEG prec "
                  f"{100 * e['deg_precision']['value']:.1f} (chance {100 * e['deg_chance']['value']:.1f})  "
                  f"P(cited) d1 {100 * e['p_cited_by_decile']['all'][0]['value']:.1f} d10 "
                  f"{100 * e['p_cited_by_decile']['all'][-1]['value']:.1f}  "
                  f"RR ribo {e['rate_ratio']['ribo_vs_nonribo']['value']:.2f}  "
                  f"RR deg|nonribo {e['rate_ratio']['deg_vs_nondeg_within_nonribo']['value']:.2f}",
                  flush=True)
    out = {"model_tag": a.model_tag, "seed": a.seed, "n_boot": a.n_boot, "n_boot_reg": a.n_boot_reg,
           "conditions": CONDS, "classes": CLASSES, "n_deciles": N_DEC,
           "ribosomal_definition": a.ribo, "results": res}
    p = os.path.join(a.out_dir, "rationale_order.json")
    json.dump(out, open(p, "w"), indent=1)
    open(os.path.join(a.out_dir, "rationale_order_summary.md"), "w").write(summary_md(res, a))
    print(f"[pos] wrote {p}")


if __name__ == "__main__":
    main()

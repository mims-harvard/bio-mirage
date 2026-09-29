#!/usr/bin/env python
"""Tests whether a Cell Ontology description of cell type B in the C2S-Scale prompt changes the
prediction for a cell sentence of type A.

Per model run it reports accuracy on the cell sentence alone and with the description of A, of B, or
a neutral description, the rate at which the prediction becomes B (over all cells and over cells
answered correctly without text), the rate at which the text of B alone yields B, and a least
squares fit of the per-pair rate on the information content of the lowest common ancestor of A and
B. This analysis is not reported in the paper.

    python -m input_use.core.analyze_text_conflict --rundir <run>/c2s_scale_27b --rundir <run>/c2s_scale_2b --pairs <run>/pairs.json [--out outputs/analysis/text_conflict.json]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from input_use.core.records import read_jsonl
from input_use.core.stats import bootstrap_ci

XA, XA_TA, XA_TB, XA_TN, TB_ONLY, REDUCTIO, TWOWAY = (
    "tc_xA", "tc_xA_tA", "tc_xA_tB", "tc_xA_tneutral", "tc_tB_only", "tc_xA_tB_reductio",
    "tc_xA_tB_2way")


def _ols(X, y):
    """Least squares with an intercept; returns coefficients. Pair-level rates, so ols is adequate and
    avoids a hard dependency on a glm library for a two-covariate fit.
    """
    y = np.asarray(y, float)
    # len(y), not len(X): X is the list of covariate columns, so len(X) is the covariate count (2).
    A = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return beta


def analyze(rundir: Path, pairs_by_id: dict) -> dict:
    by = defaultdict(dict)
    for r in read_jsonl(rundir / "records.jsonl"):
        by[r["example_id"]][r["condition"]] = r
    cells = [c for c, v in by.items() if {XA, XA_TA, XA_TB, XA_TN, TB_ONLY} <= set(v)]
    if not cells:
        raise SystemExit(f"no complete cells in {rundir}")
    pred = lambda c, k: by[c][k]["output"]["parsed"]["cell_type"]
    # exists only on the upstream Example.
    B_of = lambda c: by[c][XA]["provenance"]["B"]
    gt = lambda c: by[c][XA]["ground_truth"]["cell_type"]
    pair_of = lambda c: by[c][XA]["provenance"]["pair"]

    out = {"model": by[cells[0]][XA].get("model"), "n_cells": len(cells),
           "n_pairs": len({pair_of(c) for c in cells})}

    base_ok = [c for c in cells if pred(c, XA) == gt(c)]
    out["baseline_accuracy"] = round(len(base_ok) / len(cells), 4)
    out["n_conditioning_set"] = len(base_ok)

    # --- tor, conditional and unconditional -------------------------------------------------
    for name, pool in (("cond", base_ok), ("uncond", cells)):
        if not pool:
            out[f"TOR_{name}"] = None
            continue
        v = [1.0 if pred(c, XA_TB) == B_of(c) else 0.0 for c in pool]
        lo, hi = bootstrap_ci(v)
        out[f"TOR_{name}"] = round(sum(v) / len(v), 4)
        out[f"TOR_{name}_ci95"] = [lo, hi]
        out[f"TOR_{name}_n"] = len(pool)

    # --- TextEffect: B-rate under conflict minus under the matched neutral description ------
    tb = sum(pred(c, XA_TB) == B_of(c) for c in cells) / len(cells)
    tn = sum(pred(c, XA_TN) == B_of(c) for c in cells) / len(cells)
    ta = sum(pred(c, XA_TA) == gt(c) for c in cells) / len(cells)
    out["P_B_under_tB"] = round(tb, 4)
    out["P_B_under_tneutral"] = round(tn, 4)
    out["TextEffect"] = round(tb - tn, 4)
    out["acc_xA_tA"] = round(ta, 4)
    out["acc_xA_tB"] = round(sum(pred(c, XA_TB) == gt(c) for c in cells) / len(cells), 4)
    out["acc_xA_tneutral"] = round(sum(pred(c, XA_TN) == gt(c) for c in cells) / len(cells), 4)
    out["acc_tB_only_is_B"] = round(sum(pred(c, TB_ONLY) == B_of(c) for c in cells) / len(cells), 4)

    # --- dmr ---------------------------------------------------------------------------------
    changed = [c for c in cells if pred(c, XA_TB) != pred(c, XA)]
    if changed:
        hit = sum(pred(c, XA_TB) == pred(c, TB_ONLY) for c in changed)
        marg = Counter(pred(c, XA_TB) for c in cells)
        tot = sum(marg.values())
        exp = sum(marg[pred(c, TB_ONLY)] / tot for c in changed) / len(changed)
        out.update(DMR=round(hit / len(changed), 4), DMR_chance=round(exp, 4),
                   n_changed=len(changed), flip_rate=round(len(changed) / len(cells), 4))
    else:
        out.update(DMR=None, DMR_chance=None, n_changed=0, flip_rate=0.0)

    # --- two-way arm: the measure gate 2 validated ----------------------------------- The
    # 31-option tor structurally undercounts override.
    two = [c for c in cells if TWOWAY in by[c]]
    if two:
        v = [1.0 if pred(c, TWOWAY) == B_of(c) else 0.0 for c in two]
        lo, hi = bootstrap_ci(v)
        out["TOR_2way"] = round(sum(v) / len(v), 4)
        out["TOR_2way_ci95"] = [lo, hi]
        out["TOR_2way_n"] = len(two)
        base2 = [c for c in two if pred(c, XA) == gt(c)]
        if base2:
            vc = [1.0 if pred(c, TWOWAY) == B_of(c) else 0.0 for c in base2]
            out["TOR_2way_cond"] = round(sum(vc) / len(vc), 4)
            out["TOR_2way_cond_n"] = len(base2)
        by_str2 = defaultdict(list)
        for c in two:
            by_str2[pairs_by_id[pair_of(c)]["stratum"]].append(
                1.0 if pred(c, TWOWAY) == B_of(c) else 0.0)
        out["TOR_2way_by_stratum"] = {k: [round(float(np.mean(v)), 4), len(v)]
                                      for k, v in sorted(by_str2.items())}

    # --- reductio ----------------------------------------------------------------------------
    red = [c for c in cells if REDUCTIO in by[c]]
    if red:
        rb = [c for c in red if pred(c, REDUCTIO) != pred(c, XA)]
        out["reductio_flip_rate"] = round(len(rb) / len(red), 4)
        out["reductio_acc"] = round(sum(pred(c, REDUCTIO) == gt(c) for c in red) / len(red), 4)
        out["reductio_n"] = len(red)

    # --- per-pair tor, and the ic / |D| regression -------------------------------------------
    per_pair = defaultdict(list)
    for c in base_ok:
        per_pair[pair_of(c)].append(1.0 if pred(c, XA_TB) == B_of(c) else 0.0)
    rows = [(p, sum(v) / len(v), len(v)) for p, v in per_pair.items() if len(v) >= 5]
    if len(rows) >= 10:
        ic = [pairs_by_id[p]["ic_lca"] for p, _, _ in rows]
        nd = [pairs_by_id[p]["n_contrastive"] for p, _, _ in rows]
        tor = [t for _, t, _ in rows]
        b = _ols([ic, nd], tor)
        out["regression"] = {"n_pairs": len(rows), "intercept": round(b[0], 4),
                             "beta_ic_lca": round(b[1], 4), "beta_n_contrastive": round(b[2], 4),
                             "corr_ic_vs_nD": round(float(np.corrcoef(ic, nd)[0, 1]), 4)}
        by_str = defaultdict(list)
        for (p, t, _), i in zip(rows, ic):
            by_str[pairs_by_id[p]["stratum"]].append(t)
        out["TOR_by_stratum"] = {k: [round(float(np.mean(v)), 4), len(v)]
                                 for k, v in sorted(by_str.items())}
    out["per_pair_TOR"] = {p: round(t, 4) for p, t, _ in sorted(rows, key=lambda r: -r[1])[:8]}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rundir", action="append", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    pairs = {f"{r['cl_B']}__vs__{r['cl_A']}": r for r in json.load(open(a.pairs))}
    allr = {}
    for rd in a.rundir:
        r = analyze(Path(rd), pairs)
        allr[str(rd)] = r
        print(f"\n=== {r['model']}  {r['n_cells']} cells / {r['n_pairs']} pairs ===")
        print(f"  baseline acc(xA)      = {r['baseline_accuracy']:.3f}   "
              f"(conditioning set n={r['n_conditioning_set']})")
        print(f"  acc  xA_tA={r['acc_xA_tA']:.3f}  xA_tB={r['acc_xA_tB']:.3f}  "
              f"xA_tneutral={r['acc_xA_tneutral']:.3f}")
        print(f"  TOR  conditional={r['TOR_cond']}  (n={r.get('TOR_cond_n')})   "
              f"unconditional={r['TOR_uncond']}")
        print(f"  TextEffect = {r['TextEffect']:+.3f}   "
              f"(P(B|tB)={r['P_B_under_tB']:.3f} vs P(B|tneutral)={r['P_B_under_tneutral']:.3f})")
        print(f"  text alone yields B in {r['acc_tB_only_is_B']:.3f} of cells")
        print(f"  flip={r['flip_rate']:.3f}  DMR={r['DMR']} vs chance {r['DMR_chance']} "
              f"(n_changed={r['n_changed']})")
        if "reductio_n" in r:
            print(f"  reductio: flip={r['reductio_flip_rate']:.3f} acc={r['reductio_acc']:.3f} "
                  f"(n={r['reductio_n']})")
        if "TOR_2way" in r:
            print(f"  TWO-WAY (A vs B only, chance 0.500): TOR={r['TOR_2way']:.3f} "
                  f"(n={r['TOR_2way_n']})  conditional={r.get('TOR_2way_cond')} "
                  f"(n={r.get('TOR_2way_cond_n')})")
            print(f"     by stratum: {r['TOR_2way_by_stratum']}")
        if "regression" in r:
            g = r["regression"]
            print(f"  TOR ~ IC(LCA) + |D|:  beta_IC={g['beta_ic_lca']:+.4f}  "
                  f"beta_|D|={g['beta_n_contrastive']:+.4f}  "
                  f"(corr(IC,|D|)={g['corr_ic_vs_nD']:+.2f}, {g['n_pairs']} pairs)")
            print(f"  TOR by stratum: {r['TOR_by_stratum']}")
    if a.out:
        Path(a.out).write_text(json.dumps(allr, indent=2))
        print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""95% cluster bootstrap intervals for the evidence conflict bars of Figure 3 (BioReason, BioReason-Pro
and C2S-Scale panels).

For each bar, the shares of answers that follow A, follow B, or follow neither are resampled over
clusters: genomes (variant keys) for BioReason, individual conflicts for BioReason-Pro (counts taken
from the scorer's metrics.json), and cell type pairs for C2S-Scale. The default is 10,000 draws with
seed 20260918. The script checks every point estimate against fig_rq1_swap_numbers.csv, written by
figures/evidence_swap_panels.py, and writes outputs/analysis/ci_swap.json.

    python analysis/bootstrap_evidence_swap.py [--B 10000] [--out outputs/analysis/ci_swap.json]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.environ.get("INPUT_USE_OUTPUT_DIR", "outputs"), "analysis")
os.makedirs(OUT_DIR, exist_ok=True)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "figures"))
import rq1_data as M  # noqa: E402
import evidence_swap_panels as S  # noqa: E402
from input_use.metrics.dna import label_of, normalize_label_text, canonical_gold  # noqa: E402
from input_use.core import paths as RD  # noqa: E402

CSV = os.path.join(M.OUT_DEFAULT, "fig_rq1_swap_numbers.csv")


def boot(items, B, rng):
    """items: list of (cluster, outcome). Returns {outcome: (lo, hi)} from resampled clusters."""
    by = defaultdict(list)
    for c, o in items:
        by[c].append(o)
    keys = list(by)
    outcomes = ["A", "B", "neither"]
    # per cluster: count of each outcome, and its size
    counts = np.array([[sum(1 for o in by[k] if o == w) for w in outcomes] for k in keys], float)
    sizes = counts.sum(1)
    idx = rng.integers(0, len(keys), size=(B, len(keys)))
    num = counts[idx].sum(1)                       # B x 3
    den = sizes[idx].sum(1)[:, None]
    shares = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    lo, hi = np.percentile(shares, [2.5, 97.5], axis=0)
    return {w: (float(lo[i]), float(hi[i])) for i, w in enumerate(outcomes)}


def point(items):
    c = Counter(o for _, o in items)
    n = sum(c.values())
    return n, {w: c[w] / n for w in ("A", "B", "neither")}, c


# ------------------------------------------------------------------------------------------------
def dna_items():
    """{(arm, rowset): [(variant_key, outcome)]} for the RL checkpoint."""
    D = M.KEGG
    pairs = json.load(open(f"{D}/pairs.json"))
    donors, sens = pairs["donors"], set(pairs["strata"]["sensitivity"]["rows"])
    recs = list(S.read_jsonl(RD.dna_records(D, 'rl')))
    labels = sorted({(r.get("ground_truth") or {}).get("answer", "") for r in recs} - {""})
    wt = {r["example_id"]: r for r in recs if r["condition"] == "wt"}
    arms = {"swap_variant_donor": {r["example_id"]: r for r in recs
                                   if r["condition"] == "swap_variant_donor"}}
    f2 = RD.dna_records(S.SWAPBOTH, 'rl')
    if os.path.exists(f2):
        new = list(S.read_jsonl(f2))
        for arm, _ in S.PANEL_ARMS:
            sb = {r["example_id"]: r for r in new if r["condition"] == arm}
            if sb:
                arms[arm] = sb

    def cls(r):
        pred = label_of(r["output"].get("raw", ""), labels)
        d = donors.get(r["example_id"], {}).get("donor_answer")
        g = r["ground_truth"].get("answer")
        if d is None or g is None or pred is None:
            return None
        p = normalize_label_text(pred)
        return "A" if p == canonical_gold(d) else ("B" if p == canonical_gold(g) else "neither")

    def ok(eid):
        pred = label_of(wt[eid]["output"].get("raw", ""), labels)
        return pred is not None and \
            normalize_label_text(pred) == canonical_gold(wt[eid]["ground_truth"].get("answer"))

    out = {}
    for arm, sw in arms.items():
        for key in ("all", "dep"):
            rows = set(sw) if key == "all" else set(sw) & sens
            items = []
            for e in rows:
                if not ok(e):
                    continue
                o = cls(sw[e])
                if o is not None:
                    items.append((sw[e]["input"]["variant_key"], o))
            out[(arm, key)] = items
    return out


# ------------------------------------------------------------------------------------------------
# Panel b is pooled from metrics.json rather than re-derived from records: the scorer's gate ("both
# members answered correctly with no conflict") tests the predicted GO terms against each protein's
# require/forbid spec under GO ancestor closure, which a plain set-membership test does not
# reproduce -- it passes 60 of the 189 pairs the scorer passes.
def protein_items():
    """{family: [(item_id, outcome)]} -- one cluster per conflict, so boot() resamples conflicts."""
    m = M.load(f"{RD.PROTEIN_EVIDENCE_CONFLICTS['rl']}/metrics.json")
    cats = [c for c, _ in M.FAM_CATS]
    out = {}
    for key, _, _ in M.CONFLICT_ARMS:
        items = []
        for c in cats:
            fam = m["by_category"][c]["by_conflict_family"][key]
            n = fam["context_override_rate"]["n"]
            ks = {}
            for which, rk in (("B", "context_override_rate"), ("A", "protein_retention_rate"),
                              ("neither", "other_rate")):
                k = fam[rk]["rate"] * n
                assert abs(k - round(k)) < 1e-6, (key, c, rk, k, n)
                ks[which] = int(round(k))
            assert sum(ks.values()) == n, (key, c, ks, n)
            for which, k in ks.items():
                items += [(f"{c}:{key}:{which}:{i}", which) for i in range(k)]
        out[key] = items
    return out


# ------------------------------------------------------------------------------------------------
def sc_items():
    """{arm: [(pair, outcome)]} over the cells annotated as A with no description."""
    by = defaultdict(dict)
    for r in S.read_jsonl(f"{S.SC_RUN}/c2s_scale_27b/records.jsonl"):
        by[r["example_id"]][r["condition"]] = r
    need = {"tc_xA", "tc_xA_tB", "tc_xA_tB_2way"}
    cells = [c for c, v in by.items() if need <= set(v)]
    pred = lambda c, k: by[c][k]["output"]["parsed"]["cell_type"]      # noqa: E731
    gt = lambda c: by[c]["tc_xA"]["ground_truth"]["cell_type"]         # noqa: E731
    B_of = lambda c: by[c]["tc_xA"]["provenance"]["B"]                 # noqa: E731
    cond = [c for c in cells if pred(c, "tc_xA") == gt(c)]
    out = {}
    for arm, _ in S.SC_ARMS:
        out[arm] = [(by[c]["tc_xA"]["provenance"]["pair"],
                     "A" if pred(c, arm) == gt(c) else ("B" if pred(c, arm) == B_of(c) else "neither"))
                    for c in cond]
    return out


# ------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "ci_swap.json"))
    a = ap.parse_args()
    rng = np.random.default_rng(20260918)

    drawn = {}                                  # (panel, group, series) -> value, from the figure's CSV
    for r in csv.DictReader(open(CSV)):
        if r["value"] and r["series"].startswith(("matches", "neither")):
            drawn[(r["panel"], r["group"], r["series"])] = (float(r["value"]), int(r["n"] or 0))

    out, checked = {"_note": "95% cluster bootstrap, percentile; clusters are the text group (a), the "
                             "pair (b, c). B resamples of the clusters.", "_B": a.B}, 0
    for panel, items_by_key, keyfmt in (
            ("a", dna_items(), lambda k: f"{k[0]}|{k[1]}"),
            ("b", protein_items(), lambda k: k),
            ("c", sc_items(), lambda k: k)):
        out[panel] = {}
        for key, items in sorted(items_by_key.items(), key=lambda kv: str(kv[0])):
            if not items:
                continue
            n, p, c = point(items)
            ci = boot(items, a.B, rng)
            out[panel][keyfmt(key)] = {"n": n, "n_clusters": len({x for x, _ in items}),
                                       **{w: {"value": p[w], "k": c[w], "ci95": list(ci[w])}
                                          for w in ("A", "B", "neither")}}
            print(f"[{panel}] {keyfmt(key):42s} n={n:5d} clusters={len({x for x, _ in items}):4d}  " +
                  "  ".join(f"{w} {100 * p[w]:5.1f}% [{100 * ci[w][0]:.1f}, {100 * ci[w][1]:.1f}]"
                            for w in ("A", "B", "neither")))
            checked += 1
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"wrote {a.out} ({checked} bars)")

    # the point estimates must equal what the figure drew
    bad = []
    for (panel, group, series), (v, n) in sorted(drawn.items()):
        w = {"matches A (donor, swapped DNA)": "A", "matches A (ESM3 protein)": "A", "matches A": "A",
             "matches B (own text)": "B", "matches B": "B", "neither": "neither"}.get(series)
        if w is None:
            continue
        hit = [cell for cell in out[panel].values()
               if cell["n"] == n and abs(cell[w]["value"] - v) < 1e-9]
        if not hit:
            bad.append((panel, group, series, round(v, 6), n))
    if bad:
        print(f"\nWARNING: {len(bad)} drawn values have no matching bootstrap cell:")
        for b in bad:
            print("   ", b)
    else:
        print("every drawn value matches a bootstrap cell exactly")


if __name__ == "__main__":
    main()

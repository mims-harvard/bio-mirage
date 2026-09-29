#!/usr/bin/env python
"""Within-family AUROC on the InterPro-controlled protein subset for an ESM3 linear probe and for
BioReason-Pro RL and SFT with ESM3 intact and shuffled.

Uses the rows and features of protein_family_probe.py and adds the BioReason-Pro runs in which ESM3
encodes a shuffled sequence. BioReason-Pro is scored 1 when its GO terms assert the property and 0
otherwise. AUROC is computed within each InterPro family that contains both labels and averaged with
equal weight per family, per property and pooled. Intervals are 95% percentile bootstrap intervals
over families (2,000 draws, seed 0), and the probe is compared with 20 label permutations. Writes
outputs/analysis/one_metric_with_shuffled.json, which figures/rq1_data.py reads for the
BioReason-Pro bars of Figure 4 (left, panel b).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protein_family_probe as fla
from input_use.core import paths as RD

OUT_DIR = os.path.join(os.environ.get("INPUT_USE_OUTPUT_DIR", "outputs"), "analysis")
os.makedirs(OUT_DIR, exist_ok=True)

CATS, LABEL, FOLDS, NBOOT = fla.CATS, fla.LABEL, fla.FOLDS, 2000
SHUF = {"RL": RD.PROTEIN_EVIDENCE_CONFLICTS_SHUFFLED_ESM3["rl"], "SFT": RD.PROTEIN_EVIDENCE_CONFLICTS_SHUFFLED_ESM3["sft"]}
SHUF_CONDS = {"aligned_a_shuffle_esm3": "a", "aligned_b_shuffle_esm3": "b"}
ARMS = ["ESM3 linear probe", "BioReason-Pro RL", "BioReason-Pro RL, shuffled ESM3",
        "BioReason-Pro SFT", "BioReason-Pro SFT, shuffled ESM3"]


def load_shuffled(rows, dag):
    """Add r['RL_shuf'] / r['SFT_shuf']: did the model satisfy that protein's own spec when the
    embedding came from a shuffled sequence. Keyed exactly as protein_family_probe keys the intact
    arms, on (category, accession).
    """
    def satisfies(pred, spec):
        if not all(dag.any_under(pred, r) for r in spec.get("require", [])):
            return False
        return not any(dag.any_under(pred, r) for r in spec.get("forbid", []))

    index = {(r["cat"], r["acc"]): r for r in rows}
    for ck, rundir in SHUF.items():
        path = f"{rundir}/records.jsonl"
        if not os.path.exists(path):
            print(f"MISSING: {path} -- run the BioReason-Pro runner on the shuffled-ESM3 pair arm first")
            continue
        n_seen = n_empty = 0
        seen = set()
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            side = SHUF_CONDS.get(r["condition"])
            if side is None:
                continue
            gt = r["ground_truth"]
            key = (gt["category"], gt[f"accession_{side}"])
            row = index.get(key)
            if row is None or key in seen:
                continue
            seen.add(key)
            raw = (r["output"] or {}).get("raw") or ""
            terms = set((r["output"]["parsed"] or {}).get("go_terms") or [])
            if not raw.strip():
                n_empty += 1          # an empty response is a run failure, not a model answer
            n_seen += 1
            row[f"{ck}_shuf"] = float(satisfies(terms, row["spec"]))
        print(f"  {ck} shuffled: {n_seen} rows matched, {n_empty} empty responses")
        if n_empty:
            print(f"  WARNING {ck}: {n_empty} empty responses -- rerun at a lower "
                  f"--gpu_memory_utilization or refill before trusting this column")
    return rows


def probe_scores(X, y, g, permute=False, seed=0):
    s = np.full(len(y), np.nan)
    rng = np.random.default_rng(seed)
    for tr, te in GroupKFold(n_splits=min(FOLDS, len(set(g)))).split(X, y, g):
        ytr = rng.permutation(y[tr]) if permute else y[tr]
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=5000, C=1.0).fit(sc.transform(X[tr]), ytr)
        s[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return s


def within_family(score, y, g):
    return {f: float(roc_auc_score(y[g == f], score[g == f]))
            for f in sorted(set(g)) if len(set(y[g == f])) == 2}


def ci(v, nboot=NBOOT, seed=0):
    v = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    m = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(nboot)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    from input_use.core import config as cfg
    from input_use.metrics import go_dag
    rows = fla.build()
    rows = load_shuffled(rows, go_dag.load(str(cfg.GO_OBO)))
    have = {ck: all(f"{ck}_shuf" in r for r in rows) for ck in SHUF}
    print(f"\nshuffled column complete: {have}")

    res, pooled = {}, {}
    print(f"\n{'property':<21}{'prot':>6}{'fams':>6}  {'arm':<34}{'within-family AUROC':>21}"
          f"{'95% CI (family)':>18}")
    print("-" * 108)
    for cat in CATS:
        sub = [r for r in rows if r["cat"] == cat]
        X = np.stack([r["X"] for r in sub]); y = np.array([r["y"] for r in sub])
        g = np.array([f"{r['cat']}:{r['fam']}" for r in sub])
        arms = {"ESM3 linear probe": within_family(probe_scores(X, y, g), y, g)}
        # Label-permutation null on these rows, averaged over 20 permutations, because a single
        # permutation is too noisy to serve as the null.
        nulls = np.array([np.mean(list(within_family(
            probe_scores(X, y, g, permute=True, seed=100 + i), y, g).values())) for i in range(20)])
        for ck in ("RL", "SFT"):
            for key, name in ((ck, f"BioReason-Pro {ck}"),
                              (f"{ck}_shuf", f"BioReason-Pro {ck}, shuffled ESM3")):
                if not all(key in r for r in sub):
                    continue
                corr = np.array([r[key] for r in sub])
                asserted = np.where(y == 1, corr, 1.0 - corr)
                arms[name] = within_family(asserted, y, g)
        res[cat] = {"n_proteins": len(sub), "n_families": len(set(g)), "arms": {},
                    "probe_label_permutation_null": {"n_permutations": 20,
                                                     "mean": float(nulls.mean()),
                                                     "sd": float(nulls.std()),
                                                     "min": float(nulls.min()),
                                                     "max": float(nulls.max())}}
        for i, (name, fam) in enumerate(arms.items()):
            v = np.array(list(fam.values()))
            lo, hi = ci(v)
            res[cat]["arms"][name] = {"within_family_auroc": float(v.mean()),
                                      "ci95_family": [lo, hi], "n_families": len(v)}
            pooled.setdefault(name, []).append(v)
            head = f"{LABEL[cat]:<21}{len(sub):>6}{len(set(g)):>6}  " if i == 0 else f"{'':<21}{'':>6}{'':>6}  "
            print(f"{head}{name:<34}{v.mean():>21.3f}{f'[{lo:.3f}, {hi:.3f}]':>18}")
        print(f"{'':<21}{'':>6}{'':>6}  {'  probe, labels permuted':<34}{nulls.mean():>21.3f}"
              f"{f'sd {nulls.std():.3f}':>18}")
        print()
    print("-" * 108)
    res["ALL"] = {}
    for name in ARMS:
        if name not in pooled:
            continue
        v = np.concatenate(pooled[name])
        lo, hi = ci(v)
        res["ALL"][name] = {"within_family_auroc": float(v.mean()), "ci95_family": [lo, hi],
                            "n_families": len(v)}
        print(f"{'all four properties':<21}{len(rows):>6}{len(v):>6}  {name:<34}{v.mean():>21.3f}"
              f"{f'[{lo:.3f}, {hi:.3f}]':>18}")
    # the comparison the table exists for
    for ck in ("RL", "SFT"):
        a, b = f"BioReason-Pro {ck}", f"BioReason-Pro {ck}, shuffled ESM3"
        if a in res["ALL"] and b in res["ALL"]:
            d = res["ALL"][a]["within_family_auroc"] - res["ALL"][b]["within_family_auroc"]
            res["ALL"][f"{ck}_intact_minus_shuffled"] = d
            print(f"\n{ck}: intact minus shuffled ESM3 = {d:+.3f} within-family AUROC")
    with open(os.path.join(OUT_DIR, "one_metric_with_shuffled.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print("\n-> one_metric_with_shuffled.json")


if __name__ == "__main__":
    main()

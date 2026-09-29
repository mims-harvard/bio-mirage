#!/usr/bin/env python
"""Builds the protein set for the RQ1 appendix section "InterPro-controlled protein subset and
functional targets" and scores an ESM3 linear probe and BioReason-Pro on it by balanced accuracy.

One row per (property, protein) from the BioReason-Pro evidence conflict pairs, labelled by whether
the protein carries the property, with the InterPro family the pair shares. Proteins that are positive
in one pair and negative in another, or have no family, are dropped. Features are the mean and the
maximum of the cached ESM3 residue representations, concatenated. BioReason-Pro is scored by whether
its GO terms with unmodified inputs satisfy the protein's property specification under GO ancestor
closure. The probe uses five folds grouped by InterPro family. Intervals are 95% percentile bootstrap
intervals over families (2,000 draws, seed 0), and the probe is compared with 20 label permutations.
Writes outputs/analysis/family_level_analysis.json. protein_family_table.py and the two probe scripts
import build() from this file.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from input_use.core import config as cfg
from input_use.core import paths as RD
from input_use.metrics import go_dag

OUT_DIR = os.path.join(os.environ.get("INPUT_USE_OUTPUT_DIR", "outputs"), "analysis")
os.makedirs(OUT_DIR, exist_ok=True)

CACHE = RD.ESM3_CACHE
RUNS = {"RL": RD.PROTEIN_EVIDENCE_CONFLICTS["rl"], "SFT": RD.PROTEIN_EVIDENCE_CONFLICTS["sft"]}
CATS = ["pseudoenzyme", "organelle_targeted", "dna_binding", "go_not"]
LABEL = {"pseudoenzyme": "enzyme activity", "organelle_targeted": "organelle targeting",
         "dna_binding": "DNA binding", "go_not": "GO vs NOT"}
FOLDS, SEED, NBOOT, NPERM = 5, 0, 2000, 20


def pooled(seq):
    k = hashlib.sha1(seq.encode()).hexdigest()
    rows = np.load(f"{CACHE}/{k[:2]}/{k}.npy")
    body = rows[1:-1] if rows.shape[0] > 2 else rows
    return np.concatenate([body.mean(0), body.max(0)])


def build():
    """One row per (category, protein): label, InterPro family, ESM3 features, and each checkpoint's
    correctness on that protein under its own unperturbed evidence.
    """
    run = RUNS["RL"]
    fam_of = {}
    for line in open(f"{run}/pairs.jsonl"):
        p = json.loads(line)
        for side in ("a", "b"):
            fam_of[(p["category"], p[side]["accession"])] = p["shared_interpro"]
    rows = {}
    for line in open(f"{run}/examples.jsonl"):
        if '"aligned_a"' not in line and '"aligned_b"' not in line:
            continue
        e = json.loads(line)
        if e["condition"] not in ("aligned_a", "aligned_b"):
            continue
        gt, pl = e["ground_truth"], e["payload"]
        cat, acc = gt["category"], pl["accession"]
        y = 1 if e["condition"] == "aligned_a" else 0
        spec = gt["spec_a"] if y else gt["spec_b"]
        key = (cat, acc)
        if key in rows and rows[key]["y"] != y:
            rows[key]["conflict"] = True        # positive in one pair, negative in another
            continue
        if key in rows:
            continue
        rows[key] = {"cat": cat, "acc": acc, "y": y, "spec": spec,
                     "fam": fam_of.get(key), "seq": pl["sequence"], "conflict": False}
    dropped = [k for k, r in rows.items() if r["conflict"] or r["fam"] is None]
    for k in dropped:
        del rows[k]
    dag = go_dag.load(str(cfg.GO_OBO))

    def satisfies(pred, spec):
        if not all(dag.any_under(pred, r) for r in spec.get("require", [])):
            return False
        return not any(dag.any_under(pred, r) for r in spec.get("forbid", []))

    for ck, rundir in RUNS.items():
        seen = set()
        for line in open(f"{rundir}/records.jsonl"):
            if '"aligned_a"' not in line and '"aligned_b"' not in line:
                continue
            r = json.loads(line)
            if r["condition"] not in ("aligned_a", "aligned_b"):
                continue
            gt = r["ground_truth"]
            # records carry no accession field; the side is in the condition and the two accessions
            # are in ground_truth
            side = "a" if r["condition"] == "aligned_a" else "b"
            key = (gt["category"], gt[f"accession_{side}"])
            if key not in rows:
                continue
            if key in seen:
                continue
            seen.add(key)
            pred = set(r["output"]["parsed"]["go_terms"])
            rows[key][ck] = float(satisfies(pred, rows[key]["spec"]))
    out = [r for r in rows.values() if all(ck in r for ck in RUNS)]
    print(f"{len(out)} (category, protein) rows; dropped {len(dropped)} "
          f"(conflicting role in two pairs, or no family)")
    for r in out:
        r["X"] = pooled(r["seq"])
    return out


def balanced_accuracy(hit, y):
    hit, y = np.asarray(hit, float), np.asarray(y)
    if y.sum() == 0 or (1 - y).sum() == 0:
        return float("nan")
    return float(0.5 * (hit[y == 1].mean() + hit[y == 0].mean()))


def boot_families(hit, y, fams, nboot=NBOOT, seed=SEED):
    """Percentile CI resampling families, the unit the design actually replicates."""
    hit, y, fams = np.asarray(hit, float), np.asarray(y), np.asarray(fams)
    uniq = np.unique(fams)
    idx_of = {f: np.where(fams == f)[0] for f in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(nboot):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx_of[f] for f in draw])
        v = balanced_accuracy(hit[sel], y[sel])
        if not np.isnan(v):
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (None, None)


def probe_oof(sub, permute=False, seed=SEED):
    X = np.stack([r["X"] for r in sub])
    y = np.array([r["y"] for r in sub])
    g = np.array([r["fam"] for r in sub])
    pred = np.zeros(len(y))
    rng = np.random.default_rng(seed)
    for tr, te in GroupKFold(n_splits=min(FOLDS, len(set(g)))).split(X, y, g):
        ytr = rng.permutation(y[tr]) if permute else y[tr]
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=5000, C=1.0).fit(sc.transform(X[tr]), ytr)
        pred[te] = (clf.predict_proba(sc.transform(X[te]))[:, 1] >= 0.5).astype(float)
    return (pred == y).astype(float), y, g


def main():
    rows = build()
    res = {}
    hdr = (f"{'property':<21}{'prot':>5}{'fams':>5}  {'arm':<26}{'bal acc':>9}{'95% CI (family)':>18}"
           f"{'has prop':>10}{'lacks it':>10}")
    print("\n" + hdr); print("-" * len(hdr))
    for cat in CATS + ["ALL"]:
        sub = rows if cat == "ALL" else [r for r in rows if r["cat"] == cat]
        y = np.array([r["y"] for r in sub])
        fams = np.array([f"{r['cat']}:{r['fam']}" for r in sub])
        res[cat] = {"n_proteins": len(sub), "n_families": len(set(fams)),
                    "n_positive": int(y.sum()), "arms": {}}
        arms = []
        if cat == "ALL":
            # one probe per property, pooled by concatenating each property's out-of-fold hits
            hit = np.concatenate([res[c]["_probe_hit"] for c in CATS])
            yy = np.concatenate([res[c]["_probe_y"] for c in CATS])
            ff = np.concatenate([res[c]["_probe_fam"] for c in CATS])
            arms.append(("ESM3 linear probe", hit, yy, ff))
        else:
            hit, yy, ff = probe_oof(sub)
            res[cat]["_probe_hit"], res[cat]["_probe_y"], res[cat]["_probe_fam"] = hit, yy, ff
            nulls = [probe_oof(sub, permute=True, seed=100 + i)[0] for i in range(NPERM)]
            res[cat]["_null_hit"] = nulls
            arms.append(("ESM3 linear probe", hit, yy, ff))
        for ck in RUNS:
            arms.append((f"BioReason-Pro {ck}", np.array([r[ck] for r in sub]), y, fams))
        # null distribution of the probe, as mean +- sd over nperm label permutations
        if cat == "ALL":
            null_draws = [np.concatenate([res[c]["_null_hit"][i] for c in CATS])
                          for i in range(NPERM)]
            null_y = np.concatenate([res[c]["_probe_y"] for c in CATS])
        else:
            null_draws, null_y = res[cat]["_null_hit"], yy
        nv = np.array([balanced_accuracy(h, null_y) for h in null_draws])
        for name, hit, yy, ff in arms:
            ba = balanced_accuracy(hit, yy)
            lo, hi = boot_families(hit, yy, ff)
            pos = float(np.asarray(hit, float)[yy == 1].mean())
            neg = float(np.asarray(hit, float)[yy == 0].mean())
            res[cat]["arms"][name.strip()] = {"balanced_accuracy": ba, "ci95_family": [lo, hi],
                                              "acc_positive": pos, "acc_negative": neg,
                                              "n": int(len(yy))}
            head = f"{LABEL.get(cat, cat):<21}{len(sub):>5}{len(set(fams)):>5}  " if name == arms[0][0] \
                else f"{'':<21}{'':>5}{'':>5}  "
            print(f"{head}{name:<26}{ba:>9.3f}{f'[{lo:.3f}, {hi:.3f}]':>18}{pos:>10.3f}{neg:>10.3f}")
        obs = res[cat]["arms"]["ESM3 linear probe"]["balanced_accuracy"]
        res[cat]["probe_label_permutation_null"] = {
            "n_permutations": NPERM, "mean": float(nv.mean()), "sd": float(nv.std()),
            "min": float(nv.min()), "max": float(nv.max()),
            "fraction_of_draws_below_observed": float((obs > nv).mean())}
        print(f"{'':<21}{'':>5}{'':>5}  {'  probe, labels permuted':<26}"
              f"{nv.mean():>9.3f}{f'sd {nv.std():.3f}':>18}"
              f"   observed exceeds {(obs > nv).mean():.0%} of {NPERM} draws")
        print()
    for cat in CATS:
        for k in ("_probe_hit", "_probe_y", "_probe_fam", "_null_hit"):
            res[cat].pop(k, None)
    with open(os.path.join(OUT_DIR, "family_level_analysis.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print("-> family_level_analysis.json")


if __name__ == "__main__":
    main()

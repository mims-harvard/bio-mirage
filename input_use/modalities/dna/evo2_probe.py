#!/usr/bin/env python
r"""Fits linear probes on pooled Evo2 representations of the BioReason disease prediction queries.

The disease probe is a logistic regression evaluated with five folds grouped by genome, reported
with the majority class rate and the accuracy of the same probe on shuffled labels. With --pairs it
also fits a probe that separates the two queries of each pair from the genome-dependent queries (the
`sensitivity` stratum of pairs.json from kegg_pairs.py). Reads an npz with X, y and optional ids and
groups, and writes one JSON file.

    python -m input_use.modalities.dna.evo2_probe --embeddings <dir>/evo2_pooled.npz \
        --pairs <dir>/pairs.json --out <dir>/evo2_probe.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

PROBE_VERSION = "1"


def disease_probe(X, y, groups, seed=0, folds=5):
    """Grouped cv so the same genome never appears in train and test."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_score

    keep = np.isin(y, [k for k, c in Counter(y).items() if c >= folds])
    X, y, groups = X[keep], y[keep], groups[keep]
    if len(set(y)) < 2:
        return {"error": "fewer than 2 classes survive the >=folds filter"}
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    acc = cross_val_score(LogisticRegression(max_iter=3000), X, y, groups=groups, cv=cv)
    maj = max(Counter(y).values()) / len(y)
    # Falsification control: the same probe on shuffled labels must collapse to the majority rate.
    rng = np.random.default_rng(seed)
    y_shuf = y.copy(); rng.shuffle(y_shuf)
    acc_shuf = cross_val_score(LogisticRegression(max_iter=3000), X, y_shuf, groups=groups, cv=cv)
    return {"n": int(len(y)), "n_classes": int(len(set(y))),
            "accuracy": round(float(acc.mean()), 4), "accuracy_sd": round(float(acc.std()), 4),
            "majority_class_rate": round(float(maj), 4),
            "shuffled_labels_accuracy": round(float(acc_shuf.mean()), 4)}


def pair_probe(X, index_of, pairs, text_group=None, seed=0, folds=5):
    """Can a linear readout separate the two members of a sensitivity pair?"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score

    D, lab, grp, dropped = [], [], [], 0
    for p in pairs:
        ia, ib = index_of.get(p["a"]), index_of.get(p["b"])
        if ia is None or ib is None:
            dropped += 1
            continue
        first = p["answer_a"] < p["answer_b"]
        g = (text_group or {}).get(p["a"], p["a"])
        D.append(X[ia] - X[ib]); lab.append(int(first)); grp.append(g)
        D.append(X[ib] - X[ia]); lab.append(int(not first)); grp.append(g)
    if len(set(lab)) < 2:
        return {"error": "one class only", "n_dropped": dropped}
    D, lab, grp = np.asarray(D), np.asarray(lab), np.asarray(grp)
    clf = LogisticRegression(max_iter=3000)
    if text_group and len(set(grp)) >= 2:
        n = min(folds, len(set(grp)))
        acc = cross_val_score(clf, D, lab, groups=grp, cv=GroupKFold(n_splits=n))
        scheme = f"text-grouped ({n} folds, {len(set(grp))} texts)"
    else:
        acc = cross_val_score(clf, D, lab,
                              cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed))
        scheme = f"stratified ({folds} folds) -- NO text grouping, may leak"
    rng = np.random.default_rng(seed)
    lab_s = lab.copy(); rng.shuffle(lab_s)
    acc_s = cross_val_score(clf, D, lab_s, groups=grp if text_group else None,
                            cv=(GroupKFold(n_splits=min(folds, len(set(grp)))) if text_group
                                else StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)))
    return {"n_pairs_used": int(len(D) // 2), "n_dropped": int(dropped), "cv": scheme,
            "n_text_groups": int(len(set(grp))),
            "accuracy": round(float(acc.mean()), 4), "accuracy_sd": round(float(acc.std()), 4),
            "shuffled_labels_accuracy": round(float(acc_s.mean()), 4), "chance": 0.5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", required=True, help="npz with X, y, ids, groups")
    ap.add_argument("--pairs", default=None, help="pairs.json for the sensitivity-pair probe")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    z = np.load(a.embeddings, allow_pickle=True)
    X, y = z["X"], z["y"].astype(str)
    ids = z["ids"].astype(str) if "ids" in z else np.array([str(i) for i in range(len(y))])
    groups = z["groups"].astype(str) if "groups" in z else ids
    print(f"[evo2-probe] X={X.shape} labels={len(set(y))} groups={len(set(groups))}")

    res = {"probe_version": PROBE_VERSION, "embeddings": a.embeddings,
           "disease": disease_probe(X, y, groups, seed=a.seed, folds=a.folds)}
    d = res["disease"]
    if "error" not in d:
        print(f"  disease probe : acc={d['accuracy']:.3f} +-{d['accuracy_sd']:.3f} "
              f"(majority={d['majority_class_rate']:.3f}, shuffled-labels={d['shuffled_labels_accuracy']:.3f}, "
              f"n={d['n']}, {d['n_classes']} classes)")

    if a.pairs:
        pairs = json.load(open(a.pairs))["strata"]["sensitivity"]["pairs"]
        index_of = {e: i for i, e in enumerate(ids)}
        res["sensitivity_pair"] = pair_probe(X, index_of, pairs, seed=a.seed, folds=a.folds)
        p = res["sensitivity_pair"]
        if "error" not in p:
            print(f"  pair probe    : acc={p['accuracy']:.3f} +-{p['accuracy_sd']:.3f} "
                  f"(chance 0.5, {p['n_pairs_used']} pairs used, {p['n_dropped']} dropped)")

    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"  -> {a.out}")


if __name__ == "__main__":
    main()

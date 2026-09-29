#!/usr/bin/env python
"""95% percentile bootstrap intervals for the DNA bars of Figure 4 (left, panel a), on the 165
genome-dependent queries.

Resamples the 63 genomes (variant keys) with all their queries, 2,000 draws with seed 0, because
queries that share a genome share the Evo2 input. Covers the text-only predictor, BioReason RL with
Evo2 intact and shuffled, and a linear probe on the stored pooled Evo2 features
(dna/bioreason/perturbations/evo2_pooled.npz), which is checked against the stored probe accuracy
(dna/bioreason/genome_dependent/evo2_probe.json). The probe bars drawn in the figure come from
dna_probe_raw.py and dna_probe_projected.py. Writes outputs/analysis/ci_dna_probe.json.
"""
import os
import collections, json, os, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from input_use.metrics.dna import label_of, canonical_gold, normalize_label_text  # noqa: E402
from input_use.core import paths as RD  # noqa: E402

OUT_DIR = os.path.join(os.environ.get("INPUT_USE_OUTPUT_DIR", "outputs"), "analysis")
os.makedirs(OUT_DIR, exist_ok=True)

D = RD.DNA_PERTURBATIONS
REF = RD.DNA_GENOME_DEPENDENT_PROBE
N_BOOT, ALPHA = 2000, 0.05
LO, HI = int((ALPHA / 2) * N_BOOT), min(N_BOOT - 1, int((1 - ALPHA / 2) * N_BOOT))


def main():
    rows, raw, golds = {}, collections.defaultdict(dict), set()
    # the stored probe builds the label vocabulary from both checkpoints
    for ck in ("sft", "rl"):
        with open(RD.dna_records(D, ck)) as fh:
            for line in fh:
                r = json.loads(line); eid, cond = r["example_id"], r["condition"]
                golds.add(r["ground_truth"]["answer"])
                if ck != "rl":
                    continue
                if cond == "wt" and eid not in rows:
                    inp = r["input"]
                    rows[eid] = dict(text=inp["question"], stratum=inp["stratum"], genome=inp["variant_key"], gold=r["ground_truth"]["answer"])
                if cond in ("wt", "scramble"):
                    raw[cond][eid] = r["output"]["raw"]
    vocab = sorted(golds)
    dep = sorted(e for e, m in rows.items() if m["stratum"] == "sensitivity")
    assert len(dep) == 165, len(dep)
    corr = lambda l, g: float(l is not None and normalize_label_text(l) == canonical_gold(g))  # noqa: E731
    model = {arm: np.array([corr(label_of(raw[arm][e], vocab), rows[e]["gold"]) for e in dep]) for arm in ("wt", "scramble")}
    assert int(model["wt"].sum()) == 97 and int(model["scramble"].sum()) == 96, (model["wt"].sum(), model["scramble"].sum())

    z = np.load(f"{D}/evo2_pooled.npz", allow_pickle=True); X_all, ids = z["X"], z["ids"].astype(str)
    ix = {e: i for i, e in enumerate(ids)}
    F = np.stack([X_all[ix[e]] for e in dep]); y = np.array([rows[e]["gold"] for e in dep]); grp = np.array([rows[e]["genome"] for e in dep])
    texts = sorted({rows[e]["text"] for e in dep}); tid = np.array([texts.index(rows[e]["text"]) for e in dep])

    def folds(groups, k, seed):
        u = np.array(sorted(set(groups))); r = np.random.default_rng(seed); r.shuffle(u); f = {g: i % k for i, g in enumerate(u)}
        fa = np.array([f[g] for g in groups]); return [(np.where(fa != i)[0], np.where(fa == i)[0]) for i in range(k)]

    probe_hits, text_hits, probe_acc, text_acc = [], [], [], []
    for s in range(10):
        P = np.empty(len(y), dtype=object); TB = np.empty(len(y), dtype=object)
        for tr, te in folds(grp, 5, s):
            sc = StandardScaler().fit(F[tr]); clf = LogisticRegression(max_iter=5000, C=1.0).fit(sc.transform(F[tr]), y[tr])
            P[te] = clf.predict(sc.transform(F[te]))
            mlab = {t: collections.Counter(y[tr][tid[tr] == t]).most_common(1)[0][0] for t in set(tid[tr])}; gm = collections.Counter(y[tr]).most_common(1)[0][0]
            TB[te] = [mlab.get(t, gm) for t in tid[te]]
        probe_hits.append((P == y).astype(float)); text_hits.append((TB == y).astype(float))
        probe_acc.append((P == y).mean()); text_acc.append((TB == y).mean())
    ref = json.load(open(REF))["evo2 only C=1"]
    print(f"probe acc {np.mean(probe_acc):.6f} (ref {ref['acc']:.6f}), sd {np.std(probe_acc):.4f} (ref {ref['acc_sd']:.4f}); "
          f"text {np.mean(text_acc):.6f} (ref {ref['text_baseline']:.6f})", flush=True)
    assert abs(np.mean(probe_acc) - ref["acc"]) <= 0.003, "probe differs from evo2_probe.json by more than 5/1650"
    assert abs(np.mean(text_acc) - ref["text_baseline"]) < 1e-9
    bars = {"Evo2 linear probe, true labels": np.mean(probe_hits, axis=0),
            "text-only predictor": np.mean(text_hits, axis=0),
            "BioReason RL, Evo2 intact": model["wt"],
            "BioReason RL, Evo2 shuffled": model["scramble"]}
    genomes = sorted(set(grp)); rows_of = [np.where(grp == g)[0] for g in genomes]
    rng = np.random.default_rng(0)
    draws = {k: [] for k in bars}
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(genomes), len(genomes))
        rr = np.concatenate([rows_of[i] for i in pick])
        for k, v in bars.items():
            draws[k].append(v[rr].mean())
    out = {"method": "percentile bootstrap, 2000 draws, resampling genomes (variant keys) with all their rows",
           "n_rows": 165, "n_genomes": len(genomes), "bars": {}}
    for k, v in bars.items():
        s = np.sort(draws[k])
        out["bars"][k] = {"value": float(v.mean()), "ci95": [round(float(s[LO]), 4), round(float(s[HI]), 4)]}
        if k.startswith("Evo2"):
            import sklearn
            out["bars"][k].update(sd_over_seeds=float(np.std(probe_acc)), stored_value=float(ref["acc"]),
                                  rerun_minus_stored=float(np.mean(probe_acc) - ref["acc"]),
                                  sklearn=sklearn.__version__)
        print(k, out["bars"][k], flush=True)
    json.dump(out, open(f"{OUT_DIR}/ci_dna_probe.json", "w"), indent=1)
    print("->", f"{OUT_DIR}/ci_dna_probe.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Linear probes on the Evo2 representations before the BioReason projection, for the "Evo2 probe,
original" bar of Figure 4 (left, panel a) on the 165 genome-dependent queries.

Reads outputs/analysis/evo2_raw_both.npz from extract_evo2_raw.py. Probes the variant block alone and
the concatenated reference and variant blocks. Classifier, folds, text-only predictor, bootstrap and
label permutation are the same as in dna_probe_projected.py. Writes outputs/analysis/dna_raw_probe.json.
"""
import os
OUT_DIR = os.path.join(os.environ.get("INPUT_USE_OUTPUT_DIR", "outputs"), "analysis")
os.makedirs(OUT_DIR, exist_ok=True)
import json, sys, os, collections
import numpy as np
from input_use.metrics.dna import label_of, canonical_gold, normalize_label_text
from input_use.core import paths as RD  # noqa: E402

D = RD.DNA_PERTURBATIONS
OUT = OUT_DIR
CK = ("sft", "rl")
ARMS = ["wt", "no_dna", "no_modality", "scramble", "scramble_variant", "revert_variant_block", "swap_variant_donor"]
rows, pred, golds = {}, {ck: collections.defaultdict(dict) for ck in CK}, set()
for ck in CK:
    with open(RD.dna_records(D, ck)) as fh:
        for line in fh:
            r = json.loads(line); eid, cond = r["example_id"], r["condition"]
            if cond == "wt" and eid not in rows:
                inp = r["input"]
                rows[eid] = dict(text=inp["question"], stratum=inp["stratum"], genome=inp["variant_key"], split=inp["split"], gold=r["ground_truth"]["answer"])
            golds.add(r["ground_truth"]["answer"])
            if cond in ARMS: pred[ck][cond][eid] = r["output"]["raw"]
vocab = sorted(golds)
for ck in CK:
    for arm in ARMS: pred[ck][arm] = {e: label_of(g, vocab) for e, g in pred[ck][arm].items()}
def correct(l, g): return l is not None and normalize_label_text(l) == canonical_gold(g)
by_text = collections.defaultdict(list)
for e, m in rows.items(): by_text[m["text"]].append(e)
dep = sorted(e for e, m in rows.items() if m["stratum"] == "sensitivity")
inv = sorted(e for e, m in rows.items() if m["stratum"] != "sensitivity")

print("split composition  dep:", dict(collections.Counter(rows[e]["split"] for e in dep)), " inv:", dict(collections.Counter(rows[e]["split"] for e in inv)))
print("dep genes per split:", {s: sorted({rows[e]["text"].split('effect of this ')[1].split(' allele')[0] for e in dep if rows[e]['split']==s}) for s in ('train','test','val')})

# strict majority / minority / tied
kind = {}
for t, es in by_text.items():
    c = collections.Counter(rows[e]["gold"] for e in es).most_common()
    top, second = c[0][1], (c[1][1] if len(c) > 1 else 0)
    for e in es:
        n = dict(c)[rows[e]["gold"]]
        kind[e] = "tied" if (n == top and top == second) else ("major" if n == top else "minor")
maj = [e for e in dep if kind[e] == "major"]; mino = [e for e in dep if kind[e] == "minor"]; tied = [e for e in dep if kind[e] == "tied"]
print(f"dep rows: major {len(maj)} minor {len(mino)} tied {len(tied)}; minor rows span {len({rows[e]['text'] for e in mino})} texts, {len({rows[e]['genome'] for e in mino})} genomes; minor by split {dict(collections.Counter(rows[e]['split'] for e in mino))}")
# ---- probe on the pooled features; folds and classifier as in the stored genome-dependent
# Evo2 probe (evo2_probe.json)
z = np.load(f"{OUT}/evo2_raw_both.npz", allow_pickle=True); ids = z["ids"].astype(str)
idx = {e: i for i, e in enumerate(ids)}
def take(k): return np.stack([z[k][idx[e]] for e in dep])
y = np.array([rows[e]["gold"] for e in dep]); grp = np.array([rows[e]["genome"] for e in dep])
texts = sorted({rows[e]["text"] for e in dep}); tid = np.array([texts.index(rows[e]["text"]) for e in dep])
kmask = np.array([kind[e] for e in dep])
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def folds(groups, k, seed):
    u = np.array(sorted(set(groups))); r = np.random.default_rng(seed); r.shuffle(u); f = {g: i % k for i, g in enumerate(u)}
    fa = np.array([f[g] for g in groups]); return [(np.where(fa != i)[0], np.where(fa == i)[0]) for i in range(k)]
def probe(F, yy, C=1.0, seeds=range(10)):
    o, mi, tb, rowc = [], [], [], np.zeros(len(yy))
    for s in seeds:
        P = np.empty(len(yy), dtype=object); TB = np.empty(len(yy), dtype=object)
        for tr, te in folds(grp, 5, s):
            sc = StandardScaler().fit(F[tr]); clf = LogisticRegression(max_iter=5000, C=C).fit(sc.transform(F[tr]), yy[tr])
            P[te] = clf.predict(sc.transform(F[te]))
            m = {t: collections.Counter(yy[tr][tid[tr] == t]).most_common(1)[0][0] for t in set(tid[tr])}; gm = collections.Counter(yy[tr]).most_common(1)[0][0]
            TB[te] = [m.get(t, gm) for t in tid[te]]
        o.append((P == yy).mean()); mi.append((P[kmask == "minor"] == yy[kmask == "minor"]).mean()); tb.append((TB == yy).mean())
        rowc += (P == yy) / len(seeds)
    return dict(acc=float(np.mean(o)), acc_sd=float(np.std(o)), minor=float(np.mean(mi)), text_baseline=float(np.mean(tb))), rowc
def null(F, yy, C=1.0):
    out = []
    for k in range(20):
        r2 = np.random.default_rng(100 + k); ys = yy.copy()
        for t in range(len(texts)):
            mm = np.where(tid == t)[0]; ys[mm] = r2.permutation(yy[mm])
        P = np.empty(len(ys), dtype=object)
        for tr, te in folds(grp, 5, k):
            sc = StandardScaler().fit(F[tr]); P[te] = LogisticRegression(max_iter=5000, C=C).fit(sc.transform(F[tr]), ys[tr]).predict(sc.transform(F[te]))
        out.append((P == ys).mean())
    return float(np.mean(out)), float(np.std(out))
# genome-cluster bootstrap of the per-row correctness, as bootstrap_dna_probe.py
genomes = sorted(set(grp)); gi = {g: np.where(grp == g)[0] for g in genomes}
def boot(rc, n=2000, seed=0):
    r = np.random.default_rng(seed); m = []
    for _ in range(n):
        s = np.concatenate([gi[genomes[j]] for j in r.integers(0, len(genomes), len(genomes))]); m.append(rc[s].mean())
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
res = {"n_rows": len(dep), "n_genomes": len(genomes)}
for name, F in [("raw_var (unprojected Evo2 blocks.20.mlp.l3, variant block)", take("raw_var")),
                ("raw_ref||raw_var (unprojected, both blocks)", np.hstack([take("raw_ref"), take("raw_var")]))]:
    r, rc = probe(F, y); r["ci95_genome"] = boot(rc); r["null_within_text"] = null(F, y); r["dim"] = int(F.shape[1]); res[name] = r
    print(f"{name:<66} d={F.shape[1]:>5}  acc {r['acc']:.4f}+/-{r['acc_sd']:.4f}  CI {r['ci95_genome'][0]:.3f}-{r['ci95_genome'][1]:.3f}"
          f"  minor(41) {r['minor']:.3f}  text {r['text_baseline']:.4f}  null {r['null_within_text'][0]:.3f}+/-{r['null_within_text'][1]:.3f}", flush=True)
json.dump(res, open(f"{OUT}/dna_raw_probe.json", "w"), indent=1)

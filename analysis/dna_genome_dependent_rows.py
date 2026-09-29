#!/usr/bin/env python
"""Query-level analysis of BioReason disease prediction on the genome-dependent and genome-invariant
queries, for the RQ1 appendix section "Genome-dependent disease prediction subset".

Recomputes the split of the 1,449 queries into 165 genome-dependent and 1,284 genome-invariant
queries by text group, the accuracy a text-only predictor can reach, and the counts of queries
carrying the most frequent and less frequent label of their text group. For the SFT and RL
checkpoints it reports accuracy per DNA condition on each stratum, paired accuracy differences with
percentile bootstrap intervals over text groups and over genomes (5,000 draws, seed 0) and a sign-flip
p value, answer changes relative to the condition without DNA, the rate of predictions that differ
from the text group's most common prediction, and a table of the 14 genome-dependent text groups. It
ends with linear probes on the stored pooled Evo2 features of the 165 queries, with and without a
text group indicator, and a within-text-group label permutation. Writes
outputs/analysis/genome_dependent_queries.json.
"""
import os
OUT_DIR = os.path.join(os.environ.get("INPUT_USE_OUTPUT_DIR", "outputs"), "analysis")
os.makedirs(OUT_DIR, exist_ok=True)
import json, sys, os, collections, random
import numpy as np
from input_use.metrics.dna import label_of, canonical_gold, normalize_label_text
from input_use.core import paths as RD  # noqa: E402

D = RD.DNA_PERTURBATIONS
OUT = OUT_DIR
os.makedirs(OUT, exist_ok=True)
CK = ("sft", "rl")
GENOME_ARMS = ["wt", "wt_repeat", "no_dna", "no_modality", "scramble", "scramble_variant",
               "revert_variant_block", "swap_variant_donor"]

# ---------------------------------------------------------------- load
rows = {}       # eid -> meta
pred = {ck: collections.defaultdict(dict) for ck in CK}   # ck -> arm -> eid -> label/None
golds = set()
for ck in CK:
    with open(RD.dna_records(D, ck)) as fh:
        for line in fh:
            r = json.loads(line)
            eid, cond = r["example_id"], r["condition"]
            if cond == "wt" and eid not in rows:
                inp = r["input"]
                rows[eid] = dict(text=inp["question"], stratum=inp["stratum"], genome=inp["variant_key"],
                                 split=inp["split"], gold=r["ground_truth"]["answer"])
            golds.add(r["ground_truth"]["answer"])
            if cond in GENOME_ARMS:
                pred[ck][cond][eid] = r["output"]["raw"]
vocab = sorted(golds)
assert len(rows) == 1449 and len(vocab) == 37, (len(rows), len(vocab))
for ck in CK:
    for arm in GENOME_ARMS:
        assert len(pred[ck][arm]) == 1449, (ck, arm, len(pred[ck][arm]))
        pred[ck][arm] = {e: label_of(g, vocab) for e, g in pred[ck][arm].items()}

def correct(lbl, gold):
    return lbl is not None and normalize_label_text(lbl) == canonical_gold(gold)

# ---------------------------------------------------------------- strata and text groups
by_text = collections.defaultdict(list)
for e, m in rows.items():
    by_text[m["text"]].append(e)
text_id = {}
for i, (t, es) in enumerate(sorted(by_text.items(), key=lambda kv: kv[1][0])):
    for e in es:
        text_id[e] = i
dep = [e for e, m in rows.items() if m["stratum"] == "sensitivity"]
inv = [e for e, m in rows.items() if m["stratum"] != "sensitivity"]
assert len(dep) == 165 and len(inv) == 1284
# check: recompute the stratum definition from scratch
for t, es in by_text.items():
    ng = len({rows[e]["genome"] for e in es}); nl = len({rows[e]["gold"] for e in es})
    for e in es:
        assert (rows[e]["stratum"] == "sensitivity") == (ng > 1 and nl > 1), (t[:40], ng, nl)

# majority gold per text group; ties recorded
maj_label, tie = {}, {}
for t, es in by_text.items():
    c = collections.Counter(rows[e]["gold"] for e in es)
    top = c.most_common()
    maj_label[t] = top[0][0]
    tie[t] = len(top) > 1 and top[1][1] == top[0][1]
is_major = {e: rows[e]["gold"] == maj_label[rows[e]["text"]] for e in rows}

def ceiling(es):
    """Best accuracy of any predictor that is a function of the question text alone (in-sample)."""
    return sum(is_major[e] for e in es) / len(es)

def acc_raw(ck, arm, es):
    return float(np.mean([correct(pred[ck][arm][e], rows[e]["gold"]) for e in es]))

def acc_genome(ck, arm, es):
    g = collections.defaultdict(list)
    for e in es:
        g[rows[e]["genome"]].append(correct(pred[ck][arm][e], rows[e]["gold"]))
    return float(np.mean([np.mean(v) for v in g.values()]))

def acc_text(ck, arm, es):
    g = collections.defaultdict(list)
    for e in es:
        g[rows[e]["text"]].append(correct(pred[ck][arm][e], rows[e]["gold"]))
    return float(np.mean([np.mean(v) for v in g.values()]))

def boot_diff(ck, a1, a2, es, cluster="text", B=5000, seed=0):
    """Percentile bootstrap over clusters of acc(a1)-acc(a2), row-weighted, plus a sign-flip p."""
    rng = np.random.default_rng(seed)
    key = (lambda e: rows[e]["text"]) if cluster == "text" else (lambda e: rows[e]["genome"])
    cl = collections.defaultdict(lambda: [0.0, 0.0, 0])
    for e in es:
        k = key(e); c = cl[k]
        c[0] += correct(pred[ck][a1][e], rows[e]["gold"]); c[1] += correct(pred[ck][a2][e], rows[e]["gold"]); c[2] += 1
    arr = np.array([[v[0], v[1], v[2]] for v in cl.values()], dtype=float)
    obs = (arr[:, 0].sum() - arr[:, 1].sum()) / arr[:, 2].sum()
    n = len(arr)
    idx = rng.integers(0, n, size=(B, n))
    s = arr[idx]                                   # B x n x 3
    diffs = (s[:, :, 0].sum(1) - s[:, :, 1].sum(1)) / s[:, :, 2].sum(1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # sign-flip permutation on cluster-level differences (two-sided)
    d = arr[:, 0] - arr[:, 1]
    signs = rng.choice([-1, 1], size=(20000, n))
    perm = (signs * d).sum(1) / arr[:, 2].sum()
    p = float(np.mean(np.abs(perm) >= abs(obs) - 1e-12))
    return dict(diff=obs, lo=lo, hi=hi, p=p, n_clusters=n)

def changed_vs(ck, arm, ref, es):
    """Rows whose mapped label under `arm` differs from under `ref`; None counts as its own label."""
    ch = [e for e in es if pred[ck][arm][e] != pred[ck][ref][e]]
    both_mapped = [e for e in es if pred[ck][arm][e] is not None and pred[ck][ref][e] is not None]
    ch_m = [e for e in both_mapped if pred[ck][arm][e] != pred[ck][ref][e]]
    out = dict(n=len(es), n_changed=len(ch), rate=len(ch) / len(es),
               n_both_mapped=len(both_mapped), n_changed_mapped=len(ch_m),
               rate_mapped=len(ch_m) / len(both_mapped) if both_mapped else None)
    if ch:
        out["arm_correct_among_changed"] = float(np.mean([correct(pred[ck][arm][e], rows[e]["gold"]) for e in ch]))
        out["ref_correct_among_changed"] = float(np.mean([correct(pred[ck][ref][e], rows[e]["gold"]) for e in ch]))
        out["gained"] = sum(correct(pred[ck][arm][e], rows[e]["gold"]) and not correct(pred[ck][ref][e], rows[e]["gold"]) for e in ch)
        out["lost"] = sum((not correct(pred[ck][arm][e], rows[e]["gold"])) and correct(pred[ck][ref][e], rows[e]["gold"]) for e in ch)
    return out

def nonmodal(ck, arm, es):
    """Within each text group, rows whose prediction differs from the group's modal prediction."""
    g = collections.defaultdict(list)
    for e in es:
        g[rows[e]["text"]].append(e)
    nm = 0; texts_with_var = 0; n_distinct = []
    for t, ee in g.items():
        c = collections.Counter(pred[ck][arm][e] for e in ee)
        nm += len(ee) - c.most_common(1)[0][1]
        texts_with_var += len(c) > 1
        n_distinct.append(len(c))
    return dict(n_rows=len(es), n_texts=len(g), nonmodal_rows=nm, nonmodal_rate=nm / len(es),
                texts_with_more_than_one_prediction=texts_with_var, mean_distinct_preds=float(np.mean(n_distinct)))

# ---------------------------------------------------------------- report
res = {"n_rows": 1449, "n_dep": len(dep), "n_inv": len(inv),
       "n_dep_texts": len({rows[e]["text"] for e in dep}), "n_dep_genomes": len({rows[e]["genome"] for e in dep}),
       "n_inv_texts": len({rows[e]["text"] for e in inv}), "n_inv_genomes": len({rows[e]["genome"] for e in inv}),
       "ceiling_dep": ceiling(dep), "ceiling_inv": ceiling(inv),
       "n_dep_major": sum(is_major[e] for e in dep), "n_dep_minor": sum(not is_major[e] for e in dep),
       "dep_texts_with_tie": sum(tie[t] for t in {rows[e]["text"] for e in dep})}
print(json.dumps(res, indent=1))
dep_maj = [e for e in dep if is_major[e]]; dep_min = [e for e in dep if not is_major[e]]

for ck in CK:
    print(f"\n================ {ck.upper()} ================")
    print(f"{'arm':22s} | {'dep raw':>8s} {'dep/gen':>8s} {'dep/txt':>8s} | {'inv raw':>8s} {'inv/gen':>8s} {'inv/txt':>8s} | {'dep maj':>8s} {'dep min':>8s} | {'all/gen':>8s}")
    res[ck] = {"acc": {}}
    for arm in GENOME_ARMS:
        a = dict(dep_raw=acc_raw(ck, arm, dep), dep_genome=acc_genome(ck, arm, dep), dep_text=acc_text(ck, arm, dep),
                 inv_raw=acc_raw(ck, arm, inv), inv_genome=acc_genome(ck, arm, inv), inv_text=acc_text(ck, arm, inv),
                 dep_major=acc_raw(ck, arm, dep_maj), dep_minor=acc_raw(ck, arm, dep_min),
                 all_genome=acc_genome(ck, arm, list(rows)), all_raw=acc_raw(ck, arm, list(rows)))
        res[ck]["acc"][arm] = a
        print(f"{arm:22s} | {a['dep_raw']:8.3f} {a['dep_genome']:8.3f} {a['dep_text']:8.3f} | {a['inv_raw']:8.3f} {a['inv_genome']:8.3f} {a['inv_text']:8.3f} | {a['dep_major']:8.3f} {a['dep_minor']:8.3f} | {a['all_genome']:8.3f}")
    print(f"{'text-only ceiling':22s} | {ceiling(dep):8.3f} {'':8s} {'':8s} | {ceiling(inv):8.3f} {'':8s} {'':8s} | {1.0:8.3f} {0.0:8.3f} |")

    print("\n-- paired accuracy differences (row-weighted; bootstrap + sign-flip over text groups)")
    res[ck]["diff"] = {}
    for a1, a2 in [("wt", "no_dna"), ("wt", "scramble"), ("scramble", "no_dna"), ("wt", "no_modality"), ("wt", "revert_variant_block")]:
        for nm, es in [("dep", dep), ("inv", inv), ("dep_minor", dep_min)]:
            b = boot_diff(ck, a1, a2, es)
            bg = boot_diff(ck, a1, a2, es, cluster="genome")
            res[ck]["diff"][f"{a1}-{a2}:{nm}"] = {"text": b, "genome": bg}
            print(f"  {a1:>9s} - {a2:9s} {nm:9s}: {b['diff']:+.3f} [{b['lo']:+.3f},{b['hi']:+.3f}] p={b['p']:.3f} (texts={b['n_clusters']}) | genome-clustered [{bg['lo']:+.3f},{bg['hi']:+.3f}] p={bg['p']:.3f} (genomes={bg['n_clusters']})")

    print("\n-- rows whose answer differs from the genome-removed (no_dna) answer")
    res[ck]["changed"] = {}
    for arm in ["wt", "scramble", "scramble_variant", "no_modality", "revert_variant_block", "swap_variant_donor", "wt_repeat"]:
        for nm, es in [("dep", dep), ("inv", inv), ("dep_minor", dep_min), ("dep_major", dep_maj)]:
            c = changed_vs(ck, arm, "no_dna", es)
            res[ck]["changed"][f"{arm}:{nm}"] = c
            extra = f" arm-correct {c['arm_correct_among_changed']:.3f} ref-correct {c['ref_correct_among_changed']:.3f} gained {c['gained']} lost {c['lost']}" if c["n_changed"] else ""
            print(f"  {arm:22s} {nm:9s}: {c['n_changed']:4d}/{c['n']:4d} = {c['rate']:.3f}{extra}")
    print("  (wt vs wt_repeat, decoding noise):", {nm: changed_vs(ck, "wt_repeat", "wt", es)["rate"] for nm, es in [("dep", dep), ("inv", inv)]})

    print("\n-- within-text-group non-modal prediction rate")
    res[ck]["nonmodal"] = {}
    for arm in ["wt", "scramble", "no_dna", "no_modality", "wt_repeat"]:
        for nm, es in [("dep", dep), ("inv", inv)]:
            v = nonmodal(ck, arm, es)
            res[ck]["nonmodal"][f"{arm}:{nm}"] = v
            print(f"  {arm:12s} {nm:4s}: nonmodal {v['nonmodal_rows']:4d}/{v['n_rows']:4d} = {v['nonmodal_rate']:.3f}; texts with >1 prediction {v['texts_with_more_than_one_prediction']}/{v['n_texts']}; mean distinct preds {v['mean_distinct_preds']:.2f}")

# ---------------------------------------------------------------- per-text table for the 14
# dependent groups
import re
STEM = re.compile(r"effect of this (\S+) allele")
print("\n================ 14 genome-dependent text groups ================")
tab = []
for t in sorted({rows[e]["text"] for e in dep}, key=lambda t: (STEM.search(t).group(1) if STEM.search(t) else "", t)):
    es = by_text[t]
    gene = STEM.search(t).group(1) if STEM.search(t) else "?"
    c = collections.Counter(rows[e]["gold"] for e in es)
    row = dict(gene=gene, n_rows=len(es), n_genomes=len({rows[e]["genome"] for e in es}),
               splits=dict(collections.Counter(rows[e]["split"] for e in es)),
               golds=dict(c), ceiling=c.most_common(1)[0][1] / len(es), tie=tie[t])
    for ck in CK:
        for arm in ["wt", "scramble", "no_dna"]:
            row[f"{ck}_{arm}"] = acc_raw(ck, arm, es)
        row[f"{ck}_wt_npred"] = len({pred[ck]["wt"][e] for e in es})
        row[f"{ck}_wt_preds"] = dict(collections.Counter(pred[ck]["wt"][e] for e in es))
        row[f"{ck}_scr_npred"] = len({pred[ck]["scramble"][e] for e in es})
        row[f"{ck}_nodna_pred"] = dict(collections.Counter(pred[ck]["no_dna"][e] for e in es))
    tab.append(row)
    print(f"{gene:6s} n={row['n_rows']:3d} gen={row['n_genomes']:3d} ceil={row['ceiling']:.2f} golds={row['golds']}")
    for ck in CK:
        print(f"      {ck}: wt {row[ck+'_wt']:.2f} scr {row[ck+'_scramble']:.2f} nodna {row[ck+'_no_dna']:.2f} | wt preds {row[ck+'_wt_preds']} | nodna {row[ck+'_nodna_pred']}")
res["dep_groups"] = tab

# ---------------------------------------------------------------- row-level Evo2 probe on the 165
# rows
print("\n================ row-level Evo2 probe, 165 genome-dependent rows ================")
z = np.load(f"{D}/evo2_pooled.npz", allow_pickle=True)
X_all = z["X"]; ids = z["ids"].astype(str)
print("npz:", X_all.shape, "ids", len(ids), "extra keys", [k for k in z.files if k not in ("X", "ids")])
idx = {e: i for i, e in enumerate(ids)}
dep_sorted = sorted(dep)
Xe = np.stack([X_all[idx[e]] for e in dep_sorted])
y = np.array([rows[e]["gold"] for e in dep_sorted])
grp_genome = np.array([rows[e]["genome"] for e in dep_sorted])
tid = np.array([text_id[e] for e in dep_sorted])
T = np.zeros((len(dep_sorted), tid.max() + 1)); T[np.arange(len(tid)), tid] = 1.0
T = T[:, T.sum(0) > 0]
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def grouped_folds(groups, k, seed):
    u = np.array(sorted(set(groups))); rng = np.random.default_rng(seed); rng.shuffle(u)
    fold_of = {g: i % k for i, g in enumerate(u)}
    f = np.array([fold_of[g] for g in groups])
    return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(k)]

def run_probe(features, y, groups, tid, seeds=range(10), k=5, C=1.0):
    accs, text_only = [], []
    for s in seeds:
        hit = tot = hit_t = 0
        for tr, te in grouped_folds(groups, k, s):
            # text-only baseline: majority gold of the text group among train rows (fallback:
            # overall train majority)
            maj = {}
            for t in set(tid[tr]):
                maj[t] = collections.Counter(y[tr][tid[tr] == t]).most_common(1)[0][0]
            gm = collections.Counter(y[tr]).most_common(1)[0][0]
            hit_t += sum(maj.get(t, gm) == yy for t, yy in zip(tid[te], y[te]))
            if features is not None:
                sc = StandardScaler().fit(features[tr])
                clf = LogisticRegression(max_iter=5000, C=C).fit(sc.transform(features[tr]), y[tr])
                hit += (clf.predict(sc.transform(features[te])) == y[te]).sum()
            tot += len(te)
        accs.append(hit / tot); text_only.append(hit_t / tot)
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(text_only)), float(np.std(text_only))

probe = {}
for name, F in [("text onehot only (LR)", T), ("evo2 only", Xe), ("text onehot + evo2", np.hstack([T * 10.0, Xe]))]:
    for C in [0.01, 0.1, 1.0]:
        m, sd, tm, tsd = run_probe(F, y, grp_genome, tid, C=C)
        probe[f"{name} C={C}"] = dict(acc=m, sd=sd, text_majority=tm, text_majority_sd=tsd)
        print(f"  {name:26s} C={C:<5}: acc {m:.3f} +/- {sd:.3f}   (train-fold text-majority baseline {tm:.3f} +/- {tsd:.3f})")
# label-shuffle control: permute labels within text group (keeps each group's label composition)
rng = np.random.default_rng(0); y_s = y.copy()
for t in set(tid):
    m = np.where(tid == t)[0]; y_s[m] = rng.permutation(y[m])
m, sd, tm, tsd = run_probe(np.hstack([T * 10.0, Xe]), y_s, grp_genome, tid, C=0.1)
probe["shuffled-within-text text+evo2 C=0.1"] = dict(acc=m, sd=sd, text_majority=tm)
print(f"  {'shuffled-within-text control':26s} C=0.1  : acc {m:.3f} +/- {sd:.3f}   (text-majority {tm:.3f})")
res["probe_165"] = probe

# minority-row recall of the probe: does the embedding recover the rows a text lookup must get
# wrong?
def minority_recall(features, C=0.1, seeds=range(10), k=5):
    rec = []
    for s in seeds:
        h = n = 0
        for tr, te in grouped_folds(grp_genome, k, s):
            sc = StandardScaler().fit(features[tr])
            clf = LogisticRegression(max_iter=5000, C=C).fit(sc.transform(features[tr]), y[tr])
            p = clf.predict(sc.transform(features[te]))
            mn = np.array([not is_major[dep_sorted[i]] for i in te])
            h += (p[mn] == y[te][mn]).sum(); n += mn.sum()
        rec.append(h / n)
    return float(np.mean(rec)), float(np.std(rec))
mr = minority_recall(np.hstack([T * 10.0, Xe]))
print(f"  probe accuracy on the {sum(not is_major[e] for e in dep)} minority rows (text lookup = 0): {mr[0]:.3f} +/- {mr[1]:.3f}")
res["probe_165"]["minority_row_acc_text+evo2_C0.1"] = mr

with open(f"{OUT}/genome_dependent_queries.json", "w") as fh:
    json.dump(res, fh, indent=1, default=float)
print("\nwrote", f"{OUT}/genome_dependent_queries.json")

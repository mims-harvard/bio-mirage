#!/usr/bin/env python
"""Scores the generations of the three auxiliary supervision conditions (257 bp with the auxiliary
target, 2,048 bp with and without it) for training seeds 23, 24 and 25, for Figure 7 and the RQ4
appendix tables.

Per seed it reports disease prediction accuracy, accuracy on the edited base pair (the first
"Edit at" line of the generation), the share of evidence conflict generations that state the donor's
base pair, and accuracy on the edited base pair split by whether the query's genome occurs in
training. Each value has a 95% percentile bootstrap interval over the 145 held-out queries (2,000
draws, seed 20260916). Across seeds it reports the mean, standard deviation and range. Reads the
evaluate_generation.py outputs in dna/bioreason/auxiliary_supervision/generations under
INPUT_USE_RESULTS_DIR, named <condition>_seed<seed>.json, and writes
outputs/analysis/score_seeds.{json,txt}.
"""
import collections
import csv
import json
import os
import re

import numpy as np

from input_use.core import paths as RD

OUT_DIR = os.path.join(os.environ.get("INPUT_USE_OUTPUT_DIR", "outputs"), "analysis")
os.makedirs(OUT_DIR, exist_ok=True)

csv.field_size_limit(10 ** 9)
FG = RD.AUX_GENERATIONS
N = 145
IDX = np.random.default_rng(20260916).integers(0, N, size=(2000, N))
PAT = re.compile(r"Edit at (\d+): ref ([ACGT]) -> var ([ACGT])")
# generation file stem per condition
GEN = {"aux257": "aux_257bp", "aux2048c": "aux_2048bp", "noaux2048c": "no_aux_2048bp"}
ARMS = ["aux257", "aux2048c", "noaux2048c"]
SEEDS = [23, 24, 25]


def norm(t):
    if not t:
        return ""
    t = t.strip().lower().replace("\n", " ")
    t = re.sub(r"[^\w\s\-'/]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def first_diff(a, b):
    return next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), None)


def boot(x):
    x = np.asarray(x, float)
    m = x[IDX].mean(1)
    return dict(k=int(x.sum()), v=round(float(x.mean()), 4),
                lo=round(float(np.percentile(m, 2.5)), 4), hi=round(float(np.percentile(m, 97.5)), 4))


def score(path, arm):
    d = json.load(open(path))
    if "summary" not in d:
        return dict(status=f"partial, {len(d.get('rows', []))} queries")
    R, s = d["rows"], d["summary"]
    assert len(R) == N, (path, len(R))
    out = dict(status="complete", window_bp=len(R[0]["ref_wt"]),
               determinism=s["generation_identical_wt_vs_same"],
               unmapped={c: s[f"unmapped_{c}"] for c in ("wt", "shuffle", "swap")})
    dis = {c: np.array([norm(r[f"pred_{c}"]) == norm(r["truth"]) for r in R]) for c in ("wt", "shuffle", "swap")}
    for c in dis:
        out[f"disease_{c}"] = boot(dis[c])
    out["disease_wt_minus_shuffle"] = boot(dis["wt"].astype(float) - dis["shuffle"])
    out["disease_answers_changed_shuffle"] = int(sum(norm(r["pred_shuffle"]) != norm(r["pred_wt"]) for r in R))
    if arm.startswith("noaux"):
        return out
    tp = []
    for r in R:
        j = first_diff(r["ref_wt"], r["var_wt"])
        tp.append(r["ref_wt"][j] + ">" + r["var_wt"][j])
    dp = []
    for r in R:
        j = first_diff(r["ref_swap"], r["var_swap"])
        dp.append(r["ref_swap"][j] + ">" + r["var_swap"][j])
    m = {c: [PAT.search(r[f"gen_{c}"] or "") for r in R] for c in ("wt", "shuffle", "swap")}
    em = {c: [(x.group(2) + ">" + x.group(3)) if x else None for x in m[c]] for c in m}
    hit = {c: np.array([e == t for e, t in zip(em[c], tp)]) for c in em}
    for c in hit:
        out[f"pair_{c}"] = boot(hit[c])
    out["pair_wt_minus_shuffle"] = boot(hit["wt"].astype(float) - hit["shuffle"])
    out["pair_unparseable"] = {c: int(sum(x is None for x in m[c])) for c in m}
    out["written_offsets_wt"] = collections.Counter(int(x.group(1)) for x in m["wt"] if x).most_common(2)
    don = np.array([e == p for e, p in zip(em["swap"], dp)])
    out["swap_donor_matching"] = boot(don)
    out["swap_recipient_matching"] = int(hit["swap"].sum())
    out["swap_chance_donor"] = round(sum(sum(1 for x in dp if x == q) * sum(1 for x in em["swap"] if x == q)
                                         for q in set(dp)) / N, 2)
    train_genomes = {(r["reference_sequence"], r["variant_sequence"])
                     for r in csv.DictReader(open(f"{RD.AUX_TARGET_257BP}/train.csv"))}
    seen = np.array([(r["ref_wt"][first_diff(r["ref_wt"], r["var_wt"]) - 128:
                                  first_diff(r["ref_wt"], r["var_wt"]) + 129],
                      r["var_wt"][first_diff(r["ref_wt"], r["var_wt"]) - 128:
                                  first_diff(r["ref_wt"], r["var_wt"]) + 129]) in train_genomes for r in R])
    out["pair_wt_genome_in_train"] = [int(hit["wt"][seen].sum()), int(seen.sum())]
    out["pair_wt_genome_not_in_train"] = [int(hit["wt"][~seen].sum()), int((~seen).sum())]
    return out


res = {}
for arm in ARMS:
    for seed in SEEDS:
        fn = f"{GEN[arm]}_seed{seed}.json"
        p = os.path.join(FG, fn)
        res[f"{arm}/seed{seed}"] = dict(file=fn, **(score(p, arm) if os.path.exists(p) else {"status": "missing"}))

lines = []
keys = ["disease_wt", "disease_shuffle", "disease_wt_minus_shuffle", "pair_wt", "pair_shuffle",
        "pair_wt_minus_shuffle", "swap_donor_matching"]
for arm in ARMS:
    lines.append(f"== {arm}")
    for seed in SEEDS:
        r = res[f"{arm}/seed{seed}"]
        if r["status"] != "complete":
            lines.append(f"  seed{seed}: {r['status']}")
            continue
        lines.append(f"  seed{seed}: " + "  ".join(
            f"{k}={r[k]['v']:.3f} ({r[k]['k']}/145)" if k in r else f"{k}=--" for k in keys))
    done = [res[f"{arm}/seed{s}"] for s in SEEDS if res[f"{arm}/seed{s}"]["status"] == "complete"]
    if len(done) > 1:
        for k in keys:
            if k not in done[0]:
                continue
            v = [d[k]["v"] for d in done]
            lines.append(f"  MEAN over {len(v)} seeds {k}: {np.mean(v):.4f}  sd {np.std(v, ddof=1):.4f}  "
                         f"range [{min(v):.4f}, {max(v):.4f}]")
json.dump(res, open(os.path.join(OUT_DIR, "score_seeds.json"), "w"), indent=1)
open(os.path.join(OUT_DIR, "score_seeds.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

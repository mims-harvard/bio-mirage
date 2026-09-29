#!/usr/bin/env python
"""Selects the Prot2Text-V2 protein pairs for evidence conflicts and writes their generation requests.

A pair is two test proteins from the same taxon whose function texts fall in the lowest quartile of
BioBERT BERTScore similarity, whose names differ, and whose names are each closer to their own
function than to the other's. Each protein enters at most one pair, up to 700 pairs. Each pair
yields two requests, each with the sequence of one protein and the name of the other. Writes
results/prot2text_pairs.parquet, results/prot2text_pair_selection.json and runs/requests_B.jsonl
under protein/prot2text_v2 of $INPUT_USE_RESULTS_DIR.

    python input_use/models/prot2text_driver/pairs.py [--max-pairs 700] [--quantile 0.25] [--seed 0]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pairs", type=int, default=700)
    ap.add_argument("--quantile", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    ex = pd.read_parquet(os.path.join(C.RESULTS, "prot2text_examples.parquet"))
    el = ex[ex.within_1021 & ~ex.full_name_missing & ~ex.taxon_missing & ~ex.function_missing].reset_index(drop=True)
    by_id = el.set_index("protein_id")
    cands = []
    for taxon, g in el.groupby("taxon"):
        ids = g["protein_id"].tolist()
        for i, j in itertools.combinations(ids, 2):
            cands.append((i, j, taxon))
    print(f"eligible {len(el)} of {len(ex)}; taxa with >=2: {el.groupby('taxon').size().ge(2).sum()}; "
          f"candidate pairs {len(cands)}", flush=True)

    from scoring import Scorers
    sc = Scorers(device="cuda", which=("biobert",))
    fA = [by_id.loc[i, "function_text"] for i, _, _ in cands]
    fB = [by_id.loc[j, "function_text"] for _, j, _ in cands]
    _, _, sim = sc.bertscore("biobert", fA, fB)
    thr = float(np.quantile(sim, a.quantile))
    df = pd.DataFrame({"protein_A": [c[0] for c in cands], "protein_B": [c[1] for c in cands],
                       "shared_taxon": [c[2] for c in cands], "function_similarity_A_B": sim})
    df["full_name_A"] = df.protein_A.map(by_id["full_name"])
    df["full_name_B"] = df.protein_B.map(by_id["full_name"])
    df["names_differ"] = df.full_name_A.str.strip().str.lower() != df.full_name_B.str.strip().str.lower()
    df["below_quantile"] = df.function_similarity_A_B <= thr
    surv = df[df.below_quantile & df.names_differ].copy()
    print(f"similarity threshold (q={a.quantile}) = {thr:.4f}; below: {int(df.below_quantile.sum())}; "
          f"also names differ: {len(surv)}", flush=True)

    # text-evidence validation check: own-function similarity of the Full Name beats the other
    # function
    nA, nB = surv.full_name_A.tolist(), surv.full_name_B.tolist()
    FA = surv.protein_A.map(by_id["function_text"]).tolist()
    FB = surv.protein_B.map(by_id["function_text"]).tolist()
    _, _, aa = sc.bertscore("biobert", nA, FA)
    _, _, ab = sc.bertscore("biobert", nA, FB)
    _, _, bb = sc.bertscore("biobert", nB, FB)
    _, _, ba = sc.bertscore("biobert", nB, FA)
    surv["nameA_sim_own"], surv["nameA_sim_other"] = aa, ab
    surv["nameB_sim_own"], surv["nameB_sim_other"] = bb, ba
    surv["nameA_own_margin"] = aa - ab
    surv["nameB_own_margin"] = bb - ba
    surv["name_gate_pass"] = (surv.nameA_own_margin > 0) & (surv.nameB_own_margin > 0)
    ok = surv[surv.name_gate_pass]
    print(f"name gate: {len(ok)} of {len(surv)}", flush=True)

    rng = random.Random(a.seed)
    order = ok.index.tolist()
    rng.shuffle(order)
    used, chosen = set(), []
    for idx in order:
        r = ok.loc[idx]
        if r.protein_A in used or r.protein_B in used:
            continue
        used.update([r.protein_A, r.protein_B])
        chosen.append(idx)
        if len(chosen) >= a.max_pairs:
            break
    pairs = ok.loc[chosen].reset_index(drop=True)
    pairs.insert(0, "pair_id", [f"P{k:04d}" for k in range(len(pairs))])
    for col in ("sequence_gate_pass", "aligned_gate_pass"):
        pairs[col] = pd.NA
    assert pairs.protein_A.is_unique and pairs.protein_B.is_unique
    assert not set(pairs.protein_A) & set(pairs.protein_B)
    assert (pairs.protein_A.map(by_id["taxon"]) == pairs.protein_B.map(by_id["taxon"])).all()
    assert (pairs.protein_A != pairs.protein_B).all() and pairs.names_differ.all()
    pairs.to_parquet(os.path.join(C.RESULTS, "prot2text_pairs.parquet"), index=False)

    reqs = []
    for r in pairs.itertuples(index=False):
        sA, sB = by_id.loc[r.protein_A, "sequence"], by_id.loc[r.protein_B, "sequence"]
        assert len(sA) <= C.OFFICIAL_MAX_SEQ and len(sB) <= C.OFFICIAL_MAX_SEQ
        reqs.append({"request_id": C.request_id("Bseq_Aname", -1, r.pair_id), "pair_id": r.pair_id,
                     "protein_id": r.protein_B, "condition": "Bseq_Aname", "shuffle_seed": -1,
                     "name_field": r.full_name_A, "taxon_field": r.shared_taxon, "sequence": sB,
                     "sequence_hash": C.seq_hash(sB), "sequence_length": len(sB),
                     "sequence_source": r.protein_B, "name_source": r.protein_A})
        reqs.append({"request_id": C.request_id("Aseq_Bname", -1, r.pair_id), "pair_id": r.pair_id,
                     "protein_id": r.protein_A, "condition": "Aseq_Bname", "shuffle_seed": -1,
                     "name_field": r.full_name_B, "taxon_field": r.shared_taxon, "sequence": sA,
                     "sequence_hash": C.seq_hash(sA), "sequence_length": len(sA),
                     "sequence_source": r.protein_A, "name_source": r.protein_B})
    p = os.path.join(C.RUNS, "requests_B.jsonl")
    with open(p + ".tmp", "w") as fh:
        for r in reqs:
            fh.write(json.dumps(r) + "\n")
    os.replace(p + ".tmp", p)

    sel = {"n_test": int(len(ex)), "n_eligible": int(len(el)), "n_taxa_with_two_or_more": int(el.groupby("taxon").size().ge(2).sum()),
           "n_candidate_pairs": int(len(cands)), "similarity_metric": "BioBERT-large BERTScore F1 (benchmark.py settings)",
           "similarity_quantile": a.quantile, "similarity_threshold": thr,
           "candidate_similarity_quantiles": {str(q): float(np.quantile(sim, q)) for q in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9)},
           "n_below_threshold": int(df.below_quantile.sum()), "n_below_and_names_differ": int(len(surv)),
           "n_name_gate_pass": int(len(ok)), "n_pairs_proposed": int(len(pairs)), "max_pairs": a.max_pairs, "seed": a.seed,
           "n_taxa_in_pairs": int(pairs.shared_taxon.nunique()),
           "pairs_per_taxon_top": pairs.shared_taxon.value_counts().head(15).to_dict(),
           "proposed_pair_similarity": {"mean": float(pairs.function_similarity_A_B.mean()),
                                        "min": float(pairs.function_similarity_A_B.min()),
                                        "max": float(pairs.function_similarity_A_B.max())},
           "n_requests_B": len(reqs)}
    C.write_json(sel, os.path.join(C.RESULTS, "prot2text_pair_selection.json"))
    print(json.dumps(sel, indent=1), flush=True)


if __name__ == "__main__":
    main()

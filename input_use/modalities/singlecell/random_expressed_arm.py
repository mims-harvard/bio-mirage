#!/usr/bin/env python
r"""Adds the C2S-Scale condition `random_expressed`: a gene sentence of K genes drawn uniformly from
all of the cell's expressed genes, listed in the cell's own expression order.

K and the cells come from an existing run of prepare.py, whose intact sentences are regenerated from
--h5ad and checked. This is the "resampled" gene sentence of Figure 2e. Writes examples.jsonl and
meta.json to --outdir.

    python -m input_use.modalities.singlecell.random_expressed_arm \
        --dataset lung --source_run <run> --h5ad <data>/lung.h5ad --outdir <new_run>
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics as st

import anndata as ad

COND = "random_expressed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--source_run", required=True, help="the 23-arm rundir with examples.jsonl")
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from cell2sentence.utils import generate_vocabulary, generate_sentences

    wt = {}
    with open(os.path.join(a.source_run, "examples.jsonl")) as fh:
        for line in fh:
            e = json.loads(line)
            if e["condition"] == "wt":
                wt[e["example_id"]] = e
    meta = json.load(open(os.path.join(a.source_run, "meta.json")))
    K = int(meta["n_genes"])
    assert meta["dataset"] == os.path.basename(a.h5ad), (meta["dataset"], a.h5ad)

    adata = ad.read_h5ad(a.h5ad)
    vocab = generate_vocabulary(adata)
    sents = generate_sentences(adata, vocab)                # same call, same seed as prepare.py

    made, n_expr, overlap = [], [], []
    for eid, e in wt.items():
        i = int(eid[len("cell"):])
        genes = sents[i].split(" ")
        stored = e["payload"]["cell_sentence"].split(" ")
        assert genes[:K] == stored, f"{a.dataset} {eid}: regenerated sentence differs from wt"
        k = min(K, len(genes))
        rng = random.Random(f"{COND}:{a.seed}:{a.dataset}:{i}")   # its own stream, not prepare's
        pick = sorted(rng.sample(range(len(genes)), k))           # uniform over expressed genes,
        sent = " ".join(genes[j] for j in pick)                   # kept in expression order
        n = json.loads(json.dumps(e))
        n["condition"] = n["intervention"] = COND
        n["payload"]["cell_sentence"] = sent
        n["payload"]["num_genes"] = k
        n["payload"]["n_expressed"] = len(genes)
        n["payload"]["source_arm"] = "wt"
        assert len(set(sent.split(" "))) == k
        if len(genes) > K:
            assert sent != e["payload"]["cell_sentence"]
        made.append(n)
        n_expr.append(len(genes))
        overlap.append(sum(j < K for j in pick) / k)

    os.makedirs(a.outdir, exist_ok=True)
    with open(os.path.join(a.outdir, "examples.jsonl"), "w") as fh:
        for e in made:
            fh.write(json.dumps(e) + "\n")
    out_meta = {**meta, "conditions": [COND], "n_examples": len(made),
                "source_run": os.path.abspath(a.source_run), "arm_seed": a.seed,
                "n_expressed": {"min": min(n_expr), "median": int(st.median(n_expr)),
                                "max": max(n_expr), "n_cells_le_K": sum(x <= K for x in n_expr)},
                "wt_window_overlap_mean": round(st.mean(overlap), 4),
                "note": f"{COND}: {K} genes drawn uniformly from all of the cell's expressed genes, "
                        f"kept in expression order; options/ground truth copied from the wt example"}
    json.dump(out_meta, open(os.path.join(a.outdir, "meta.json"), "w"), indent=2)
    print(f"[{COND}] {a.dataset}: {len(made)} examples -> {a.outdir}; n_expressed "
          f"{out_meta['n_expressed']}; mean share of the wt window {st.mean(overlap):.3f}")


if __name__ == "__main__":
    main()

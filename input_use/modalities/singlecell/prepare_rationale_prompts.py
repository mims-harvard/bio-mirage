#!/usr/bin/env python
r"""Builds C2S-Scale prompts without an option list, in a free answer format and in the rationale
format of the RQ3 appendix ("Rationale generation").

Reuses the cell sentences of eight conditions (intact, `no_modality`, removal of 50% or 100% of the
strongest DEGs or of non-DEGs with similar expression, gene order shuffling and gene set resampling)
from an existing multiple-choice run and samples --n_cells cells that have all eight. Refuses a
source run whose DEG thresholds differ from log2 fold change >= 1.0 and adjusted p < 0.05. Writes
examples_free.jsonl, examples_rationale.jsonl and meta.json to --outdir.

    python -m input_use.modalities.singlecell.prepare_rationale_prompts \
        --mcq_examples <results>/single_cell/c2s_scale/deg_removal/<atlas>/examples.jsonl \
        --outdir <results>/single_cell/c2s_scale/rationales/<atlas>
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

# The 8 that carry the grounding gap; the intermediate DEG doses add cost without adding contrast.
CANONICAL_DEG = {"deg_min_log2fc": 1.0, "deg_max_padj": 0.05}

CONDITIONS = ["wt", "no_modality",
              "top_deg_dropout_p50", "matched_nondeg_dropout_p50",
              "top_deg_dropout_p100", "matched_nondeg_dropout_p100",
              "scramble_rank", "scramble_geneset"]

STEM = ("The following is a list of {k} gene names ordered by descending expression level in a "
        "{organism} cell. Your task is to give the cell type which this cell belongs to based on "
        "its gene expression.\nCell sentence: {genes}.\n")

FORMATS = {
    "free": STEM + "The cell type corresponding to these genes is:",
    "rationale": STEM + ("Answer using exactly this format:\nCell type: <cell type>\n"
                         "Rationale: <the genes that support the call>\n\nCell type:"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mcq_examples", required=True,
                    help="Experiment 1 examples.jsonl; its perturbed cell sentences are reused "
                         "verbatim so the format arms are exactly comparable to the MCQ arm")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n_cells", type=int, default=400)
    ap.add_argument("--organism", default="Homo sapiens")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import numpy as np
    from input_use.core.records import Example, write_jsonl, provenance

    os.makedirs(a.outdir, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    src_meta_path = os.path.join(os.path.dirname(a.mcq_examples), "meta.json")
    if not os.path.exists(src_meta_path):
        raise SystemExit(f"missing {src_meta_path}: cannot verify the DEG definition behind "
                         f"`salient`, and inheriting it unverified is how a paper ends up reporting "
                         f"one cutoff and running another")
    src_meta = json.load(open(src_meta_path))
    deg_cfg = {k: src_meta.get(k) for k in CANONICAL_DEG}
    if deg_cfg != CANONICAL_DEG:
        raise SystemExit(f"source DEG cutoffs {deg_cfg} != canonical {CANONICAL_DEG}; refusing to "
                         f"build a question set whose DEG definition differs from the rest of the "
                         f"single-cell experiments")
    print(f"[fl] inherited DEG definition: log2FC >= {deg_cfg['deg_min_log2fc']} AND "
          f"pvals_adj < {deg_cfg['deg_max_padj']} ({src_meta.get('importance_source')})")

    by_cell = defaultdict(dict)
    gt, n_src = {}, 0
    for line in open(a.mcq_examples):
        r = json.loads(line)
        n_src += 1
        if r["condition"] in CONDITIONS:
            by_cell[r["example_id"]][r["condition"]] = r
            gt[r["example_id"]] = r["ground_truth"]

    # only cells that carry every condition, so the ladder is balanced and paired
    complete = sorted(c for c, d in by_cell.items() if len(d) == len(CONDITIONS))
    dropped = len(by_cell) - len(complete)
    if dropped:
        print(f"[fl] {dropped} cells dropped: missing >=1 condition (scramble_geneset omits cells "
              f"with <1000 expressed genes)")
    take = sorted(rng.choice(complete, size=min(a.n_cells, len(complete)), replace=False).tolist())
    print(f"[fl] {n_src} source examples -> {len(complete)} complete cells -> sampling {len(take)}")

    # Per-cell-type DEG set, free: `salient` in Experiment 1 is that cell's type's DEG set, so the
    # union over cells of a type recovers it without recomputing markers or touching the h5ad.
    deg_by_type = defaultdict(set)
    for c, d in by_cell.items():
        t = (gt.get(c) or {}).get("cell_type")
        if t:
            deg_by_type[t] |= set(d[CONDITIONS[0]]["payload"].get("salient") or [])

    examples, stats = [], defaultdict(int)
    for cell in take:
        for cond in CONDITIONS:
            src = by_cell[cell][cond]
            p = src["payload"]
            for fmt, tmpl in FORMATS.items():
                prompt = tmpl.format(k=p["num_genes"], organism=a.organism,
                                     genes=p["cell_sentence"])
                examples.append(Example(
                    example_id=f"{cell}__{fmt}", modality="singlecell", condition=cond,
                    payload={"prompt": prompt, "options": [], "format": fmt,
                             "num_genes": p["num_genes"],
                             "salient": p.get("salient") or [], "control": p.get("control") or []},
                    intervention=src.get("intervention") or cond,
                    ground_truth=gt[cell],
                    provenance=provenance(seed=a.seed, format=fmt, condition=cond, cell=cell,
                                          source=os.path.basename(a.mcq_examples))))
                stats[f"{fmt}:{cond}"] += 1

    # one file per format: `free` needs ~24 new tokens and `rationale` ~256, and running them in one
    # job would pay the rationale budget on every free-format row
    for fmt in FORMATS:
        write_jsonl([e for e in examples if e.payload["format"] == fmt],
                    os.path.join(a.outdir, f"examples_{fmt}.jsonl"))
    json.dump({"experiment": "rationale_prompts", "source": a.mcq_examples,
               "deg_definition": {**deg_cfg,
                                  "importance_source": src_meta.get("importance_source"),
                                  "criteria": "log2FC >= min AND pvals_adj < max (both required)",
                                  "inherited_from": src_meta_path},
               "conditions": CONDITIONS, "formats": sorted(FORMATS),
               "n_cells": len(take), "n_examples": len(examples), "seed": a.seed,
               "cells": take,
               "deg_by_type": {k: sorted(v) for k, v in deg_by_type.items()},
               "per_arm": dict(stats)},
              open(os.path.join(a.outdir, "meta.json"), "w"), indent=2)
    print(f"[fl] {len(examples)} questions ({len(take)} cells x {len(CONDITIONS)} conditions x "
          f"{len(FORMATS)} formats) -> {a.outdir}")


if __name__ == "__main__":
    main()

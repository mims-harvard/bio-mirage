#!/usr/bin/env python
"""Builds the BioReason disease prediction inputs: every query under every condition of
interventions.py.

Loads all 1,449 queries of the KEGG-derived dataset (`kegg_data` in checkpoints.json), cuts the
reference and variant sequences to a window of --truncate_per_side bases on each side of the first
differing base, and builds the genome and text conditions for each query. Each record also carries a
genome identifier, used as the resampling unit for bootstrap intervals, and its stratum from
kegg_pairs.py. Writes examples.jsonl and meta.json to --outdir. Needs the `datasets` package.

    python -m input_use.modalities.dna.prepare --outdir results/dna/bioreason/perturbations
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter

from input_use.core import conditions as C
from input_use.core.records import Example, write_jsonl, provenance
from input_use.modalities.dna.importance import variant_span
from input_use.modalities.dna.interventions import make_conditions, remove_star_marker
from input_use.modalities.dna.kegg_pairs import DATASET, build_donors, build_strata, load_rows


def truncate_around(ref: str, var: str, per_side: int):
    """Window both sequences around the variant. Indel-safe: the window is centred on the first
    differing base and each sequence is clipped to its own bounds, so a deletion does not shift the
    variant out of the variant window.
    """
    start, _, _ = variant_span(ref, var)
    lo, hi = max(0, start - per_side), start + per_side + 1
    return ref[lo:min(len(ref), hi)], var[lo:min(len(var), hi)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--splits", default="train,test,val")
    ap.add_argument("--n_examples", type=int, default=0, help="0 = all rows (the default)")
    ap.add_argument("--truncate_per_side", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=20260812)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()

    splits = tuple(s for s in a.splits.split(",") if s)
    rows = load_rows(a.dataset, splits)
    if a.n_examples:
        rows = rows[:a.n_examples]

    donors = build_donors(rows, seed=a.seed)
    row_by_id = {r["example_id"]: r for r in rows}
    strata = build_strata(rows)
    sensitivity_rows = set(strata["sensitivity"]["rows"])
    stability_rows = set(strata["stability"]["rows"])

    examples, n_star = [], 0
    for r in rows:
        eid = r["example_id"]
        ref_t, var_t = truncate_around(r["reference_sequence"], r["variant_sequence"],
                                       a.truncate_per_side)
        donor = donors.get(eid)
        donor_var = None
        if donor:
            # The donor block is the donor's own variant window, cut against the donor's reference -
            # not one re-cut against ours, which would centre it on the wrong locus.
            d = row_by_id[donor["donor_example_id"]]
            _, donor_var = truncate_around(d["reference_sequence"], d["variant_sequence"],
                                           a.truncate_per_side)

        has_star = remove_star_marker(r["question"]) != r["question"]
        n_star += has_star
        gt = {"answer": r["answer"]}
        prov = provenance(dataset=a.dataset, split=r["split"], seed=a.seed)
        conds = make_conditions(ref_t, var_t, r["question"], seed=a.seed + r["index"],
                                donor_variant=donor_var,
                                donor_example_id=donor["donor_example_id"] if donor else None)
        for cond, payload in conds.items():
            payload.update({
                "split": r["split"], "has_star_marker": bool(has_star),
                # Cluster unit for every bootstrap: 1,449 rows collapse onto 708 distinct variants,
                # and several rows can share one genome with different pathway text.
                "variant_key": hashlib.sha1(
                    (r["reference_sequence"] + "|" + r["variant_sequence"]).encode()).hexdigest()[:16],
                "stratum": ("sensitivity" if eid in sensitivity_rows
                            else "stability" if eid in stability_rows else "unpaired"),
                "donor_answer": donor["donor_answer"] if donor else None,
            })
            examples.append(Example(example_id=eid, modality="dna", condition=cond, payload=payload,
                                    intervention=cond, ground_truth=gt, provenance=prov))

    # --- the two invariants the attribution argument rests on ------------------------------------
    by_id = {}
    for e in examples:
        by_id.setdefault(e.example_id, {})[e.condition] = e
    for eid, arms in by_id.items():
        wt = arms[C.WT]
        for cond in C.KEGG_GENOME_CONDS:
            if cond in arms:
                assert arms[cond].payload["question"] == wt.payload["question"], \
                    f"{eid}/{cond}: genome arm changed the question text"
        for cond in C.KEGG_TEXT_CONDS:
            if cond in arms:
                assert arms[cond].payload["dna_sequences"] == wt.payload["dna_sequences"], \
                    f"{eid}/{cond}: text arm changed the DNA blocks"
        assert arms[C.NO_DNA].payload["dna_sequences"] == [], f"{eid}: no_dna still carries DNA"

    os.makedirs(a.outdir, exist_ok=True)
    write_jsonl(examples, os.path.join(a.outdir, "examples.jsonl"))
    cond_counts = Counter(e.condition for e in examples)
    meta = {
        "modality": "dna", "task": "kegg", "dataset": a.dataset, "splits": list(splits),
        "n_rows": len(rows), "n_examples": len(examples),
        "n_variants": len({(r["reference_sequence"], r["variant_sequence"]) for r in rows}),
        "n_question_texts": len({r["question"] for r in rows}),
        "n_answers": len({r["answer"] for r in rows}),
        "rows_per_split": dict(Counter(r["split"] for r in rows)),
        "truncate_per_side": a.truncate_per_side, "seed": a.seed,
        "conditions": dict(sorted(cond_counts.items())),
        "genome_conds": [c for c in C.KEGG_GENOME_CONDS if c in cond_counts],
        "text_conds": [c for c in C.KEGG_TEXT_CONDS if c in cond_counts],
        "rows_with_star_marker": n_star,
        "n_donors": len(donors),
        "strata": {k: {"texts": v["texts"], "rows": len(v["rows"]), "pairs": len(v["pairs"])}
                   for k, v in strata.items()},
        "note": "all rows kept; the previous single-SNV filter discarded 357 of 1449 (24.6%)",
    }
    with open(os.path.join(a.outdir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"[dna-prepare] {len(rows)} KEGG rows ({meta['n_variants']} variants, "
          f"{meta['n_question_texts']} texts) -> {len(examples)} examples in {len(cond_counts)} arms")
    print(f"  splits: {meta['rows_per_split']} | star marker in {n_star}/{len(rows)} rows | "
          f"donors {len(donors)}")
    print(f"  strata: " + " ".join(f"{k}={v['rows']}rows/{v['pairs']}pairs"
                                   for k, v in meta["strata"].items()))
    print(f"  -> {a.outdir}/examples.jsonl")


if __name__ == "__main__":
    main()

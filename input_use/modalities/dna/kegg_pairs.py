#!/usr/bin/env python
"""Groups the BioReason disease prediction queries by prompt text and finds the genome-dependent
queries.

A text group with more than one reference/variant genome pair and more than one disease label is
genome-dependent (the `sensitivity` stratum, 165 queries). A group with several genome pairs and a
single label is the `stability` stratum. Within each stratum the script lists pairs of queries with
identical prompt text and different genomes, keeping only pairs with different labels in the
`sensitivity` stratum. It also assigns each query a donor query with a different disease label,
whose variant sequence the `swap_variant_donor` condition uses. Writes pairs.json.

    python -m input_use.modalities.dna.kegg_pairs --out results/dna/bioreason/perturbations/pairs.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from itertools import combinations
from typing import Dict, List

from input_use.core.checkpoints import ckpt

DATASET = ckpt("kegg_data")
SPLITS = ("train", "test", "val")
PAIRS_VERSION = "1"

_NETWORK_RE = re.compile(r"^\s*Network Definition of the pathway:\s*(.*)$", re.MULTILINE)


def example_id(split: str, index: int) -> str:
    """Stable id, matching modalities/dna/prepare.py. Index is the row's position in its split."""
    return f"kegg_{split}_{index}"


def load_rows(dataset: str = DATASET, splits=SPLITS) -> List[dict]:
    """Every KEGG row, with its split, row index and stable example id attached."""
    from datasets import load_dataset

    ds = load_dataset(dataset)
    rows = []
    for split in splits:
        if split not in ds:
            continue
        for i, r in enumerate(ds[split]):
            rows.append({
                "example_id": example_id(split, i), "split": split, "index": i,
                "question": r["question"], "answer": r["answer"].strip(),
                "reference_sequence": r["reference_sequence"],
                "variant_sequence": r["variant_sequence"],
            })
    return rows


def _dna_key(r: dict):
    return (r["reference_sequence"], r["variant_sequence"])


def build_strata(rows: List[dict]) -> Dict[str, dict]:
    """Group rows by question text; split the groups that contain >1 distinct DNA pair into S and T."""
    by_text = defaultdict(list)
    for r in rows:
        by_text[r["question"]].append(r)

    S, T = {"texts": 0, "rows": [], "pairs": []}, {"texts": 0, "rows": [], "pairs": []}
    for text, group in by_text.items():
        if len({_dna_key(r) for r in group}) < 2:
            continue                                   # text pinned to one genome: no contrast
        target = S if len({r["answer"] for r in group}) > 1 else T
        target["texts"] += 1
        target["rows"].extend(r["example_id"] for r in group)
        for a, b in combinations(group, 2):
            if _dna_key(a) == _dna_key(b):
                continue
            differs = a["answer"] != b["answer"]
            if target is S and not differs:
                continue                               # within S, only cross-answer pairs are informative
            target["pairs"].append({"a": a["example_id"], "b": b["example_id"],
                                    "answer_a": a["answer"], "answer_b": b["answer"],
                                    "should_change": differs})
    return {"sensitivity": S, "stability": T}


def build_donors(rows: List[dict], seed: int = 20260812) -> Dict[str, dict]:
    """Map example_id -> a donor example whose variant block replaces this row's."""
    rng = random.Random(seed)
    by_answer = defaultdict(list)
    for r in rows:
        by_answer[r["answer"]].append(r)
    answers = sorted(by_answer)

    donors = {}
    for r in rows:
        pool = [a for a in answers if a != r["answer"]]
        if not pool:
            continue
        for _ in range(8):                              # a handful of draws; genomes rarely collide
            cand = rng.choice(by_answer[rng.choice(pool)])
            if _dna_key(cand) != _dna_key(r):
                donors[r["example_id"]] = {"donor_example_id": cand["example_id"],
                                           "donor_answer": cand["answer"]}
                break
    return donors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--seed", type=int, default=20260812)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = load_rows(a.dataset)
    strata = build_strata(rows)
    donors = build_donors(rows, seed=a.seed)

    # Invariants the whole experiment rests on - cheap, and each one has a way of being violated.
    assert all(d["donor_example_id"] != eid for eid, d in donors.items()), "self-donor"
    by_id = {r["example_id"]: r for r in rows}
    for eid, d in donors.items():
        assert by_id[d["donor_example_id"]]["answer"] != by_id[eid]["answer"], "donor shares the answer"
    for p in strata["sensitivity"]["pairs"]:
        assert p["answer_a"] != p["answer_b"], "non-discriminating pair in the sensitivity stratum"
        assert by_id[p["a"]]["question"] == by_id[p["b"]]["question"], "text differs inside a pair"

    out = {
        "pairs_version": PAIRS_VERSION, "dataset": a.dataset, "seed": a.seed,
        "n_rows": len(rows), "n_variants": len({_dna_key(r) for r in rows}),
        "n_question_texts": len({r["question"] for r in rows}),
        "strata": strata, "donors": donors,
    }
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)

    S, T = strata["sensitivity"], strata["stability"]
    print(f"[kegg-pairs] {len(rows)} rows | {out['n_variants']} variants | "
          f"{out['n_question_texts']} question texts")
    print(f"  sensitivity S: {S['texts']:3d} texts | {len(S['rows']):4d} rows | "
          f"{len(S['pairs']):4d} cross-answer pairs")
    print(f"  stability   T: {T['texts']:3d} texts | {len(T['rows']):4d} rows | "
          f"{len(T['pairs']):4d} pairs")
    print(f"  donors: {len(donors)} of {len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()

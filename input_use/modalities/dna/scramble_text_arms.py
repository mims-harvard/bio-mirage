#!/usr/bin/env python
r"""Adds BioReason conditions that pair shuffled DNA with the text conditions `no_gene`, `no_pathway`
and `no_textkey`.

Each new query copies the prompt text of the text condition and takes the DNA of the `scramble`
condition for the same query, so Figure 2a can compare intact and shuffled DNA under each text
condition. Reads the examples.jsonl written by prepare.py and writes only the new conditions to
--outdir. Score them together with the source run's records.

    python -m input_use.modalities.dna.scramble_text_arms \
        --source results/dna/bioreason/perturbations/examples.jsonl \
        --outdir results/dna/bioreason/text_conditions
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

from input_use.core import conditions as C

TEXT_RUNGS = (C.NO_GENE, C.NO_PATHWAY, C.NO_TEXTKEY)
GENOME_ARM = C.SCRAMBLE


def new_name(text_cond: str) -> str:
    return f"{GENOME_ARM}_{text_cond}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="the 16-arm examples.jsonl")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()

    by = {}
    with open(a.source) as fh:
        for line in fh:
            e = json.loads(line)
            by.setdefault(e["example_id"], {})[e["condition"]] = e
    assert all({C.WT, GENOME_ARM, *TEXT_RUNGS} <= set(v) for v in by.values()), "source lacks an arm"

    made = []
    for eid, arms in by.items():
        wt, scr = arms[C.WT], arms[GENOME_ARM]
        assert scr["payload"]["question"] == wt["payload"]["question"]
        assert scr["payload"]["dna_sequences"] != wt["payload"]["dna_sequences"]
        for tcond in TEXT_RUNGS:
            t = arms[tcond]
            assert t["payload"]["dna_sequences"] == wt["payload"]["dna_sequences"]
            assert t["payload"]["question"] != wt["payload"]["question"]
            n = json.loads(json.dumps(t))                     # deep copy of the text-arm example
            n["condition"] = n["intervention"] = new_name(tcond)
            n["payload"]["dna_sequences"] = list(scr["payload"]["dna_sequences"])
            n["payload"]["channels"] = {**n["payload"].get("channels", {}), "dna": GENOME_ARM}
            n["payload"]["source_arms"] = {"dna": GENOME_ARM, "text": tcond}
            # invariants: the text is the text arm's, the genome is the scramble arm's
            assert n["payload"]["question"] == t["payload"]["question"]
            assert n["payload"]["dna_sequences"] == scr["payload"]["dna_sequences"]
            assert len(n["payload"]["dna_sequences"]) == 2
            made.append(n)

    os.makedirs(a.outdir, exist_ok=True)
    path = os.path.join(a.outdir, "examples.jsonl")
    with open(path, "w") as fh:
        for e in made:
            fh.write(json.dumps(e) + "\n")
    counts = Counter(e["condition"] for e in made)
    json.dump({"source": os.path.abspath(a.source), "arms": dict(sorted(counts.items())),
               "n_examples": len(made), "n_rows": len(by),
               "note": "text-arm example with dna_sequences replaced by the row's own `scramble` "
                       "blocks verbatim; score merged with the source run's records"},
              open(os.path.join(a.outdir, "meta.json"), "w"), indent=2)
    print(f"[scramble_text_arms] {len(made)} examples over {len(by)} rows -> {path}")
    for k, v in sorted(counts.items()):
        print(f"   {k:22s} {v}")


if __name__ == "__main__":
    main()

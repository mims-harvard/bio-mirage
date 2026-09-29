#!/usr/bin/env python
"""Computes text-only lookup baselines for BioReason disease prediction, without a model or DNA.

Each baseline maps a key taken from the prompt text (the gene named in the query, the full prompt
text, or the pathway network string) to the most frequent disease label among training split queries
with that key. Queries with an unseen key get the most frequent training label. Reports accuracy and
key coverage on the test and validation splits and writes one JSON file.

    python -m input_use.modalities.dna.lookup_baseline --out results/dna/bioreason/perturbations/lookup.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from typing import Callable, Dict, List

from input_use.modalities.dna.kegg_pairs import DATASET, load_rows
from input_use.modalities.dna.interventions import (NETWORK_PREFIX, field_line, stem_gene)

BASELINE_VERSION = "1"


def _network_string(question: str) -> str:
    ln = field_line(question, NETWORK_PREFIX)
    return ln.split(":", 1)[1].strip() if ln and ":" in ln else ""


KEYS: Dict[str, Callable[[dict], str]] = {
    "stem_gene": lambda r: stem_gene(r["question"]) or "",
    "full_question": lambda r: r["question"],
    "network_string": lambda r: _network_string(r["question"]),
}


def fit(rows: List[dict], keyfn) -> Dict[str, str]:
    """key -> modal answer among the training rows carrying that key."""
    tally = defaultdict(Counter)
    for r in rows:
        tally[keyfn(r)][r["answer"]] += 1
    return {k: c.most_common(1)[0][0] for k, c in tally.items()}


def evaluate(table: Dict[str, str], fallback: str, rows: List[dict], keyfn) -> dict:
    hit = seen = 0
    for r in rows:
        k = keyfn(r)
        if k in table:
            seen += 1
            pred = table[k]
        else:
            pred = fallback
        hit += pred == r["answer"]
    n = len(rows)
    return {"n": n, "accuracy": hit / n if n else 0.0,
            "key_coverage": seen / n if n else 0.0, "n_keys": len(table)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--fit_split", default="train")
    ap.add_argument("--eval_splits", default="test,val")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = load_rows(a.dataset)
    by_split = defaultdict(list)
    for r in rows:
        by_split[r["split"]].append(r)
    train = by_split[a.fit_split]
    assert train, f"no rows in fit split {a.fit_split!r}"
    fallback = Counter(r["answer"] for r in train).most_common(1)[0][0]

    result = {"baseline_version": BASELINE_VERSION, "dataset": a.dataset,
              "fit_split": a.fit_split, "n_fit": len(train),
              "majority_class": fallback,
              "majority_class_rate": Counter(r["answer"] for r in rows)[fallback] / len(rows),
              "n_answers": len({r["answer"] for r in rows}), "keys": {}}

    for name, keyfn in KEYS.items():
        table = fit(train, keyfn)
        result["keys"][name] = {split: evaluate(table, fallback, by_split[split], keyfn)
                                for split in a.eval_splits.split(",") if by_split[split]}
        result["keys"][name]["n_keys"] = len(table)

    with open(a.out, "w") as fh:
        json.dump(result, fh, indent=2)

    print(f"[lookup] fit on {a.fit_split} (n={len(train)}); majority class "
          f"{fallback!r} at {result['majority_class_rate']:.3f} over all {len(rows)} rows")
    for name, res in result["keys"].items():
        parts = " | ".join(f"{sp}: acc={v['accuracy']:.3f} cov={v['key_coverage']:.3f} (n={v['n']})"
                           for sp, v in res.items() if isinstance(v, dict))
        print(f"  {name:16s} keys={res['n_keys']:4d}  {parts}")
    print(f"  -> {a.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Rechecks the readout GO terms of an existing BioReason-Pro pairs run and rewrites it without
rerunning the model.

For the feature categories each pair gets the one readout term that the positive protein carries,
and a pair is dropped when the negative protein carries that term or a descendant in its curated GO
annotation. `go_not` and the remaining categories are checked the same way against their stated
terms. Pairs whose identifier occurs in two categories are dropped. Rewrites records.jsonl,
pairs.jsonl, examples.jsonl and meta.json into --outdir.

    python -m input_use.modalities.protein.pairs_repair --rundir <run> --outdir <run>_fix
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter
from pathlib import Path

from input_use.core import config as cfg
from input_use.modalities.protein.pairs import FEATURE_CONTRASTS


def repair_pair(p, dag):
    """(repaired pair, reason-dropped or None) under the corrected rules. Categories with a single
    unambiguous readout term and a verified negative are returned unchanged.
    """
    cat = p["category"]
    a_go = set(p["a"].get("go") or [])
    b_go = set(p["b"].get("go") or [])

    if cat in FEATURE_CONTRASTS:
        terms = FEATURE_CONTRASTS[cat]["go"]
        carried = [t for t in terms if dag.any_under(a_go, t)]
        if len(carried) != 1:
            return None, ("positive carries none of the readout terms" if not carried
                          else f"positive carries {len(carried)} readout terms (ambiguous)")
        t = carried[0]
        if dag.any_under(b_go, t):
            return None, "negative carries the readout term"
        label = dag.name.get(t, terms[0])
        q = dict(p)
        q["spec_a"] = {"require": [t], "forbid": [], "label": label}
        q["spec_b"] = {"require": [], "forbid": [t], "label": f"no {label}"}
        q["property"] = f"{label} present vs absent ({FEATURE_CONTRASTS[cat]['determinant']})"
        return q, None

    if cat == "go_not":
        t = (p["spec_b"].get("forbid") or [None])[0]
        if t is None:
            return None, "no forbidden term on the negative spec"
        if dag.any_under(b_go, t):
            return None, ("negative carries the denied term itself" if t in b_go
                          else "negative carries a descendant of the denied term")
        if not dag.any_under(a_go, t):
            return None, "positive does not carry the term"
        return p, None

    # pseudoenzyme / cofactor / merops_activity: single readout term, both members already checked
    # so a future category cannot slip through unchecked.
    for side, spec in (("a", p["spec_a"]), ("b", p["spec_b"])):
        go = a_go if side == "a" else b_go
        if not all(dag.any_under(go, r) for r in spec.get("require", [])):
            return None, f"member {side} lacks its required term"
        if any(dag.any_under(go, r) for r in spec.get("forbid", [])):
            return None, f"member {side} carries its forbidden term"
    return p, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rundir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--go_obo", default=str(cfg.GO_OBO))
    a = ap.parse_args()

    from input_use.metrics import go_dag
    dag = go_dag.load(a.go_obo)

    rundir, outdir = Path(a.rundir), Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pairs = [json.loads(l) for l in open(rundir / "pairs.jsonl")]

    # `example_id` in these runs is `A__B`, with no category, so two categories that drew the same
    # protein pair share one group of records and one of them is already lost.
    id_count = Counter(f"{p['a']['accession']}__{p['b']['accession']}" for p in pairs)
    collided = {k for k, v in id_count.items() if v > 1}

    kept, dropped = [], Counter()
    spec_by_pair = {}
    for p in pairs:
        pid = f"{p['a']['accession']}__{p['b']['accession']}"
        if pid in collided:
            dropped[(p["category"], "pair id shared with another category - records ambiguous")] += 1
            continue
        q, why = repair_pair(p, dag)
        if q is None:
            dropped[(p["category"], why)] += 1
            continue
        kept.append(q)
        spec_by_pair[pid] = (q["spec_a"], q["spec_b"], q["property"])

    before, after = Counter(p["category"] for p in pairs), Counter(p["category"] for p in kept)
    print(f"pairs: {len(pairs)} -> {len(kept)}")
    for c in sorted(before):
        print(f"  {c:<22}{before[c]:>5} -> {after.get(c, 0):<5}")
    if dropped:
        print("dropped, by reason:")
        for (c, why), n in sorted(dropped.items()):
            print(f"  {c:<22}{n:>5}  {why}")

    # Rewrite the records with the corrected spec, dropping records for dropped pairs. `example_id`
    # is the pair id, and `ground_truth` carries spec_a/spec_b - that is all the scorer reads.
    n_in = n_out = 0
    with open(outdir / "records.jsonl", "w") as out:
        for f in sorted(glob.glob(str(rundir / "records.jsonl*"))):
            for line in open(f):
                r = json.loads(line)
                n_in += 1
                got = spec_by_pair.get(r["example_id"])
                if not got:
                    continue
                r["ground_truth"]["spec_a"], r["ground_truth"]["spec_b"], r["ground_truth"]["property"] = got
                out.write(json.dumps(r) + "\n")
                n_out += 1
    print(f"records: {n_in} -> {n_out}")

    with open(outdir / "pairs.jsonl", "w") as fh:
        for p in kept:
            fh.write(json.dumps(p) + "\n")

    # examples.jsonl is filtered the same way: the ESM3 probe reads it (not records.jsonl) to decide
    # which proteins to encode, so the premise check has to be fit on the pairs reported.
    ex_in = rundir / "examples.jsonl"
    if ex_in.exists():
        n_ex_in = n_ex_out = 0
        with open(outdir / "examples.jsonl", "w") as out:
            for line in open(ex_in):
                e = json.loads(line)
                n_ex_in += 1
                got = spec_by_pair.get(e["example_id"])
                if not got:
                    continue
                e["ground_truth"]["spec_a"], e["ground_truth"]["spec_b"], e["ground_truth"]["property"] = got
                out.write(json.dumps(e) + "\n")
                n_ex_out += 1
        print(f"examples: {n_ex_in} -> {n_ex_out}")

    meta = json.load(open(rundir / "meta.json")) if (rundir / "meta.json").exists() else {}
    meta.update({"n_pairs": len(kept), "n_examples": n_out, "pairs_by_category": dict(after),
                 "repaired_from": str(rundir),
                 "repair": "per-pair readout term for feature contrasts; negatives verified against "
                           "full curated GO (see modalities/protein/pairs_repair.py)",
                 "repair_dropped": {f"{c}:{why}": n for (c, why), n in sorted(dropped.items())}})
    json.dump(meta, open(outdir / "meta.json", "w"), indent=2)
    print(f"-> {outdir}")


if __name__ == "__main__":
    main()

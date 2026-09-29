#!/usr/bin/env python
"""Computes the information accretion (IA) weights used for the weighted F_max of GO term predictions.

    python -m input_use.metrics.make_ia --gaf <goa_human.gaf> --obo <go-basic.obo> --out IA_goa_human.txt

A term's weight is ia(t) = -log2(|proteins annotated with t| / |proteins annotated with every parent
of t|), computed over GOA annotations propagated to the roots (Clark and Radivojac, 2013), so root
terms have weight 0. The weights are estimated from the same GOA snapshot the predictions are scored
against, so they differ from weights estimated on another snapshot. Writes one
`GO:xxxxxxx<TAB>value` line per term, the format cafaeval reads. input_use.core.score reads the file
from INPUT_USE_GO_IA (default data/IA_goa_human.txt).
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

from input_use.core import config as cfg
from input_use.metrics import go_dag
from input_use.modalities.protein.sources import goa


def build_ia(gaf_path, obo_path, taxon=None, include_htp=False):
    dag = go_dag.load(str(obo_path))
    ann = goa.load_annotations(gaf_path, taxon=taxon, include_htp=include_htp)
    gt = ann.ground_truth()          # {accession: {direct GO terms}}

    # Propagate each protein's annotations to the roots, exactly as cafaeval's prop='max' does over
    # is_a + part_of. Counting unpropagated terms would understate every ancestor and inflate its
    # weight.
    # Each term gets a bitmask over protein indices. A term's parent set can have more than one
    # member, and then the denominator is the size of the INTERSECTION of the parents' protein sets,
    # not the smallest of them --- min() would overstate the denominator and so overstate ia. Bitmask
    # AND + popcount makes the exact intersection cheap (~20k proteins = 2.5 kB per term).
    masks = defaultdict(int)
    # Per-ASPECT protein masks. `dag.ancestors` deliberately excludes the three namespace roots, so a
    # term whose only parent is a root would otherwise be left with an empty parent set, be mistaken
    # for an orphan, and be handed ia = 0. Those are exactly the most general terms -- `catalytic
    # activity`, `binding` -- and zeroing them silently removes the commonest predictions from the
    # weighted score. The correct denominator for a direct child of a root is the number of proteins
    # annotated anywhere in that aspect, which is what these masks hold.
    aspect_mask = defaultdict(int)
    n_proteins = 0
    for acc, terms in sorted(gt.items()):
        closure = set()
        for t in terms:
            n = dag.norm(t)
            if n:
                closure |= dag.ancestors(n) | {n}
        if not closure:
            continue
        bit = 1 << n_proteins
        n_proteins += 1
        for t in closure:
            masks[t] |= bit
            a = dag.aspect_of.get(t)
            if a:
                aspect_mask[a] |= bit

    counts = {t: m.bit_count() for t, m in masks.items()}

    ia = {}
    zero_parent = root_child = skipped = 0
    for term, c in counts.items():
        if term in go_dag.ROOTS:
            ia[term] = 0.0           # a root asserts nothing beyond the corpus itself
            zero_parent += 1
            continue
        parents = {p for p in dag.parents.get(term, set()) if p in masks}
        if not parents:
            # Parents are roots (or unannotated): condition on the aspect instead.
            denom = aspect_mask.get(dag.aspect_of.get(term, ""), 0).bit_count()
            root_child += 1
            if denom <= 0 or c <= 0:
                skipped += 1
                continue
            pr = c / denom
            ia[term] = 0.0 if pr >= 1.0 else -math.log2(pr)
            continue
        inter = None
        for p in parents:
            inter = masks[p] if inter is None else (inter & masks[p])
        denom = inter.bit_count()
        if denom <= 0 or c <= 0:
            skipped += 1
            continue
        # P(t | Pa(t)) <= 1: annotations are propagated, so every protein carrying t carries all of
        # its parents and is inside the intersection.
        p = c / denom
        ia[term] = 0.0 if p >= 1.0 else -math.log2(p)
    return ia, {"n_proteins": n_proteins, "n_terms": len(ia),
                "n_roots": zero_parent, "n_root_children": root_child, "n_skipped": skipped}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gaf", default=str(Path(cfg.DATA_DIR) / "goa_human.gaf"))
    ap.add_argument("--obo", default=str(cfg.GO_OBO))
    ap.add_argument("--taxon", default="taxon:9606")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ia, stats = build_ia(a.gaf, a.obo, taxon=a.taxon)
    with open(a.out, "w") as f:
        for term in sorted(ia):
            f.write(f"{term}\t{ia[term]:.6f}\n")

    vals = sorted(ia.values())
    nz = [v for v in vals if v > 0]
    print(f"[ia] proteins used        : {stats['n_proteins']}")
    print(f"[ia] terms with a weight  : {stats['n_terms']}  (roots at 0: {stats['n_roots']}, "
          f"direct root children conditioned on aspect: {stats['n_root_children']}, "
          f"skipped: {stats['n_skipped']})")
    if nz:
        q = lambda p: nz[int(p * (len(nz) - 1))]
        print(f"[ia] nonzero weights      : n={len(nz)}  min={nz[0]:.3f}  "
              f"q25={q(.25):.3f}  median={q(.5):.3f}  q75={q(.75):.3f}  max={nz[-1]:.3f}")
    print(f"[ia] wrote {a.out}")
    print(f"[ia] use it with:  INPUT_USE_GO_IA={a.out} python -m input_use.core.score --rundir <run>")
    return 0


if __name__ == "__main__":
    sys.exit(main())

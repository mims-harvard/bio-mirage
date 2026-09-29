r"""Adds the BioReason-Pro conditions `interpro_only_wt_esm3` and `gogpt_only_wt_esm3` to an existing
run.

They are `interpro_only` and `gogpt_only` with the ESM3 input set back to the wild-type sequence, so
each single text source can be read with both intact and shuffled ESM3 representations. Reads the
examples.jsonl of a prepare.py run and writes only the new conditions. Score them together with the
source run's records.

    python -m input_use.modalities.protein.derive_wt_esm3_arms \
        --src <run>/examples.jsonl --out <new_run>/examples.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from input_use.core import conditions as C

DERIVED = {
    C.INTERPRO_ONLY_WT_ESM3: (C.INTERPRO_ONLY, {"interpro": "wt", "gogpt": "none"}),
    C.GOGPT_ONLY_WT_ESM3: (C.GOGPT_ONLY, {"interpro": "none", "gogpt": "on"}),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="examples.jsonl of the run to derive from")
    ap.add_argument("--out", required=True, help="examples.jsonl to write (new arms only)")
    a = ap.parse_args()

    # Two streaming passes rather than one in-memory index: the proteome file is 124,816 rows / 287
    # MB and holding it as dicts is several GB, which the login node kills.
    wt_seq = {}
    for line in open(a.src):
        d = json.loads(line)
        if d["condition"] == C.WT:
            wt_seq[d["example_id"]] = d["payload"]["sequence"]
    print(f"[derive] pass 1: wild-type sequence for {len(wt_seq)} proteins")

    src_of = {src: new for new, (src, _) in DERIVED.items()}
    channels_of = {new: ch for new, (_, ch) in DERIVED.items()}
    counts, skipped = defaultdict(int), defaultdict(int)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for line in open(a.src):
            d = json.loads(line)
            new_cond = src_of.get(d["condition"])
            if new_cond is None:
                continue
            ex_id = d["example_id"]
            if ex_id not in wt_seq:
                skipped["no wt row"] += 1
                continue
            p = d["payload"]
            # The whole edit: the embedding goes back to wild type.
            assert p["symbolic_sequence"] == wt_seq[ex_id], f"{ex_id}/{d['condition']}: sym seq != WT"
            assert p["channels"] == channels_of[new_cond], f"{ex_id}: channels {p['channels']}"
            src_cond = d["condition"]
            p["sequence"] = wt_seq[ex_id]
            d["condition"] = new_cond
            d["intervention"] = {"operator": new_cond,
                                 "note": f"{src_cond} with the ESM3 embedding left INTACT; "
                                         "intact-ESM3 reference for that arm"}
            d["provenance"] = dict(d.get("provenance") or {},
                                   derived_from=f"{Path(a.src).name}::{src_cond}")
            f.write(json.dumps(d) + "\n")
            counts[new_cond] += 1

    print(f"[derive] {sum(counts.values())} examples -> {a.out}")
    for cond in DERIVED:
        print(f"[derive]   {cond}: {counts[cond]}")
    for k, v in skipped.items():
        print(f"[derive]   SKIPPED {v} ({k})")


if __name__ == "__main__":
    main()

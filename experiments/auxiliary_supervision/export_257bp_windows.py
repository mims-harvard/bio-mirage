"""Cuts the pathway network split to 257-bp windows with the edit at offset 128 and writes an
intermediate training set that export_ref_var_target.py rewrites into the 257-bp auxiliary condition
of Figure 7.

Queries whose window does not fit inside both sequences are dropped. The reasoning trace is replaced
with "Edit at 128: ref <21 bp> | var <21 bp> (class)", where the class comes from the full sequences.
Reads INPUT_USE_DATA_DIR/kegg and writes train/val/test CSVs to windows_257bp in
dna/bioreason/auxiliary_supervision under INPUT_USE_RESULTS_DIR.
"""
import os
import collections, csv, json, os, re

from input_use.core import paths as RD

REPO_DIR = os.environ.get("INPUT_USE_HOME", ".")
DATA_DIR = os.environ.get("INPUT_USE_DATA_DIR", "data")

csv.field_size_limit(10 ** 9)
SRC = f"{DATA_DIR}/kegg"
DST = RD.AUX_WINDOWS_257BP
FLANK, HALF = 10, 128          # 2*half+1 = 257 bp window, edit at offset 128
SPLITS = [("train", "train_network_split.csv"),
          ("val", "id_test_network_split.csv"),
          ("test", "ood_test_network_split.csv")]


def first_diff(a, b):
    return next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), None)


def classify(a, b, i):
    if len(a) == len(b):
        d = [k for k, (x, y) in enumerate(zip(a, b)) if x != y]
        if len(d) == 1:
            return f"substitution {a[d[0]]}>{b[d[0]]}"
        return f"multi-base substitution, {len(d)} positions"
    n = abs(len(a) - len(b))
    return f"{'insertion' if len(b) > len(a) else 'deletion'} of {n} bp"


def main():
    os.makedirs(DST, exist_ok=True)
    kinds, counts = collections.Counter(), collections.Counter()
    for split, fname in SPLITS:
        rows = list(csv.DictReader(open(os.path.join(SRC, fname), newline="")))
        out = []
        for r in rows:
            A, B = r["reference_sequence"], r["variant_sequence"]
            i = first_diff(A, B)
            counts[f"{split}_rows"] += 1
            if i is None or i - HALF < 0 or i + HALF + 1 > min(len(A), len(B)):
                counts[f"{split}_drop"] += 1
                continue
            lo = i - HALF
            a, b = A[lo:i + HALF + 1], B[lo:i + HALF + 1]   # 257 bp each, edit at index half
            assert len(a) == len(b) == 2 * HALF + 1
            j = HALF
            klass = classify(A, B, i)                        # class from the full sequences
            ref_q, var_q = a[j - FLANK:j + FLANK + 1], b[j - FLANK:j + FLANK + 1]
            assert len(ref_q) == len(var_q) == 2 * FLANK + 1
            assert a[j] == A[i] and b[j] == B[i]             # the cut kept the edited base
            rr = dict(r)
            rr["reference_sequence"], rr["variant_sequence"] = a, b
            rr["reasoning"] = f"Edit at {j}: ref {ref_q} | var {var_q} ({klass})"
            kinds[klass.split()[0]] += 1
            out.append(rr)
        with open(os.path.join(DST, f"{split}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            for rr in out:
                w.writerow(rr)
        counts[f"{split}_written"] = len(out)
        print(f"{split:<6} -> {split}.csv  {len(out):>5,} rows  ({2*HALF+1} bp windows, edit at {HALF})")
    print("\nedit classes:", dict(kinds))
    json.dump(dict(source=SRC, window=2 * HALF + 1, edit_offset=HALF, flank=FLANK,
                   counts=dict(counts), edit_classes=dict(kinds)),
              open(os.path.join(DST, "meta.json"), "w"), indent=2)
    print(f"wrote {DST}")


if __name__ == "__main__":
    main()

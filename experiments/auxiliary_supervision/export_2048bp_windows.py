"""Writes the training data of the two 2,048-bp conditions of Figure 7: windows cut at the same
coordinates in both sequences, with the edit at offset 1,024.

The auxiliary condition replaces the reasoning trace with "Edit at 1024: ref X -> var Y (class)". The
condition without auxiliary supervision keeps the benchmark's reasoning trace. Reads the pathway
network split CSVs under INPUT_USE_DATA_DIR/kegg and writes train/val/test CSVs to
windows_2048bp_ref_var_target and windows_2048bp in dna/bioreason/auxiliary_supervision under
INPUT_USE_RESULTS_DIR.
"""
import collections
import csv
import json
import os

from input_use.core import paths as RD

DATA_DIR = os.environ.get("INPUT_USE_DATA_DIR", "data")

csv.field_size_limit(10 ** 9)
SRC = f"{DATA_DIR}/kegg"
DST_AUX = RD.AUX_TARGET_2048BP
DST_NOAUX = RD.AUX_NO_TARGET_2048BP
WIN, HALF = 2048, 1024          # edit at offset half; half bp before it, win - half - 1 after
SPLITS = [("train", "train_network_split.csv"), ("val", "id_test_network_split.csv"),
          ("test", "ood_test_network_split.csv")]


def first_diff(a, b):
    return next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), None)


def classify(a, b):
    """Same four classes as export_ref_var_target.py, from the full windows."""
    if len(a) == len(b):
        d = [k for k, (x, y) in enumerate(zip(a, b)) if x != y]
        return "substitution" if len(d) == 1 else "multi-base"
    return "insertion" if len(b) > len(a) else "deletion"


def main():
    for d in (DST_AUX, DST_NOAUX):
        os.makedirs(d, exist_ok=True)
    meta = dict(source=SRC, window=WIN, edit_offset=HALF, counts={}, classes=collections.Counter(),
                pairs=collections.Counter())
    for split, fname in SPLITS:
        rows = list(csv.DictReader(open(os.path.join(SRC, fname), newline="")))
        aux, noaux = [], []
        for r in rows:
            A, B = r["reference_sequence"], r["variant_sequence"]
            i = first_diff(A, B)
            lo = i - HALF
            assert i is not None and lo >= 0 and lo + WIN <= min(len(A), len(B)), (split, i, len(A), len(B))
            a, b = A[lo:lo + WIN], B[lo:lo + WIN]
            assert len(a) == len(b) == WIN
            assert a[:HALF] == b[:HALF] and a[HALF] != b[HALF]          # the edit is the first difference, at half
            assert a[HALF] == A[i] and b[HALF] == B[i]
            assert a[HALF] in "ACGT" and b[HALF] in "ACGT"
            klass = classify(A, B)
            base = dict(r)
            base["reference_sequence"], base["variant_sequence"] = a, b
            noaux.append(dict(base))
            base["reasoning"] = f"Edit at {HALF}: ref {a[HALF]} -> var {b[HALF]} ({klass})"
            aux.append(base)
            meta["classes"][f"{split}:{klass}"] += 1
            meta["pairs"][f"{a[HALF]}>{b[HALF]}"] += 1
        for d, out in ((DST_AUX, aux), (DST_NOAUX, noaux)):
            with open(os.path.join(d, f"{split}.csv"), "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                for rr in out:
                    w.writerow(rr)
        meta["counts"][split] = dict(source=len(rows), written=len(aux))
        print(f"  {split:<6} {len(rows):>5,} source rows -> {len(aux):>5,} written ({WIN} bp, edit at {HALF})")
    meta["classes"], meta["pairs"] = dict(meta["classes"]), dict(meta["pairs"])
    for d in (DST_AUX, DST_NOAUX):
        json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=2)
    print(f"  wrote {DST_AUX}\n  wrote {DST_NOAUX}")


if __name__ == "__main__":
    main()

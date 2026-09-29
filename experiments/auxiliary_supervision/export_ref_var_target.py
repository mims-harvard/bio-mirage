"""Writes the training data of the 257-bp auxiliary condition of Figure 7, with the auxiliary target
"Edit at 128: ref X -> var Y (class)".

X and Y are the reference and variant bases at the edited position. Reads the windows written by
export_257bp_windows.py (windows_257bp) and writes train/val/test CSVs to
windows_257bp_ref_var_target, both in dna/bioreason/auxiliary_supervision under
INPUT_USE_RESULTS_DIR.
"""
import os
import collections, csv, json, os

from input_use.core import paths as RD


csv.field_size_limit(10 ** 9)
SRC = RD.AUX_WINDOWS_257BP
DST = RD.AUX_TARGET_257BP
OFF = 128


def main():
    os.makedirs(DST, exist_ok=True)
    kinds, pairs = collections.Counter(), collections.Counter()
    for split in ("train", "val", "test"):
        rows = list(csv.DictReader(open(os.path.join(SRC, f"{split}.csv"), newline="")))
        out = []
        for r in rows:
            a, b = r["reference_sequence"], r["variant_sequence"]
            # the class was already derived from the full sequences by export_257bp_windows.py
            klass = r["reasoning"].rsplit("(", 1)[1].rstrip(")")
            klass = klass.split()[0]                       # substitution / insertion / deletion / multi-base
            assert a[OFF] != b[OFF], "the edited base must differ at the stated offset"
            rr = dict(r)
            rr["reasoning"] = f"Edit at {OFF}: ref {a[OFF]} -> var {b[OFF]} ({klass})"
            kinds[klass] += 1
            pairs[f"{a[OFF]}>{b[OFF]}"] += 1
            out.append(rr)
        with open(os.path.join(DST, f"{split}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            for rr in out:
                w.writerow(rr)
        print(f"  {split:<6} {len(out):>5,} rows")
    print(f"\n  distinct ref->var pairs: {len(pairs)}   {dict(pairs.most_common(6))}")
    print(f"  classes: {dict(kinds)}")
    json.dump(dict(source=SRC, offset=OFF, distinct_pairs=len(pairs),
                   pairs=dict(pairs), classes=dict(kinds)),
              open(os.path.join(DST, "meta.json"), "w"), indent=2)
    print(f"  wrote {DST}")


if __name__ == "__main__":
    main()

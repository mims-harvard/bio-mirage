"""Rebuilds the KEGG pathway network split from wanglab/kegg.

    python build_network_split.py --out <dir>
"""
import argparse, hashlib, os
import pandas as pd
from datasets import load_dataset

REVISION = "9f0ef941a362f0e308e11861ed82c36338e23586"
MD5 = {"train": "067c273308a0bba057a3c44ed24656ee", "id_test": "37c2708d3db3e82d1ce864253e391e59",
       "ood_test": "5edbf42b62d0ae8825382ca1fea39f0c"}
COLS = ["question", "answer", "reasoning", "reference_sequence", "variant_sequence"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".")
    ap.add_argument("--map", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "network_split_to_wanglab_kegg.csv"))
    a = ap.parse_args()
    ds = load_dataset("wanglab/kegg", revision=REVISION)
    src = {s: ds[s].to_pandas()[COLS] for s in ("train", "val", "test")}
    m = pd.read_csv(a.map)
    os.makedirs(a.out, exist_ok=True)
    for split in ("train", "id_test", "ood_test"):
        mm = m[m.network_split == split].sort_values("network_row")
        rows = [src[s].iloc[i] for s, i in zip(mm.wanglab_kegg_split, mm.wanglab_kegg_row)]
        path = os.path.join(a.out, f"{split}_network_split.csv")
        pd.DataFrame(rows, columns=COLS).to_csv(path, index=False)
        md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
        print(f"{split:<9} {len(rows):>5} rows  {'md5 OK' if md5 == MD5[split] else 'md5 MISMATCH'}")


if __name__ == "__main__":
    main()

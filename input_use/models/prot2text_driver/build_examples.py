#!/usr/bin/env python
"""Builds the Prot2Text-V2 test query table and the generation requests for the shuffled sequence
conditions.

Conditions: the released prompt (name, taxon, sequence truncated at 1,021 residues), the sequence
alone with name and taxon set to "unknown", and each of these with the sequence shuffled under three
seeds (proteins of at most 1,021 residues only). Reads data/test.parquet and writes
results/prot2text_examples.parquet, results/prot2text_shuffle_integrity.json and
runs/requests_A.jsonl under protein/prot2text_v2 of $INPUT_USE_RESULTS_DIR.

    python input_use/models/prot2text_driver/build_examples.py
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

SEEDS = (0, 1, 2)


def main():
    os.makedirs(C.RESULTS, exist_ok=True)
    os.makedirs(C.RUNS, exist_ok=True)
    df = pd.read_parquet(os.path.join(C.DATA, "test.parquet"))
    assert list(df.columns) == ["accession", "name", "Full Name", "taxon", "sequence", "function",
                                "AlphaFoldDB"], df.columns
    assert df["accession"].is_unique
    n_test = len(df)

    ex = pd.DataFrame({
        "protein_id": df["accession"],
        "mnemonic": df["name"],
        "alphafolddb": df["AlphaFoldDB"],
        "full_name": df["Full Name"],
        "taxon": df["taxon"],
        "sequence": df["sequence"],
        "sequence_hash": df["sequence"].map(C.seq_hash),
        "sequence_length": df["sequence"].str.len(),
        "function_text": df["function"],
        "split": "test",
    })
    ex["full_name_missing"] = ex["full_name"].isna() | (ex["full_name"].str.strip() == "") \
        | (ex["full_name"].str.lower() == C.UNKNOWN)
    ex["taxon_missing"] = ex["taxon"].isna() | (ex["taxon"].str.strip() == "")
    ex["function_missing"] = ex["function_text"].isna() | (ex["function_text"].str.strip() == "")
    ex["within_1021"] = ex["sequence_length"] <= C.OFFICIAL_MAX_SEQ
    ex["nonstandard_aa"] = ex["sequence"].str.contains(r"[^ACDEFGHIKLMNPQRSTVWY]", regex=True)
    ex.to_parquet(os.path.join(C.RESULTS, "prot2text_examples.parquet"), index=False)

    # name / taxon as dataset.py would feed them (NaN -> "unknown")
    name_field = ex["full_name"].where(~ex["full_name"].isna(), C.UNKNOWN)
    taxon_field = ex["taxon"].where(~ex["taxon"].isna(), C.UNKNOWN)

    reqs, integrity = [], []

    def add(cond, seed, row, name, taxon, seq):
        reqs.append({"request_id": C.request_id(cond, seed, row.protein_id), "protein_id": row.protein_id,
                     "condition": cond, "shuffle_seed": seed, "name_field": name, "taxon_field": taxon,
                     "sequence": seq, "sequence_hash": C.seq_hash(seq), "sequence_length": len(seq)})

    for i, row in enumerate(ex.itertuples(index=False)):
        nm, tx = name_field.iloc[i], taxon_field.iloc[i]
        add("native_full", -1, row, nm, tx, row.sequence[:C.OFFICIAL_MAX_SEQ])
        if not row.within_1021:
            continue
        add("seq_only_intact", -1, row, C.UNKNOWN, C.UNKNOWN, row.sequence)
        for s in SEEDS:
            shuf = C.shuffle_sequence(row.sequence, s, row.protein_id)
            chk = C.shuffle_ok(row.sequence, shuf)
            integrity.append({"protein_id": row.protein_id, "seed": s, **chk})
            assert chk["len_equal"] and chk["composition_equal"], (row.protein_id, s)
            add("seq_only_shuffle", s, row, C.UNKNOWN, C.UNKNOWN, shuf)
            add("native_shuffle", s, row, nm, tx, shuf)

    p = os.path.join(C.RUNS, "requests_A.jsonl")
    with open(p + ".tmp", "w") as fh:
        for r in reqs:
            fh.write(json.dumps(r) + "\n")
    os.replace(p + ".tmp", p)

    n_within = int(ex["within_1021"].sum())
    same_hash = sum(1 for r in integrity if not r["hash_differs"])
    summary = {
        "n_test": int(n_test), "n_within_1021": n_within, "n_over_1021_truncated_in_native_full": int(n_test - n_within),
        "n_full_name_missing": int(ex["full_name_missing"].sum()), "n_taxon_missing": int(ex["taxon_missing"].sum()),
        "n_function_missing": int(ex["function_missing"].sum()), "n_nonstandard_aa": int(ex["nonstandard_aa"].sum()),
        "n_requests": len(reqs), "requests_per_condition": pd.Series([r["condition"] for r in reqs]).value_counts().to_dict(),
        "n_shuffles": len(integrity), "n_shuffles_len_equal": sum(r["len_equal"] for r in integrity),
        "n_shuffles_composition_equal": sum(r["composition_equal"] for r in integrity),
        "n_shuffles_hash_unchanged": same_hash,
        "n_shuffles_degenerate_composition": sum(r["degenerate"] for r in integrity),
        "seeds": list(SEEDS), "vendored_prompt_sha256_of_dataset_py": C.VENDORED_PROMPT_SHA,
    }
    C.write_json({"summary": summary, "per_shuffle": integrity},
                 os.path.join(C.RESULTS, "prot2text_shuffle_integrity.json"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Scores the Prot2Text-V2 generations for the shuffled sequence conditions against each protein's own
function text.

Computes BERTScore (BioBERT and RoBERTa, as in the released benchmark script) and ROUGE-L per
generation, and compares the released prompt condition with the BERTScore F1 reported by the
Prot2Text-V2 authors. Writes prot2text_shuffle_predictions.parquet and prot2text_reproduction.json.

    python input_use/models/prot2text_driver/score_A.py [--gen-dir $INPUT_USE_RESULTS_DIR/protein/prot2text_v2/runs/genA]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from scoring import Scorers, rouge_l, bleu_corpus  # noqa: E402


def load_generations(gen_dir):
    rows = []
    for p in sorted(glob.glob(os.path.join(gen_dir, "generations_shard*.jsonl"))):
        rows += C.read_jsonl(p)
    g = pd.DataFrame(rows)
    dup = g.request_id.duplicated().sum()
    if dup:
        print(f"WARNING: {dup} duplicate request ids (resume overlap); keeping the last", flush=True)
        g = g.drop_duplicates("request_id", keep="last")
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", default=os.path.join(C.RUNS, "genA"))
    a = ap.parse_args()
    ex = pd.read_parquet(os.path.join(C.RESULTS, "prot2text_examples.parquet")).set_index("protein_id")
    req = pd.DataFrame(C.read_jsonl(os.path.join(C.RUNS, "requests_A.jsonl")))
    gen = load_generations(a.gen_dir)
    df = req.merge(gen, on="request_id", how="left")
    missing = df.generation.isna().sum()
    print(f"requests {len(req)}, generations {len(gen)}, missing {missing}", flush=True)
    df = df[~df.generation.isna()].reset_index(drop=True)
    df["function_text"] = df.protein_id.map(ex["function_text"])

    sc = Scorers(device="cuda")
    cands, refs = df.generation.tolist(), df.function_text.tolist()
    for name in ("biobert", "roberta"):
        P, R, F = sc.bertscore(name, cands, refs)
        df[f"{name}_p"], df[f"{name}_r"], df[f"{name}_score_to_own_function"] = P, R, F
        print(f"{name} done", flush=True)
    df["rougeL"] = rouge_l(cands, refs)

    out = df[["protein_id", "condition", "shuffle_seed", "sequence_hash", "sequence_length", "name_field",
              "taxon_field", "generation", "n_new_tokens", "ended_with_eos", "prompt_len", "batch_size",
              "biobert_score_to_own_function", "roberta_score_to_own_function", "rougeL",
              "biobert_p", "biobert_r", "roberta_p", "roberta_r"]].copy()
    out.to_parquet(os.path.join(C.RESULTS, "prot2text_shuffle_predictions.parquet"), index=False)

    nat = df[df.condition == "native_full"]
    rep = {"n": int(len(nat)), "n_test": int(len(ex)),
           "roberta-large": {k: float(nat[f"roberta_{s}"].mean()) for k, s in (("precision", "p"), ("recall", "r"), ("f1", "score_to_own_function"))},
           "biobert-large": {k: float(nat[f"biobert_{s}"].mean()) for k, s in (("precision", "p"), ("recall", "r"), ("f1", "score_to_own_function"))},
           "rougeL_mean": float(nat.rougeL.mean()),
           "bleu2_corpus": bleu_corpus(nat.generation.tolist(), nat.function_text.tolist(), 2),
           "bleu4_corpus": bleu_corpus(nat.generation.tolist(), nat.function_text.tolist(), 4),
           "paper_reported": {"roberta_bertscore_f1": 0.9195, "biobert_bertscore_f1": 0.8681},
           "n_empty_generations": int((nat.generation.str.strip() == "").sum()),
           "n_hit_max_new_tokens": int((~nat.ended_with_eos).sum()),
           "mean_new_tokens": float(nat.n_new_tokens.mean()),
           "per_condition_mean_biobert_f1": df.groupby(["condition", "shuffle_seed"]).biobert_score_to_own_function.mean().round(4).to_dict().__repr__()}
    C.write_json(rep, os.path.join(C.RESULTS, "prot2text_reproduction.json"))
    print(json.dumps(rep, indent=1), flush=True)


if __name__ == "__main__":
    main()

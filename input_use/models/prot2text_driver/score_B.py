#!/usr/bin/env python
"""Scores the Prot2Text-V2 evidence conflict generations against the function texts of both proteins in
each pair.

For each pair it scores the two conflict generations, the two generations with each protein's own
prompt and the two sequence-only generations (both from the shuffle run) by BERTScore to protein A's
function minus BERTScore to protein B's function. Each conflict generation is labelled as following
the sequence, the text (protein name), or neither when the difference is below --epsilon, and each
pair is marked by whether its sequence-only generations, and its generations with each protein's own
prompt, are closer to their own protein. Writes prot2text_conflict_predictions.parquet and updates
prot2text_pairs.parquet.

    python input_use/models/prot2text_driver/score_B.py [--gen-dir $INPUT_USE_RESULTS_DIR/protein/prot2text_v2/runs/genB]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from scoring import Scorers  # noqa: E402
from score_A import load_generations  # noqa: E402


def follow(pref, seq_sign, eps):
    """seq_sign = +1 if a positive preference (s_A - s_B) means following the sequence source."""
    if abs(pref) < eps:
        return "ambiguous"
    return "sequence" if pref * seq_sign > 0 else "text"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", default=os.path.join(C.RUNS, "genB"))
    ap.add_argument("--epsilon", type=float, default=0.005)
    a = ap.parse_args()
    ex = pd.read_parquet(os.path.join(C.RESULTS, "prot2text_examples.parquet")).set_index("protein_id")
    pairs = pd.read_parquet(os.path.join(C.RESULTS, "prot2text_pairs.parquet"))
    predA = pd.read_parquet(os.path.join(C.RESULTS, "prot2text_shuffle_predictions.parquet"))
    nat = predA[predA.condition == "native_full"].set_index("protein_id")
    seq = predA[predA.condition == "seq_only_intact"].set_index("protein_id")
    genB = load_generations(a.gen_dir)
    reqB = pd.DataFrame(C.read_jsonl(os.path.join(C.RUNS, "requests_B.jsonl")))
    gb = reqB.merge(genB, on="request_id", how="left")
    print(f"B requests {len(reqB)}, generations {len(genB)}, missing {gb.generation.isna().sum()}", flush=True)
    gb = gb[~gb.generation.isna()].set_index("request_id")

    rows = []
    for r in pairs.itertuples(index=False):
        specs = [("aligned_A", r.protein_A, r.protein_A, nat.loc[r.protein_A] if r.protein_A in nat.index else None),
                 ("aligned_B", r.protein_B, r.protein_B, nat.loc[r.protein_B] if r.protein_B in nat.index else None),
                 ("seqonly_A", r.protein_A, None, seq.loc[r.protein_A] if r.protein_A in seq.index else None),
                 ("seqonly_B", r.protein_B, None, seq.loc[r.protein_B] if r.protein_B in seq.index else None)]
        for cond, seq_src, name_src in (("Bseq_Aname", r.protein_B, r.protein_A), ("Aseq_Bname", r.protein_A, r.protein_B)):
            rid = C.request_id(cond, -1, r.pair_id)
            specs.append((cond, seq_src, name_src, gb.loc[rid] if rid in gb.index else None))
        for cond, seq_src, name_src, g in specs:
            if g is None:
                continue
            rows.append({"pair_id": r.pair_id, "condition": cond, "sequence_source": seq_src, "name_source": name_src,
                         "generation": g["generation"], "n_new_tokens": int(g["n_new_tokens"]),
                         "F_A": ex.loc[r.protein_A, "function_text"], "F_B": ex.loc[r.protein_B, "function_text"]})
    df = pd.DataFrame(rows)
    sc = Scorers(device="cuda")
    for name in ("biobert", "roberta"):
        _, _, fa = sc.bertscore(name, df.generation.tolist(), df.F_A.tolist())
        _, _, fb = sc.bertscore(name, df.generation.tolist(), df.F_B.tolist())
        df[f"{name}_score_to_A"], df[f"{name}_score_to_B"] = fa, fb
    df["score_to_A"], df["score_to_B"] = df.biobert_score_to_A, df.biobert_score_to_B
    df["preference_A_minus_B"] = df.score_to_A - df.score_to_B
    df["roberta_preference_A_minus_B"] = df.roberta_score_to_A - df.roberta_score_to_B
    df["direction"] = df.condition.map({"Bseq_Aname": "seqB_nameA", "Aseq_Bname": "seqA_nameB"}).fillna("")
    # sequence source A -> a positive preference is sequence-following; sequence source B ->
    # negative
    df["seq_sign"] = np.where(df.condition.isin(["Aseq_Bname"]), 1, np.where(df.condition.isin(["Bseq_Aname"]), -1, 0))
    df["follow_class"] = [follow(p, s, a.epsilon) if s else "" for p, s in zip(df.preference_A_minus_B, df.seq_sign)]
    df["follow_class_eps0"] = [follow(p, s, 0.0) if s else "" for p, s in zip(df.preference_A_minus_B, df.seq_sign)]
    df["roberta_follow_class"] = [follow(p, s, a.epsilon) if s else "" for p, s in zip(df.roberta_preference_A_minus_B, df.seq_sign)]
    # oriented preference: positive = towards the sequence source's function
    df["pref_toward_sequence"] = df.preference_A_minus_B * df.seq_sign
    df.drop(columns=["F_A", "F_B"]).to_parquet(os.path.join(C.RESULTS, "prot2text_conflict_predictions.parquet"), index=False)

    pv = df.pivot(index="pair_id", columns="condition", values="preference_A_minus_B")
    pairs = pairs.set_index("pair_id")
    pairs["seqonly_A_pref"], pairs["seqonly_B_pref"] = pv.get("seqonly_A"), pv.get("seqonly_B")
    pairs["aligned_A_pref"], pairs["aligned_B_pref"] = pv.get("aligned_A"), pv.get("aligned_B")
    pairs["sequence_gate_pass"] = (pairs.seqonly_A_pref > 0) & (pairs.seqonly_B_pref < 0)
    pairs["aligned_gate_pass"] = (pairs.aligned_A_pref > 0) & (pairs.aligned_B_pref < 0)
    pairs["has_both_conflicts"] = pv.get("Bseq_Aname").notna() & pv.get("Aseq_Bname").notna()
    pairs.reset_index().to_parquet(os.path.join(C.RESULTS, "prot2text_pairs.parquet"), index=False)
    print(f"pairs {len(pairs)}: sequence gate {int(pairs.sequence_gate_pass.sum())}, aligned gate "
          f"{int(pairs.aligned_gate_pass.sum())}, both conflicts generated {int(pairs.has_both_conflicts.sum())}", flush=True)
    g = df[df.seq_sign != 0]
    print(g.groupby("condition").follow_class.value_counts(), flush=True)


if __name__ == "__main__":
    main()

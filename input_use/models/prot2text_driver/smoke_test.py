#!/usr/bin/env python
"""Checks the Prot2Text-V2 model setup before the evaluation runs.

Verifies tokenization (a name change alters only the prompt tokens, a shuffle alters only the
protein tokens, and the prompt holds one placeholder token per residue plus two), generates the
model card example, and generates descriptions for up to 64 test proteins under each Llama chat
template, comparing batched with single generation and scoring BERTScore and ROUGE-L. Writes
smoke_<template>.jsonl and smoke_summary.json to --out.

    python input_use/models/prot2text_driver/smoke_test.py --n 64 --out $INPUT_USE_RESULTS_DIR/protein/prot2text_v2/runs/smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import generate as G  # noqa: E402

EXAMPLE_SEQ = (
    "MCYSANGNTFLIVDNTQKRIPEEKKPDFVRENVGDLDGVIFVELVDGKYFMDYYNRDGSMAAFCGNGARAFSQ"
    "YLIDRGWIKEKEFTFLSRAGEIKVIVDDSIWVRMPGVSEKKEMKVDGYEGYFVVVGVPHFVMEVKGIDELDVE"
    "KLGRDLRYKTGANVDFYEVLPDRLKVRTYERGVERETKACGTGVTSVFVVYRDKTGAKEVKIQVPGGTLFLKE"
    "ENGEIFLRGDVKRCSEE"
)


def model_card_example(model, llama_tok, esm_tok):
    """The readme example, character for character (user message without name/taxon fields)."""
    placeholder = "<|reserved_special_token_1|>"
    user_message = "Sequence embeddings: " + placeholder * (len(EXAMPLE_SEQ) + 2)
    tokenized_prompt = llama_tok.apply_chat_template(
        [{"role": "system", "content": C.SYSTEM_MESSAGE}, {"role": "user", "content": user_message}],
        add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True)
    tokenized_sequence = esm_tok(EXAMPLE_SEQ, return_tensors="pt")
    with torch.no_grad():
        generated = model.generate(
            inputs=tokenized_prompt["input_ids"].to(model.device),
            attention_mask=tokenized_prompt["attention_mask"].to(model.device),
            protein_input_ids=tokenized_sequence["input_ids"].to(model.device),
            protein_attention_mask=tokenized_sequence["attention_mask"].to(model.device),
            max_new_tokens=1024, eos_token_id=128009, pad_token_id=128002,
            return_dict_in_generate=False, num_beams=4, do_sample=False)
    return llama_tok.decode(generated[0], skip_special_tokens=True)


def invariances(esm_tok, llama_tok, ex):
    a, b = ex.iloc[0], ex.iloc[1]
    p1, s1 = C.tokenize_request(esm_tok, llama_tok, a.full_name, a.taxon, a.sequence)
    p1b, s1b = C.tokenize_request(esm_tok, llama_tok, a.full_name, a.taxon, a.sequence)
    p2, s2 = C.tokenize_request(esm_tok, llama_tok, a.full_name, a.taxon, b.sequence)
    p3, s3 = C.tokenize_request(esm_tok, llama_tok, "unknown", a.taxon, a.sequence)
    shuf = C.shuffle_sequence(a.sequence, 0, a.protein_id)
    p4, s4 = C.tokenize_request(esm_tok, llama_tok, a.full_name, a.taxon, shuf)
    out = {
        "same_sequence_same_protein_ids": bool(torch.equal(s1, s1b) and torch.equal(p1, p1b)),
        "different_sequence_different_protein_ids": bool(not torch.equal(s1, s2)),
        "name_change_leaves_protein_ids": bool(torch.equal(s1, s3)),
        "name_change_changes_prompt_ids": bool(not torch.equal(p1, p3)),
        "name_change_keeps_placeholder_count": int((p1 == C.PLACEHOLDER_ID).sum()) == int((p3 == C.PLACEHOLDER_ID).sum()),
        "shuffle_changes_protein_ids": bool(not torch.equal(s1, s4)),
        "shuffle_keeps_prompt_ids": bool(torch.equal(p1, p4)),
        "placeholder_count_equals_protein_tokens": int((p1 == C.PLACEHOLDER_ID).sum()) == s1.numel() == len(a.sequence) + 2,
        "prompt_decoded_head": llama_tok.decode(p1[:80]),
    }
    assert all(v for k, v in out.items() if isinstance(v, bool)), out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--out", default=os.path.join(C.RUNS, "smoke"))
    ap.add_argument("--templates", nargs="+", default=["llama31", "llama3_old"])
    ap.add_argument("--batch-size", type=int, default=12)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ex = pd.read_parquet(os.path.join(C.RESULTS, "prot2text_examples.parquet"))
    ex = ex[ex.within_1021 & ~ex.full_name_missing].head(a.n).reset_index(drop=True)
    esm_tok, llama_tok = C.load_tokenizers("llama31")
    summary = {"invariances": invariances(esm_tok, llama_tok, ex), "versions": C.versions(), "n": len(ex)}
    print(json.dumps(summary["invariances"], indent=1), flush=True)

    t0 = time.time()
    model = G.load_model()
    summary["load_seconds"] = time.time() - t0
    summary["gpu"] = torch.cuda.get_device_name(0)
    print(f"loaded in {time.time() - t0:.0f}s on {summary['gpu']}", flush=True)

    t0 = time.time()
    summary["model_card_example"] = model_card_example(model, llama_tok, esm_tok)
    summary["model_card_example_seconds"] = time.time() - t0
    print("MODEL CARD EXAMPLE:", summary["model_card_example"], flush=True)

    from scoring import Scorers, rouge_l
    scorers = Scorers(device="cuda")
    for tpl in a.templates:
        esm_tok, llama_tok = C.load_tokenizers(tpl)
        torch.manual_seed(C.GEN_SEED)
        reqs = []
        for r in ex.itertuples(index=False):
            p, s = C.tokenize_request(esm_tok, llama_tok, r.full_name, r.taxon, r.sequence)
            reqs.append({"request_id": C.request_id("native_full", -1, r.protein_id), "prompt_ids": p,
                         "prot_ids": s, "prompt_sha": "", "protein_id": r.protein_id})
        t0 = time.time()
        res = []
        order = sorted(range(len(reqs)), key=lambda i: -reqs[i]["prompt_ids"].numel())
        for i in range(0, len(order), a.batch_size):
            batch = [reqs[j] for j in order[i:i + a.batch_size]]
            res += G.generate_batch(model, llama_tok, batch, esm_tok.pad_token_id, C.GEN_KWARGS)
        gen_s = time.time() - t0
        by_id = {r["request_id"]: r for r in res}
        # first 8 one at a time: does batching change the text?
        single = {}
        for j in range(min(8, len(reqs))):
            single[reqs[j]["request_id"]] = G.generate_batch(model, llama_tok, [reqs[j]], esm_tok.pad_token_id,
                                                             C.GEN_KWARGS)[0]["generation"]
        same = sum(single[k] == by_id[k]["generation"] for k in single)
        refs = ex.set_index("protein_id").loc[[r["protein_id"] for r in reqs], "function_text"].tolist()
        cands = [by_id[r["request_id"]]["generation"] for r in reqs]
        rows = []
        sc = {}
        for name in ("biobert", "roberta"):
            P, R, F = scorers.bertscore(name, cands, refs)
            sc[name] = F
        rl = rouge_l(cands, refs)
        for k, r in enumerate(reqs):
            rows.append({**by_id[r["request_id"]], "protein_id": r["protein_id"], "template": tpl,
                         "biobert_f1": float(sc["biobert"][k]), "roberta_f1": float(sc["roberta"][k]),
                         "rougeL": float(rl[k]), "reference": refs[k]})
        with open(os.path.join(a.out, f"smoke_{tpl}.jsonl"), "w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        summary[tpl] = {"n": len(rows), "gen_seconds": gen_s, "s_per_gen": gen_s / len(rows),
                        "biobert_f1_mean": float(np.mean(sc["biobert"])), "roberta_f1_mean": float(np.mean(sc["roberta"])),
                        "rougeL_mean": float(np.mean(rl)), "mean_new_tokens": float(np.mean([r["n_new_tokens"] for r in res])),
                        "n_hit_max_new_tokens": int(sum(not r["ended_with_eos"] for r in res)),
                        "n_empty": int(sum(len(r["generation"].strip()) == 0 for r in res)),
                        "single_vs_batched_identical": f"{same}/{len(single)}",
                        "max_mem_gb": torch.cuda.max_memory_allocated() / 1024**3,
                        "examples": [{"protein_id": rows[k]["protein_id"], "generation": rows[k]["generation"][:400],
                                      "reference": rows[k]["reference"][:400]} for k in range(3)]}
        print(json.dumps({tpl: summary[tpl]}, indent=1), flush=True)
    C.write_json(summary, os.path.join(a.out, "smoke_summary.json"))
    print("SMOKE DONE", flush=True)


if __name__ == "__main__":
    main()

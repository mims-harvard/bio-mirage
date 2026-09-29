#!/usr/bin/env python
"""Checks the ChatNT model setup on the PyTorch (Hugging Face) model used for the evaluation.

Checks that the notebook example is answered Yes, that DNA tokens are identical for identical
sequences and change under a shuffle while prompt tokens stay fixed, that the Yes/No log-likelihoods
read from the first answer position match teacher-forced scoring within 0.05, that a query's Yes/No
difference is unchanged when it is batched, and that likelihood accuracy on up to 100 unmodified
queries per task is at least 0.6. Writes smoke_torch.json and exits 1 if any check fails.

    python input_use/models/chatnt_driver/smoke_torch.py [--n_integrity 100] [--batch_size 16]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chatnt_common import (RESULTS, NOTEBOOK_ENGLISH, NOTEBOOK_DNA, NOTEBOOK_EXPECTED, TASK_ORDER,  # noqa: E402
                           TASKS, MODEL_REV, DATA_REV, ChatNTTorch, build_prompt, sha, shuffle_mono,
                           rng_for, parse_generation_label, package_versions, repo_commit)

README_EXPECTED = "Yes, an acceptor splice site is without question present in the sequence."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_integrity", type=int, default=100)
    ap.add_argument("--n_gen_print", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new", type=int, default=24)
    a = ap.parse_args()
    t0 = time.time()
    m = ChatNTTorch()
    torch = m.torch
    out = {"versions": package_versions(), "torch_cuda": torch.version.cuda,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "dtype": str(m.dtype), "model_revision": MODEL_REV, "data_revision": DATA_REV,
           "repo_commit": repo_commit(), "loading_info": m.loading_info,
           "answer_token_ids": m.answer_token_ids, "answer_token_ids_nospace": m.answer_token_ids_nospace,
           "answer_tokens": {k: m.english_tokenizer.convert_ids_to_tokens(v) for k, v in m.answer_token_ids.items()},
           "checks": {}}
    print(f"loaded in {time.time() - t0:.0f}s on {out['gpu']}; answer ids {out['answer_token_ids']} "
          f"{out['answer_tokens']}; loading_info {m.loading_info}", flush=True)
    failures = []

    def check(name, ok, detail=None):
        out["checks"][name] = {"ok": bool(ok), "detail": detail}
        print(f"[{'ok' if ok else 'FAIL'}] {name}: {detail}", flush=True)
        if not ok:
            failures.append(name)

    # 1. notebook / model-card example
    r = m.run_batch([NOTEBOOK_ENGLISH], [NOTEBOOK_DNA], max_new_tokens=20)[0]
    r["generation_token_ids"] = list(r["generation_token_ids"])
    r.update({"expected_notebook": NOTEBOOK_EXPECTED, "expected_readme": README_EXPECTED,
              "parsed_label": parse_generation_label(r["raw_generation"]),
              "margin": r["L_yes"] - r["L_no"]})
    out["notebook_example"] = r
    check("notebook_example_label_yes", r["parsed_label"] == "Yes",
          {"generated": r["raw_generation"], "margin": r["margin"],
           "exact_notebook": r["raw_generation"].strip() == NOTEBOOK_EXPECTED,
           "exact_readme": r["raw_generation"].strip() == README_EXPECTED})

    # 3 / 4. bio-token hashes
    ex = pd.read_parquet(os.path.join(RESULTS, "chatnt_examples.parquet"))
    e0 = ex.iloc[0]
    h_same = [m.bio_hash(e0.sequence_A) for _ in range(3)]
    h_shuf = m.bio_hash(shuffle_mono(e0.sequence_A, rng_for(0, e0.sample_id, 0)))
    h_nb = [m.bio_hash(NOTEBOOK_DNA), m.bio_hash(shuffle_mono(NOTEBOOK_DNA, rng_for(0, 0, 0)))]
    p0 = build_prompt(e0.question)
    eh = [sha(m.tokenize_english([p0]).numpy().tobytes()) for _ in range(2)]
    check("bio_tokens_identical_for_identical_dna", len(set(h_same)) == 1, h_same[0][:16])
    check("bio_tokens_change_when_dna_shuffled", h_shuf != h_same[0] and h_nb[0] != h_nb[1],
          {"intact": h_same[0][:16], "shuffled": h_shuf[:16]})
    check("english_tokens_identical_across_conditions", eh[0] == eh[1], eh[0][:16])
    out["token_counts_example0"] = {"prompt_tokens": m.prompt_len(p0), "bio_tokens": m.bio_len(e0.sequence_A),
                                    "sequence_length": int(e0.sequence_length_A)}

    # fast path == general teacher-forced path (3 examples)
    diffs = []
    for _, e in ex.groupby("task").head(1).iterrows():
        p = build_prompt(e.question)
        fast = m.run_batch([p], [e.sequence_A], generate=False)[0]
        gy, ids_y = m.teacher_forced_logprob(p, e.sequence_A, " Yes")
        gn, ids_n = m.teacher_forced_logprob(p, e.sequence_A, " No")
        diffs.append({"example_id": e.example_id, "fast_yes": fast["L_yes"], "general_yes": gy,
                      "fast_no": fast["L_no"], "general_no": gn, "ids": [ids_y, ids_n]})
    mx = max(max(abs(d["fast_yes"] - d["general_yes"]), abs(d["fast_no"] - d["general_no"])) for d in diffs)
    check("single_token_fast_path_matches_teacher_forcing", mx < 0.05, {"max_abs_diff": mx, "rows": diffs})

    # batch invariance (a row alone vs inside a batch of 8)
    sub = ex.head(8)
    prompts = [build_prompt(q) for q in sub.question]
    alone = m.run_batch(prompts[:1], list(sub.sequence_A)[:1], generate=False)[0]
    batched = m.run_batch(prompts, list(sub.sequence_A), generate=False)[0]
    dm = abs((alone["L_yes"] - alone["L_no"]) - (batched["L_yes"] - batched["L_no"]))
    check("batch_invariance_margin", dm < 0.1 and alone["first_token_argmax"] == batched["first_token_argmax"],
          {"alone": alone, "batched": batched, "abs_margin_diff": dm})

    # 5. integrity: n unmodified examples per task
    integ, gens = {}, []
    for task in TASK_ORDER:
        sub = ex[ex.task == task].head(a.n_integrity)
        rows = []
        for i in range(0, len(sub), a.batch_size):
            b = sub.iloc[i:i + a.batch_size]
            rr = m.run_batch([build_prompt(q) for q in b.question], list(b.sequence_A), max_new_tokens=a.max_new)
            for (_, e), r in zip(b.iterrows(), rr):
                r = {**r, "example_id": e.example_id, "gold": e.gold_A,
                     "pred_lik": "Yes" if r["L_yes"] - r["L_no"] > 0 else "No",
                     "pred_gen": parse_generation_label(r["raw_generation"])}
                del r["generation_token_ids"]
                rows.append(r)
        df = pd.DataFrame(rows)
        integ[task] = {"n": int(len(df)),
                       "acc_likelihood": float((df.pred_lik == df.gold).mean()),
                       "acc_generation": float((df.pred_gen == df.gold).mean()),
                       "n_generation_unparsed": int(df.pred_gen.isna().sum()),
                       "likelihood_generation_agreement": float((df.pred_lik == df.pred_gen).mean()),
                       "first_token_is_yes_or_no": float(df.first_token_argmax.isin(
                           [m.answer_token_ids["Yes"][0], m.answer_token_ids["No"][0]]).mean()),
                       "mean_margin_yes": float(df[df.gold == "Yes"].eval("L_yes - L_no").mean()),
                       "mean_margin_no": float(df[df.gold == "No"].eval("L_yes - L_no").mean()),
                       "rows": rows}
        print(f"{task}: n={len(df)} acc_lik={integ[task]['acc_likelihood']:.3f} "
              f"acc_gen={integ[task]['acc_generation']:.3f} unparsed={integ[task]['n_generation_unparsed']}",
              flush=True)
        check(f"integrity_accuracy_above_chance_{task}", integ[task]["acc_likelihood"] >= 0.6,
              integ[task]["acc_likelihood"])
        k = a.n_gen_print // 3 + (1 if TASK_ORDER.index(task) < a.n_gen_print % 3 else 0)
        gens += [{"task": task, "example_id": r["example_id"], "gold": r["gold"],
                  "margin": r["L_yes"] - r["L_no"], "generation": r["raw_generation"]} for r in rows[:k]]
    out["integrity"] = integ
    out["raw_generations_for_inspection"] = gens
    print("\n=== raw generations (task | gold | margin | generation)")
    for g in gens:
        print(f"{TASKS[g['task']]:16s} | {g['gold']:3s} | {g['margin']:+7.3f} | {g['generation']!r}")
    out["seconds_total"] = time.time() - t0
    out["failures"] = failures
    p = os.path.join(RESULTS, "smoke_torch.json")
    with open(p, "w") as fh:
        json.dump(out, fh, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("wrote", p, "| failures:", failures)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

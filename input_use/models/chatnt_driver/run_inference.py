#!/usr/bin/env python
"""Runs ChatNT on every prompt of one task in conditions.parquet and writes the predictions in
resumable chunks.

For each prompt it stores the log-likelihoods of " Yes" and " No", their difference and the
resulting label, and a greedy generation with its parsed label. Writes predictions/<task>.parquet
and run_manifest_<task>.json under dna/chatnt/results under $INPUT_USE_RESULTS_DIR.

    python input_use/models/chatnt_driver/run_inference.py --task NT_splice_sites_donors [--batch_size 16] [--chunk 256]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chatnt_common import (RESULTS, TASKS, MODEL_ID, MODEL_REV, DATA_ID, DATA_REV, REPO_URL,  # noqa: E402
                           ENGLISH_MAX_LENGTH, BIO_MAX_LENGTH, MAX_NEW_TOKENS, CONTEXT, ChatNTTorch,
                           parse_generation_label, package_versions, repo_commit)

OUT_COLS = ["example_id", "task", "condition", "shuffle_seed", "question_hash", "sequence_hash", "L_yes",
            "L_no", "margin", "predicted_label", "gold_source_label", "donor_label", "raw_generation",
            "parsed_generation_label", "donor_id", "degenerate", "L_yes_nospace", "L_no_nospace",
            "first_token_argmax", "n_english_tokens", "n_bio_tokens", "generation_token_ids",
            "batch_index", "chunk_index"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--limit", type=int, default=None, help="debug: only the first N prompts")
    ap.add_argument("--reverse", action="store_true",
                    help="process the chunks last-to-first, so a second job on the same task meets this one in the middle")
    a = ap.parse_args()
    t0 = time.time()
    cond = pd.read_parquet(os.path.join(RESULTS, "conditions.parquet"))
    cond = cond[cond.task == a.task].reset_index(drop=True)
    if a.limit:
        cond = cond.head(a.limit)
    out_dir = os.path.join(RESULTS, "predictions", a.task)
    os.makedirs(out_dir, exist_ok=True)
    n_chunks = (len(cond) + a.chunk - 1) // a.chunk
    def done_path(k):
        return os.path.join(out_dir, f"chunk_{k:04d}.parquet")
    todo = [k for k in range(n_chunks) if not os.path.exists(done_path(k))]
    if a.reverse:
        todo = todo[::-1]
    print(f"{a.task}: {len(cond)} prompts, {n_chunks} chunks, {len(todo)} to do{' (reverse)' if a.reverse else ''}", flush=True)
    m = ChatNTTorch()
    torch = m.torch
    print(f"model loaded in {time.time() - t0:.0f}s on {torch.cuda.get_device_name(0)}", flush=True)
    t_inf = time.time()
    n_done = 0
    n_skipped = 0
    for k in todo:
        if os.path.exists(done_path(k)):          # written by the other job on this task meanwhile
            n_skipped += 1
            continue
        rows = []
        sub = cond.iloc[k * a.chunk:(k + 1) * a.chunk]
        for i in range(0, len(sub), a.batch_size):
            b = sub.iloc[i:i + a.batch_size]
            rr = m.run_batch(list(b.prompt), list(b.sequence), max_new_tokens=a.max_new_tokens)
            for (_, c), r in zip(b.iterrows(), rr):
                margin = r["L_yes"] - r["L_no"]
                rows.append({
                    "example_id": c.example_id, "task": c.task, "condition": c.condition,
                    "shuffle_seed": None if pd.isna(c.shuffle_seed) else int(c.shuffle_seed),
                    "question_hash": c.question_hash, "sequence_hash": c.sequence_hash,
                    "L_yes": r["L_yes"], "L_no": r["L_no"], "margin": margin,
                    "predicted_label": "Yes" if margin > 0 else "No",
                    "gold_source_label": c.gold_source_label, "donor_label": c.donor_label,
                    "raw_generation": r["raw_generation"],
                    "parsed_generation_label": parse_generation_label(r["raw_generation"]),
                    "donor_id": c.donor_id, "degenerate": bool(c.degenerate),
                    "L_yes_nospace": r["L_yes_nospace"], "L_no_nospace": r["L_no_nospace"],
                    "first_token_argmax": r["first_token_argmax"],
                    "n_english_tokens": r["n_english_tokens"], "n_bio_tokens": r["n_bio_tokens"],
                    "generation_token_ids": json.dumps(r["generation_token_ids"]),
                    "batch_index": i // a.batch_size, "chunk_index": k,
                })
        df = pd.DataFrame(rows, columns=OUT_COLS)
        df["shuffle_seed"] = df["shuffle_seed"].astype("Int64")
        tmp = os.path.join(out_dir, f"chunk_{k:04d}.parquet.{os.getpid()}.tmp")
        df.to_parquet(tmp, index=False)
        if os.path.exists(done_path(k)):          # lost the race on this chunk: keep the first writer's file
            os.remove(tmp)
            n_skipped += 1
        else:
            os.replace(tmp, done_path(k))
        n_done += len(df)
        el = time.time() - t_inf
        print(f"chunk {k + 1}/{n_chunks} done: {n_done} prompts in {el:.0f}s "
              f"({el / max(n_done, 1):.2f} s/prompt); eta {(len(todo) * a.chunk - n_done) * el / max(n_done, 1) / 60:.0f} min",
              flush=True)
    parts = sorted(glob.glob(os.path.join(out_dir, "chunk_[0-9][0-9][0-9][0-9].parquet")))
    if len(parts) < n_chunks:
        print(f"{a.task}: {len(parts)}/{n_chunks} chunks present after this job ({n_skipped} skipped); "
              f"the job that writes the last chunk merges. done in {time.time() - t0:.0f}s")
        return
    assert len(parts) == n_chunks, (len(parts), n_chunks)
    merged = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    assert len(merged) == len(cond), (len(merged), len(cond))
    assert (merged.example_id.to_numpy() == cond.example_id.to_numpy()).all()
    assert (merged.sequence_hash.to_numpy() == cond.sequence_hash.to_numpy()).all()
    merged.to_parquet(os.path.join(RESULTS, "predictions", f"{a.task}.parquet"), index=False)
    manifest = {
        "task": a.task, "task_name": TASKS[a.task], "n_prompts": int(len(merged)),
        "n_prompts_run_this_job": int(n_done), "n_chunks_skipped_this_job": int(n_skipped),
        "seconds_inference_this_job": time.time() - t_inf, "reverse": a.reverse,
        "model": {"id": MODEL_ID, "revision": MODEL_REV, "implementation": "HF custom code chatNT.py (TorchMultiOmicsModel)",
                  "dtype": str(m.dtype), "loading_info": m.loading_info},
        "data": {"id": DATA_ID, "revision": DATA_REV},
        "official_repo": {"url": REPO_URL, "commit": repo_commit()},
        "inference": {"context_prompt": CONTEXT, "english_max_length": ENGLISH_MAX_LENGTH,
                      "bio_max_length": BIO_MAX_LENGTH, "batch_size": a.batch_size, "chunk": a.chunk,
                      "likelihood": "log-softmax of the first answer token at the last prompt position; no sampling",
                      "answer_token_ids": m.answer_token_ids, "answer_token_ids_nospace": m.answer_token_ids_nospace,
                      "generation": f"greedy (argmax), max {a.max_new_tokens} new tokens, stop at </s>, "
                                    "projected DNA embeddings cached from the first step (released pipeline loop)"},
        "n_english_tokens": {"min": int(merged.n_english_tokens.min()), "max": int(merged.n_english_tokens.max())},
        "n_bio_tokens": {"min": int(merged.n_bio_tokens.min()), "max": int(merged.n_bio_tokens.max())},
        "gpu": torch.cuda.get_device_name(0), "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "python": platform.python_version(), "hostname": platform.node(), "versions": package_versions(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(os.path.join(RESULTS, f"run_manifest_{a.task}.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"done {a.task}: {len(merged)} rows in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

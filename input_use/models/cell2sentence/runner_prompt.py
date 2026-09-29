#!/usr/bin/env python
"""Runs C2S-Scale 2B or 27B on examples whose full prompt is stored in the example
(`payload["prompt"]`), such as the rationale queries, and writes records.jsonl.

A generation is mapped onto the option list when the example has one and kept as free text
otherwise. Decoding is greedy with eager attention, and the 27B model is split across the visible
GPUs. `--n_shards` splits the examples across jobs, each writing the records file it is given.

    python -m input_use.models.cell2sentence.runner_prompt --model_size 27b --examples <run>/examples.jsonl --records_out <run>/records.jsonl
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from input_use.core.records import read_examples, record_from_example, append_jsonl, read_jsonl
from input_use.metrics.mcq import parse_choice
from input_use.core.checkpoints import ckpt

REPOS = {"2b": ckpt("c2s_2b"), "27b": ckpt("c2s_27b")}


def load_model(repo, size, reserve_gib=14):
    """`reserve_gib` is per-device headroom withheld from the weight placement."""
    tok = AutoTokenizer.from_pretrained(repo)
    kw = dict(dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="eager")
    if size == "27b":
        n = torch.cuda.device_count()
        max_memory = {}
        for i in range(n):
            tot = torch.cuda.get_device_properties(i).total_memory / 2**30
            max_memory[i] = f"{max(6, int(tot - reserve_gib))}GiB"
        budget = sum(int(v[:-3]) for v in max_memory.values())
        print(f"[prompt-runner] {n} GPU(s), weight budget {max_memory} (total {budget} GiB); "
              f"27B bf16 needs ~55 GiB")
        if budget < 56:
            raise SystemExit(f"weight budget {budget} GiB < 56 GiB needed; device_map would offload "
                             f"to CPU and crawl. Request more or larger GPUs.")
        model = AutoModelForCausalLM.from_pretrained(repo, device_map="auto",
                                                     max_memory=max_memory, **kw)
    else:
        model = AutoModelForCausalLM.from_pretrained(repo, **kw).to(
            "cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    print(f"[prompt-runner] attn={getattr(model.config,'_attn_implementation','?')}")
    return tok, model


@torch.no_grad()
def generate(tok, model, prompt, max_new_tokens, ctx=None):
    enc = tok(prompt, return_tensors="pt").to(model.device)
    n_in = enc["input_ids"].shape[1]
    # Check before generating. eager attention materialises a (heads x L x L) fp32 buffer, so an
    # over-long prompt OOMs during prefill rather than reaching a post-hoc length assertion.
    if ctx and n_in + max_new_tokens > ctx:
        raise SystemExit(f"prompt {n_in} + {max_new_tokens} new exceeds context {ctx}")
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return n_in, tok.decode(out[0, n_in:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", required=True)
    ap.add_argument("--records_out", required=True)
    ap.add_argument("--model_size", choices=["2b", "27b"], default="2b")
    ap.add_argument("--experiment", default="text_probe")
    ap.add_argument("--max_new_tokens", type=int, default=24,
                    help="24 suits a bare label; free-text summaries need ~180")
    ap.add_argument("--conditions", default="")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1,
                    help="split examples round-robin across independent jobs; each writes its own "
                         "records file, so resume-by-record_id still works and the shards concatenate")
    ap.add_argument("--shard_group", type=int, default=1,
                    help="assign shards in consecutive blocks of this many examples instead of one "
                         "at a time. Set it to the number of conditions emitted per matched slot so "
                         "a slot's arms stay in ONE shard -- otherwise any paired analysis is "
                         "impossible until every shard finishes, which matters on a preemptible "
                         "partition where one shard may lag far behind.")
    a = ap.parse_args()

    repo, model_name = REPOS[a.model_size], f"c2s_scale_gemma2_{a.model_size}"
    examples = read_examples(a.examples)
    if a.conditions:
        want = {c.strip() for c in a.conditions.split(",") if c.strip()}
        examples = [e for e in examples if e.condition in want]

    if a.n_shards > 1:
        g = max(1, a.shard_group)
        examples = [e for i, e in enumerate(examples) if (i // g) % a.n_shards == a.shard]
        print(f"[prompt-runner] shard {a.shard}/{a.n_shards} (group={g}): {len(examples)} examples")

    out_path = Path(a.records_out)

    # Refuse to start if another live job is already appending to this file.
    lock = out_path.with_suffix(out_path.suffix + ".lock")
    me = os.environ.get("SLURM_JOB_ID", f"pid{os.getpid()}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if lock.exists():
        holder = lock.read_text().strip()
        alive = (os.popen(f"squeue -h -j {holder} -o %i 2>/dev/null").read().strip()
                 if holder.isdigit() else "")
        if holder == me:
            # a preempted-and-requeued job keeps its id: the lock is our own earlier incarnation,
            print(f"[prompt-runner] lock held by this job id {me} (requeue); continuing")
            alive = ""
        if alive:
            raise SystemExit(f"{out_path.name} is already being written by job {holder}; refusing to "
                             f"start a second writer. Cancel one, then repair the file if it already "
                             f"contains duplicates.")
        print(f"[prompt-runner] clearing stale lock from job {holder} (no longer running)")
    lock.write_text(me)

    done = set()
    if out_path.exists():
        done = {r["record_id"] for r in read_jsonl(out_path)}
        print(f"[prompt-runner] resuming: {len(done)} records present")
    print(f"[prompt-runner] {len(examples)} examples; loading {repo}")
    tok, model = load_model(repo, a.model_size)
    ctx = getattr(model.config, "max_position_embeddings", None)

    n = 0
    for k, ex in enumerate(examples):
        rid = f"{a.experiment}::{model_name}::{ex.example_id}::{ex.condition}"
        if rid in done:
            continue
        prompt = ex.payload["prompt"]
        n_in, raw = generate(tok, model, prompt, a.max_new_tokens, ctx)
        # Free-text tasks (Experiment 5's abstract summaries) carry no option list.
        opts = ex.payload.get("options") or []
        parsed = parse_choice(raw, opts) if opts else {"cell_type": raw.strip(),
                                                       "in_options": None, "match_kind": "free_text"}
        rec = record_from_example(
            ex, a.experiment, model_name,
            input={"prompt": prompt, "options": opts, "n_options": len(opts),
                   "num_genes": ex.payload.get("num_genes"),
                   "n_cells": ex.payload.get("n_cells"),
                   "stated_cell_type": ex.payload.get("stated_cell_type"),
                   "n_prompt_tokens": int(n_in)},
            output={"raw": raw, "parsed": parsed})
        append_jsonl(rec, out_path)
        n += 1
        if n % 100 == 0:
            print(f"  {n} run ({k+1}/{len(examples)} seen)", flush=True)
    print(f"[prompt-runner] DONE: {n} new records -> {out_path}")


if __name__ == "__main__":
    main()

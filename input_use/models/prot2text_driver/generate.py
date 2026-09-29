#!/usr/bin/env python
"""Runs the released Prot2Text-V2 model on a generation request file, with sharding and resume.

Uses the released generation settings (beam search with 4 beams, up to 1,024 new tokens) and writes
one generations JSONL file and one manifest per shard to --out.

    python input_use/models/prot2text_driver/generate.py --requests $INPUT_USE_RESULTS_DIR/protein/prot2text_v2/runs/requests_A.jsonl --out $INPUT_USE_RESULTS_DIR/protein/prot2text_v2/runs/genA --shard 0 --nshards 4 [--template llama31] [--batch-size 12] [--limit N] [--ids file]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


def load_model(device="cuda"):
    from transformers import AutoModelForCausalLM
    path = C.snapshot(C.MODEL_ID, C.MODEL_REV)
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True,
                                                 torch_dtype=torch.bfloat16, device_map=device)
    model.eval()
    return model


def pad_batch(items, pad_id_prompt, pad_id_prot, device):
    """items: list of (prompt_ids, prot_ids). Prompts left-padded, proteins right-padded."""
    Lp = max(p.numel() for p, _ in items)
    Ls = max(s.numel() for _, s in items)
    n = len(items)
    ids = torch.full((n, Lp), pad_id_prompt, dtype=torch.long)
    am = torch.zeros((n, Lp), dtype=torch.long)
    pid = torch.full((n, Ls), pad_id_prot, dtype=torch.long)
    pam = torch.zeros((n, Ls), dtype=torch.long)
    for i, (p, s) in enumerate(items):
        ids[i, Lp - p.numel():] = p
        am[i, Lp - p.numel():] = 1
        pid[i, :s.numel()] = s
        pam[i, :s.numel()] = 1
    return ids.to(device), am.to(device), pid.to(device), pam.to(device)


@torch.no_grad()
def generate_batch(model, llama_tok, batch, esm_pad_id, gen_kwargs):
    ids, am, pid, pam = pad_batch([(b["prompt_ids"], b["prot_ids"]) for b in batch],
                                  C.PAD_ID, esm_pad_id, model.device)
    out = model.generate(inputs=ids, attention_mask=am, protein_input_ids=pid,
                         protein_attention_mask=pam, **gen_kwargs)
    res = []
    for i, b in enumerate(batch):
        toks = out[i]
        keep = toks[toks != C.PAD_ID]
        ended = bool((keep == C.EOS_ID).any())
        text = llama_tok.decode(toks, skip_special_tokens=True)
        res.append({"request_id": b["request_id"], "generation": text, "n_new_tokens": int(keep.numel()),
                    "ended_with_eos": ended, "prompt_len": int(b["prompt_ids"].numel()),
                    "protein_len": int(b["prot_ids"].numel()), "prompt_sha": b["prompt_sha"],
                    "batch_size": len(batch)})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--template", default="llama31", choices=list(C.LLAMA_TOK))
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--token-budget", type=int, default=14000, help="max summed prompt tokens per batch")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids", default=None, help="file with request_ids to restrict to (one per line)")
    ap.add_argument("--max-new-tokens", type=int, default=None, help="override (smoke tests only)")
    ap.add_argument("--tag", default="", help="suffix for this copy's output file (a hedge copy on another partition)")
    ap.add_argument("--reverse", action="store_true", help="work the shard shortest-first so two copies meet in the middle")
    ap.add_argument("--sync-every", type=int, default=0, help="every N batches re-read the other copies' files and skip what they finished")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    out_path = os.path.join(a.out, f"generations_shard{a.shard:02d}of{a.nshards:02d}{a.tag}.jsonl")
    # every copy of this shard counts as done, whichever partition wrote it
    done = set()
    for p in glob.glob(os.path.join(a.out, f"generations_shard{a.shard:02d}of{a.nshards:02d}*.jsonl")):
        done |= {r["request_id"] for r in C.read_jsonl(p)}

    reqs = C.read_jsonl(a.requests)
    if a.ids:
        keep = {l.strip() for l in open(a.ids) if l.strip()}
        reqs = [r for r in reqs if r["request_id"] in keep]
    reqs = [r for i, r in enumerate(reqs) if i % a.nshards == a.shard]
    if a.limit:
        reqs = reqs[:a.limit]
    todo = [r for r in reqs if r["request_id"] not in done]
    print(f"shard {a.shard}/{a.nshards}: {len(reqs)} requests, {len(done)} done, {len(todo)} to do", flush=True)
    if not todo:
        return

    torch.manual_seed(C.GEN_SEED)
    torch.cuda.manual_seed_all(C.GEN_SEED)
    esm_tok, llama_tok = C.load_tokenizers(a.template)
    esm_pad_id = esm_tok.pad_token_id
    assert esm_pad_id == 1, esm_pad_id
    t0 = time.time()
    model = load_model()
    print(f"model loaded in {time.time() - t0:.0f}s; {torch.cuda.get_device_name(0)}; "
          f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB", flush=True)

    gen_kwargs = dict(C.GEN_KWARGS)
    if a.max_new_tokens:
        gen_kwargs["max_new_tokens"] = a.max_new_tokens

    # tokenise everything up front, then batch longest-first so an OOM shows in the first minute
    for r in todo:
        p, s = C.tokenize_request(esm_tok, llama_tok, r["name_field"], r["taxon_field"], r["sequence"])
        r["prompt_ids"], r["prot_ids"] = p, s
        r["prompt_sha"] = hashlib.sha256(p.numpy().tobytes()).hexdigest()[:16]
    todo.sort(key=lambda r: (-r["prompt_ids"].numel(), r["request_id"]))
    if a.reverse:
        todo.reverse()

    batches, cur, cur_tok = [], [], 0
    for r in todo:
        L = r["prompt_ids"].numel()
        if cur and (len(cur) >= a.batch_size or cur_tok + L > a.token_budget):
            batches.append(cur)
            cur, cur_tok = [], 0
        cur.append(r)
        cur_tok += L
    if cur:
        batches.append(cur)

    manifest = {"model": C.MODEL_ID, "model_revision": C.MODEL_REV, "esm_tokenizer": [C.ESM_ID, C.ESM_REV],
                "llama_tokenizer": list(C.LLAMA_TOK[a.template]) + [a.template], "gen_kwargs": gen_kwargs,
                "gen_seed": C.GEN_SEED, "batch_size": a.batch_size, "token_budget": a.token_budget,
                "device": torch.cuda.get_device_name(0), "versions": C.versions(),
                "requests_file": os.path.abspath(a.requests), "shard": a.shard, "nshards": a.nshards,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "restart_count": os.environ.get("SLURM_RESTART_COUNT"),
                "partition": os.environ.get("SLURM_JOB_PARTITION"), "tag": a.tag, "reverse": a.reverse}
    C.write_json(manifest, os.path.join(a.out, f"manifest_shard{a.shard:02d}of{a.nshards:02d}{a.tag}.json"))

    def others_done():
        d = set()
        for p in glob.glob(os.path.join(a.out, f"generations_shard{a.shard:02d}of{a.nshards:02d}*.jsonl")):
            if os.path.abspath(p) != os.path.abspath(out_path):
                d |= {r["request_id"] for r in C.read_jsonl(p)}
        return d

    n_done, t0, skipped = 0, time.time(), 0
    with open(out_path, "a") as fh:
        for bi, batch in enumerate(batches):
            tb = time.time()
            if a.sync_every and bi % a.sync_every == 0:
                od = others_done()
                before = len(batch)
                batch = [b for b in batch if b["request_id"] not in od]
                skipped += before - len(batch)
                if not batch:
                    remaining = [b for bb in batches[bi + 1:] for b in bb if b["request_id"] not in od]
                    if not remaining:
                        print(f"all remaining requests done by another copy; skipped {skipped}; stopping", flush=True)
                        break
                    continue
            try:
                res = generate_batch(model, llama_tok, batch, esm_pad_id, gen_kwargs)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"OOM on batch of {len(batch)} (max prompt {batch[0]['prompt_ids'].numel()}); "
                      f"retrying one at a time", flush=True)
                res = []
                for b in batch:
                    res += generate_batch(model, llama_tok, [b], esm_pad_id, gen_kwargs)
            for r in res:
                fh.write(json.dumps(r) + "\n")
            fh.flush()
            n_done += len(res)
            if bi % 10 == 0 or bi == len(batches) - 1:
                el = time.time() - t0
                print(f"batch {bi + 1}/{len(batches)} n={len(batch)} maxlen={batch[0]['prompt_ids'].numel()} "
                      f"{time.time() - tb:.1f}s | {n_done}/{len(todo)} done, {el / 60:.1f} min, "
                      f"{el / n_done:.2f} s/gen, mem {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

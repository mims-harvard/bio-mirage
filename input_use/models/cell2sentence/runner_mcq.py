#!/usr/bin/env python
"""Runs C2S-Scale 2B or 27B on multiple-choice cell type annotation queries from examples.jsonl and
writes records.jsonl, inside the Cell2Sentence environment.

The prompt is the C2S cell type prediction template with the option list inserted before the cell
sentence. Decoding is greedy, the model loads with eager attention (Gemma-2 needs it for logit
soft-capping), and each generation is mapped onto one option.

    python -m input_use.models.cell2sentence.runner_mcq --model_size 2b --examples <run>/examples.jsonl --records_out <run>/records.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from input_use.core.records import read_examples, record_from_example, append_jsonl, read_jsonl
from input_use.metrics.mcq import parse_choice

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from input_use.core.checkpoints import ckpt

REPOS = {"2b": ckpt("c2s_2b"), "27b": ckpt("c2s_27b")}

# Built from the C2S cell-type-prediction template's own opening sentence and answer cue, so the
# framing stays in-distribution for these instruction-tuned checkpoints; only the middle (the option
# block + the "copy one verbatim" instruction) is new.
MCQ_TEMPLATE = (
    "The following is a list of {num_genes} gene names ordered by descending expression level in a "
    "{organism} cell. Your task is to give the cell type which this cell belongs to based on its "
    "gene expression.\n"
    "Choose your answer from the following list of cell types:\n"
    "{option_block}\n"
    "Cell sentence: {cell_sentence}.\n"
    "The cell type corresponding to these genes, copied verbatim from the list above, is:"
)


def build_prompt(payload) -> str:
    opts = payload["options"]
    return MCQ_TEMPLATE.format(
        num_genes=payload["num_genes"], organism=payload["organism"],
        cell_sentence=payload["cell_sentence"],
        option_block="\n".join(f"- {o}" for o in opts))


def config_fingerprint(model_size: str, attn: str = "eager") -> str:
    """Short hash of everything that changes what the model is asked and how it is run."""
    import hashlib
    blob = f"{MCQ_TEMPLATE}|{model_size}|{attn}"
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def load_model(repo: str, size: str):
    """`attn_implementation="eager"` is required for Gemma-2, not a tuning choice."""
    tok = AutoTokenizer.from_pretrained(repo)
    kw = dict(dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="eager")
    if size == "27b":
        model = AutoModelForCausalLM.from_pretrained(repo, device_map="auto", **kw)
    else:
        model = AutoModelForCausalLM.from_pretrained(repo, **kw)
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    print(f"[c2s-mcq] attn={getattr(model.config,'_attn_implementation','?')} "
          f"softcap attn={getattr(model.config,'attn_logit_softcapping',None)} "
          f"final={getattr(model.config,'final_logit_softcapping',None)}")
    return tok, model


@torch.no_grad()
def generate(tok, model, prompt: str, max_new_tokens: int):
    """Greedy (temperature 0) decode, matching the paper's eval. Returns (n_prompt_tokens, text)."""
    enc = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.pad_token_id or tok.eos_token_id)
    n_in = enc["input_ids"].shape[1]
    return n_in, tok.decode(out[0, n_in:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", required=True)
    ap.add_argument("--records_out", required=True)
    ap.add_argument("--model_size", choices=["2b", "27b"], default="2b")
    ap.add_argument("--experiment", default="singlecell_c2s_mcq")
    ap.add_argument("--max_new_tokens", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0, help="if >0, only run the first N examples")
    ap.add_argument("--conditions", default="",
                    help="comma-separated condition filter (e.g. 'wt' for a clean accuracy read)")
    ap.add_argument("--no_resume", action="store_true")
    a = ap.parse_args()

    repo = REPOS[a.model_size]
    model_name = f"c2s_scale_gemma2_{a.model_size}"

    examples = read_examples(a.examples)
    if a.conditions:
        want = {c.strip() for c in a.conditions.split(",") if c.strip()}
        examples = [e for e in examples if e.condition in want]
    missing = [e.example_id for e in examples if "options" in e.payload][:1]
    if not any("options" in e.payload for e in examples):
        raise SystemExit("no example carries payload['options'] -- re-run prepare with "
                         "--multiple_choice (this runner is forced-choice only)")
    examples = [e for e in examples if "options" in e.payload]
    if a.limit > 0:
        examples = examples[: a.limit]

    fp = config_fingerprint(a.model_size)
    out_path = Path(a.records_out)
    done = set()
    if a.no_resume:
        out_path.unlink(missing_ok=True)
    elif out_path.exists():
        existing = read_jsonl(out_path)
        # Refuse to resume onto records produced under a different prompt/config. Without this the
        # run silently mixes two experiments in one file (see config_fingerprint).
        stale = 0
        for r in existing:
            got = (r.get("input") or {}).get("config_fingerprint")
            if got is None:
                pr = (r.get("input") or {}).get("prompt", "")
                i, j = pr.find("Choose your answer"), pr.find("Cell sentence:")
                if not (0 <= i < j):
                    stale += 1
            elif got != fp:
                stale += 1
        if stale:
            raise SystemExit(
                f"{out_path} holds {stale}/{len(existing)} records from a DIFFERENT prompt/config "
                f"(current fingerprint {fp}). Resuming would mix two experiments in one file.\n"
                f"Fix: python -m input_use.scripts.strip_stale_records --records {out_path} --apply")
        done = {r["record_id"] for r in existing}
        print(f"[c2s-mcq] resuming: {len(done)} records already present in {out_path} "
              f"(config fingerprint {fp} verified)")

    print(f"[c2s-mcq] {len(examples)} examples ({len(examples) - len(done)} to run); loading {repo}")
    tok, model = load_model(repo, a.model_size)
    # not tok.model_max_length -- the Gemma-2 tokenizer reports the int64 sentinel (1e18), so a
    # guard against it never fires. The real window is on the model config (8192 for Gemma-2).
    max_len = getattr(model.config, "max_position_embeddings", None)

    n_run = 0
    for n, ex in enumerate(examples):
        rid = f"{a.experiment}::{model_name}::{ex.example_id}::{ex.condition}"
        if rid in done:
            continue
        prompt = build_prompt(ex.payload)
        n_in, raw = generate(tok, model, prompt, a.max_new_tokens)
        if n_run == 0:
            print(f"[c2s-mcq] first prompt: {n_in} tokens (tokenizer model_max_length={max_len}), "
                  f"{len(ex.payload['options'])} options, "
                  f"{ex.payload['num_genes']} genes in the cell sentence")
            # A truncated prompt silently drops the tail of the cell sentence -- exactly the genes
            # K=1000 exists to include -- so fail loudly instead of reporting a corrupted result.
            if isinstance(max_len, int) and 0 < max_len < n_in:
                raise SystemExit(f"prompt ({n_in} tokens) exceeds the model window ({max_len}); "
                                 f"lower --n_genes in prepare or the option count")
        parsed = parse_choice(raw, ex.payload["options"])
        rec = record_from_example(
            ex, a.experiment, model_name,
            # `cell_sentence` is deliberately not duplicated here: it is embedded verbatim in
            # `prompt`, and at K=1000 it is 6.2 KB -- ~40% of every record, for a field recoverable
            # from the prompt with one split.
            input={"prompt": prompt,
                   "num_genes": ex.payload["num_genes"], "organism": ex.payload["organism"],
                   "options": ex.payload["options"], "n_options": len(ex.payload["options"]),
                   "n_prompt_tokens": int(n_in), "answer_mode": "multiple_choice",
                   "config_fingerprint": fp},
            output={"raw": raw, "parsed": parsed})
        append_jsonl(rec, out_path)
        n_run += 1
        if n_run % 25 == 0:
            print(f"  {n_run} run ({n+1}/{len(examples)} seen)", flush=True)
    print(f"[c2s-mcq] DONE: {n_run} new records -> {out_path}")


if __name__ == "__main__":
    main()

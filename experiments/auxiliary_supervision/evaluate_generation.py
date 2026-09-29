#!/usr/bin/env python
"""Generates BioReason reasoning traces and answers for a trained auxiliary supervision checkpoint on
held-out queries, with the DNA intact, shuffled, or replaced by another query's windows.

Loads the LoRA checkpoint into the BioReason DNA model (code taken from INPUT_USE_DNA_REPO), decodes
greedily, maps each answer to a disease label, and stores per query the windows given to Evo2, the
generation and the predicted label. Conditions are `wt` (intact), `shuffle` (both windows permuted),
`swap` (both windows of a random other query, the evidence conflict) and `same` (the intact input
decoded again). score_seeds.py expects `--split test` and the conditions wt, shuffle, swap and same.
Writes the JSON given by --out.
"""
from __future__ import annotations
import os
REPO_DIR = os.environ.get("INPUT_USE_HOME", ".")
import argparse, csv, json, os, random, re, sys, time
from collections import defaultdict
import torch

from input_use.core.checkpoints import ckpt
from input_use.core import paths as RD

sys.path.insert(0, os.environ.get("INPUT_USE_DNA_REPO", f"{REPO_DIR}/BioReason"))
from bioreason.models.dna_llm import DNALLMModel, get_target_modules   # noqa: E402
from bioreason.dataset.utils import truncate_dna                       # noqa: E402



def norm_label(text):
    if not text:
        return ""
    t = text.strip().lower().replace("\n", " ")
    t = re.sub(r"[^\w\s\-'/]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def extract_label(generation, labels_sorted):
    """The test phase's matcher: prefer the Answer: segment, then fall back to full text."""
    seg = generation or ""
    if "<|im_start|>assistant" in seg:
        seg = seg.rsplit("<|im_start|>assistant", 1)[-1]
    if "Answer:" in seg:
        seg = seg.rsplit("Answer:", 1)[-1]
    if "<|im_end|>" in seg:
        seg = seg.split("<|im_end|>", 1)[0]
    seg = seg.strip()
    na, nf = norm_label(seg), norm_label(generation or "")
    for nl, raw in labels_sorted:
        if nl in na:
            return raw, seg
    for nl, raw in labels_sorted:
        if nl in nf:
            return raw, seg
    return None, seg


def load_trained(ckpt, text_model, dna_model, lora_r, lora_alpha, cache_dir):
    model = DNALLMModel(text_model_name=text_model, dna_model_name=dna_model,
                        cache_dir=cache_dir, max_length_dna=2048, max_length_text=1280,
                        dna_is_evo2=True, dna_embedding_layer="blocks.20.mlp.l3")
    from peft import LoraConfig, get_peft_model
    model.text_model = get_peft_model(model.text_model, LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0,
        target_modules=get_target_modules(model), init_lora_weights="gaussian",
        bias="none", task_type="CAUSAL_LM"))
    sd = torch.load(os.path.join(ckpt, "checkpoint", "mp_rank_00_model_states.pt"),
                    map_location="cpu", weights_only=False)["module"]
    # DeepSpeed writes every tensor twice: once under the LightningModule's `model.` attribute with
    # peft's wrapper flattened away, and once under the real module path.
    clean = {k: v for k, v in sd.items() if not k.startswith("model.")}
    missing, unexpected = model.load_state_dict(clean, strict=False)
    frac = 1.0 - len(unexpected) / max(1, len(clean))
    print(f"loaded {len(clean)} tensors; {len(missing)} missing, {len(unexpected)} unexpected "
          f"({frac:.1%} matched)", flush=True)
    assert frac > 0.99, (f"only {frac:.1%} of checkpoint tensors matched the model - the "
                         f"adapters are NOT loaded and any result would describe an "
                         f"untrained model")
    return model.eval()


def perturb(ref, var, mode, rng, donor):
    if mode == "wt":
        return ref, var
    if mode == "same":
        # control: identical input, regenerated.
        return ref, var
    if mode == "shuffle":
        a, b = list(ref), list(var)
        rng.shuffle(a); rng.shuffle(b)
        return "".join(a), "".join(b)
    if mode == "swap":
        return donor
    raise ValueError(mode)


@torch.no_grad()
def generate(model, proc, question, ref, var, max_new_tokens, arm="embed"):
    """Prompt ends at the assistant marker, so the model writes the trace and the answer."""
    if arm == "text":
        question = (f"Reference sequence: {ref}\nVariant sequence: {var}\n"
                    f"Question: {question}")
        ref, var = "ACGT", "ACGT"
    prompt = (f"<|im_start|>user\n<|dna_start|><|dna_pad|><|dna_end|>"
              f"<|dna_start|><|dna_pad|><|dna_end|>{question}<|im_end|>\n"
              f"<|im_start|>assistant\n")
    batch = proc(text=[prompt], batch_dna_sequences=[[ref, var]], return_tensors="pt",
                 add_special_tokens=False, max_length_text=1280, max_length_dna=2048)
    out = model.generate(input_ids=batch["input_ids"].to("cuda"),
                         attention_mask=batch["attention_mask"].to("cuda"),
                         dna_tokenized=batch["dna_tokenized"],
                         batch_idx_map=batch["batch_idx_map"],
                         max_new_tokens=max_new_tokens, do_sample=False)
    return proc.tokenizer.decode(out[0], skip_special_tokens=False).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True,
                    help=f"a training export with test.csv, e.g. {RD.AUX_TARGET_257BP}")
    ap.add_argument("--text-model", default=ckpt("dna_text_base"))
    ap.add_argument("--dna-model", default="evo2_1b_base")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--n-rows", type=int, default=150)
    ap.add_argument("--max-new-tokens", type=int, default=800)
    ap.add_argument("--conditions", default="wt,shuffle,swap")
    ap.add_argument("--truncate-per-side", type=int, default=1024,
                    help="0 for datasets whose sequences are already cut to size")
    ap.add_argument("--arm", choices=["embed", "text"], default="embed",
                    help="embed = --model_type dna-llm; text = --model_type llm")
    ap.add_argument("--split", choices=["test", "valtest"], default="valtest",
                    help="test = test.csv only (clean held-out); valtest = the legacy pooled behaviour")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    csv.field_size_limit(10 ** 7)
    # --split test reads test.csv alone.
    _files = ("test.csv",) if a.split == "test" else ("val.csv", "test.csv")
    rows = []
    for f in _files:
        rows += list(csv.DictReader(open(os.path.join(a.data, f))))
    print(f"[split] {a.split}: {len(rows)} rows from {list(_files)}", flush=True)
    train = list(csv.DictReader(open(os.path.join(a.data, "train.csv"))))
    labels = sorted({r["answer"] for r in rows + train})
    labels_sorted = sorted(((norm_label(x), x) for x in labels if norm_label(x)),
                           key=lambda t: len(t[0]), reverse=True)
    print(f"{len(rows)} eval rows, {len(labels)} labels", flush=True)

    rng = random.Random(a.seed)
    pool = list(rows); rng.shuffle(pool)
    sample = pool[: a.n_rows]

    model = load_trained(a.ckpt, a.text_model, a.dna_model, a.lora_r, a.lora_alpha,
                         os.environ.get("HF_HOME")).to("cuda")
    proc = model.processor
    conds = a.conditions.split(",")

    recs, t0 = [], time.time()
    out_path = a.out
    for i, r in enumerate(sample):
        t = (truncate_dna(dict(r), truncate_dna_per_side=a.truncate_per_side)
             if a.truncate_per_side else dict(r))
        ref, var = t["reference_sequence"], t["variant_sequence"]
        # the auxiliary supervision exports carry only question/answer/reasoning/ref/var, with no
        # context_id, variant_key or gene.
        _cid = lambda x: x.get("context_id") or x["question"]
        other = rng.choice([x for x in pool if _cid(x) != _cid(r)])
        ot = (truncate_dna(dict(other), truncate_dna_per_side=a.truncate_per_side)
              if a.truncate_per_side else dict(other))
        donor = (ot["reference_sequence"], ot["variant_sequence"])

        rec = {"context_id": r.get("context_id", ""), "variant_key": r.get("variant_key", ""),
               "gene": r.get("gene", ""), "truth": r["answer"]}
        for c in conds:
            rr, vv = perturb(ref, var, c, rng, donor)
            # store the sequence fed to Evo2, so a later scorer can check the emitted bases against
            # this window rather than inferring from composition
            rec[f"ref_{c}"] = rr
            rec[f"var_{c}"] = vv
            g = generate(model, proc, r["question"], rr, vv, a.max_new_tokens, a.arm)
            lab, seg = extract_label(g, labels_sorted)
            rec[f"pred_{c}"] = lab
            rec[f"answer_text_{c}"] = seg[:120]
            rec[f"gen_{c}"] = g
        recs.append(rec)
        if (i + 1) % 10 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(sample)}  {el/60:.1f} min  "
                  f"eta {el/(i+1)*(len(sample)-i-1)/60:.0f} min", flush=True)
            json.dump({"partial": True, "rows": recs}, open(out_path, "w"))

    n = len(recs)
    summ = {"n_rows": n, "conditions": conds, "max_new_tokens": a.max_new_tokens,
            "decoding": "greedy (do_sample=False)"}
    for c in conds:
        summ[f"accuracy_{c}"] = round(
            sum(norm_label(r[f"pred_{c}"]) == norm_label(r["truth"]) for r in recs) / n, 4)
        summ[f"unmapped_{c}"] = sum(1 for r in recs if r[f"pred_{c}"] is None)
    for c in conds[1:]:
        summ[f"answer_agreement_wt_vs_{c}"] = round(
            sum(norm_label(r[f"pred_{c}"]) == norm_label(r["pred_wt"]) for r in recs) / n, 4)
        summ[f"generation_identical_wt_vs_{c}"] = round(
            sum(r[f"gen_{c}"] == r["gen_wt"] for r in recs) / n, 4)
    json.dump({"summary": summ, "rows": recs}, open(out_path, "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()

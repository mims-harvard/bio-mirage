#!/usr/bin/env python
"""Runs a BioReason DNA checkpoint on disease prediction queries from examples.jsonl and writes
records.jsonl, inside the BioReason environment.

Loads the Qwen3 language model (vLLM) and the frozen Evo2 encoder once, projects the Evo2
representation of each DNA block, and inserts it at the `<|dna_pad|>` positions, so a condition can
change, remove or add DNA blocks. Decoding is greedy. Each record holds hashes of the text tokens
and DNA embeddings, which input_use.core.score_bioreason uses to check each perturbation.

    python -m input_use.models.bioreason.runner --examples <run>/examples.jsonl --records_out <run>/records_rl.jsonl --checkpoint <dna_sft or dna_rl id from checkpoints.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import torch

from input_use.core import config as cfg
from input_use.core.records import read_examples, record_from_example, append_jsonl

from input_use.core.checkpoints import ckpt  # noqa: E402
BRDNA = cfg.require(cfg.DNA_REPO, "BioReason DNA repo (INPUT_USE_DNA_REPO)")
sys.path.insert(0, str(BRDNA))
from bioreason.models.dna_vllm import DNALLMModel          # noqa: E402  (vLLM Qwen3 + Evo2)
from bioreason.models.dl.processing_dl import DLProcessor   # noqa: E402
from trl.data_utils import maybe_apply_chat_template        # noqa: E402
from vllm import SamplingParams                             # noqa: E402

RUNNER_VERSION = "2"
DNA_PAD_ID = 151670          # <|dna_pad|>; identical in the local and released checkpoints


def _sha1_tensor(t: torch.Tensor) -> str:
    """sha1 over the raw bytes of a tensor, in fp32 so bf16/fp16 storage differences do not show up as
    spurious changes between arms.
    """
    return hashlib.sha1(t.detach().to(torch.float32).cpu().numpy().tobytes()).hexdigest()


def _sha1_ids(ids: torch.Tensor) -> str:
    return hashlib.sha1(ids.detach().cpu().numpy().tobytes()).hexdigest()


def resolve_checkpoint(name: str) -> str:
    """Absolute path -> itself; HF repo id -> local mirror if present, else download; bare name ->
    checkpoints_hf/<name>.
    """
    if os.path.isdir(name):
        return name
    if "/" in name:
        mirror = os.environ.get("INPUT_USE_DNA_MODELS_DIR")
        if mirror:
            local = os.path.join(mirror, name.split("/")[-1])
            if os.path.isdir(local):
                return local
        from huggingface_hub import snapshot_download
        return snapshot_download(repo_id=name)
    return str(BRDNA / "checkpoints_hf" / name)


def _assert_dna_checkpoint(ckpt_dir: str, name: str) -> dict:
    """Refuse anything that is not an Evo2 DNA-LLM."""
    # The released configs come from transformers 5.1.0, which moved `rope_theta` into
    # `rope_parameters`.
    cfg_path = os.path.join(ckpt_dir, "config.json")
    if os.path.exists(cfg_path):
        cc = json.load(open(cfg_path))
        rp = cc.get("rope_parameters")
        if isinstance(rp, dict) and rp.get("rope_theta") is not None:
            assert cc.get("rope_theta") == rp["rope_theta"], (
                f"{name}: config.json carries rope_parameters.rope_theta={rp['rope_theta']} but "
                f"rope_theta={cc.get('rope_theta')}. vLLM would silently use 10000. "
                f"Copy rope_parameters.rope_theta into the top-level rope_theta field.")

    meta_path = os.path.join(ckpt_dir, "dna_model_config.json")
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        assert meta.get("dna_is_evo2") is True, f"{name}: dna_is_evo2 is not true ({meta})"
        assert meta.get("dna_model_name") == "evo2_1b_base", f"{name}: unexpected DNA encoder ({meta})"
        return meta
    assert os.path.exists(os.path.join(ckpt_dir, "dna_projection.pt")), (
        f"{name}: no dna_model_config.json and no dna_projection.pt -- this is not a DNA checkpoint. "
        f"The CPT, RNA SFT and protein SFT LoRA checkpoints of the same release land here.")
    return {"dna_is_evo2": True, "dna_model_name": "evo2_1b_base", "stage": "local-converted"}


def format_item(question: str, dna_sequences, answer: str) -> dict:
    """The repo's KEGG chat format, but with a variable number of DNA blocks."""
    return {
        "prompt": [{"role": "user", "content": [
            *({"type": "dna", "text": None} for _ in dna_sequences),
            {"type": "text", "text": question.strip()},
        ]}],
        "dna_sequences": list(dna_sequences),
        "answer": answer,
    }


class DNARowCache:
    """lru of projected Evo2 rows, keyed by sha1(sequence). One entry is ~8 MB at 2048x2048 bf16."""

    def __init__(self, model, proc, max_length_dna: int, capacity: int = 24):
        self.model, self.proc, self.max_length_dna = model, proc, max_length_dna
        self.capacity = capacity
        self._d: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self.hits = self.misses = 0

    def rows(self, seq: str) -> torch.Tensor:
        key = hashlib.sha1(seq.encode()).hexdigest()
        if key in self._d:
            self.hits += 1
            self._d.move_to_end(key)
            return self._d[key]
        self.misses += 1
        tok = self.proc.tokenize_dna_sequences([[seq]], max_length=self.max_length_dna,
                                               return_tensors="pt", device=self.model.device)
        # `tokenize_dna_sequences` accepts a `device` argument and ignores it - it only calls the
        # DNA tokenizer - so the ids come back on CPU and Evo2's embedding lookup dies with a device
        # mismatch.
        dna_tok = {k: v.to(self.model.device) for k, v in tok["dna_tokenized"].items()}
        out = self.model.process_dna_embeddings(dna_tok, tok["batch_idx_map"], 1)[0]
        self._d[key] = out
        if len(self._d) > self.capacity:
            self._d.popitem(last=False)
        return out


@torch.no_grad()
def build_prompt_embeds(ex, model, proc, cache, max_length_text: int):
    """One un-padded prompt: token ids -> text embeddings -> Evo2 rows spliced into the pad slots."""
    seqs = ex.payload["dna_sequences"]
    item = format_item(ex.payload["question"], seqs, ex.ground_truth.get("answer", ""))
    text = maybe_apply_chat_template(item, proc)["prompt"]
    prepared = proc(text=[text], batch_dna_sequences=[seqs], return_tensors="pt",
                    add_special_tokens=False, max_length_text=max_length_text,
                    max_length_dna=cache.max_length_dna)
    ids = prepared["input_ids"][0].to(model.device)

    mask = ids == model.dna_token_id
    n_pad = int(mask.sum().item())
    # The stem is the last thing in the prompt, and the tokenizer truncates on the right, so an
    # overflow silently removes the question rather than some padding.
    assert ids.shape[0] < max_length_text + 2 * cache.max_length_dna, (
        f"{ex.example_id}/{ex.condition}: prompt hit the truncation ceiling "
        f"({ids.shape[0]} tokens) -- raise --max_length_text")

    emb = model._embedding_layer(ids)
    dna_sha = None
    if seqs:
        rows = torch.cat([cache.rows(s) for s in seqs], dim=0)
        assert rows.shape[0] == n_pad, (
            f"{ex.example_id}/{ex.condition}: {rows.shape[0]} DNA rows vs {n_pad} pad slots")
        dna_sha = _sha1_tensor(rows)
        emb[mask] = rows.to(emb.dtype)
    else:
        assert n_pad == 0, f"{ex.example_id}/{ex.condition}: no_dna arm still rendered pad slots"
    return emb, dna_sha, _sha1_ids(ids[~mask]), n_pad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", required=True)
    ap.add_argument("--records_out", required=True)
    ap.add_argument("--checkpoint", default=ckpt("dna_sft"))
    ap.add_argument("--text_model_name", default=ckpt("dna_text_base"))
    # The generation budget must clear the model's rationale: a truncated row emits no "Answer:"
    # and scores 0 on budget rather than on knowledge.
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--max_length_text", type=int, default=1280)
    ap.add_argument("--max_length_dna", type=int, default=2048)
    ap.add_argument("--chunk_size", type=int, default=32)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.60)
    ap.add_argument("--limit", type=int, default=0, help="0 = every example (the default)")
    ap.add_argument("--resume", action="store_true", help="skip example/condition pairs already written")
    a = ap.parse_args()

    # The released DNA checkpoints' own evaluation launchers export this prelude verbatim.
    _need = {"BIOREASON_USE_VORTEX_PYTORCH_LINEAR": ("1", "true", "yes"),
             "TORCHDYNAMO_DISABLE": ("1", "true", "yes")}
    for var, ok in _need.items():
        got = os.environ.get(var, "")
        assert got.lower() in ok, (
            f"{var}={got!r} -- source setup/dna_runtime_env.sh before running. Without "
            f"BIOREASON_USE_VORTEX_PYTORCH_LINEAR=1 the Evo2 encoder runs FP8 input projections that "
            f"the trained dna_projection never saw, and the run is silently invalid.")

    ckpt_dir = resolve_checkpoint(a.checkpoint)
    ckpt_meta = _assert_dna_checkpoint(ckpt_dir, a.checkpoint)
    print(f"[dna-runner] v{RUNNER_VERSION} loading {a.checkpoint}\n"
          f"             -> {ckpt_dir}\n"
          f"             stage={ckpt_meta.get('stage')} encoder={ckpt_meta.get('dna_model_name')} "
          f"text={a.text_model_name}")

    model = DNALLMModel(
        ckpt_dir=ckpt_dir, text_model_name=a.text_model_name, dna_model_name="evo2_1b_base",
        cache_dir=os.environ.get("HF_HOME"),
        max_length_dna=a.max_length_dna, max_length_text=a.max_length_text,
        text_model_finetune=False, dna_model_finetune=False,
        dna_is_evo2=True, dna_embedding_layer="blocks.20.mlp.l3",
        gpu_memory_utilization=a.gpu_memory_utilization, max_model_len=8192,
    )
    proc = DLProcessor(tokenizer=model.text_tokenizer, dna_tokenizer=model.dna_tokenizer)

    pad_id = model.text_tokenizer.convert_tokens_to_ids("<|dna_pad|>")
    assert pad_id == DNA_PAD_ID == model.dna_token_id, (
        f"<|dna_pad|> is {pad_id}, expected {DNA_PAD_ID} (model says {model.dna_token_id}) -- the "
        f"checkpoint's tokenizer does not match the one the projection was trained against")

    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=a.max_new_tokens,
                              stop=["<|im_end|>"])
    cache = DNARowCache(model, proc, a.max_length_dna)

    examples = read_examples(a.examples)
    if a.limit:
        examples = examples[:a.limit]
    out = Path(a.records_out)
    done = set()
    if a.resume and out.exists():
        with open(out) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    done.add((r["example_id"], r["condition"]))
                except Exception:
                    continue
        examples = [e for e in examples if (e.example_id, e.condition) not in done]
        print(f"[dna-runner] resume: {len(done)} records already written, {len(examples)} to go")
    else:
        out.unlink(missing_ok=True)

    experiment = out.parent.name
    model_name = f"bioreason_dna_{a.checkpoint.split('/')[-1]}"
    t0, n_done = time.time(), 0

    for start in range(0, len(examples), a.chunk_size):
        chunk = examples[start:start + a.chunk_size]
        embeds, metas = [], []
        for ex in chunk:
            emb, dna_sha, text_sha, n_pad = build_prompt_embeds(ex, model, proc, cache,
                                                                a.max_length_text)
            embeds.append(emb)
            metas.append({"dna_embed_sha1": dna_sha, "text_token_sha1": text_sha,
                          "n_dna_tokens": n_pad, "n_prompt_tokens": int(emb.shape[0])})
        # Ragged batch: each request keeps its own length, so nothing is padded into anything else.
        outs = model.text_model.generate([{"prompt_embeds": e} for e in embeds],
                                         sampling_params=sampling)
        for ex, meta, o in zip(chunk, metas, outs):
            raw = o.outputs[0].text
            record = record_from_example(
                ex, experiment=experiment, model=model_name,
                input={"question": ex.payload["question"],
                       "dna_lengths": [len(s) for s in ex.payload["dna_sequences"]],
                       "channels": ex.payload["channels"],
                       "variant_key": ex.payload.get("variant_key"),
                       "stratum": ex.payload.get("stratum"),
                       "split": ex.payload.get("split"),
                       # Two-choice grouping keys; absent for the free-generation battery. The
                       # scorer can recover these from the example_id, but carrying them makes the
                       # record self-describing.
                       **{k: ex.payload[k] for k in
                          ("pair_key", "option_order", "member", "text_key")
                          if k in ex.payload},
                       **meta},
                # No parsed field: the label mapping needs the closed vocabulary and is done at
                # score time, which also means a scoring change never requires re-generating.
                output={"raw": raw, "parsed": {}})
            record.provenance = {**(record.provenance or {}), "runner_version": RUNNER_VERSION,
                                 "checkpoint": a.checkpoint, "checkpoint_stage": ckpt_meta.get("stage"),
                                 "max_length_text": a.max_length_text,
                                 "max_length_dna": a.max_length_dna,
                                 # Recorded because the answer rate depends on the budget.
                                 "max_new_tokens": a.max_new_tokens}
            append_jsonl(record, a.records_out)
        # Drop the chunk's prompt tensors before building the next one. Each is ~4,500 x 2,048, and
        # holding two chunks at once is what turns a comfortable footprint into an OOM.
        embeds.clear()
        del outs
        n_done += len(chunk)
        rate = n_done / max(1e-9, time.time() - t0)
        print(f"  [{n_done}/{len(examples)}] {rate:.2f} rows/s | evo2 cache "
              f"{cache.hits}h/{cache.misses}m | last: {ex.example_id} [{ex.condition}] -> "
              f"{raw.strip()[-60:]!r}", flush=True)

    print(f"[dna-runner] wrote {a.records_out} ({n_done} records in "
          f"{(time.time() - t0) / 60:.1f} min; Evo2 cache {cache.hits} hits / {cache.misses} misses)")


if __name__ == "__main__":
    main()

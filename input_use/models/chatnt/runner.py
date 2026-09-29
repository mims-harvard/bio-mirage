#!/usr/bin/env python
"""Runs the jax release of ChatNT on examples.jsonl with greedy decoding and writes records.jsonl.

`--dna_mode` sets how DNA enters the prompt: `inline` for ChatNT's own Yes/No tasks, whose question
already holds the `<DNA>` placeholder, `ref_var` for the reference and variant blocks of a disease
prediction query, and a single variant block otherwise. Yes/No answers are parsed here and disease
answers at scoring time. The ChatNT results in the paper come from input_use/models/chatnt_driver.

    python -m input_use.models.chatnt.runner --examples <run>/examples.jsonl --records_out <run>/records.jsonl --dna_mode inline
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from input_use.core.records import read_examples, record_from_example, append_jsonl
from input_use.metrics.seqtask import extract_yesno

# ChatNT's training-time system prefix (from the repo's inference notebook) - keeps generations
# in-distribution. The assistant answers after "assistant:".
SYS_PREFIX = (
    "A chat between a curious user and an artificial intelligence assistant that can handle bio "
    "sequences. The assistant gives helpful, detailed, and polite answers to the user's questions. "
    "USER: "
)


def _assemble(question: str, dna_mode: str) -> str:
    """Full prompt string: ChatNT system prefix + question + DNA cue + 'assistant:'."""
    if dna_mode == "inline":          # <DNA> already inside `question` (ChatNT native format)
        return SYS_PREFIX + question + " ASSISTANT:"
    if dna_mode == "ref_var":
        return SYS_PREFIX + question + "\nReference DNA sequence: <DNA>\nVariant DNA sequence: <DNA> ASSISTANT:"
    return SYS_PREFIX + question + "\nDNA sequence: <DNA> ASSISTANT:"


def build_english_ids(question: str, dna_mode: str, n_dna: int, eng_tok, eng_max: int,
                      max_new: int) -> np.ndarray:
    """Tokenize the full prompt string in one call (piecewise tokenization injects spurious space
    tokens at boundaries and pushes ChatNT off-distribution), right-padded to eng_max with `max_new`
    slots reserved for the greedy decode.
    """
    dna_id = eng_tok.convert_tokens_to_ids("<DNA>")
    budget = eng_max - max_new - 1
    q = question
    ids = eng_tok(_assemble(q, dna_mode), add_special_tokens=True).input_ids
    if len(ids) > budget or ids.count(dna_id) != n_dna:
        # shrink the question (token-trim from its end), keeping prefix/suffix/<DNA> intact.
        over = max(0, len(ids) - budget)
        qids = eng_tok(q, add_special_tokens=False).input_ids
        qids = qids[: max(0, len(qids) - over - 2)]
        q = eng_tok.decode(qids, skip_special_tokens=True)
        ids = eng_tok(_assemble(q, dna_mode), add_special_tokens=True,
                      truncation=True, max_length=budget).input_ids
    ids = ids + [eng_tok.pad_token_id] * (eng_max - len(ids))               # right-pad to eng_max
    assert ids.count(dna_id) == n_dna, f"<DNA> count {ids.count(dna_id)} != {n_dna} (truncation?)"
    return np.asarray(ids, dtype=np.int32)[None, :]                         # (1, eng_max)


def make_generate(apply_fn, eng_tok):
    import jax.numpy as jnp

    def generate(params, key, english_ids: np.ndarray, bio_tokens: np.ndarray, max_new: int) -> str:
        """Greedy autoregressive decode. Caches projected_bio_embeddings so the DNA encoder runs once;
        writes each argmax token into the first pad slot until eos or max_new.
        """
        pad_id = eng_tok.pad_token_id
        pads = np.where(english_ids[0] == pad_id)[0]
        if len(pads) == 0:
            return ""
        idx_begin = int(pads[0])
        eng = jnp.asarray(english_ids)
        bio = jnp.asarray(bio_tokens)
        projected = None
        n_steps = 0
        for _ in range(max_new):
            outs = apply_fn(params, key, multi_omics_tokens_ids=(eng, bio),
                            projection_english_tokens_ids=eng, projected_bio_embeddings=projected)
            projected = outs["projected_bio_embeddings"]
            cur_pads = np.where(np.asarray(eng[0]) == pad_id)[0]
            if len(cur_pads) == 0:
                break
            first_pad = int(cur_pads[0])
            next_tok = int(np.argmax(np.asarray(outs["logits"][0, first_pad - 1])))
            if next_tok == eng_tok.eos_token_id:
                break
            eng = eng.at[0, first_pad].set(next_tok)
            n_steps += 1
        gen = np.asarray(eng[0, idx_begin: idx_begin + n_steps])
        return eng_tok.decode(gen, skip_special_tokens=True).strip()

    return generate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", required=True)
    ap.add_argument("--records_out", required=True)
    ap.add_argument("--dna_mode", choices=["ref_var", "variant_only", "inline"], default="ref_var",
                    help="ref_var: present [reference, variant] as two <DNA> blocks (BioReason "
                         "analog). variant_only: the variant alone. inline: question already carries "
                         "the <DNA> placeholder(s) (ChatNT native seq-tasks).")
    ap.add_argument("--english_max_length", type=int, default=1024)
    ap.add_argument("--bio_max_length", type=int, default=512)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="if >0, only run the first N examples (smoke).")
    a = ap.parse_args()

    import haiku as hk
    import jax
    from nucleotide_transformer.chatNT.pretrained import get_chatNT

    # important: warm jax's dtype-promotion lattice before joblib-loading the params.
    print(f"[chatnt-runner] devices: {jax.devices()}")
    key = jax.random.PRNGKey(a.seed)

    forward_fn, params, english_tok, bio_tok = get_chatNT()
    english_tok.padding_side = "right"

    # The joblib checkpoint stores 18 (of 756) leaves as ml_dtypes.bfloat16, which this jaxlib build
    # treats as a foreign dtype ("not a valid jax array type").
    def _fix(x):
        x = np.asarray(x)
        return x.astype(np.float32) if x.dtype.name == "bfloat16" else x
    params = jax.device_put(jax.tree_util.tree_map(_fix, params))
    forward_fn = hk.transform(forward_fn)
    apply_fn = jax.jit(forward_fn.apply)
    generate = make_generate(apply_fn, english_tok)

    examples = read_examples(a.examples)
    if a.limit > 0:
        examples = examples[: a.limit]
    out = Path(a.records_out)
    out.unlink(missing_ok=True)
    experiment, model_name = out.parent.name, f"chatnt_dna_{a.dna_mode}"

    for ex in examples:
        seqs = ex.payload["dna_sequences"]
        if a.dna_mode == "inline":
            dna_seqs = seqs                                  # question carries the inline <DNA>(s)
        elif a.dna_mode == "ref_var":
            dna_seqs = [seqs[0], seqs[1]]                    # [reference, variant]
        else:
            dna_seqs = [seqs[-1]]                            # variant only
        english_ids = build_english_ids(ex.payload["question"], a.dna_mode, len(dna_seqs), english_tok,
                                        a.english_max_length, a.max_new_tokens)
        bio_tokens = bio_tok(dna_seqs, return_tensors="np", padding="max_length",
                             max_length=a.bio_max_length, truncation=True).input_ids[None, ...]
        raw = generate(params, key, english_ids, bio_tokens, a.max_new_tokens)
        # Disease labels are mapped onto the closed gold vocabulary at score time (see
        # metrics/dna.py), so only the yes/no seqtasks are parsed here.
        parsed = extract_yesno(raw) if ex.modality == "dna_seqtask" else ""
        output = {"raw": raw, "parsed": {"answer": parsed}}
        inp = {"question": ex.payload["question"], "dna_lengths": [len(s) for s in seqs],
               "dna_mode": a.dna_mode, "channels": ex.payload["channels"]}
        append_jsonl(record_from_example(ex, experiment=experiment, model=model_name,
                                         input=inp, output=output), a.records_out)
        print(f"  {ex.example_id} [{ex.condition}] -> {output['parsed']['answer'][:45]!r}")
    print(f"[chatnt-runner] wrote {a.records_out} ({len(examples)} records)")


if __name__ == "__main__":
    main()

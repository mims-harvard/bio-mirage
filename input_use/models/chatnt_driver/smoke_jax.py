#!/usr/bin/env python
"""Checks the ChatNT model setup by running the official jax inference notebook code unchanged.

Generates the answer for the notebook's example and compares it with the notebook's published
output, then records the generation and the log-likelihoods of " Yes" and " No" as the first answer
token for the first queries of each task. Writes smoke_jax.json to
dna/chatnt/results under $INPUT_USE_RESULTS_DIR, which compare_smoke.py reads.

    python input_use/models/chatnt_driver/smoke_jax.py [--backend cpu] [--n_per_task 3] [--max_new 20]
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
from chatnt_common import (ROOT, RESULTS, NOTEBOOK_ENGLISH, NOTEBOOK_DNA, NOTEBOOK_EXPECTED,  # noqa: E402
                           TASK_ORDER, build_prompt, package_versions, repo_commit,
                           parse_generation_label)

# --- notebook (imports)
import haiku as hk
import jax
import jax.numpy as jnp
from tqdm import tqdm
from nucleotide_transformer.chatNT.pretrained import get_chatNT
# --- notebook end
import hashlib
import ml_dtypes
import nucleotide_transformer.chatNT.pretrained as _pretrained


def load_clean_params():
    """The released jax_params/params.joblib, re-read from the per-leaf .npy dump diag_jax.py writes
    (bfloat16 leaves stored as uint16 bit patterns), every leaf checked against the sha256 of the
    original array's bytes.
    """
    d = os.path.join(ROOT, "xdg", "params_clean")
    man = json.load(open(os.path.join(d, "manifest.json")))
    params = {}
    for m in man:
        a = np.load(os.path.join(d, m["file"]), allow_pickle=False)
        assert hashlib.sha256(a.tobytes()).hexdigest() == m["sha256"], m
        if m["dtype"] == "bfloat16":
            a = a.view(ml_dtypes.bfloat16)
        assert str(a.dtype) == m["dtype"] and list(a.shape) == m["shape"], m
        params.setdefault(m["module"], {})[m["name"]] = a
    print(f"clean params: {len(man)} leaves verified against the joblib's sha256 manifest", flush=True)
    return params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--n_per_task", type=int, default=3)
    ap.add_argument("--max_new", type=int, default=20)
    ap.add_argument("--out", default="smoke_jax.json", help="output file name under results/")
    ap.add_argument("--params_source", default="clean", choices=["clean", "joblib"],
                    help="clean = the sha256-verified .npy dump of params.joblib (default); joblib = the file itself")
    a = ap.parse_args()
    t0 = time.time()
    if a.params_source == "clean":
        _pretrained.download_ckpt = load_clean_params     # get_chatNT() below is otherwise untouched

    # --- notebook
    jax.config.update("jax_platform_name", a.backend)
    backend = a.backend
    devices = jax.devices(backend)
    num_devices = len(devices)
    print(f"Devices found: {devices}")

    def generate_answer(apply_fn, parameters, random_keys, english_tokenizer, english_tokens, bio_tokens, max_num_tokens_to_decode):
        """Note: the function expects that pmap is already applied to the forward function, the inputs
        and the parameters
        """
        english_tokens = english_tokens.copy()

        idx_begin_generation = np.where(
            english_tokens[0, 0] == english_tokenizer.pad_token_id
        )[0][0]
        projected_bio_embeddings = jax.device_put_replicated(None, devices=devices)
        actual_nb_steps = 0

        for _ in tqdm(range(max_num_tokens_to_decode)):
            outs = apply_fn(
                parameters,
                random_keys,
                multi_omics_tokens_ids=(english_tokens, bio_tokens),
                projection_english_tokens_ids=english_tokens,
                projected_bio_embeddings=projected_bio_embeddings,
            )
            projected_bio_embeddings = outs["projected_bio_embeddings"]
            logits = outs["logits"]

            first_idx_pad_token = np.where(
                english_tokens[0, 0] == english_tokenizer.pad_token_id
            )[0][0]
            predicted_token = np.argmax(logits[0, 0, first_idx_pad_token - 1])

            if predicted_token == english_tokenizer.eos_token_id:
                break
            else:
                english_tokens = english_tokens.at[0, 0, first_idx_pad_token].set(
                    predicted_token
                )
                actual_nb_steps += 1

        decoded_generated_sentence = english_tokenizer.decode(
            english_tokens[0, 0, idx_begin_generation : idx_begin_generation + actual_nb_steps]
        )

        return decoded_generated_sentence

    forward_fn, parameters, english_tokenizer, bio_tokenizer = get_chatNT()
    forward_fn = hk.transform(forward_fn)
    apply_fn = jax.pmap(forward_fn.apply, devices=devices, donate_argnums=(0,))

    random_key = jax.random.PRNGKey(seed=0)
    random_keys = jax.numpy.stack([random_key for _ in range(len(devices))])
    keys = jax.device_put_replicated(random_key, devices=devices)
    parameters = jax.device_put_replicated(parameters, devices=devices)

    english_max_length = 512 # length of the tokenized english sequence
    bio_tokenized_sequence_length = 512 # length of the tokenized DNA sequences

    def tokenize(english_sequence, dna_sequences):
        english_tokens = english_tokenizer(
            [english_sequence],
            return_tensors="np",
            max_length=english_max_length,
            padding="max_length",
            truncation=True,
        ).input_ids

        bio_tokens = bio_tokenizer(
            dna_sequences,
            return_tensors="np",
            padding="max_length",
            max_length=bio_tokenized_sequence_length,
            truncation=True,
        ).input_ids
        bio_tokens = np.expand_dims(bio_tokens, axis=0) # Add batch dimension -> result: (1, num_dna_sequences,
        # bio_tokenized_sequence_length)

        # Replicate over devices
        english_tokens = jnp.stack([jnp.asarray(english_tokens, dtype=jnp.int32)]*num_devices, axis=0)
        bio_tokens = jnp.stack([jnp.asarray(bio_tokens, dtype=jnp.int32)]*num_devices, axis=0)
        return english_tokens, bio_tokens
    # --- notebook end

    print(f"model loaded in {time.time() - t0:.0f}s", flush=True)
    yes_id = english_tokenizer(NOTEBOOK_ENGLISH + " Yes").input_ids[len(english_tokenizer(NOTEBOOK_ENGLISH).input_ids):]
    no_id = english_tokenizer(NOTEBOOK_ENGLISH + " No").input_ids[len(english_tokenizer(NOTEBOOK_ENGLISH).input_ids):]
    assert len(yes_id) == 1 and len(no_id) == 1, (yes_id, no_id)
    yes_id, no_id = yes_id[0], no_id[0]

    def first_step(english_tokens, bio_tokens):
        """Next-token log-probabilities at the last prompt position (the first generation step)."""
        outs = apply_fn(parameters, keys, multi_omics_tokens_ids=(english_tokens, bio_tokens),
                        projection_english_tokens_ids=english_tokens,
                        projected_bio_embeddings=jax.device_put_replicated(None, devices=devices))
        fp = int(np.where(np.asarray(english_tokens[0, 0]) == english_tokenizer.pad_token_id)[0][0])
        lg = np.asarray(outs["logits"][0, 0, fp - 1], dtype=np.float64)
        lp = lg - np.log(np.exp(lg - lg.max()).sum()) - lg.max()
        top = np.argsort(-lg)[:10]
        return {"n_english_tokens": fp, "L_yes": float(lp[yes_id]), "L_no": float(lp[no_id]),
                "margin": float(lp[yes_id] - lp[no_id]), "first_token_argmax": int(top[0]),
                "top10_ids": top.tolist(), "top10_logits": lg[top].tolist(),
                "logits_dtype": str(outs["logits"].dtype)}

    def run(english_sequence, dna):
        et, bt = tokenize(english_sequence, [dna])
        t = time.time()
        gen = generate_answer(apply_fn=apply_fn, parameters=parameters, random_keys=random_keys,
                              english_tokenizer=english_tokenizer, english_tokens=et, bio_tokens=bt,
                              max_num_tokens_to_decode=a.max_new)
        r = {"generated": gen, "parsed_label": parse_generation_label(gen), "gen_seconds": time.time() - t}
        r.update(first_step(et, bt))
        return r

    out = {"backend": backend, "devices": [str(d) for d in devices], "repo_commit": repo_commit(),
           "params_source": a.params_source,
           "versions": package_versions(), "yes_token_id": yes_id, "no_token_id": no_id,
           "max_new": a.max_new}
    nb = run(NOTEBOOK_ENGLISH, NOTEBOOK_DNA)
    nb["expected_notebook_output"] = NOTEBOOK_EXPECTED
    nb["matches_notebook_output"] = nb["generated"].strip() == NOTEBOOK_EXPECTED
    out["notebook_example"] = nb
    print("NOTEBOOK EXAMPLE ->", repr(nb["generated"]), "| expected:", repr(NOTEBOOK_EXPECTED),
          "| exact:", nb["matches_notebook_output"], f"| margin {nb['margin']:+.3f}", flush=True)
    p = os.path.join(RESULTS, a.out)

    def dump():                     # incremental: a CPU run is slow enough to hit the wall clock
        out["seconds_so_far"] = time.time() - t0
        tmp = p + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(out, fh, indent=2)
        os.replace(tmp, p)
    out["examples"] = []
    dump()

    ex = pd.read_parquet(os.path.join(RESULTS, "chatnt_examples.parquet"))
    for task in TASK_ORDER:
        sub = ex[ex.task == task].head(a.n_per_task)
        for _, r in sub.iterrows():
            res = run(build_prompt(r.question), r.sequence_A)
            res.update({"example_id": r.example_id, "task": task, "gold_A": r.gold_A,
                        "sequence_A_hash": r.sequence_A_hash})
            out["examples"].append(res)
            print(f"{r.example_id} gold {r.gold_A} margin {res['margin']:+.3f} -> {res['generated']!r}", flush=True)
            dump()
    out["seconds_total"] = time.time() - t0
    out["complete"] = True
    dump()
    print("wrote", p)


if __name__ == "__main__":
    main()

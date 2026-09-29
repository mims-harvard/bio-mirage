"""Scores the candidate answers of one atlas with the CellWhisperer chat model or the base Mistral
model. Third step of four (build_cells, embed_cells, score_llm, analyze). Runs on one GPU in the
LLaVA environment of the CellWhisperer repository; run the three tasks for each atlas.

  --task expA      chat model, representation-only prompt: every cell x {intact, shuffle_s0-2,
                   rankonly_s0} x every answer of the atlas. Supports Fig. 2f.
  --task expB      chat model, prompt with the top-k gene text: every pair x k x {Z_CW and gene
                   text of A, both of B, Z_CW of B with gene text of A, Z_CW of A with gene text of
                   B} x {answer of A, answer of B}. Supports Fig. 3e.
  --task textgate  base Mistral, gene text without Z_CW: every pair x k x gene text of A or B x
                   {answer of A, answer of B}. analyze.py reads it for its secondary pair sets.

Reads cells.jsonl, pairs.jsonl and embeddings.npz and appends one record per scored answer to
single_cell/cellwhisperer/<atlas>/<model>__<task>.jsonl under $INPUT_USE_RESULTS_DIR, with a
manifest beside it. A rerun skips records already written, so an interrupted job resumes.

    python input_use/models/cellwhisperer_driver/score_llm.py --atlas immune1 --task expA --model default
    python input_use/models/cellwhisperer_driver/score_llm.py --atlas immune1 --task expB --model default
    python input_use/models/cellwhisperer_driver/score_llm.py --atlas immune1 --task textgate --model base
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CHAT_MODELS, KS, answer_text, atlas_dir, dump_json, mistral_base, read_jsonl, sha256_array)
from llava_scoring import Scorer  # noqa: E402

EMB_CONDITIONS = ["intact", "shuffle_s0", "shuffle_s1", "shuffle_s2", "rankonly_s0"]
B_CONDITIONS = {  # condition -> (embedding source, gene text source)
    "aligned_A": ("A", "A"), "aligned_B": ("B", "B"),
    "Bembed_Agenes": ("B", "A"), "Aembed_Bgenes": ("A", "B"),
}


def load_tables(atlas, need_embeddings=True):
    d = atlas_dir(atlas)
    cells = read_jsonl(d / "cells.jsonl")
    pairs = read_jsonl(d / "pairs.jsonl")
    if not need_embeddings:
        return d, cells, pairs, {}, {}
    z = np.load(d / "embeddings.npz", allow_pickle=True)
    ids = list(z["cell_ids"].astype(str))
    emb = {c: z[f"emb_{c}"] for c in EMB_CONDITIONS}
    pos = {cid: i for i, cid in enumerate(ids)}
    assert all(c["cell_id"] in pos for c in cells)
    return d, cells, pairs, emb, pos


def append_rows(path, rows):
    with open(path, "a") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def done_keys(path, keyf):
    """Keys already scored. A job killed mid-write can leave a truncated last line; such lines are
    dropped and the file is rewritten with the complete rows only."""
    if not Path(path).exists():
        return set()
    good, bad = [], 0
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                good.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    if bad:
        print(f"[score] dropping {bad} unparsable line(s) from {path}", flush=True)
        tmp = str(path) + ".tmp"
        with open(tmp, "w") as fh:
            for r in good:
                fh.write(json.dumps(r) + "\n")
        os.replace(tmp, path)
    return {keyf(r) for r in good}


def acquire_lock(out_path) -> bool:
    """One writer per output file. The lock names the Slurm job holding it, and a lock whose job is
    no longer running is taken over. Returns False if another job is running this task, in which
    case the caller exits without touching the file."""
    import subprocess
    lock = Path(str(out_path) + ".lock")
    me = os.environ.get("SLURM_JOB_ID", f"pid{os.getpid()}")
    for _ in range(3):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, me.encode()); os.close(fd)
            return True
        except FileExistsError:
            holder = lock.read_text().strip() if lock.exists() else ""
            if holder == me:
                return True
            state = ""
            if holder and not holder.startswith("pid"):
                try:
                    state = subprocess.run(["squeue", "-h", "-j", holder, "-o", "%T"], capture_output=True,
                                           text=True, timeout=60).stdout.strip()
                except Exception:  # noqa: BLE001
                    state = "UNKNOWN"
            if state == "RUNNING" or state == "UNKNOWN":
                print(f"[score] {out_path} is held by job {holder} ({state}); exiting", flush=True)
                return False
            print(f"[score] taking over stale lock from job {holder!r} ({state or 'gone'})", flush=True)
            try:
                lock.unlink()
            except FileNotFoundError:
                pass
    return False


def release_lock(out_path):
    lock = Path(str(out_path) + ".lock")
    me = os.environ.get("SLURM_JOB_ID", f"pid{os.getpid()}")
    if lock.exists() and lock.read_text().strip() == me:
        lock.unlink()


def run_chunks(scorer, work, keyf, out_path, chunk, images_of, make_row, log_every=1):
    """work: list of (key, encode-kwargs, meta). Encodes + scores in chunks, appending rows."""
    done = done_keys(out_path, keyf)
    todo = [w for w in work if w[0] not in done]
    print(f"[score] {len(work)} items, {len(done)} done, {len(todo)} to do", flush=True)
    t0 = time.time()
    for s in range(0, len(todo), chunk):
        part = todo[s:s + chunk]
        items = [scorer.encode(**w[1]) for w in part]
        imgs = [images_of(w[2]) for w in part] if images_of else None
        res = scorer.score(items, imgs)
        rows = [make_row(w, it, r) for w, it, r in zip(part, items, res)]
        append_rows(out_path, rows)
        el = time.time() - t0
        print(f"[score] {s + len(part)}/{len(todo)} rows  {el:.0f}s  ({(s + len(part)) / max(el, 1e-9):.1f} it/s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", required=True)
    ap.add_argument("--task", required=True, choices=["expA", "expB", "textgate"])
    ap.add_argument("--model", default="default", help="default | celltype | base")
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--token_budget", type=int, default=24000)
    ap.add_argument("--max_batch", type=int, default=64)
    ap.add_argument("--limit_pairs", type=int, default=0)
    ap.add_argument("--limit_cells", type=int, default=0)
    ap.add_argument("--ks", nargs="+", type=int, default=KS)
    a = ap.parse_args()

    d, cells, pairs, emb, pos = load_tables(a.atlas, need_embeddings=(a.task != "textgate"))
    if a.limit_cells:
        cells = cells[:a.limit_cells]
    if a.limit_pairs:
        pairs = pairs[:a.limit_pairs]
    cell_by_id = {c["cell_id"]: c for c in cells}
    labels = sorted({c["canonical_label"] for c in cells})
    t0 = time.time()
    if a.task == "textgate":
        assert a.model == "base"
        scorer = Scorer(mistral_base(), kind="base")
    else:
        scorer = Scorer(CHAT_MODELS[a.model], kind="chat", lift_cap=True)
    print(f"[score] loaded {a.model} in {time.time()-t0:.0f}s: {scorer.manifest()}", flush=True)
    out_path = d / f"{a.model}__{a.task}.jsonl"
    if not acquire_lock(out_path):
        return
    man = {"atlas": a.atlas, "task": a.task, "model": a.model, "scorer": scorer.manifest(),
           "n_cells": len(cells), "n_pairs": len(pairs), "labels": labels, "ks": a.ks,
           "token_budget": a.token_budget, "max_batch": a.max_batch,
           "embedding_matrix_hash": {c: sha256_array(v) for c, v in emb.items()}}
    dump_json(d / f"{a.model}__{a.task}_manifest.json", man)

    def emb_vec(cid, cond):
        return emb[cond][pos[cid]]

    if a.task == "expA":
        work = []
        for c in cells:
            for cond in EMB_CONDITIONS:
                for lab in labels:
                    work.append(((c["cell_id"], cond, lab),
                                 dict(genes=None, answer=answer_text(lab), image=True),
                                 dict(cell_id=c["cell_id"], cond=cond, label=lab, correct=c["canonical_label"])))
        keyf = lambda r: (r["cell_id"], r["condition"], r["candidate"])  # noqa: E731

        def make_row(w, it, r):
            m = w[2]
            v = emb_vec(m["cell_id"], m["cond"])
            return {"cell_id": m["cell_id"], "condition": m["cond"], "candidate": m["label"],
                    "is_correct": m["label"] == m["correct"], "correct_label": m["correct"],
                    "logp_sum": r["logp_sum"], "logp_mean": r["logp_mean"], "n_tokens": r["n_tokens"],
                    "prompt_token_count": it["prompt_token_count"], "spliced_len": it["spliced_len"],
                    "within_released_cap": it["within_released_cap"], "emb_hash": sha256_array(v)}
        run_chunks(scorer, work, keyf, out_path, a.chunk, lambda m: emb_vec(m["cell_id"], m["cond"]), make_row)

    elif a.task == "expB":
        work = []
        for p in pairs:
            A, B = cell_by_id[p["cell_A"]], cell_by_id[p["cell_B"]]
            ans = {"A": answer_text(A["canonical_label"]), "B": answer_text(B["canonical_label"])}
            for k in a.ks:
                for cond, (esrc, gsrc) in B_CONDITIONS.items():
                    genes = (A if gsrc == "A" else B)[f"G_{k}"]
                    assert len(genes) == k
                    ecell = A["cell_id"] if esrc == "A" else B["cell_id"]
                    for cand in ("A", "B"):
                        work.append(((p["pair_id"], k, cond, cand),
                                     dict(genes=genes, answer=ans[cand], image=True),
                                     dict(pair_id=p["pair_id"], k=k, cond=cond, esrc=esrc, gsrc=gsrc, cand=cand,
                                          ecell=ecell, answer=ans[cand])))
        keyf = lambda r: (r["pair_id"], r["k"], r["condition"], r["candidate"])  # noqa: E731

        def make_row(w, it, r):
            m = w[2]
            return {"pair_id": m["pair_id"], "k": m["k"], "condition": m["cond"],
                    "embedding_source": m["esrc"], "gene_text_source": m["gsrc"], "candidate": m["cand"],
                    "answer": m["answer"], "embedding_cell_id": m["ecell"],
                    "logp_sum": r["logp_sum"], "logp_mean": r["logp_mean"], "n_tokens": r["n_tokens"],
                    "prompt_token_count": it["prompt_token_count"], "spliced_len": it["spliced_len"],
                    "within_released_cap": it["within_released_cap"],
                    "emb_hash": sha256_array(emb_vec(m["ecell"], "intact"))}
        run_chunks(scorer, work, keyf, out_path, a.chunk, lambda m: emb_vec(m["ecell"], "intact"), make_row)

    else:  # textgate
        work = []
        for p in pairs:
            A, B = cell_by_id[p["cell_A"]], cell_by_id[p["cell_B"]]
            ans = {"A": answer_text(A["canonical_label"]), "B": answer_text(B["canonical_label"])}
            for k in a.ks:
                for gsrc in ("A", "B"):
                    genes = (A if gsrc == "A" else B)[f"G_{k}"]
                    for cand in ("A", "B"):
                        work.append(((p["pair_id"], k, gsrc, cand),
                                     dict(genes=genes, answer=ans[cand], image=False),
                                     dict(pair_id=p["pair_id"], k=k, gsrc=gsrc, cand=cand, answer=ans[cand])))
        keyf = lambda r: (r["pair_id"], r["k"], r["gene_text_source"], r["candidate"])  # noqa: E731

        def make_row(w, it, r):
            m = w[2]
            return {"pair_id": m["pair_id"], "k": m["k"], "gene_text_source": m["gsrc"], "candidate": m["cand"],
                    "answer": m["answer"], "logp_sum": r["logp_sum"], "logp_mean": r["logp_mean"],
                    "n_tokens": r["n_tokens"], "prompt_token_count": it["prompt_token_count"]}
        run_chunks(scorer, work, keyf, out_path, a.chunk, None, make_row)

    man["seconds"] = time.time() - t0
    man["n_rows"] = sum(1 for _ in open(out_path))
    dump_json(d / f"{a.model}__{a.task}_manifest.json", man)
    release_lock(out_path)
    print(f"[score] done {a.atlas} {a.task} {a.model}: {man['n_rows']} rows in {man['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()

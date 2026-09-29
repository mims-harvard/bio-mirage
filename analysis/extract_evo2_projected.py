#!/usr/bin/env python
"""Encodes the reference and variant windows of all 1,449 BioReason queries with Evo2, applies the
released SFT and RL DNA projections, and mean-pools each window over positions.

Checks that the SFT variant block reproduces the stored features in
dna/bioreason/perturbations/evo2_pooled.npz. Requires a GPU and the BioReason checkpoints. Writes
outputs/analysis/evo2_projected_both.npz, or the path given as the first argument, for
dna_probe_projected.py.
"""
import os
import os, sys, time
import numpy as np, torch
from input_use.core.checkpoints import ckpt
from input_use.core import paths as RD
from input_use.core.records import read_examples
from input_use.models.bioreason.diagnose_evo2 import rows_for, DNALLMModel, DLProcessor, resolve_checkpoint

OUT_DIR = os.path.join(os.environ.get("INPUT_USE_OUTPUT_DIR", "outputs"), "analysis")
os.makedirs(OUT_DIR, exist_ok=True)

EX = f"{RD.DNA_PERTURBATIONS}/examples.jsonl"
OLD = f"{RD.DNA_PERTURBATIONS}/evo2_pooled.npz"
OUT = sys.argv[1] if len(sys.argv) > 1 else f"{OUT_DIR}/evo2_projected_both.npz"
ex = [e for e in read_examples(EX) if e.condition == "wt"]
print(f"{len(ex)} wt rows", flush=True)

ck = {c: resolve_checkpoint(ckpt(f"dna_{c}")) for c in ("rl", "sft")}
model = DNALLMModel(ckpt_dir=ck["rl"], text_model_name=ckpt("dna_text_base"), dna_model_name="evo2_1b_base",
                    cache_dir=os.environ.get("HF_HOME"), max_length_dna=2048, max_length_text=1280,
                    text_model_finetune=False, dna_model_finetune=False, dna_is_evo2=True,
                    dna_embedding_layer="blocks.20.mlp.l3", gpu_memory_utilization=0.35, max_model_len=8192)
proc = DLProcessor(tokenizer=model.text_tokenizer, dna_tokenizer=model.dna_tokenizer)
proj = {c: {k: v.to(model.dna_projection.weight.device, model.dna_projection.weight.dtype)
            for k, v in torch.load(os.path.join(ck[c], "dna_projection.pt"), map_location="cpu").items()}
        for c in ck}

res = {}
for c in ("rl", "sft"):
    model.dna_projection.load_state_dict(proj[c], strict=True)
    for side, j in (("ref", 0), ("var", 1)):
        t0 = time.time(); X, L = [], []
        for e in ex:
            r = rows_for(model, proc, e.payload["dna_sequences"][j], 2048)
            X.append(r.mean(0).cpu().numpy()); L.append(r.shape[0])
        res[f"{c}_{side}"] = np.stack(X).astype(np.float32); res[f"{c}_{side}_len"] = np.array(L)
        print(f"{c}_{side}: {res[f'{c}_{side}'].shape}, positions min/median/max "
              f"{min(L)}/{int(np.median(L))}/{max(L)}, {time.time()-t0:.0f}s", flush=True)

z = np.load(OLD, allow_pickle=True)
ids = np.array([e.example_id for e in ex])
assert (z["ids"].astype(str) == ids).all(), "row order differs from evo2_pooled.npz"
d = float(np.abs(res["sft_var"] - z["X"]).max())
print(f"reproduction: max|sft_var - evo2_pooled.X| = {d:.3e} (scale {np.abs(z['X']).max():.3e})")
print(f"max|rl_var - sft_var| = {np.abs(res['rl_var'] - res['sft_var']).max():.3e}")
np.savez(OUT, ids=ids, y=z["y"], groups=z["groups"], repro_maxdiff=d, **res)
print("->", OUT)

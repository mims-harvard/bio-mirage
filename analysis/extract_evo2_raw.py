#!/usr/bin/env python
"""Mean-pools the Evo2 blocks.20.mlp.l3 output (1,920 dimensions, the input to the BioReason DNA
projection) over positions for the reference and variant windows of all 1,449 BioReason queries.

The tensor is captured with a forward pre-hook on the projection, so it is the exact input the
projection receives. The script checks its RL-projected rerun against
outputs/analysis/evo2_projected_both.npz, so run extract_evo2_projected.py first.
Requires a GPU. Writes outputs/analysis/evo2_raw_both.npz for dna_probe_raw.py.
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
ex = [e for e in read_examples(EX) if e.condition == "wt"]
ck = resolve_checkpoint(ckpt("dna_rl"))
model = DNALLMModel(ckpt_dir=ck, text_model_name=ckpt("dna_text_base"), dna_model_name="evo2_1b_base",
                    cache_dir=os.environ.get("HF_HOME"), max_length_dna=2048, max_length_text=1280,
                    text_model_finetune=False, dna_model_finetune=False, dna_is_evo2=True,
                    dna_embedding_layer="blocks.20.mlp.l3", gpu_memory_utilization=0.35, max_model_len=8192)
proc = DLProcessor(tokenizer=model.text_tokenizer, dna_tokenizer=model.dna_tokenizer)
cap = {}
model.dna_projection.register_forward_pre_hook(lambda m, a: cap.__setitem__("x", a[0].detach()))
res = {}
for side, j in (("ref", 0), ("var", 1)):
    t0 = time.time(); R, P, S = [], [], []
    for e in ex:
        p = rows_for(model, proc, e.payload["dna_sequences"][j], 2048)
        x = cap["x"].reshape(-1, cap["x"].shape[-1]).float()
        assert x.shape[0] == p.shape[0], (x.shape, p.shape)
        R.append(x.mean(0).cpu().numpy()); P.append(p.mean(0).cpu().numpy()); S.append(tuple(cap["x"].shape))
    res[f"raw_{side}"] = np.stack(R).astype(np.float32); res[f"rl_{side}"] = np.stack(P).astype(np.float32)
    print(f"raw_{side}: {res[f'raw_{side}'].shape}, hook input shape {S[0]}, {time.time()-t0:.0f}s", flush=True)
z = np.load(f"{OUT_DIR}/evo2_projected_both.npz", allow_pickle=True)
ids = np.array([e.example_id for e in ex]); assert (z["ids"].astype(str) == ids).all()
for s in ("ref", "var"):
    print(f"determinism: max|rl_{s} rerun - stored| = {np.abs(res[f'rl_{s}'] - z[f'rl_{s}']).max():.3e}")
W = model.dna_projection.weight.float().cpu().numpy(); b = model.dna_projection.bias.float().cpu().numpy()
lin = res["raw_var"] @ W.T + b
print(f"linearity: max|W*mean(raw)+b - mean(W*raw+b)| = {np.abs(lin - z['rl_var']).max():.3e} (scale {np.abs(z['rl_var']).max():.3e})")
np.savez(f"{OUT_DIR}/evo2_raw_both.npz", ids=ids, raw_ref=res["raw_ref"], raw_var=res["raw_var"])
print("->", f"{OUT_DIR}/evo2_raw_both.npz")

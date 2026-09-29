#!/usr/bin/env python
"""Within-family AUROC of linear probes on ESM3 representations passed through the BioReason-Pro RL
and SFT projections, on the InterPro-controlled protein subset.

Features are the mean and maximum of each representation concatenated, taken over residues or over
all positions: the ESM3 representation itself and the output of each projection. Folds, AUROC and
bootstrap follow protein_family_table.py, whose output outputs/analysis/one_metric_with_shuffled.json
must exist, since the unprojected features are checked against it. Writes
outputs/analysis/protein_projected_probe.json. Figure 4 uses the mean-pooled variant in
protein_probe_mean_embedding.py.
"""
import os
OUT_DIR = os.path.join(os.environ.get("INPUT_USE_OUTPUT_DIR", "outputs"), "analysis")
os.makedirs(OUT_DIR, exist_ok=True)
HF_DIR = os.environ.get("HF_HOME", "~/.cache/huggingface")
import json, os, sys, hashlib
import numpy as np, torch
from input_use.core.checkpoints import hub_dir
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protein_family_probe as fla
import protein_family_table as om


torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
HUB = f"{HF_DIR}/datasets/hub"
SNAP = {"rl": f"{HUB}/{hub_dir('protein_rl')}/snapshots/24b20477670fb27e35e469b9a73197a81f9c53d0",
        "sft": f"{HUB}/{hub_dir('protein_sft')}/snapshots/b77bd65fdc3c3e95224dc977cc4d4480fdbf8f2b"}
PROJ = {}
for ck, d in SNAP.items():
    m = torch.nn.Sequential(torch.nn.Linear(1536, 2560), torch.nn.GELU(), torch.nn.Linear(2560, 2560))
    m.load_state_dict(torch.load(f"{d}/protein_projection.pt", map_location="cpu"), strict=True)
    PROJ[ck] = m.to(torch.bfloat16).eval()

def feats(seq):
    k = hashlib.sha1(seq.encode()).hexdigest()
    rows = np.load(f"{fla.CACHE}/{k[:2]}/{k}.npy")
    assert rows.shape[1] == 1536 and rows.shape[0] == len(seq) + 2, (rows.shape, len(seq))
    out = {"raw_body": np.concatenate([rows[1:-1].mean(0), rows[1:-1].max(0)])}
    x = torch.from_numpy(rows).to(torch.bfloat16)
    assert torch.equal(x.float(), torch.from_numpy(rows)), "cache is not an exact bf16 copy"
    with torch.no_grad():
        for ck, m in PROJ.items():
            z = m(x).float().numpy()
            out[f"{ck}_all"] = np.concatenate([z.mean(0), z.max(0)])
            out[f"{ck}_body"] = np.concatenate([z[1:-1].mean(0), z[1:-1].max(0)])
    return out

rows = fla.build()
for r in rows:
    r["F"] = feats(r["seq"])
FEATS = ["raw_body", "rl_all", "rl_body", "sft_all", "sft_body"]
FIG_CATS = ["pseudoenzyme", "organelle_targeted", "dna_binding"]
res, pooled = {}, {f: {"fig3": [], "all4": []} for f in FEATS}
for cat in fla.CATS:
    sub = [r for r in rows if r["cat"] == cat]
    y = np.array([r["y"] for r in sub]); g = np.array([f"{r['cat']}:{r['fam']}" for r in sub])
    res[cat] = {"n_proteins": len(sub), "n_families": len(set(g))}
    for f in FEATS:
        X = np.stack([r["F"][f] for r in sub])
        v = np.array(list(om.within_family(om.probe_scores(X, y, g), y, g).values()))
        nulls = [np.mean(list(om.within_family(om.probe_scores(X, y, g, permute=True, seed=100 + i), y, g).values()))
                 for i in range(20)]
        lo, hi = om.ci(v)
        res[cat][f] = {"within_family_auroc": float(v.mean()), "ci95_family": [lo, hi], "n_families": len(v),
                       "dim": int(X.shape[1]), "perm_null_mean": float(np.mean(nulls)), "perm_null_sd": float(np.std(nulls))}
        pooled[f]["all4"].append(v)
        if cat in FIG_CATS:
            pooled[f]["fig3"].append(v)
        print(f"{cat:<20}{len(sub):>5}{len(v):>5}  {f:<10}{X.shape[1]:>6}  {v.mean():.4f} [{lo:.3f}, {hi:.3f}]"
              f"   null {np.mean(nulls):.3f}+/-{np.std(nulls):.3f}", flush=True)
for f in FEATS:
    for k, vs in pooled[f].items():
        v = np.concatenate(vs); lo, hi = om.ci(v)
        res.setdefault(f"POOLED_{k}", {})[f] = {"within_family_auroc": float(v.mean()), "ci95_family": [lo, hi],
                                                "n_families": len(v)}
        print(f"POOLED {k:<5} {f:<10} {v.mean():.4f} [{lo:.3f}, {hi:.3f}] ({len(v)} families)")
pub = json.load(open(f"{OUT_DIR}/one_metric_with_shuffled.json"))
for cat in fla.CATS:
    d = res[cat]["raw_body"]["within_family_auroc"] - pub[cat]["arms"]["ESM3 linear probe"]["within_family_auroc"]
    res[cat]["raw_body_minus_published"] = d
    print(f"reproduction {cat}: raw_body - published = {d:+.2e}")
json.dump(res, open(os.path.join(OUT_DIR, "protein_projected_probe.json"), "w"), indent=1)

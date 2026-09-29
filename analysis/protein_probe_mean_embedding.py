#!/usr/bin/env python
"""Within-family AUROC of linear probes on mean-pooled ESM3 representations, with and without the
BioReason-Pro RL projection, for the ESM3 probe bars of Figure 4 (left, panel b).

Five feature sets are probed per property: the mean and maximum over residues concatenated, the mean
over residues, the mean over all positions including the start and end tokens, and the RL projection
output averaged over all positions and over residues. The figure draws the mean over all positions
before and after the RL projection. Folds, AUROC and bootstrap follow protein_family_table.py, whose
output outputs/analysis/one_metric_with_shuffled.json must exist, since the first feature set is
checked against it. Writes outputs/analysis/protein_meanonly_probe.json.
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
    out = {"raw_body_meanmax": np.concatenate([rows[1:-1].mean(0), rows[1:-1].max(0)]),
           "raw_body_mean": rows[1:-1].mean(0), "raw_all_mean": rows.mean(0)}
    x = torch.from_numpy(rows).to(torch.bfloat16)
    with torch.no_grad():
        z = PROJ["rl"](x).float().numpy()
    out["rl_all_mean"] = z.mean(0); out["rl_body_mean"] = z[1:-1].mean(0)
    return out

rows = fla.build()
for r in rows:
    r["F"] = feats(r["seq"])
FEATS = ["raw_body_meanmax", "raw_body_mean", "raw_all_mean", "rl_all_mean", "rl_body_mean"]
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
    print(f"reproduction {cat}: raw_body_meanmax - published = "
          f"{res[cat]['raw_body_meanmax']['within_family_auroc'] - pub[cat]['arms']['ESM3 linear probe']['within_family_auroc']:+.2e}")
json.dump(res, open(os.path.join(OUT_DIR, "protein_meanonly_probe.json"), "w"), indent=1)

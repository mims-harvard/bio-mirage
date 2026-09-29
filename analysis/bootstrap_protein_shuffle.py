#!/usr/bin/env python
"""95% percentile bootstrap interval for the BioReason-Pro "ESM3 shuffled" bar with GO-GPT and InterPro
text in Figure 2c, which is the mean weighted F_max over three ESM3 shuffling seeds.

Resamples proteins, 2,000 draws split over 8 worker processes (seeds 0 to 7), with all four
conditions (intact and the three shuffles) scored on the same draw. Each shuffle is scored only over
the proteins it was run on. Point estimates are asserted against the run's metrics.json. Runs the RL
and SFT checkpoints, or those named as arguments, and writes
outputs/analysis/ci_protein_meanperm.json.

    python analysis/bootstrap_protein_shuffle.py [RL] [SFT]
"""
import os
OUT_DIR = os.path.join(os.environ.get("INPUT_USE_OUTPUT_DIR", "outputs"), "analysis")
os.makedirs(OUT_DIR, exist_ok=True)
REPO_DIR = os.environ.get("INPUT_USE_HOME", ".")
import glob, json, os, random, sys
from collections import defaultdict
from multiprocessing import Pool
sys.path.insert(0, REPO_DIR)
from input_use.core import config as cfg                                       # noqa: E402
from input_use.core import paths as RD                                         # noqa: E402
from input_use.metrics import go_dag                                           # noqa: E402
from input_use.metrics.cafa import _cafa_f1_parts, _f1_from_parts, load_ia     # noqa: E402


RUNS = {"RL": RD.PROTEIN_PERTURBATIONS["rl"], "SFT": RD.PROTEIN_PERTURBATIONS["sft"]}
PERM = ["shuffle_esm3_only", "shuffle_esm3_only_s1", "shuffle_esm3_only_s2"]
CONDS = ["wt"] + PERM
ASPECTS = ("MF", "BP", "CC")
N_BOOT, N_CHUNK, ALPHA = 2000, 8, 0.05
G = {}


def build(run):
    by_ex = defaultdict(dict)
    parts = [p for p in sorted(glob.glob(f"{run}/records.jsonl*")) if p.endswith("records.jsonl") or p.split(".")[-1].isdigit()]
    for p in parts:
        with open(p) as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    if r["condition"] in CONDS:
                        by_ex[r["example_id"]][r["condition"]] = r
    preds, gt = defaultdict(dict), {}
    for ex, conds in by_ex.items():            # same rule as core.score.score_protein
        if "wt" not in conds:
            continue
        gt[ex] = set(conds["wt"]["ground_truth"].get("go_terms", []))
        preds["wt"][ex] = set(conds["wt"]["output"]["parsed"]["go_terms"])
        for c in PERM:
            if c in conds:
                preds[c][ex] = set(conds[c]["output"]["parsed"]["go_terms"])
    return preds, gt


def chunk(args):
    seed, n_draws = args
    parts, members, proteins = G["parts"], G["members"], G["proteins"]
    rng, n = random.Random(seed), len(proteins)
    out = []
    for _ in range(n_draws):
        sample = [proteins[rng.randrange(n)] for _ in range(n)]
        idx = {a: [p for p in sample if p in members[a]] for a in ASPECTS}
        v = {}
        for c in CONDS:
            idx_c = {a: [p for p in idx[a] if p in parts[c][a]] for a in ASPECTS}
            v[c] = _f1_from_parts(parts[c], ASPECTS, idx_c)
        v["mean_perm"] = sum(v[c] for c in PERM) / len(PERM)
        out.append(v)
    return out


def pct(vals):
    s = sorted(vals)
    return [float(s[int((ALPHA / 2) * len(s))]), float(s[min(len(s) - 1, int((1 - ALPHA / 2) * len(s)))])]


def main():
    cks = sys.argv[1:] or ["RL", "SFT"]
    dag, ia = go_dag.load(str(cfg.GO_OBO)), load_ia(str(cfg.GO_IA))
    res = {"method": "percentile bootstrap, 2000 draws over proteins (8 chunks, seeds 0-7); arms paired within draw; "
                     "IA-weighted F1 as in metrics.cafa.cafa_f1_bootstrap", "ia_file": str(cfg.GO_IA), "runs": {}}
    for ck in cks:
        run = RUNS[ck]
        metrics = json.load(open(f"{run}/metrics.json"))
        preds, gt = build(run)
        # score each arm over the proteins it was run on (metrics.cafa.coverage_groups): _s1/_s2
        # cover a subset, and scoring them against every protein's ground truth would count the
        # uncovered ones as recall misses
        parts = {c: _cafa_f1_parts(preds[c], {p: gt[p] for p in preds[c]}, dag, ASPECTS, ia=ia) for c in CONDS}
        cover = {c: len(preds[c]) for c in CONDS}
        point = {c: _f1_from_parts(parts[c], ASPECTS) for c in CONDS}
        for c in CONDS:
            assert abs(point[c] - metrics["go_fmax_weighted"][c]) < 1e-4, (ck, c, point[c], metrics["go_fmax_weighted"][c])
        G.update(parts=parts, proteins=sorted(gt),
                 members={a: {p for c in parts for p in parts[c][a]} for a in ASPECTS})
        print(f"[{ck}] {len(gt)} proteins; coverage {cover}; points reproduce metrics.json: "
              + ", ".join(f"{c} {point[c]:.4f}" for c in CONDS), flush=True)
        per = [N_BOOT // N_CHUNK + (1 if i < N_BOOT % N_CHUNK else 0) for i in range(N_CHUNK)]
        with Pool(N_CHUNK) as pool:
            draws = [d for part in pool.map(chunk, list(zip(range(N_CHUNK), per))) for d in part]
        stored = metrics["go_f1_bootstrap"]
        entry = {"n_proteins": len(gt), "coverage": cover, "n_draws": len(draws),
                 "mean_perm": {"value": sum(point[c] for c in PERM) / 3, "ci95": pct([d["mean_perm"] for d in draws])}}
        for c in CONDS:
            entry[c] = {"value": point[c], "ci95": pct([d[c] for d in draws]), "stored_ci95": stored[c]["ci"]}
        res["runs"][ck] = entry
        print(f"[{ck}] " + json.dumps(entry), flush=True)
    json.dump(res, open(f"{OUT_DIR}/ci_protein_meanperm.json", "w"), indent=1)
    print("->", f"{OUT_DIR}/ci_protein_meanperm.json")


if __name__ == "__main__":
    main()

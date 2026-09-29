#!/usr/bin/env python
"""Tabulates C2S-Scale 2B and 27B cell type annotation accuracy under DEG removal on five atlases.

For each atlas and model it reads `<atlas>/c2s_scale_<size>/records.jsonl` from --root (default
single_cell/c2s_scale/deg_removal under INPUT_USE_RESULTS_DIR). It reports accuracy on the
unmodified cell sentence, without the cell sentence, with the gene order permuted and with a random
gene set, then accuracy after removing 10% to 100% of the strongest DEGs, the weakest DEGs, random
genes, or non-DEGs with similar expression. These are the DEG removal conditions of RQ1 (Fig. 4
right).

    python -m input_use.core.analyze_deg_removal [--out outputs/analysis/deg_removal_summary.json]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from input_use.core import paths as RD
from input_use.core.stats import bootstrap_ci

DATASETS = ["immune1", "immune2", "immune3", "pancreas", "lung"]
MODELS = ["2b", "27b"]
DOSES = ["p10", "p25", "p50", "p75", "p100"]
ARMS = {"top": "top_deg_dropout", "bottom": "bottom_deg_dropout",
        "random": "random_dropout", "matched": "matched_nondeg_dropout"}


def load(rundir: Path):
    by = defaultdict(dict)
    p = rundir / "records.jsonl"
    if not p.exists():
        return by
    for line in open(p):
        if not line.strip():
            continue
        r = json.loads(line)
        by[r["example_id"]][r["condition"]] = r
    return by


def analyze(rundir: Path) -> dict | None:
    by = load(rundir)
    if not by:
        return None
    core = ["wt", "no_modality", "scramble_rank", "scramble_geneset"]
    arms = [f"{pre}_{d}" for name, pre in ARMS.items() for d in DOSES
            if not (name == "bottom" and d == "p100")]      # bottom@100% == top@100%, never emitted
    need = core + arms
    cells = [c for c, v in by.items() if all(k in v for k in need)]
    if len(cells) < 50:
        return {"n_cells": len(cells), "incomplete": True}

    def acc(cond):
        return sum(by[c][cond]["output"]["parsed"]["cell_type"]
                   == by[c][cond]["ground_truth"]["cell_type"] for c in cells) / len(cells)

    def vals(cond):
        return [1.0 if by[c][cond]["output"]["parsed"]["cell_type"]
                == by[c][cond]["ground_truth"]["cell_type"] else 0.0 for c in cells]

    n_opts = len(by[cells[0]]["wt"]["input"]["options"])
    chance = 1.0 / n_opts
    wt, nm = acc("wt"), acc("no_modality")
    lo, hi = bootstrap_ci(vals("wt"))
    out = {
        "n_cells": len(cells), "n_options": n_opts, "chance": round(chance, 4),
        "wt": round(wt, 4), "wt_ci95": [lo, hi], "no_modality": round(nm, 4),
        "M0": round(nm / wt, 4) if wt else None,
        "M0_adj": round(max(0.0, (nm - chance)) / (wt - chance), 4) if wt > chance else None,
        "scramble_rank": round(acc("scramble_rank"), 4),
        "scramble_geneset": round(acc("scramble_geneset"), 4),
        "rank_cost": round(wt - acc("scramble_rank"), 4),
        "identity_cost": round(wt - acc("scramble_geneset"), 4),
        "doses": {},
    }
    for d in DOSES:
        row = {name: round(acc(f"{pre}_{d}"), 4) for name, pre in ARMS.items()
               if not (name == "bottom" and d == "p100")}
        row["G_grounding_gap"] = round(row["matched"] - row["top"], 4)
        row["top_minus_random"] = round(row["top"] - row["random"], 4)
        if "bottom" in row:
            row["top_minus_bottom"] = round(row["top"] - row["bottom"], 4)
        out["doses"][d] = row
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=RD.C2S_DEG_REMOVAL, help="directory with one run per atlas")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    res = {}
    for ds in DATASETS:
        for m in MODELS:
            r = analyze(Path(a.root) / ds / f"c2s_scale_{m}")
            if r and not r.get("incomplete"):
                res[f"{ds}|{m}"] = r

    if not res:
        print("no complete (dataset, model) cells yet"); return

    print("=== HEADLINE: accuracy, prior baseline, rank vs identity ===")
    print(f"{'dataset':<9} {'mdl':<4} {'n':>5} {'chance':>7} {'wt':>6} {'noMod':>6} {'M0':>6} "
          f"{'M0adj':>6} {'rankCost':>9} {'identCost':>10}")
    for k, r in res.items():
        ds, m = k.split("|")
        print(f"{ds:<9} {m:<4} {r['n_cells']:>5} {r['chance']:>7.3f} {r['wt']:>6.3f} "
              f"{r['no_modality']:>6.3f} {(r['M0'] if r['M0'] is not None else float('nan')):>6.3f} "
              f"{(r['M0_adj'] if r['M0_adj'] is not None else float('nan')):>6.3f} "
              f"{r['rank_cost']:>9.3f} {r['identity_cost']:>10.3f}")

    print("\n=== TOP vs RANDOM vs BOTTOM vs MATCHED (accuracy at each dose) ===")
    for k, r in res.items():
        ds, m = k.split("|")
        print(f"\n{ds} / {m}  (wt {r['wt']:.3f}, n={r['n_cells']})")
        print(f"  {'dose':<6} {'top':>7} {'bottom':>7} {'random':>7} {'matched':>8} | "
              f"{'G':>7} {'top-rand':>9} {'top-bot':>8}")
        for d, row in r["doses"].items():
            # bottom@p100 is deliberately not emitted (it deletes the same set as top@p100)
            bot = f"{row['bottom']:.3f}" if "bottom" in row else "  --"
            tmb = f"{row['top_minus_bottom']:+.3f}" if "top_minus_bottom" in row else "   --"
            print(f"  {d:<6} {row['top']:>7.3f} {bot:>7} {row['random']:>7.3f} "
                  f"{row['matched']:>8.3f} | {row['G_grounding_gap']:>+7.3f} "
                  f"{row['top_minus_random']:>+9.3f} {tmb:>8}")

    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2))
        print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""95% percentile bootstrap intervals for the C2S-Scale accuracies in Figure 2e and Figure 4 (right).

Resamples cells within each atlas, 2,000 draws per atlas and model (seed 1000 * model index + atlas
index). Intervals for the mean over the five atlases average the per-atlas draws. Covers intact
accuracy, gene order shuffling and resampling from all expressed genes (27B only) per atlas, and the
accuracy and the accuracy drop at every DEG removal fraction for the four removal conditions. Point
estimates are asserted against the stored deg_removal_summary.json. Writes
outputs/analysis/ci_c2s.json, which the figure scripts read to draw error bars.
"""
import os
OUT_DIR = os.path.join(os.environ.get("INPUT_USE_OUTPUT_DIR", "outputs"), "analysis")
os.makedirs(OUT_DIR, exist_ok=True)
REPO_DIR = os.environ.get("INPUT_USE_HOME", ".")
import json, os, sys
import numpy as np
sys.path.insert(0, REPO_DIR)
from input_use.core.analyze_deg_removal import ARMS, DOSES, DATASETS, MODELS  # noqa: E402
from input_use.core import paths as RD  # noqa: E402


RX = RD.C2S_RANDOM_EXPRESSED
N_BOOT, ALPHA = 2000, 0.05
LO, HI = int((ALPHA / 2) * N_BOOT), min(N_BOOT - 1, int((1 - ALPHA / 2) * N_BOOT))


def pct(draws):
    s = np.sort(np.asarray(draws))
    return [round(float(s[LO]), 4), round(float(s[HI]), 4)]


def load_correct(path, keep=None):
    by = {}
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if keep is not None and r["condition"] not in keep:
                continue
            by.setdefault(r["example_id"], {})[r["condition"]] = float(
                r["output"]["parsed"]["cell_type"] == r["ground_truth"]["cell_type"])
    return by


def main():
    ref = json.load(open(RD.C2S_DEG_REMOVAL_SUMMARY))
    rx_ref = json.load(open(f"{RX}/random_expressed_27b.json"))
    core = ["wt", "no_modality", "scramble_rank", "scramble_geneset"]
    arms = [f"{pre}_{d}" for name, pre in ARMS.items() for d in DOSES if not (name == "bottom" and d == "p100")]
    need = core + arms
    out = {"method": "percentile bootstrap, 2000 draws, resampling cells within atlas; rule = core.stats.bootstrap_ci",
           "per_atlas": {}, "singlecell_macro_drop": {}, "singlecell_macro_acc": {}}
    boots = {}                                    # (ds, m) -> {cond: draws}
    for mi, m in enumerate(MODELS):
        for di, ds in enumerate(DATASETS):
            key = f"{ds}|{m}"
            by = load_correct(f"{RD.deg_removal(ds)}/c2s_scale_{m}/records.jsonl")
            cells = [c for c, v in by.items() if all(k in v for k in need)]
            r = ref[key]
            assert len(cells) == r["n_cells"], (key, len(cells), r["n_cells"])
            conds = ["wt", "scramble_rank"] + arms
            M = {c: np.array([by[x][c] for x in cells]) for c in conds}
            if m == "27b":
                rx = load_correct(f"{RX}/{ds}/c2s_scale_27b/records.jsonl", keep={"random_expressed"})
                M["random_expressed"] = np.array([rx[x]["random_expressed"] for x in cells])
                assert round(M["random_expressed"].mean(), 4) == rx_ref[ds]["random_expressed"], (ds, "random_expressed")
            assert round(M["wt"].mean(), 4) == r["wt"], (key, "wt")
            assert round(M["scramble_rank"].mean(), 4) == r["scramble_rank"], (key, "scramble_rank")
            for name, pre in ARMS.items():
                for d in DOSES:
                    if name == "bottom" and d == "p100":
                        continue
                    assert round(M[f"{pre}_{d}"].mean(), 4) == r["doses"][d][name], (key, pre, d)
            rng = np.random.default_rng(1000 * mi + di)
            idx = rng.integers(0, len(cells), size=(N_BOOT, len(cells)))
            boots[key] = {c: v[idx].mean(axis=1) for c, v in M.items()}
            entry = {"n_cells": len(cells)}
            for c in ("wt", "scramble_rank", "random_expressed"):
                if c in M:
                    entry[c] = {"acc": round(float(M[c].mean()), 4), "ci95": pct(boots[key][c])}
            out["per_atlas"][key] = entry
            print(key, {c: (e["acc"], e["ci95"]) for c, e in entry.items() if c != "n_cells"}, flush=True)
            boots[key]["_point"] = {c: float(v.mean()) for c, v in M.items()}
    for m in MODELS:
        res, acc = {}, {}
        wt_rows = np.stack([boots[f"{ds}|{m}"]["wt"] for ds in DATASETS])
        res["wt_macro"] = {"value": float(np.mean([boots[f"{ds}|{m}"]["_point"]["wt"] for ds in DATASETS])),
                           "ci95": pct(wt_rows.mean(axis=0))}
        for name, pre in ARMS.items():
            pts = []
            for d in DOSES:
                if name == "bottom" and d == "p100":
                    continue
                c = f"{pre}_{d}"
                draws = np.mean([boots[f"{ds}|{m}"]["wt"] - boots[f"{ds}|{m}"][c] for ds in DATASETS], axis=0)
                val = float(np.mean([boots[f"{ds}|{m}"]["_point"]["wt"] - boots[f"{ds}|{m}"]["_point"][c] for ds in DATASETS]))
                plotted = float(np.mean([ref[f"{ds}|{m}"]["wt"] - ref[f"{ds}|{m}"]["doses"][d][name] for ds in DATASETS]))
                assert abs(val - plotted) < 1e-3, (m, name, d, val, plotted)      # plot uses 4-dp rounded inputs
                pts.append({"dose": d, "value": val, "plotted_value": plotted, "ci95": pct(draws)})
                a_draws = np.mean([boots[f"{ds}|{m}"][c] for ds in DATASETS], axis=0)
                a_val = float(np.mean([boots[f"{ds}|{m}"]["_point"][c] for ds in DATASETS]))
                a_plotted = float(np.mean([ref[f"{ds}|{m}"]["doses"][d][name] for ds in DATASETS]))
                assert abs(a_val - a_plotted) < 1e-3, (m, name, d, a_val, a_plotted)
                acc.setdefault(name, []).append({"dose": d, "value": a_val, "plotted_value": a_plotted,
                                                 "ci95": pct(a_draws)})
            res[name] = pts
        out["singlecell_macro_drop"][m] = res
        out["singlecell_macro_acc"][m] = acc
        print(m, {k: [(p["dose"], round(p["plotted_value"], 3), p["ci95"]) for p in v] if isinstance(v, list) else v
                  for k, v in res.items()}, flush=True)
    json.dump(out, open(f"{OUT_DIR}/ci_c2s.json", "w"), indent=1)
    print("->", f"{OUT_DIR}/ci_c2s.json")


if __name__ == "__main__":
    main()

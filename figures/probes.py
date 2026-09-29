#!/usr/bin/env python
"""Draws Figure 4 left ("Targets remain predictable from biological representations"): linear probes
against BioReason and BioReason-Pro.

Panel a is disease prediction accuracy on the 165 genome-dependent queries for linear probes on the
Evo2 representations, a text-only predictor and BioReason (RL) with Z_Evo2 intact and shuffled.
Panel b is the mean within-family AUROC per protein category for linear probes on the ESM3
representations and BioReason-Pro. Reads dna/bioreason/linear_probes and
protein/bioreason_pro/linear_probes under INPUT_USE_RESULTS_DIR and the rq1_data.py loaders. Writes
fig_rq1_probes and its numbers CSV to outputs/figures, or to --out_dir.

    python figures/probes.py
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figure_style import (init_print_style, center_ylabel, ax_in, finish, bar_label,  # noqa: E402
                          align_xlabels, panel_letter, emit, DNA, PROTEIN, GREY,
                          TICK_FS, LEGEND_FS, EDGE_LW, ERR_LW, _mix)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rq1_data as M  # noqa: E402

OUT_DEFAULT = M.OUT_DEFAULT
STEM = "fig_rq1_probes"
# palette's own probe/baseline tints, so the ordering probe < non-model predictor < model <
# perturbed reads as lightness alone.
PROBE_FILL = {"dna": _mix(DNA.model, "#FFFFFF", 0.82), "protein": _mix(PROTEIN.model, "#FFFFFF", 0.78)}
# Both stay in the DNA hue: the probe is the lightest tint and the text-only predictor a darker
# one, so the two pale bars stay distinguishable.
BASELINE_FILL = _mix(DNA.model, "#FFFFFF", 0.45)

# probes -- on the foundation-model output (lightest tint) and on the RL checkpoint's projected rows
# (the model's actual
PROJ_FILL = {"dna": _mix(DNA.model, "#FFFFFF", 0.64), "protein": _mix(PROTEIN.model, "#FFFFFF", 0.50)}
PROT_ARMS = [("ESM3 probe,\noriginal", PROBE_FILL["protein"], None),                    # two-line labels: the legend
             ("ESM3 probe,\nRL-projected", PROJ_FILL["protein"], None),
             ("BioReason-Pro RL,\nESM3 intact", PROTEIN.model, None),                   # column right of panel b is 1.3 in
             ("BioReason-Pro RL,\nESM3 shuffled", PROTEIN.perturbed, None)]
DNA_PROBES = [("Evo2 probe, original", f"{M.RD.DNA_LINEAR_PROBES}/dna_raw_probe.json", "raw_ref||raw_var (unprojected, both blocks)"),
              ("Evo2 probe, RL-projected", f"{M.RD.DNA_LINEAR_PROBES}/dna_projected_probe.json",
               "rl_ref||rl_var (RL projection, both blocks the model receives)")]
PROT_PROBES = [("ESM3 probe, original", "raw_all_mean"), ("ESM3 probe, RL-projected", "rl_all_mean")]
PROT_JSON = f"{M.RD.PROTEIN_LINEAR_PROBES}/protein_meanonly_probe.json"


def err(ax, x, v, ci):
    """95% bootstrap interval as a plain black line (no caps); returns the height a value label should clear."""
    if not ci:
        return v
    ax.errorbar([x], [v], yerr=[[v - ci[0]], [ci[1] - v]], fmt="none", ecolor="black",
                elinewidth=ERR_LW, capsize=0, zorder=5)
    return max(v, ci[1])


DNA_CI_KEYS = ("Evo2 linear probe, true labels", "text-only predictor",
               "BioReason RL, Evo2 intact", "BioReason RL, Evo2 shuffled")
A_TICKS = ["Evo2 probe, original", "Evo2 probe, RL-projected", "text-only predictor", "BioReason, Evo2 intact",
           "BioReason, Evo2 shuffled"]   # one line each: two-line labels interleave at 45 degrees with
           # five bars


def dna_cis(dna):
    """Genome-cluster bootstrap CIs for panel a (outputs/analysis/ci_dna_probe.json, from analysis/bootstrap_dna_probe.py), asserted against the plotted values."""
    j = M.ci_json("ci_dna_probe.json")
    if not j:
        return {k: None for k in DNA_CI_KEYS}
    plotted = dict(zip(DNA_CI_KEYS, (dna["probe"]["acc"], dna["probe"]["text_baseline"],
                                     dna["acc"]["RL"]["wt"]["dep"], dna["acc"]["RL"]["scramble"]["dep"])))
    del plotted[DNA_CI_KEYS[0]]             # the probe bar is drawn from the linear_probes JSON,
    # not from this JSON
    for k, v in plotted.items():
        tol = 0.003 if k == DNA_CI_KEYS[0] else 1e-9
        assert abs(j["bars"][k]["value"] - v) <= tol, (k, j["bars"][k]["value"], v)
    return {k: j["bars"][k]["ci95"] for k in DNA_CI_KEYS}


def draw(dna, fam, out_dir):
    W = 4.20
    TOP = 0.10
    TIT = 0.20                   # two-line titles
    AXH = 1.25
    BOT = 1.20                   # 45-degree tick labels only (~0.38 in): no x-axis titles, no
                                 # legend band, and trimmed to the labels (user: less white space at
                                 # the bottom)
    H = TOP + TIT + AXH + BOT
    fig = plt.figure(figsize=(W, H))
    L_AX = 0.46
    row_top = TOP + TIT
    a_w, b_w = 1.45, 1.62        # panel b widened into the gap; the legend column is tighter (user:
    # less white
    gap = 0.46                   # space to the right of panel b)
    a_x = L_AX
    b_x = a_x + a_w + gap
    leg_x = b_x + b_w + 0.08     # left edge of panel b's legend column
    assert b_x + b_w <= W, (b_x, b_w, W)

    def title(ax, text):
        ax.set_title(text, fontsize=TICK_FS, fontweight="normal", pad=3.0, linespacing=1.15)

    # ---- a: DNA, one group of four bars on the 165 genome-dependent rows, RL only
    pr = dna["probe"]
    cis = dna_cis(dna)
    fills = [PROBE_FILL["dna"], PROJ_FILL["dna"]]
    a_bars = [(name, dna["new_probes"][name]["acc"], col, None) for (name, _, _), col in zip(DNA_PROBES, fills)]
    for name, *_ in DNA_PROBES:
        cis[name] = dna["new_probes"][name]["ci95_genome"]
    a_bars += [("text-only predictor", pr["text_baseline"], BASELINE_FILL, None),
              ("BioReason RL, Evo2 intact", dna["acc"]["RL"]["wt"]["dep"], DNA.model, None),
              ("BioReason RL, Evo2 shuffled", dna["acc"]["RL"]["scramble"]["dep"], DNA.perturbed, None)]
    ax = ax_in(fig, a_x, row_top, a_w, AXH)
    finish(ax, "Accuracy", ylim=(0, 1.10), yticks=np.arange(0, 1.01, 0.25), spine_top=1.0)
    center_ylabel(ax, 1.0)
    bw, step = 0.117, 0.19       # bar width and spacing; no legend: each
                                 # bar carries its own tick label
    xs = [(k - (len(a_bars) - 1) / 2) * step for k in range(len(a_bars))]
    for (label, v, col, hatch), xb in zip(a_bars, xs):
        ax.bar(xb, v, bw, color=col, edgecolor="black", linewidth=EDGE_LW, hatch=hatch)
        bar_label(ax, xb, err(ax, xb, v, cis[label]), f"{v:.3f}")      # rotated: four thin bars, flat labels collide
    ax.set_xticks(xs)
    ax.set_xticklabels(A_TICKS, rotation=45, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", labelsize=6.0)
    ax.set_xlim(xs[0] - step * 0.9, xs[-1] + step * 0.9)
    ax.set_xlabel("")
    title(ax, "DNA")
    ax_a = ax
    panel_letter(fig, 0.02, TOP, "a")

    # ---- b: protein, mean per-family AUROC
    ax = ax_in(fig, b_x, row_top, b_w, AXH)
    xs = np.arange(len(fam))
    w = 0.2
    finish(ax, "Mean per-family AUROC", ylim=(0, 1.32), yticks=np.arange(0, 1.01, 0.25),   # 1.32: room for labels above CIs reaching 1.0
           spine_top=1.0)
    center_ylabel(ax, 1.0)
    for i, r in enumerate(fam):
        for k, ((name, v, ci, _), (_, col, hatch)) in enumerate(zip(r["vals"], PROT_ARMS)):
            xb = xs[i] + (k - 1.5) * w
            ax.bar(xb, v, w, color=col, edgecolor="black", linewidth=EDGE_LW, hatch=hatch)
            bar_label(ax, xb, err(ax, xb, v, ci), f"{v:.3f}")    # ci = protein_family_table ci95_family
    ax.set_xticks(xs)
    ax.set_xticklabels([r["label"] for r in fam], rotation=45, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", labelsize=6.0)
    ax.set_xlim(-0.58, len(fam) - 0.42)
    ax.set_xlabel("")
    n_prot = sum(r["n_proteins"] for r in fam)
    n_fam = sum(r["n_families"] for r in fam)
    title(ax, "Protein")
    ax_b = ax
    panel_letter(fig, b_x - 0.50, TOP, "b")

    b_handles = [Rectangle((0, 0), 1, 1, facecolor=col, edgecolor="black", lw=EDGE_LW, hatch=hatch,
                           label=label) for label, col, hatch in PROT_ARMS]
    fig.legend(handles=b_handles, loc="upper center", ncol=2,
               bbox_to_anchor=((b_x + b_w / 2) / W, 1 - (row_top + AXH + 0.50) / H),
               frameon=False, fontsize=LEGEND_FS, handlelength=1.1, handletextpad=0.4, labelspacing=0.3,
               columnspacing=0.8)
    return emit(fig, STEM, out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=OUT_DEFAULT)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    dna = M.data_dna_rows()              # records ("dna", "b")
    cis = dna_cis(dna)
    for r in M.ROWS:
        if (r["figure"], r["panel"]) != ("dna", "b"):
            continue
        key = None
        if r.get("model") == "Evo2 row probe":
            key = {"Evo2 probe, all genome-dependent rows": DNA_CI_KEYS[0],
                   "text-only predictor (training folds), all rows": DNA_CI_KEYS[1]}.get(r["series"])
        elif r.get("checkpoint") == "RL" and r.get("group") == "all 165 genome-dependent rows":
            key = {"wt": DNA_CI_KEYS[2], "scramble": DNA_CI_KEYS[3]}.get(r.get("x"))
        if key and cis[key]:
            r["ci_low"], r["ci_high"] = cis[key]
    fam = M.data_family_auroc()          # records ("protein", "b")

    import json
    new_rows = []
    dna["new_probes"] = {}
    for name, path, key in DNA_PROBES:
        r = json.load(open(path))[key]
        assert r["dim"] in (3840, 4096) and abs(r["text_baseline"] - dna["probe"]["text_baseline"]) < 1e-12, (name, r)
        dna["new_probes"][name] = r
        new_rows.append({"figure": "probes", "panel": "a", "model": name, "checkpoint": "RL", "group":
                         "all 165 genome-dependent rows", "series": "mean-pooled reference||variant block", "x": "",
                         "value": r["acc"], "n": 165, "ci_low": r["ci95_genome"][0], "ci_high": r["ci95_genome"][1],
                         "sd_over_folds": r["acc_sd"], "source": path, "key": f"{key}.acc"})
    pj = json.load(open(PROT_JSON))
    for row in fam:
        assert row["vals"][0][0] == "ESM3 linear probe", row["vals"][0]
        probes = []
        for name, key in PROT_PROBES:
            pa = pj[row["cat"]][key]
            probes.append((name, pa["within_family_auroc"], pa["ci95_family"], None))
            new_rows.append({"figure": "probes", "panel": "b", "model": name, "group": row["cat"],
                             "series": "within-family AUROC", "x": row["cat"], "value": pa["within_family_auroc"],
                             "n": row["n_proteins"], "ci_low": pa["ci95_family"][0], "ci_high": pa["ci95_family"][1],
                             "source": PROT_JSON, "key": f"{row['cat']}.{key}.within_family_auroc"})
        row["vals"] = probes + row["vals"][1:]
        print("[protein b, new] " + row["cat"] + " " + " ".join(f"{v:.3f}" for _, v, _, _ in row["vals"]))

    init_print_style()
    draw(dna, fam, a.out_dir)

    panel_of = {("dna", "b"): "a", ("protein", "b"): "b"}

    def drawn(r):
        if (r["figure"], r["panel"]) != ("dna", "b"):
            return True
        if r.get("model") == "Evo2 row probe":
            return r["series"] == "text-only predictor (training folds), all rows"
        return r.get("checkpoint") == "RL" and r.get("x") in ("wt", "scramble") \
            and r.get("group") == "all 165 genome-dependent rows"
    keys = ["figure", "panel", "model", "checkpoint", "group", "series", "x", "value", "n", "k",
            "ci_low", "ci_high", "sd_over_folds", "source", "key"]
    rows = [{**{k: r.get(k, "") for k in keys}, "figure": "probes",
             "panel": panel_of[(r["figure"], r["panel"])]}
            for r in M.ROWS if (r["figure"], r["panel"]) in panel_of and drawn(r)
            and r.get("model") != "ESM3 linear probe"]
    rows += [{k: r.get(k, "") for k in keys} for r in new_rows]
    csv_path = os.path.join(a.out_dir, f"{STEM}_numbers.csv")
    with open(csv_path, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=keys)
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"wrote {csv_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

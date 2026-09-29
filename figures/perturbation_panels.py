#!/usr/bin/env python
"""Biological input perturbation data loaders for Figure 2, and a standalone figure fig_rq1_perturbation
with one panel per modality.

Panel a is BioReason (RL) disease prediction accuracy with Z_Evo2 from intact or shuffled DNA under
four text conditions, panel b BioReason-Pro weighted F_max with ESM3 intact or shuffled under four
text conditions, and panel c C2S-Scale 27B cell type accuracy per atlas with the gene sentence intact
or perturbed. perturbations_and_conflicts.py uses data_dna_rl() and data_c2s_atlas(). Writes
fig_rq1_perturbation and a numbers CSV to outputs/figures, or to --out_dir.
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics as st
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figure_style import (init_print_style, center_ylabel, ax_in, finish, bar_label,  # noqa: E402
                          align_xlabels, panel_letter, emit, DNA, PROTEIN, CELL,
                          GREY, TICK_FS, LEGEND_FS, EDGE_LW, ERR_LW)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rq1_data as M  # noqa: E402  (data loaders + the shared CSV recorder)

OUT_DEFAULT = M.OUT_DEFAULT
STEM = "fig_rq1_perturbation"


# ------------------------------------------------------------------------------------------------
# a.
DNA_CKPT = "RL"
DNA_RUNGS = [("wt", "all text"),
             ("no_gene", "pathway fields,\nsymbols masked"),
             ("no_pathway", "query\nnaming the gene"),
             ("no_textkey", "query,\nsymbol masked")]
DNA_SHUFFLED = {"wt": "scramble", "no_gene": "scramble_no_gene",
                "no_pathway": "scramble_no_pathway", "no_textkey": "scramble_no_textkey"}
DNA_REMOVED = "no_dna"                     # quoted in the caption, not drawn
DNA_MERGED = M.RD.DNA_TEXT_CONDITIONS   # text arms scored together with the perturbation run


def data_dna_rl():
    f = f"{DNA_MERGED}/metrics_{DNA_CKPT.lower()}.json"
    m = M.load(f)["experiment1_2_per_arm"]
    orig = M.load(f"{M.KEGG}/metrics_{DNA_CKPT.lower()}.json")["experiment1_2_per_arm"]
    val = M.load(f"{DNA_MERGED}/validity_{DNA_CKPT.lower()}.json")
    for arm in ("scramble_no_gene", "scramble_no_pathway", "scramble_no_textkey"):
        v = val[arm]
        assert v["n"] == 1449 and v["dna_changed_vs_wt_rate"] == 1.0 \
            and v["text_identical_to_text_arm_rate"] == 1.0, (arm, v)

    # per-query accuracy and its cluster bootstrap interval (dna_per_query_accuracy.py)
    pq = {}
    for d in (M.KEGG, DNA_MERGED):
        pq.update(M.load(M.RD.dna_per_query(d, DNA_CKPT))["per_arm"])

    def get(arm):
        v = m[arm]
        assert v["n_rows"] == 1449 and v["n_clusters"] == 708, (arm, v["n_rows"], v["n_clusters"])
        if arm in orig:                      # merged scoring must reproduce the source run exactly
            assert abs(v["accuracy"] - orig[arm]["accuracy"]) < 1e-9, (arm, v["accuracy"], orig[arm])
        a = pq[arm]["accuracy_per_query"]
        assert abs(a - v["accuracy_unclustered"]) < 5e-5, (arm, a, v["accuracy_unclustered"])
        assert pq[arm]["n_queries"] == 1449 and pq[arm]["n_genomes"] == 708, (arm, pq[arm])
        return a

    def ci(arm):                             # percentile bootstrap resampling the 708 genomes
        return pq[arm]["ci_cluster"]

    out = []
    for arm, label in DNA_RUNGS:
        row = {"arm": arm, "label": label, "intact": get(arm), "shuffled": None, "intact_ci": ci(arm), "shuffled_ci": None}
        M.record("perturbation", "a", model="BioReason", checkpoint=DNA_CKPT, group="all 1,449 rows",
                 series=f"Evo2 intact, {label.replace(chr(10), ' ')}", x=arm, value=row["intact"],
                 n=1449, k=708, source=f, key=f"dna_per_query_accuracy.per_arm.{arm}.accuracy_per_query",
                 ci_low=row["intact_ci"][0], ci_high=row["intact_ci"][1])
        if arm in DNA_SHUFFLED:
            sarm = DNA_SHUFFLED[arm]
            row["shuffled"], row["shuffled_ci"] = get(sarm), ci(sarm)
            M.record("perturbation", "a", model="BioReason", checkpoint=DNA_CKPT,
                     group="all 1,449 rows", series=f"Evo2 shuffled, {label.replace(chr(10), ' ')}", x=sarm,
                     value=row["shuffled"], n=1449, k=708, source=f,
                     key=f"dna_per_query_accuracy.per_arm.{sarm}.accuracy_per_query",
                     ci_low=row["shuffled_ci"][0], ci_high=row["shuffled_ci"][1])
        out.append(row)
    removed = get(DNA_REMOVED)
    M.record("perturbation", "a", model="BioReason", checkpoint=DNA_CKPT, group="all 1,449 rows",
             series="Evo2 removed, all text (caption only)", x=DNA_REMOVED, value=removed, n=1449,
             k=708, source=f, key=f"dna_per_query_accuracy.per_arm.{DNA_REMOVED}.accuracy_per_query")
    print(f"[a] {DNA_CKPT} " + " ".join(f"{r['arm']}={r['intact']:.3f}/{r['shuffled']:.3f}" for r in out)
          + f" (intact/shuffled) | no_dna={removed:.3f}")
    return out


# ------------------------------------------------------------------------------------------------
# c.
C2S_MODEL = ("27b", "C2S-Scale 27B")
C2S_SHUFFLED = "scramble_rank"             # same visible gene set as wt, order permuted within it
C2S_RESAMPLED = "random_expressed"         # 1,000 genes drawn from all expressed genes, in order
C2S_RANDEXPR = f"{M.RD.C2S_RANDOM_EXPRESSED}/{C2S_RESAMPLED}_27b.json"


def data_c2s_atlas():
    f = M.RD.C2S_DEG_REMOVAL_SUMMARY
    d = M.load(f)
    m, name = C2S_MODEL
    ref = M.data_c2s()[m]                  # the mean rq1_data draws; asserted below
    rx = M.load(C2S_RANDEXPR)              # the resampled arm, scored on the DEG removal cells
    cis = M.ci_json("ci_c2s.json")
    out = []
    for ds in M.C2S_DS:
        e = d[f"{ds}|{m}"]
        r = rx[ds]
        assert r["n_cells"] == e["n_cells"] and abs(r["wt"] - e["wt"]) < 1e-9 \
            and abs(r[C2S_SHUFFLED] - e[C2S_SHUFFLED]) < 1e-9, (ds, r, e["wt"])
        row = {"atlas": ds, "n_cells": e["n_cells"], "n_options": e["n_options"],
               "chance": e["chance"], "intact": e["wt"], "shuffled": e[C2S_SHUFFLED],
               "resampled": r[C2S_RESAMPLED], "intact_ci": None, "shuffled_ci": None, "resampled_ci": None}
        if cis:
            q = cis["per_atlas"][f"{ds}|{m}"]
            for key, arm in (("intact", "wt"), ("shuffled", C2S_SHUFFLED), ("resampled", C2S_RESAMPLED)):
                assert abs(q[arm]["acc"] - round(row[key], 4)) < 1e-9, (ds, arm, q[arm]["acc"], row[key])
                row[f"{key}_ci"] = q[arm]["ci95"]
        assert abs(row["chance"] - 1 / row["n_options"]) < 5e-4, (ds, row["chance"], row["n_options"])
        out.append(row)
        for series, key, v, c in (("cell sentence intact", "wt", row["intact"], row["intact_ci"]),
                                  ("gene order shuffled (top 1,000)", C2S_SHUFFLED, row["shuffled"], row["shuffled_ci"])):
            M.record("perturbation", "c", model=name, group=ds, series=series, x=key, value=v,
                     n=e["n_cells"], k=e["n_options"], source=f, key=f"{ds}|{m}.{key}",
                     ci_low=c[0] if c else "", ci_high=c[1] if c else "")
        M.record("perturbation", "c", model=name, group=ds,
                 series="genes resampled from all expressed (1,000, expression order)",
                 x=C2S_RESAMPLED, value=row["resampled"], n=r["n_cells"], k=e["n_options"],
                 ci_low=row["resampled_ci"][0] if row["resampled_ci"] else "",
                 ci_high=row["resampled_ci"][1] if row["resampled_ci"] else "",
                 source=C2S_RANDEXPR, key=f"{ds}.{C2S_RESAMPLED}")
        M.record("perturbation", "c", model=name, group=ds, series="random reference (1/n_options)",
                 x="chance", value=e["chance"], n=e["n_cells"], k=e["n_options"], source=f,
                 key=f"{ds}|{m}.chance")
    mean_i = st.mean(r["intact"] for r in out)
    mean_s = st.mean(r["shuffled"] for r in out)
    mean_r = st.mean(r["resampled"] for r in out)
    assert abs(mean_i - ref["wt_mean"]) < 1e-9, (mean_i, ref["wt_mean"])
    for series, key, v, src in (("cell sentence intact", "wt", mean_i, f),
                                ("gene order shuffled (top 1,000)", C2S_SHUFFLED, mean_s, f),
                                ("genes resampled from all expressed (1,000, expression order)",
                                 C2S_RESAMPLED, mean_r, C2S_RANDEXPR)):
        M.record("perturbation", "c", model=name, group="mean over 5 atlases (caption only)",
                 series=series, x=key, value=v, n=sum(r["n_cells"] for r in out), source=src,
                 key=f"mean over atlases of {key}")
    print(f"[c] {name} " + " ".join(f"{r['atlas']}={r['intact']:.3f}/{r['shuffled']:.3f}/{r['resampled']:.3f}"
                                    for r in out)
          + f" (intact/order shuffled/genes resampled) | mean {mean_i:.3f}/{mean_s:.3f}/{mean_r:.3f}")
    return out


# ------------------------------------------------------------------------------------------------
# the figure: one row
# ------------------------------------------------------------------------------------------------
def draw(dna, prot_ladder, c2s, out_dir):
    W = 5.5
    TOP = 0.10                   # band for the panel letters
    TIT = 0.30                   # two-line titles
    AXH = 1.25
    BOT = 1.33
    H = TOP + TIT + AXH + BOT
    fig = plt.figure(figsize=(W, H))
    L_AX = 0.46
    row_top = TOP + TIT
    a_w, b_w, c_w = 1.12, 0.92, 1.58
    gap_ab, gap_bc = 0.60, 0.68
    a_x = L_AX
    b_x = a_x + a_w + gap_ab
    c_x = b_x + b_w + gap_bc
    assert c_x + c_w <= W - 0.14, c_x + c_w

    def title(ax, text):
        ax.set_title(text, fontsize=TICK_FS, fontweight="normal", pad=3.0, linespacing=1.15)

    def err(ax, x, v, ci):
        """95% bootstrap interval as a plain black line (no caps); returns the height a value label should clear."""
        if not ci:
            return v
        ax.errorbar([x], [v], yerr=[[v - ci[0]], [ci[1] - v]], fmt="none", ecolor="black",
                    elinewidth=ERR_LW, capsize=0, zorder=5)
        return max(v, ci[1])

    def ticks45(ax, labels):
        ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
        ax.tick_params(axis="x", labelsize=6.0)

    # ---- a: DNA, RL: intact genome down the text ladder; scrambled genome at the top rung
    ax = ax_in(fig, a_x, row_top, a_w, AXH)
    xs = np.arange(len(dna))
    w = 0.38
    finish(ax, "Accuracy", ylim=(0, 1.18), yticks=np.arange(0, 1.01, 0.25), spine_top=1.0)
    center_ylabel(ax, 1.0)
    for i, r in enumerate(dna):
        paired = r["shuffled"] is not None
        xi = xs[i] - w / 2 if paired else xs[i]
        ax.bar(xi, r["intact"], w, color=DNA.model, edgecolor="black", linewidth=EDGE_LW)
        bar_label(ax, xi, err(ax, xi, r["intact"], r["intact_ci"]), f"{r['intact']:.3f}")
        if paired:
            ax.bar(xs[i] + w / 2, r["shuffled"], w, color=DNA.perturbed, edgecolor="black",
                   linewidth=EDGE_LW)
            bar_label(ax, xs[i] + w / 2, err(ax, xs[i] + w / 2, r["shuffled"], r["shuffled_ci"]), f"{r['shuffled']:.3f}")
    ax.set_xticks(xs)
    ticks45(ax, [r["label"] for r in dna])
    ax.set_xlim(-0.62, len(dna) - 0.38)
    ax.set_xlabel("Text supplied with Evo2", fontsize=TICK_FS)
    title(ax, "DNA: BioReason\nRL, 1,449 KEGG questions")
    ax_a = ax
    a_handles = [Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="black", lw=EDGE_LW, label=l)
                 for l, c in (("Evo2 intact", DNA.model), ("Evo2 shuffled", DNA.perturbed))]
    panel_letter(fig, 0.02, TOP, "a")

    # ---- b: protein, RL: ESM3 intact vs shuffled as the text sources are removed
    ax = ax_in(fig, b_x, row_top, b_w, AXH)
    rows = prot_ladder["RL"]
    xs = np.arange(len(rows))
    finish(ax, r"$F_{\max}$", ylim=(0, 1.18), yticks=np.arange(0, 1.01, 0.25), spine_top=1.0)
    center_ylabel(ax, 1.0)
    for i, r in enumerate(rows):
        ax.bar(xs[i] - w / 2, r["intact"], w, color=PROTEIN.model, edgecolor="black",
               linewidth=EDGE_LW)
        ax.bar(xs[i] + w / 2, r["shuffled"], w, color=PROTEIN.perturbed, edgecolor="black",
               linewidth=EDGE_LW)
        bar_label(ax, xs[i] - w / 2, err(ax, xs[i] - w / 2, r["intact"], r.get("intact_ci")), f"{r['intact']:.3f}")
        bar_label(ax, xs[i] + w / 2, err(ax, xs[i] + w / 2, r["shuffled"], r.get("shuffled_ci")), f"{r['shuffled']:.3f}")
    ax.set_xticks(xs)
    # one line for the top rung: at this panel width two-line 45-degree labels overlap their
    # neighbour
    ticks45(ax, [r["label"].replace("\n", " ") for r in rows])
    ax.set_xlim(-0.62, len(rows) - 0.38)
    ax.set_xlabel("Text evidence sources", fontsize=TICK_FS)
    title(ax, "Protein: BioReason-Pro\nRL, 14,102 proteins")
    ax_b = ax
    b_handles = [Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="black", lw=EDGE_LW, label=l)
                 for l, c in (("ESM3 intact", PROTEIN.model), ("ESM3 shuffled", PROTEIN.perturbed))]
    panel_letter(fig, b_x - 0.32, TOP, "b")

    # ---- c: single cell, per atlas: cell sentence intact vs gene order shuffled
    ax = ax_in(fig, c_x, row_top, c_w, AXH)
    xs = np.arange(len(c2s))
    finish(ax, "Cell-type annotation\naccuracy", ylim=(0, 1.18), yticks=np.arange(0, 1.01, 0.25),
           spine_top=1.0)
    ax.yaxis.label.set_linespacing(1.1)
    center_ylabel(ax, 1.0)
    w3 = 0.27
    for i, r in enumerate(c2s):
        for k, (key, col) in enumerate((("intact", CELL.model), ("shuffled", CELL.perturbed),
                                        ("resampled", CELL.perturbed2))):
            xb = xs[i] + (k - 1) * w3
            ax.bar(xb, r[key], w3, color=col, edgecolor="black", linewidth=EDGE_LW)
            bar_label(ax, xb, err(ax, xb, r[key], r[f"{key}_ci"]), f"{r[key]:.3f}")
    ax.set_xticks(xs)
    ticks45(ax, [r["atlas"] for r in c2s])
    ax.set_xlim(-0.58, len(c2s) - 0.42)
    ax.set_xlabel("Atlas", fontsize=TICK_FS)
    title(ax, f"Single cell: {C2S_MODEL[1]}\n{sum(r['n_cells'] for r in c2s):,} cells, 5 atlases")
    c_handles = [Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="black", lw=EDGE_LW, label=l)
                 for l, c in (("cell sentence intact", CELL.model),
                              ("cell sentence gene order shuffled", CELL.perturbed),
                              ("genes in cell sentence\nresampled from all expressed",
                               CELL.perturbed2))]
    ax_c = ax
    panel_letter(fig, c_x - 0.40, TOP, "c")
    align_xlabels(fig, [ax_a, ax_b, ax_c])

    # c against the right edge. c is the tallest (three entries, one of them wrapped).
    y_top = 0.52 / H
    leg = dict(frameon=False, fontsize=LEGEND_FS, handletextpad=0.4, labelspacing=0.3)
    fig.legend(handles=a_handles, loc="upper center", bbox_to_anchor=((a_x + a_w / 2) / W, y_top),
               ncol=1, handlelength=1.1, **leg)
    fig.legend(handles=b_handles, loc="upper center", bbox_to_anchor=((b_x + b_w / 2) / W, y_top),
               ncol=1, handlelength=1.1, **leg)
    fig.legend(handles=c_handles, loc="upper right", bbox_to_anchor=(1 - 0.06 / W, y_top),
               ncol=1, handlelength=1.4, **leg)
    return emit(fig, STEM, out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=OUT_DEFAULT)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    dna = data_dna_rl()                  # records ("perturbation", "a")
    prot_ladder = M.data_ladder()        # records ("protein", "a"), both checkpoints
    c2s = data_c2s_atlas()               # records ("perturbation", "c"); data_c2s adds "singlecell"
    for ck, rows in prot_ladder.items():
        print(f"[b] {ck}", [(r["label"].replace(chr(10), " "), round(r["intact"], 3),
                            round(r["shuffled"], 3)) for r in rows])

    init_print_style()
    draw(dna, prot_ladder, c2s, a.out_dir)

    # one CSV of every plotted number, re-keyed to this figure's panels
    panel_of = {("perturbation", "a"): "a", ("protein", "a"): "b", ("perturbation", "c"): "c"}
    keys = ["figure", "panel", "model", "checkpoint", "group", "series", "x", "value", "n", "k",
            "ci_low", "ci_high", "sd_over_folds", "source", "key"]
    rows = []
    for r in M.ROWS:
        p = panel_of.get((r["figure"], r["panel"]))
        if p is None:
            continue                     # rq1_data's singlecell rows, used only for the assertion
        rows.append({**{k: r.get(k, "") for k in keys}, "figure": "perturbation", "panel": p})
    csv_path = os.path.join(a.out_dir, f"{STEM}_numbers.csv")
    with open(csv_path, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=keys)
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"wrote {csv_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

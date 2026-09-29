#!/usr/bin/env python
"""Draws Figure 5a and 5c ("Biological representations contribute similarly little to SFT and RL
checkpoint performance"): SFT and RL checkpoints with intact and shuffled foundation model
representations under each text condition.

    python figures/sft_vs_rl_panels.py [--out_dir <dir>]

draw_dna_panel() draws BioReason disease prediction accuracy with Z_Evo2 intact against shuffled on
1,449 queries (Figure 5a), and draw_protein_panel() BioReason-Pro weighted F_max with ESM3 intact
against shuffled on 14,102 proteins (Figure 5c). Writes both panels in one figure,
fig_sft_vs_rl_panels, and its numbers CSV to outputs/figures, or to --out_dir.
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
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figure_style import (init_print_style, ax_in, panel_letter, emit, DNA, PROTEIN,  # noqa: E402
                          GREY, TICK_FS, LEGEND_FS, EDGE_LW)
import seaborn as sns  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rq1_data as M  # noqa: E402

OUT_DEFAULT = M.OUT_DEFAULT
STEM = "fig_sft_vs_rl_panels"
CKPTS = ["SFT", "RL"]
SHAPES = ["o", "^", "s", "D"]              # one per text rung, top of the ladder first
DNA_MERGED = M.RD.DNA_TEXT_CONDITIONS
# (rung label, intact arm, shuffled arm)
DNA_RUNGS = [("all text", "wt", "scramble"),
             ("pathway fields, symbols masked", "no_gene", "scramble_no_gene"),
             ("query naming the gene", "no_pathway", "scramble_no_pathway"),
             ("query, symbol masked", "no_textkey", "scramble_no_textkey")]
PROT_RUNGS = ["GO-GPT & InterPro", "GO-GPT", "InterPro", "None"]
MS = 4.4                                   # marker size, points
DNA_RUNG_LABELS = [lab for lab, *_ in DNA_RUNGS]
TITLE_DNA = "DNA: BioReason\n1,449 queries"
TITLE_PROT = "Protein: BioReason-Pro\n14,102 proteins"
# legend keywords shared by this figure and any figure that reuses one of its panels
LEG_KW = dict(frameon=False, fontsize=LEGEND_FS, handletextpad=0.4, labelspacing=0.28,
              handlelength=1.0)


def data_dna():
    """Per checkpoint, (intact, shuffled) at each rung. The all-text pair is in the 16-arm metrics; the
    reduced-rung shuffled twins are in the merged scoring of the text condition run, whose intact
    values are asserted equal to the 16-arm file. A checkpoint without that scoring gets only its
    all-text pair.
    """
    out = {}
    for ck in CKPTS:
        base_path = f"{M.KEGG}/metrics_{ck.lower()}.json"
        base = M.load(base_path)["experiment1_2_per_arm"]
        merged_path = f"{DNA_MERGED}/metrics_{ck.lower()}.json"
        merged = M.load(merged_path)["experiment1_2_per_arm"] if os.path.exists(merged_path) else None
        if merged is not None:
            val = M.load(f"{DNA_MERGED}/validity_{ck.lower()}.json")
            for arm in ("scramble_no_gene", "scramble_no_pathway", "scramble_no_textkey"):
                v = val[arm]
                assert v["n"] == 1449 and v["dna_changed_vs_wt_rate"] == 1.0 \
                    and v["text_identical_to_text_arm_rate"] == 1.0, (ck, arm, v)
        pq = {}
        for d in (M.KEGG, DNA_MERGED):
            pq_path = M.RD.dna_per_query(d, ck)
            if os.path.exists(pq_path):
                pq.update(M.load(pq_path)["per_arm"])
        out[ck] = {}
        for label, intact, shuffled in DNA_RUNGS:
            src, f = (base, base_path) if shuffled in base else (merged, merged_path)
            if src is None:
                continue
            for arm in (intact, shuffled):
                vv = src[arm]
                assert vv["n_rows"] == 1449 and vv["n_clusters"] == 708, (ck, arm)
                if arm in base:
                    assert abs(vv["accuracy"] - base[arm]["accuracy"]) < 1e-9, (ck, arm)
                assert abs(pq[arm]["accuracy_per_query"] - vv["accuracy_unclustered"]) < 5e-5, (ck, arm)
                assert pq[arm]["n_queries"] == 1449 and pq[arm]["n_genomes"] == 708, (ck, arm)
            out[ck][label] = (pq[intact]["accuracy_per_query"], pq[shuffled]["accuracy_per_query"])
            for state, arm in (("Evo2 intact", intact), ("Evo2 shuffled", shuffled)):
                M.record("sft_vs_rl", "a", model="BioReason", checkpoint=ck, group=label,
                         series=state, x=arm, value=pq[arm]["accuracy_per_query"], n=1449, k=708,
                         source=f, key=f"dna_per_query_accuracy.per_arm.{arm}.accuracy_per_query")
        print(f"[a] {ck}: " + "; ".join(f"{lab}=({i:.3f},{s:.3f})" for lab, (i, s) in out[ck].items()))
        missing = [lab for lab, *_ in DNA_RUNGS if lab not in out[ck]]
        if missing:
            print(f"[a] {ck}: NOT DRAWN (no scrambled-genome arm scored yet): {missing}")
    return out


def data_protein():
    ladder = M.data_ladder()             # records ("protein", "a") for both checkpoints
    out = {}
    for ck, rows in ladder.items():
        byl = {r["label"].replace("\n", " "): r for r in rows}
        out[ck] = {lab: (byl[lab]["intact"], byl[lab]["shuffled"]) for lab in PROT_RUNGS}
        print(f"[b] {ck}: " + "; ".join(f"{lab}=({i:.3f},{s:.3f})" for lab, (i, s) in out[ck].items()))
    return out


def draw_panel(ax, pal, rungs, vals, metric, mod, s=1.0):
    """One panel of intact against shuffled performance, drawn into `ax`. `s` multiplies every
    length and font size, so another figure can draw the panel at its own scale."""
    lim = (-0.02, 1.02)
    ax.plot([0, 1], [0, 1], color=GREY, lw=0.7 * s, ls=":", zorder=1)
    for k, lab in enumerate(rungs):
        pts = {ck: vals[ck][lab] for ck in CKPTS if lab in vals[ck]}
        if len(pts) == 2:
            (x1, y1), (x2, y2) = pts["SFT"], pts["RL"]
            ax.plot([x1, x2], [y1, y2], color="#B8B8B8", lw=0.6 * s, zorder=2)
        for ck, (x, y) in pts.items():
            ax.plot(x, y, marker=SHAPES[k], ms=MS * s, mfc=pal.model if ck == "RL" else "white",
                    mec="black", mew=EDGE_LW * s, ls="none", zorder=4 if ck == "SFT" else 5)
    sns.despine(ax=ax)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xticks(np.arange(0, 1.01, 0.25))
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.set_aspect("equal")
    ax.set_xlabel(f"{metric}, {mod} intact", fontsize=TICK_FS * s)
    ax.set_ylabel(f"{metric}, {mod} shuffled", fontsize=TICK_FS * s)


def draw_dna_panel(ax, dna, s=1.0):
    """BioReason accuracy with Evo2 intact against shuffled, with its title."""
    draw_panel(ax, DNA, DNA_RUNG_LABELS, dna, "Accuracy", "Evo2", s=s)
    panel_title(ax, TITLE_DNA, s=s)


def draw_protein_panel(ax, prot, s=1.0):
    """BioReason-Pro F_max with ESM3 intact against shuffled, with its title."""
    draw_panel(ax, PROTEIN, PROT_RUNGS, prot, r"$F_{\max}$", "ESM3", s=s)
    panel_title(ax, TITLE_PROT, s=s)


def panel_title(ax, text, s=1.0):
    ax.set_title(text, fontsize=TICK_FS * s, fontweight="normal", pad=3.0 * s, linespacing=1.15)


def mk(shape, fill, label, s=1.0):
    return Line2D([], [], marker=shape, ms=MS * s, mfc=fill, mec="black", mew=EDGE_LW * s,
                  ls="none", label=label)


def rung_handles(pal, rungs, s=1.0):
    """Legend of the text conditions of one panel: one marker shape per condition, in the panel's
    hue."""
    return [mk(SHAPES[k], pal.model, lab, s) for k, lab in enumerate(rungs)]


def ckpt_handles(s=1.0):
    """Checkpoint legend: filled for RL, hollow for SFT."""
    return [mk("o", "#606060", "RL", s), mk("o", "white", "SFT", s)]


def draw(dna, prot, out_dir):
    W = 4.32                     # = the printed width; no dead space right of panel b
    TOP = 0.08
    TIT = 0.28
    AX = 1.52                    # square panels
    BOT = 1.02                   # x label + the legends
    H = TOP + TIT + AX + BOT
    fig = plt.figure(figsize=(W, H))
    L_AX = 0.48
    gap = 0.62
    a_x = L_AX
    b_x = a_x + AX + gap
    assert b_x + AX <= W - 0.15 + 1e-9, b_x + AX   # 0.15 holds the last x tick label
    row_top = TOP + TIT

    ax = ax_in(fig, a_x, row_top, AX, AX)
    draw_dna_panel(ax, dna)
    panel_letter(fig, 0.02, TOP, "a")

    ax = ax_in(fig, b_x, row_top, AX, AX)
    draw_protein_panel(ax, prot)
    panel_letter(fig, b_x - 0.46, TOP, "b")

    # three legend columns on one baseline: the text rungs of each panel, then the checkpoint fill
    leg = dict(LEG_KW)
    y0 = 0.03 / H
    cols = [0.20, 1.95, 3.42]                   # inches from the left edge
    fig.legend(handles=rung_handles(DNA, DNA_RUNG_LABELS),
               loc="lower left", bbox_to_anchor=(cols[0] / W, y0), ncol=1, **leg)
    fig.legend(handles=rung_handles(PROTEIN, PROT_RUNGS),
               loc="lower left", bbox_to_anchor=(cols[1] / W, y0), ncol=1, **leg)
    fig.legend(handles=ckpt_handles(),
               loc="upper left", bbox_to_anchor=(cols[2] / W, (0.03 + 0.50) / H), ncol=1, **leg)
    hdr = dict(va="bottom", fontsize=LEGEND_FS, color="black", ha="left")
    fig.text((cols[0] + 0.04) / W, (0.03 + 0.54) / H, "BioReason text sources", **hdr)
    fig.text((cols[1] + 0.04) / W, (0.03 + 0.54) / H, "BioReason-Pro text sources", **hdr)
    fig.text((cols[2] + 0.04) / W, (0.03 + 0.54) / H, "Checkpoint", **hdr)
    return emit(fig, STEM, out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=OUT_DEFAULT)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    dna = data_dna()
    prot = data_protein()
    init_print_style()
    draw(dna, prot, a.out_dir)

    panel_of = {("sft_vs_rl", "a"): "a", ("protein", "a"): "b"}
    keys = ["figure", "panel", "model", "checkpoint", "group", "series", "x", "value", "n", "k",
            "ci_low", "ci_high", "sd_over_folds", "source", "key"]
    rows = [{**{k: r.get(k, "") for k in keys}, "figure": "sft_vs_rl",
             "panel": panel_of[(r["figure"], r["panel"])]}
            for r in M.ROWS if (r["figure"], r["panel"]) in panel_of]
    csv_path = os.path.join(a.out_dir, f"{STEM}_numbers.csv")
    with open(csv_path, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=keys)
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"wrote {csv_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

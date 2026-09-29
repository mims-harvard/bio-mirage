#!/usr/bin/env python
"""Draws the Prot2Text-V2 panels of Figures 2 and 3: BioBERT BERTScore F1 to the reference description
with the ESM2 representation intact or shuffled, with name and taxon removed or given, and the answer
Prot2Text-V2 follows when the ESM2 representation of protein A is paired with the name of protein B.

perturbations_and_conflicts.py calls data_prot2text(), panel_destroyed() and panel_swapped(). Run on
its own, the module reads protein/prot2text_v2/results under INPUT_USE_RESULTS_DIR and writes
fig_rq1_prot2text and its numbers CSV to outputs/figures, or to --out_dir.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figure_style import (init_print_style, center_ylabel, ax_in, finish, bar_label,  # noqa: E402
                          align_xlabels, panel_letter, emit, PROTEIN,
                          TICK_FS, LEGEND_FS, EDGE_LW, ERR_LW)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rq1_data as M  # noqa: E402
import evidence_swap_panels as S  # noqa: E402

P2T = f"{M.RD.PROT2TEXT}/results"
OUT_DEFAULT = M.OUT_DEFAULT
STEM = "fig_rq1_prot2text"
MODEL = "Prot2Text-V2"
XLABEL_DESTROYED = "Text with ESM2"
XLABEL_SWAPPED = "Protein source"
UNIT_DESTROYED = "proteins"
YL = (0, 1.18)
YT = np.arange(0, 1.01, 0.25)
GROUPS = [("seq_only", "Name and\ntaxon removed"), ("native", "Name and\ntaxon given")]


def data_prot2text():
    fig = M.load(os.path.join(P2T, "prot2text_figure.json"))
    ana = M.load(os.path.join(P2T, "prot2text_analysis.json"))
    src = os.path.join(P2T, "prot2text_figure.json")
    d = fig["destroyed"]
    n = fig["n_proteins"]
    rows = []
    for key, label in GROUPS:
        ik, sk, dk = (("seq_only_intact", "seq_only_shuffle", "seq_only_delta") if key == "seq_only"
                      else ("native_intact", "native_shuffle", "native_delta"))
        r = {"key": key, "label": label, "n": n,
             "intact": d[ik]["mean"], "intact_ci": tuple(d[ik]["ci95"]),
             "shuffled": d[sk]["mean"], "shuffled_ci": tuple(d[sk]["ci95"]),
             "delta": d[dk]["mean"], "delta_ci": tuple(d[dk]["ci95"])}
        rows.append(r)
        for series, vk, ck in (("ESM2 intact", "intact", "intact_ci"), ("ESM2 shuffled (mean of 3 seeds)", "shuffled", "shuffled_ci"),
                               ("intact - shuffled (paired)", "delta", "delta_ci")):
            M.record("prot2text", "a", model=MODEL, checkpoint="released", group=label.replace("\n", " "),
                     series=series, x="BioBERT BERTScore F1", value=r[vk], n=n, ci_low=r[ck][0], ci_high=r[ck][1],
                     source=src, key=f"destroyed.{ik if vk == 'intact' else sk if vk == 'shuffled' else dk}")
    sw = None
    if "swapped" in fig and fig["swapped"].get("n_pairs", 0) > 0:
        b = fig["swapped"]
        eps = ana["epsilon"]
        e = b[f"eps_{eps}"]
        sw = {"A": e["sequence_follow_rate"]["mean"], "B": e["text_follow_rate"]["mean"], "neither": e["ambiguous_rate"]["mean"],
              "ci": {"A": e["sequence_follow_rate"]["ci95"], "B": e["text_follow_rate"]["ci95"], "neither": e["ambiguous_rate"]["ci95"]},
              "n": b["n_conflicts"], "n_pairs": b["n_pairs"], "epsilon": eps, "label": "Full Name(B)\nESM2(A)",
              "shift": b["mean_preference_shift_from_aligned"], "toward": b["mean_preference_toward_sequence"]}
        for which, name in (("A", "sequence"), ("B", "text"), ("neither", "ambiguous")):
            M.record("prot2text", "b", model=MODEL, checkpoint="released", group="ESM2(A) + Full Name(B), both directions",
                     series=f"matching {which} ({name})", x="share of conflicts", value=sw[which], n=sw["n"],
                     ci_low=sw["ci"][which][0], ci_high=sw["ci"][which][1], source=src,
                     key=f"swapped.eps_{eps}.{name}_follow_rate" if which != "neither" else f"swapped.eps_{eps}.ambiguous_rate")
        M.record("prot2text", "b", model=MODEL, checkpoint="released", group="ESM2(A) + Full Name(B), both directions",
                 series="mean preference shift from aligned prompt, toward sequence source", x="BioBERT F1 difference",
                 value=sw["shift"]["mean"], n=sw["n"], ci_low=sw["shift"]["ci95"][0], ci_high=sw["shift"]["ci95"][1],
                 source=src, key="swapped.mean_preference_shift_from_aligned")
    return rows, sw


def err(ax, x, v, ci):
    if not ci:
        return v
    ax.errorbar([x], [v], yerr=[[max(0.0, v - ci[0])], [max(0.0, ci[1] - v)]], fmt="none", ecolor="black",
                elinewidth=ERR_LW, capsize=0, zorder=5)
    return max(v, ci[1])


def panel_destroyed(ax, rows, fs=4.5, ylabel="BioBERT F1 to reference", tick_fs=6.0):
    """Two groups (name removed / name given), intact vs shuffled, drawn like the combined figure's
    ladder panels: rotated value labels, 95% interval as a plain black line.
    """
    finish(ax, ylabel, ylim=YL, yticks=YT, spine_top=1.0)
    center_ylabel(ax, 1.0)
    xs = np.arange(len(rows))
    w = 0.38
    for i, r in enumerate(rows):
        for j, (k, col) in enumerate((("intact", PROTEIN.model), ("shuffled", PROTEIN.perturbed))):
            x = xs[i] + (j - 0.5) * w
            ax.bar(x, r[k], w, color=col, edgecolor="black", linewidth=EDGE_LW)
            bar_label(ax, x, err(ax, x, r[k], r.get(f"{k}_ci")), f"{r[k]:.3f}", fs=fs)
    ax.set_xticks(xs)
    ax.set_xticklabels([r["label"] for r in rows], rotation=45, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", labelsize=tick_fs)
    ax.set_xlim(-0.62, len(rows) - 0.38)
    return [Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="black", lw=EDGE_LW, label=l)
            for l, c in (("ESM2 intact", PROTEIN.model), ("ESM2 shuffled", PROTEIN.perturbed))]


def panel_swapped(ax, sw, slots=None, fs=4.5, ylabel="Higher BioBERT F1\nanswer", tick_fs=4.5):
    S.pct_axis(ax, ylabel)
    S.bars(ax, PROTEIN, [sw], slots=slots, fs=fs)
    ax.set_xticklabels([sw["label"]], linespacing=1.1, rotation=45, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", labelsize=tick_fs)
    return [Rectangle((0, 0), 1, 1, facecolor=S.shade(PROTEIN, w), edgecolor="black", lw=EDGE_LW, label=l)
            for w, l in (("A", "closer to A's function (sequence)"), ("B", "closer to B's function (name)"),
                         ("neither", f"within {sw['epsilon']} of both"))]


def title(ax, text):
    ax.set_title(text, fontsize=TICK_FS, fontweight="normal", pad=3.0, linespacing=1.15)


def draw(rows, sw, out_dir):
    W = 3.4
    TOP, TIT, AXH, BOT = 0.30, 0.30, 1.25, 1.30
    H = TOP + TIT + AXH + BOT
    fig = plt.figure(figsize=(W, H))
    row_top = TOP + TIT
    xa, wa, xb, wb = 0.52, 1.20, 2.50, 0.62
    ax = ax_in(fig, xa, row_top, wa, AXH)
    panel_letter(fig, xa - 0.42, TOP - 0.14, "a")
    a_handles = panel_destroyed(ax, rows)
    ax.set_xlabel("Text with ESM2", fontsize=TICK_FS)
    title(ax, f"Protein: {MODEL}\n{rows[0]['n']:,} proteins")
    axes = [ax]
    y_top = 0.62 / H
    leg = dict(frameon=False, fontsize=LEGEND_FS, handletextpad=0.4, labelspacing=0.3)
    fig.legend(handles=a_handles, loc="upper center", ncol=1, handlelength=1.1,
               bbox_to_anchor=((xa + wa / 2) / W, y_top), **leg)
    if sw:
        ax = ax_in(fig, xb, row_top, wb, AXH)
        panel_letter(fig, xb - 0.42, TOP - 0.14, "b")
        b_handles = panel_swapped(ax, sw, slots=1.0)
        ax.set_xlabel("Protein source", fontsize=TICK_FS)
        title(ax, f"Protein: {MODEL}\n{sw['n']:,} conflicts")
        axes.append(ax)
        fig.legend(handles=b_handles, loc="upper center", ncol=1, handlelength=1.1,
                   bbox_to_anchor=((xb + wb / 2 - 0.25) / W, y_top), **leg)
    align_xlabels(fig, axes)
    return emit(fig, STEM, out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=OUT_DEFAULT)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    rows, sw = data_prot2text()
    init_print_style()
    draw(rows, sw, a.out_dir)
    keys = ["figure", "panel", "model", "checkpoint", "group", "series", "x", "value", "n", "k",
            "ci_low", "ci_high", "sd_over_folds", "source", "key"]
    p = os.path.join(a.out_dir, f"{STEM}_numbers.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in keys} for r in M.ROWS if r["figure"] == "prot2text"])
    print(f"wrote {p}")


if __name__ == "__main__":
    main()

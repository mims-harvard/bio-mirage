"""Draws the CellWhisperer panels of Figures 2 and 3: 1 - perplexity quantile of the correct cell type
with the transcriptome intact or shuffled, per C2S-Scale atlas, and the answer CellWhisperer prefers
when the transcriptome of cell A is paired with gene text of cell B.

perturbations_and_conflicts.py calls data_cellwhisperer(), panel_destroyed() and panel_swapped().
Run on its own, the module reads single_cell/cellwhisperer/analysis_default under
INPUT_USE_RESULTS_DIR and writes fig_rq1_cellwhisperer and its numbers CSV to outputs/figures, or to
--out_dir.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figure_style import (init_print_style, center_ylabel, ax_in, finish, bar_label,  # noqa: E402
                          align_xlabels, panel_letter, emit, CELL, GREY,
                          TICK_FS, LEGEND_FS, EDGE_LW, ERR_LW)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rq1_data as M  # noqa: E402
import evidence_swap_panels as S  # noqa: E402

CWPC = f"{M.RD.CELLWHISPERER}/analysis_default"
OUT_DEFAULT = M.OUT_DEFAULT
STEM = "fig_rq1_cellwhisperer"
MODEL = "CellWhisperer"
XLABEL_DESTROYED = "C2S-Scale atlas"
XLABEL_SWAPPED = "Cell source"
UNIT_DESTROYED = "cells"
UNIT_SWAPPED = "conflicts"
SWAP_K = 1000                      # the gene-text length that equals the C2S-Scale input
YL = (0, 1.18)
YT = np.arange(0, 1.01, 0.25)
ATLASES = ["immune1", "immune2", "immune3", "pancreas", "lung"]
LABELS = {"immune1": "immune 1", "immune2": "immune 2", "immune3": "immune 3", "pancreas": "pancreas",
          "lung": "lung", "macro": "mean of\natlases"}


def data_cellwhisperer():
    src = os.path.join(CWPC, "results.json")
    if not os.path.exists(src):
        return None
    R = M.load(src)
    per, macro, pooled = R["per_atlas"], R["macro"], R["pooled"]
    if any(g not in per for g in ATLASES) or "n_embgate_ppl" not in pooled:   # partial (immune-only) analysis
        return None
    rows = []
    n_total = sum(per[g]["expA"]["n_cells"] for g in ATLASES)
    for g in ATLASES + ["macro"]:
        e = per[g]["expA"] if g in per else macro["expA"]
        r = {"key": g, "label": LABELS[g], "n": per[g]["expA"]["n_cells"] if g in per else n_total,
             "n_total": n_total,        # all cells over the five atlases, for a panel title
             "n_labels": per[g]["expA"]["n_labels"] if g in per else None,
             "chance": per[g]["expA"]["chance_uniform"] if g in per else None,
             "intact": e["auc_intact"], "intact_ci": tuple(e["auc_intact_ci"]),
             "shuffled": e["auc_shuffle"], "shuffled_ci": tuple(e["auc_shuffle_ci"]),
             "delta": e["auc_delta_shuffle"], "delta_ci": tuple(e["auc_delta_shuffle_ci"]),
             "acc_intact": e["acc_ppl_intact"], "acc_shuffled": e["acc_ppl_shuffle"]}
        rows.append(r)
        base = f"per_atlas.{g}.expA" if g in per else "macro.expA"
        for series, vk, ck, kk in (("transcriptome intact", "intact", "intact_ci", "auc_intact"),
                                   ("transcriptome count-shuffled (mean of 3 seeds)", "shuffled", "shuffled_ci", "auc_shuffle"),
                                   ("intact - shuffled (paired)", "delta", "delta_ci", "auc_delta_shuffle")):
            M.record("cellwhisperer", "a", model=MODEL, checkpoint="default (released chat model)",
                     group=r["label"].replace("\n", " "), series=series,
                     x="1 - perplexity quantile of the correct answer, matched vs all other-type transcriptomes (chance 0.5)",
                     value=r[vk], n=r["n"], ci_low=r[ck][0], ci_high=r[ck][1], source=src, key=f"{base}.{kk}")
    k = SWAP_K
    n_pairs = pooled["n_embgate_ppl"]       # embedding gate by per-token log-likelihood, no text gate, k
    # = 1,000
    sw = None
    if n_pairs > 0:
        g = f"embgate_ppl_k{k}"
        sw = {"A": pooled[f"{g}_emb_ppl"], "B": pooled[f"{g}_text_ppl"], "neither": pooled[f"{g}_amb_ppl"],
              "ci": {"A": pooled[f"{g}_emb_ppl_ci"], "B": pooled[f"{g}_text_ppl_ci"], "neither": pooled[f"{g}_amb_ppl_ci"]},
              "n": 2 * n_pairs, "n_pairs": n_pairs, "k": k,
              "label": f"Gene text k={k} (B)\nEmbedding (A)",
              "toward": pooled[f"{g}_pref_text_ppl"], "toward_ci": pooled[f"{g}_pref_text_ppl_ci"],
              "emb_effect": pooled[f"{g}_embedding_effect_ppl"], "emb_effect_ci": pooled[f"{g}_embedding_effect_ppl_ci"],
              "text_effect": pooled[f"{g}_text_effect_ppl"], "text_effect_ci": pooled[f"{g}_text_effect_ppl_ci"]}
        grp = f"Embedding(A) + {k}-gene text(B), both directions, embedding-gated pairs (per-token log-likelihood, no text gate)"
        for which, name, key in (("A", "embedding", f"pooled.{g}_emb_ppl"), ("B", "gene text", f"pooled.{g}_text_ppl"),
                                 ("neither", "exact perplexity tie", f"pooled.{g}_amb_ppl")):
            M.record("cellwhisperer", "b", model=MODEL, checkpoint="default (released chat model)", group=grp,
                     series=f"matching {which} ({name}), lower perplexity", x="share of conflicts", value=sw[which], n=sw["n"], k=k,
                     ci_low=sw["ci"][which][0], ci_high=sw["ci"][which][1], source=src, key=key)
        for series, vk, ck, key in (("mean per-token log-likelihood preference toward the gene-text source", "toward", "toward_ci", f"pooled.{g}_pref_text_ppl"),
                                    ("2x2: per-token shift of log p(A)-log p(B) from swapping the transcriptome", "emb_effect", "emb_effect_ci", f"pooled.{g}_embedding_effect_ppl"),
                                    ("2x2: per-token shift of log p(A)-log p(B) from swapping the gene text", "text_effect", "text_effect_ci", f"pooled.{g}_text_effect_ppl")):
            M.record("cellwhisperer", "b", model=MODEL, checkpoint="default (released chat model)", group=grp,
                     series=series, x="nats per token", value=sw[vk], n=sw["n"], k=k, ci_low=sw[ck][0], ci_high=sw[ck][1], source=src, key=key)
    return rows, sw


def err(ax, x, v, ci):
    if not ci:
        return v
    ax.errorbar([x], [v], yerr=[[max(0.0, v - ci[0])], [max(0.0, ci[1] - v)]], fmt="none", ecolor="black",
                elinewidth=ERR_LW, capsize=0, zorder=5)
    return max(v, ci[1])


def panel_destroyed(ax, rows, fs=4.5, ylabel="1 \u2212 perplexity\nquantile", tick_fs=6.0):
    """One group per atlas (plus the mean), intact vs count-shuffled transcriptome embedding; the
    metric is 1 - perplexity quantile: the correct answer's perplexity given the cell's own transcriptome,
    ranked among its perplexities given transcriptomes of other cell types (0.5 = no preference).
    No chance line is drawn.
    """
    finish(ax, ylabel, ylim=YL, yticks=YT, spine_top=1.0)
    center_ylabel(ax, 1.0)
    xs = np.arange(len(rows))
    w = 0.38
    for i, r in enumerate(rows):
        for j, (k, col) in enumerate((("intact", CELL.model), ("shuffled", CELL.perturbed))):
            x = xs[i] + (j - 0.5) * w
            ax.bar(x, r[k], w, color=col, edgecolor="black", linewidth=EDGE_LW)
            bar_label(ax, x, err(ax, x, r[k], r.get(f"{k}_ci")), f"{r[k]:.2f}", fs=fs)
    ax.set_xticks(xs)
    ax.set_xticklabels([r["label"] for r in rows], rotation=45, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", labelsize=tick_fs)
    ax.set_xlim(-0.62, len(rows) - 0.38)
    return [Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="black", lw=EDGE_LW, label=l)
            for l, c in (("transcriptome intact", CELL.model), ("transcriptome count-shuffled", CELL.perturbed))]


def panel_swapped(ax, sw, slots=None, fs=4.5, ylabel="Lower-perplexity\nanswer", tick_fs=4.5):
    S.pct_axis(ax, ylabel)
    S.bars(ax, CELL, [sw], slots=slots, fs=fs)
    ax.set_xticklabels([sw["label"]], linespacing=1.1, rotation=45, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", labelsize=tick_fs)
    return [Rectangle((0, 0), 1, 1, facecolor=S.shade(CELL, w), edgecolor="black", lw=EDGE_LW, label=l)
            for w, l in (("A", "A's type lower perplexity (embedding)"), ("B", "B's type lower perplexity (gene text)"),
                         ("neither", "exact tie"))]


def title(ax, text):
    ax.set_title(text, fontsize=TICK_FS, fontweight="normal", pad=3.0, linespacing=1.15)


def draw(rows, sw, out_dir):
    W = 4.2
    TOP, TIT, AXH, BOT = 0.30, 0.30, 1.25, 1.30
    H = TOP + TIT + AXH + BOT
    fig = plt.figure(figsize=(W, H))
    row_top = TOP + TIT
    xa, wa, xb, wb = 0.52, 2.10, 3.30, 0.62
    ax = ax_in(fig, xa, row_top, wa, AXH)
    panel_letter(fig, xa - 0.42, TOP - 0.14, "a")
    a_handles = panel_destroyed(ax, rows)
    ax.set_xlabel(XLABEL_DESTROYED, fontsize=TICK_FS)
    title(ax, f"Single cell: {MODEL}\n{rows[-1]['n']:,} cells, 5 atlases")
    axes = [ax]
    y_top = 0.62 / H
    leg = dict(frameon=False, fontsize=LEGEND_FS, handletextpad=0.4, labelspacing=0.3)
    fig.legend(handles=a_handles, loc="upper center", ncol=1, handlelength=1.1,
               bbox_to_anchor=((xa + wa / 2) / W, y_top), **leg)
    if sw:
        ax = ax_in(fig, xb, row_top, wb, AXH)
        panel_letter(fig, xb - 0.42, TOP - 0.14, "b")
        b_handles = panel_swapped(ax, sw, slots=1.0)
        ax.set_xlabel(XLABEL_SWAPPED, fontsize=TICK_FS)
        title(ax, f"Single cell: {MODEL}\n{sw['n']:,} conflicts")
        axes.append(ax)
        fig.legend(handles=b_handles, loc="upper center", ncol=1, handlelength=1.1,
                   bbox_to_anchor=((xb + wb / 2 - 0.25) / W, y_top), **leg)
    align_xlabels(fig, axes)
    return emit(fig, STEM, out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=OUT_DEFAULT)
    a = ap.parse_args()
    data = data_cellwhisperer()
    if data is None:
        print("NOT DRAWN: cellwhisperer numbers missing (analysis_default/results.json)")
        return
    rows, sw = data
    init_print_style()
    draw(rows, sw, a.out_dir)
    keys = ["figure", "panel", "model", "checkpoint", "group", "series", "x", "value", "n", "k", "ci_low", "ci_high", "source", "key"]
    import csv
    p = os.path.join(a.out_dir, f"{STEM}_numbers.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in M.ROWS:
            if r["figure"] == "cellwhisperer":
                w.writerow({k: r.get(k, "") for k in keys})
    print(f"wrote {p}")


if __name__ == "__main__":
    main()

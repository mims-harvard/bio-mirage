#!/usr/bin/env python
"""Draws fig_rq1_combined, which is not in the paper: the biological input perturbation and evidence
conflict panels of BioReason, BioReason-Pro and C2S-Scale in one row, with the ChatNT, Prot2Text-V2
and CellWhisperer panels in a second row (omitted with --no_prot2text).

perturbations_and_conflicts.py imports ladder_panel(), the shared axis limits and the three model
modules from here. Writes the figure and its numbers CSV to outputs/figures, or to --out_dir.
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
import matplotlib.ticker
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figure_style import (init_print_style, center_ylabel, ax_in, finish, bar_label,  # noqa: E402
                          align_xlabels, panel_letter, emit, DNA, PROTEIN, CELL,
                          TICK_FS, LEGEND_FS, EDGE_LW, ERR_LW)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rq1_data as M          # noqa: E402
import perturbation_panels as P  # noqa: E402
import evidence_swap_panels as S          # noqa: E402
import prot2text_v2 as T     # noqa: E402
try:                                    # the ChatNT and CellWhisperer modules, when present
    import chatnt as N        # noqa: E402  (DNA)
except ImportError:
    N = None
try:
    import cellwhisperer as CW  # noqa: E402  (single cell)
except ImportError:
    CW = None

OUT_DEFAULT = M.OUT_DEFAULT
STEM = "fig_rq1_combined"
YL_LADDER = (0, 1.18)
YT = np.arange(0, 1.01, 0.25)


def err(ax, x, v, ci):
    """95% interval as a plain black line without caps; returns the height a value label must clear."""
    if not ci:
        return v
    ax.errorbar([x], [v], yerr=[[v - ci[0]], [ci[1] - v]], fmt="none", ecolor="black",
                elinewidth=ERR_LW, capsize=0, zorder=5)
    return max(v, ci[1])


def ladder_panel(ax, rows, series, label_key, fmt="{:.3f}", fs=4.5):
    """One group per row, one bar per series. series: [(key, colour)] -- a row whose value is None (the
    DNA ladder's unpaired top rung) simply has no bar for that series.
    """
    xs = np.arange(len(rows))
    n = len(series)
    w = 0.38 if n == 2 else 0.27
    for i, r in enumerate(rows):
        present = [k for k, _ in series if r.get(k) is not None]
        for k, col in series:
            if r.get(k) is None:
                continue
            off = 0.0 if len(present) == 1 else (series.index((k, col)) - (n - 1) / 2) * w
            x = xs[i] + off
            ax.bar(x, r[k], w, color=col, edgecolor="black", linewidth=EDGE_LW)
            bar_label(ax, x, err(ax, x, r[k], r.get(f"{k}_ci")), fmt.format(r[k]), fs=fs)
    ax.set_xticks(xs)
    ax.set_xticklabels([r[label_key] for r in rows], rotation=45, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", labelsize=6.0)
    ax.set_xlim(-0.62, len(rows) - 0.38)
    return xs


def draw(pert, swap, out_dir, p2t=None):
    dna, prot_ladder, c2s = pert
    dna_swap, prot_conf, sc = swap
    specs = S.panel_specs(dna_swap, prot_conf, sc)          # d, e, f, already built by the swap script

    W = 8.0
    TOP, TIT, AXH = 0.44, 0.30, 1.25   # top holds the two spanning headers and the panel letters
    BOT = 1.46                    # 45-degree ticks (a-c) + x labels on one level + the legend block
    H1 = TOP + TIT + AXH + BOT    # the one-row figure; second row (ChatNT, Prot2Text-V2, CellWhisperer): its
    # own header band, titles, axes and tick/legend block
    TOP2, TIT2, BOT2 = 0.34, 0.30, 1.46
    H = H1 + (TOP2 + TIT2 + AXH + BOT2 if p2t else 0)
    fig = plt.figure(figsize=(W, H))
    row_top = TOP + TIT
    L_AX = 0.46
    # widths, then the gutter that follows each: a new y axis needs room for its ticks and label, e
    # and f share d's, so they only need a thin gap Every gutter after a, b and c has to hold the
    # next panel's y axis label and its tick numbers, which is ~0.35 in before any white space at
    # all; 0.60 leaves about a quarter inch of real gap.
    W_GUT = [(0.76, 0.60), (0.68, 0.60), (1.12, 0.60), (1.08, 0.18), (1.08, 0.18), (0.58, 0.08)]
    # ^ a and b give d and e the width their three-line tick labels need
    xs, x = [], L_AX
    for w, g in W_GUT:
        xs.append(x)
        x += w + g
    assert x <= W + 1e-9, x
    letters = "abcdef"

    def title(ax, text):
        ax.set_title(text, fontsize=TICK_FS, fontweight="normal", pad=3.0, linespacing=1.15)

    def new_ax(i):
        ax = ax_in(fig, xs[i], row_top, W_GUT[i][0], AXH)
        panel_letter(fig, xs[i] - (0.02 if i == 0 else 0.26), TOP - 0.14, letters[i])
        return ax

    # what separates the two halves, said once instead of in six titles
    for lo, hi, text in ((0, 2, "Input source destroyed"), (3, 5, "Input source swapped")):
        fig.text((xs[lo] + (xs[hi] + W_GUT[hi][0])) / 2 / W, 1 - 0.12 / H, text, ha="center",
                 va="top", fontsize=TICK_FS, fontweight="bold")

    axes = []
    # ---- a: DNA text ladder, Evo2 intact vs shuffled
    ax = new_ax(0)
    finish(ax, "Accuracy", ylim=YL_LADDER, yticks=YT, spine_top=1.0)
    center_ylabel(ax, 1.0)
    ladder_panel(ax, dna, [("intact", DNA.model), ("shuffled", DNA.perturbed)], "label")
    ax.set_xlabel("Text with Evo2", fontsize=TICK_FS)
    title(ax, "DNA\n1,449 questions")
    a_handles = [Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="black", lw=EDGE_LW, label=l)
                 for l, c in (("Evo2 intact", DNA.model), ("Evo2 shuffled", DNA.perturbed))]
    axes.append(ax)

    # ---- b: protein text-source ladder, ESM3 intact vs shuffled
    ax = new_ax(1)
    rows = prot_ladder["RL"]
    finish(ax, r"$F_{\max}$", ylim=YL_LADDER, yticks=YT, spine_top=1.0)
    center_ylabel(ax, 1.0)
    rows = [{**r, "label1": r["label"].replace("\n", " ")} for r in rows]   # one line at this width
    ladder_panel(ax, rows, [("intact", PROTEIN.model), ("shuffled", PROTEIN.perturbed)], "label1")
    ax.set_xlabel("Text sources", fontsize=TICK_FS)
    title(ax, "Protein\n14,102 proteins")
    b_handles = [Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="black", lw=EDGE_LW, label=l)
                 for l, c in (("ESM3 intact", PROTEIN.model), ("ESM3 shuffled", PROTEIN.perturbed))]
    axes.append(ax)

    # ---- c: single cell per atlas, three cell-sentence arms
    ax = new_ax(2)
    finish(ax, "Annotation accuracy", ylim=YL_LADDER, yticks=YT, spine_top=1.0)   # one line here:
    center_ylabel(ax, 1.0)                    # two lines reach across the gutter into panel b
    ladder_panel(ax, c2s, [("intact", CELL.model), ("shuffled", CELL.perturbed),
                           ("resampled", CELL.perturbed2)], "atlas")
    ax.set_xlabel("Atlas", fontsize=TICK_FS)
    title(ax, f"Single cell\n{sum(r['n_cells'] for r in c2s):,} cells")
    c_handles = [Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="black", lw=EDGE_LW, label=l)
                 for l, c in (("cell sentence intact", CELL.model),
                              ("cell sentence gene order shuffled", CELL.perturbed),
                              ("genes in cell sentence\nresampled from all expressed",
                               CELL.perturbed2))]
    axes.append(ax)

    # ---- d, e, f: the evidence-swap panels, drawn by the swap script's own helpers Bar width =
    # panel width / slots.
    unit = W_GUT[3][0] / max(len(v) for _, _, v, _, _, _ in specs)
    XLAB = {3: "DNA source", 4: "Protein source", 5: "Cell source"}
    for i, (_, pal, vals, ticks, xlab, ttl) in enumerate(specs):
        j = 3 + i
        ax = new_ax(j)
        S.pct_axis(ax, "Exact-match\nanswer" if i == 0 else None)
        S.bars(ax, pal, vals, slots=W_GUT[j][0] / unit, fs=4.5)
        ax.set_xticklabels(ticks, linespacing=1.1, rotation=45, ha="right", rotation_mode="anchor")
        ax.tick_params(axis="x", labelsize=4.5)   # rotated, as in a-c: horizontal they overlap here
        ax.set_xlabel(XLAB[j], fontsize=TICK_FS)
        n = vals[0]["n"] if len(vals) == 1 else max(v["n"] for v in vals)
        title(ax, f"{['DNA', 'Protein', 'Single cell'][i]}\n{n:,} conflicts")
        axes.append(ax)
    align_xlabels(fig, axes)

    # ---- legends: a, b, c under their own panels; the shared A/B/neither block under d-f
    y_top = (H - H1 + 0.62) / H      # 0.62 in below the row's own bottom edge
    leg = dict(frameon=False, fontsize=LEGEND_FS, handletextpad=0.4, labelspacing=0.3)
    for handles, i, hl in ((a_handles, 0, 1.1), (b_handles, 1, 1.1)):
        fig.legend(handles=handles, loc="upper center", ncol=1, handlelength=hl,
                   bbox_to_anchor=((xs[i] + W_GUT[i][0] / 2) / W, y_top), **leg)
    fig.legend(handles=c_handles, loc="upper center", ncol=1, handlelength=1.4,
               bbox_to_anchor=((xs[2] + W_GUT[2][0] / 2) / W, y_top), **leg)
    S.ab_legend(fig, 0.05 / H, ncol=1).set_bbox_to_anchor(
        (((xs[3] + xs[5] + W_GUT[5][0]) / 2) / W, y_top), transform=fig.transFigure)
    fig.legends[-1].set_loc("upper center")

    # ---- second row: the ChatNT, Prot2Text-V2 and CellWhisperer panels, one "destroyed" panel per
    # model centred as a block under a-c and one "swapped" panel per model under d-f, in controls order (DNA before
    # protein, as in row 1).
    if p2t:
        top2 = H1 + TOP2 + TIT2
        fig.text(0.5, 1 - (H1 + 0.06) / H, "ChatNT, Prot2Text-V2 and CellWhisperer", ha="center", va="top",
                 fontsize=TICK_FS, fontweight="bold")
        y_top2 = 0.62 / H
        axes2 = []

        def destroyed(c, x, w):
            ax = ax_in(fig, x, top2, w, AXH)
            panel_letter(fig, x - 0.40, H1 + TOP2 - 0.04, c["letter_destroyed"])
            handles = c["module"].panel_destroyed(ax, c["rows"], fs=4.5)
            ax.set_xlabel(c["xlabel_destroyed"], fontsize=TICK_FS)
            n_d = c["rows"][0].get("n_total", c["rows"][0]["n"])     # cellwhisperer: cells over all atlases
            title(ax, f"{c['modality']}: {c['module'].MODEL}\n{n_d:,} {c['unit_destroyed']}")
            axes2.append(ax)
            fig.legend(handles=handles, loc="upper center", ncol=1, handlelength=1.1,
                       bbox_to_anchor=((x + w / 2) / W, y_top2), **leg)

        def swapped(c, x, w):
            ax = ax_in(fig, x, top2, w, AXH)
            panel_letter(fig, x - 0.40, H1 + TOP2 - 0.04, c["letter_swapped"])
            c["module"].panel_swapped(ax, c["sw"], slots=w / unit, fs=4.5)   # its legend is replaced below
            ax.set_xlabel(c["xlabel_swapped"], fontsize=TICK_FS)
            title(ax, f"{c['modality']}: {c['module'].MODEL}\n{c['sw']['n']:,} {c['unit_swapped']}")
            axes2.append(ax)

        # Laid out left to right like row 1: the destroyed panels (one per model, in controls
        # order), a wider gap, then the swapped panels, each with its own y axis (their units
        # differ).
        GUT2, BLOCK_GAP = 0.62, 0.10
        GUT2_SW = 1.00   # the swapped panels are one bar-group wide but carry two-line titles ~1.1
        # in wide
        x = L_AX
        for c in p2t:
            destroyed(c, x, c["width_destroyed"])
            x += c["width_destroyed"] + GUT2
        x += BLOCK_GAP
        x_sw0 = x
        pals = []
        for c in p2t:
            if not c["sw"]:
                continue
            swapped(c, x, unit)          # one group, bars the same physical width as in d-f
            pals.append(c["palette"])
            x += unit + GUT2_SW
        assert x - GUT2_SW <= W + 1e-9, f"second row overflows: {x - GUT2_SW:.2f} in > {W} in"
        if pals:
            # B = the one whose text it gets; "neither" is the caption's per-model definition
            from matplotlib.legend_handler import HandlerTuple
            handles = [tuple(Rectangle((0, 0), 1, 1, facecolor=S.shade(pal, which), edgecolor="black", lw=EDGE_LW)
                             for pal in pals) for which, _ in S.WHICH]
            labels = ["matching A", "matching B", "matching neither"]
            fig.legend(handles=handles, labels=labels, loc="upper center",
                       bbox_to_anchor=(((x_sw0 + x - GUT2_SW - unit) / 2 + unit / 2) / W, y_top2), ncol=1,
                       frameon=False, fontsize=LEGEND_FS, handlelength=0.8 * len(pals) + 0.4, handleheight=0.8,
                       handletextpad=0.5, labelspacing=0.3, handler_map={tuple: HandlerTuple(ndivide=len(pals), pad=0)})
        align_xlabels(fig, axes2)
    return emit(fig, STEM, out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=OUT_DEFAULT)
    ap.add_argument("--no_prot2text", action="store_true", help="the one-row figure without g-h")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    pert = (P.data_dna_rl(), M.data_ladder(), P.data_c2s_atlas())
    swap = (S.data_dna_swap(), S.data_protein_abn(M.data_conflict()), S.data_sc_swap())

    # The ChatNT, Prot2Text-V2 and CellWhisperer panels drawn on the second row, DNA before protein as in row 1.
    p2t = None
    if not a.no_prot2text:
        p2t = []
        for name, mod, spec in (
                ("ChatNT", N, dict(csv_figure="chatnt", palette=DNA, width_destroyed=1.28, modality="DNA", xlabel_destroyed="Native tasks",
                                   xlabel_swapped="DNA source", unit_destroyed="questions", unit_swapped="gated pairs")),
                ("Prot2Text-V2", T, dict(csv_figure="prot2text", palette=PROTEIN, width_destroyed=1.10, modality="Protein", xlabel_destroyed="Text with ESM2",
                                         xlabel_swapped="Protein source", unit_destroyed="proteins", unit_swapped="conflicts")),
                ("CellWhisperer", CW, dict(csv_figure="cellwhisperer", palette=CELL, width_destroyed=1.15, modality="Cell", xlabel_destroyed="Atlas",
                                           xlabel_swapped="Cell source", unit_destroyed="cells", unit_swapped="conflicts"))):
            if mod is None:
                print(f"NOT DRAWN: {name} (module missing)")
                continue
            loader = getattr(mod, {"ChatNT": "data_chatnt", "Prot2Text-V2": "data_prot2text",
                                   "CellWhisperer": "data_cellwhisperer"}[name])
            try:
                data = loader()
            except Exception as e:                       # a model whose results are half-written
                print(f"NOT DRAWN: {name} (its loader raised {type(e).__name__}: {e})")
                continue
            if data is None:
                print(f"NOT DRAWN: {name} (its loader returned None: numbers missing)")
                continue
            rows_c, sw_c = data
            p2t.append(dict(module=mod, rows=rows_c, sw=sw_c, csv_figure=spec.pop("csv_figure"), **spec))
        # letters g, h, ... over the destroyed panels present, then over the swapped ones
        letters2 = iter("ghijklmn")
        for c in p2t:
            c["letter_destroyed"] = next(letters2)
        for c in p2t:
            c["letter_swapped"] = next(letters2) if c["sw"] else None

    init_print_style()
    draw(pert, swap, a.out_dir, p2t)

    panel_of = {("perturbation", "a"): "a", ("protein", "a"): "b", ("perturbation", "c"): "c",
                ("swap", "a"): "d", ("protein", "c"): "e", ("swap", "b"): "e", ("swap", "c"): "f"}
    for c in p2t or []:
        panel_of[(c["csv_figure"], "a")] = c["letter_destroyed"]
        if c["letter_swapped"]:
            panel_of[(c["csv_figure"], "b")] = c["letter_swapped"]
    keys = ["figure", "panel", "model", "checkpoint", "group", "series", "x", "value", "n", "k",
            "ci_low", "ci_high", "sd_over_folds", "source", "key"]
    rows = [{**{k: r.get(k, "") for k in keys}, "figure": "combined",
             "panel": panel_of[(r["figure"], r["panel"])]}
            for r in M.ROWS if (r["figure"], r["panel"]) in panel_of]
    p = os.path.join(a.out_dir, f"{STEM}_numbers.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {p} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

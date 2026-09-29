#!/usr/bin/env python
"""Draws Figure 2 (fig_rq1_destroyed, "Reasoning model performance with representations of intact
and shuffled sequences") for all six models, and holds draw_swapped(), the drawing code of Figure 3.

Data come from perturbation_panels.py, evidence_swap_panels.py, rq1_data.py and the chatnt.py,
prot2text_v2.py and cellwhisperer.py modules, which read INPUT_USE_RESULTS_DIR. Writes
fig_rq1_destroyed and fig_rq1_destroyed_numbers.csv to outputs/figures, or to --out_dir.
evidence_conflicts.py draws Figure 3 with draw_swapped().

    python figures/perturbations_and_conflicts.py
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
from matplotlib.legend_handler import HandlerTuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figure_style import (init_print_style, center_ylabel, ax_in, finish, align_xlabels,  # noqa: E402
                          panel_letter, emit, DNA, PROTEIN, CELL, GREY, TICK_FS, LEGEND_FS, EDGE_LW)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rq1_data as M          # noqa: E402
import perturbation_panels as P  # noqa: E402
import evidence_swap_panels as S          # noqa: E402
import combined_panels as Cb     # noqa: E402  (ladder_panel, and the ChatNT, Prot2Text-V2 and CellWhisperer modules N, T, CW)

OUT_DEFAULT = M.OUT_DEFAULT
W = 8.0
MODALITIES = ["DNA", "Protein", "Single cell"]
AXH = 1.25
LEG = dict(frameon=False, fontsize=LEGEND_FS, handletextpad=0.4, labelspacing=0.3)


def title(ax, text, dx_in=0.0):
    """dx_in shifts the centred title sideways (inches) when it would overhang into a neighbour's letter."""
    w_in = ax.get_position().width * ax.figure.get_size_inches()[0]
    ax.set_title(text, fontsize=TICK_FS, fontweight="normal", pad=3.0, linespacing=1.15, x=0.5 + dx_in / w_in)


def headings(fig, H, spans, top_in=0.08):
    """Bold modality headings, each centred over the span (x_left, x_right in inches) of its panels."""
    for m, (x0, x1) in zip(MODALITIES, spans):
        fig.text(((x0 + x1) / 2) / W, 1 - top_in / H, m, ha="center", va="top", fontsize=TICK_FS, fontweight="bold")


def rect(c, label):
    return Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="black", lw=EDGE_LW, label=label)


def positions(x0, widths, gut):
    """gut: one gutter width, or one per gap (len(widths) - 1)."""
    guts = [gut] * (len(widths) - 1) if not isinstance(gut, (list, tuple)) else list(gut)
    xs, x = [], x0
    for w, g in zip(widths, guts + [0.0]):
        xs.append(x); x += w + g
    assert xs[-1] + widths[-1] <= W + 1e-9, f"row overflows: {xs[-1] + widths[-1]:.2f} in > {W} in"
    return xs


# ------------------------------------------------------------------------------------------------
# figure 1: destroyed -- one row, grouped by modality: BioReason, ChatNT | BioReason-Pro, Prot2Text
# | C2S-Scale, CellWhisperer.
def draw_destroyed(dna, prot_ladder, c2s, controls, out_dir):
    HEAD, LET, TIT, BOT = 0.18, 0.04, 0.30, 1.74     # headings just above the letters; per-panel legends under the
    # x labels
    H = HEAD + LET + TIT + AXH + BOT
    fig = plt.figure(figsize=(W, H))
    widths = [0.92, 0.60, 0.76, 0.46, 1.10, 0.82]     # a, c: two-line tick labels; e, f: many value labels
    xs = positions(0.56, widths, [0.46, 0.56, 0.56, 0.56, 0.61])
    top = HEAD + LET + TIT
    headings(fig, H, [(xs[0], xs[1] + widths[1]), (xs[2], xs[3] + widths[3]), (xs[4], xs[5] + widths[5])], top_in=0.04)
    letters = iter("abcdef")
    ctrl = {c["csv_figure"]: c for c in controls}
    axes = []

    def new_ax(i):
        ax = ax_in(fig, xs[i], top, widths[i], AXH)
        panel_letter(fig, xs[i] - 0.42, top - TIT + 0.01, next(letters))   # top edge level with the title's first line
        axes.append(ax)
        return ax

    legends = {}                                     # panel index -> legend handles (what was shuffled, per panel)

    def control_panel(i, c, rows=None, dx_in=0.0, **kw):
        ax = new_ax(i)
        legends[i] = c["module"].panel_destroyed(ax, rows if rows is not None else c["rows"], fs=4.5, **kw)
        ax.set_xlabel(c["xlabel_destroyed"], fontsize=TICK_FS)
        n_d = c["rows"][0].get("n_total", c["rows"][0]["n"])
        title(ax, f"{c['module'].MODEL}\n{n_d:,} {c['unit_destroyed']}", dx_in=dx_in)
        return ax

    ax = new_ax(0)                                    # a BioReason (RL)
    finish(ax, "Accuracy", ylim=Cb.YL_LADDER, yticks=Cb.YT, spine_top=1.0)
    center_ylabel(ax, 1.0)
    Cb.ladder_panel(ax, dna, [("intact", DNA.model), ("shuffled", DNA.perturbed)], "label")
    ax.tick_params(axis="x", labelsize=5.5)          # four two-line rotated labels in 0.86 in ...
    for t in ax.get_xticklabels():                   # ... need a steeper angle than 45 degrees to clear each other
        t.set_rotation(60)
    ax.set_xlabel("Text with Evo2", fontsize=TICK_FS)
    title(ax, "BioReason (RL)\n1,449 queries")
    legends[0] = [rect(DNA.model, "Evo2 intact"), rect(DNA.perturbed, "Evo2 shuffled")]
    ax = control_panel(1, ctrl["chatnt"], tick_fs=5.5)   # b ChatNT (three two-line rotated labels in 0.66 in)
    for t in ax.get_xticklabels():
        t.set_rotation(60)

    ax = new_ax(2)                                    # c BioReason-Pro (RL)
    rows = [{**r, "label1": (r["label"] if "\n" in r["label"]                      # two-line tick labels
                             else r["label"].replace("GO-GPT & InterPro", "GO-GPT &\nInterPro"))}
            for r in prot_ladder["RL"]]
    finish(ax, r"$F_{\max}$", ylim=Cb.YL_LADDER, yticks=Cb.YT, spine_top=1.0)
    center_ylabel(ax, 1.0)
    Cb.ladder_panel(ax, rows, [("intact", PROTEIN.model), ("shuffled", PROTEIN.perturbed)], "label1")
    ax.set_xlabel("Text sources", fontsize=TICK_FS)
    title(ax, "BioReason-Pro (RL)\n14,102 proteins", dx_in=-0.07)   # clear of letter d
    import matplotlib.transforms as mtrans                            # first tick label ("GO-GPT &\nInterPro")
    t0 = ax.get_xticklabels()[0]                                      # nudged left so its 2nd line clears "GO-GPT"
    t0.set_transform(t0.get_transform() + mtrans.ScaledTranslation(-0.06, 0.0, fig.dpi_scale_trans))
    legends[2] = [rect(PROTEIN.model, "ESM3 intact"), rect(PROTEIN.perturbed, "ESM3 shuffled")]
    ax = control_panel(3, ctrl["prot2text"], dx_in=-0.05)  # d Prot2Text-V2 (title clear of letter e)
    for t in ax.get_xticklabels():                         # two two-line labels in 0.46 in
        t.set_rotation(60)

    ax = new_ax(4)                                    # e C2S-Scale 27B
    finish(ax, "Annotation accuracy", ylim=Cb.YL_LADDER, yticks=Cb.YT, spine_top=1.0)
    center_ylabel(ax, 1.0)
    Cb.ladder_panel(ax, c2s, [("intact", CELL.model), ("shuffled", CELL.perturbed),
                              ("resampled", CELL.perturbed2)], "atlas")
    ax.set_xlabel("Atlas", fontsize=TICK_FS)
    title(ax, f"C2S-Scale 27B\n{sum(r['n_cells'] for r in c2s):,} cells")
    legends[4] = [rect(CELL.model, "cell sentence intact"), rect(CELL.perturbed, "cell sentence\ngene order shuffled"),
                  rect(CELL.perturbed2, "genes in cell sentence\nresampled from all expressed")]
    cw_rows = [r for r in ctrl["cellwhisperer"]["rows"] if r["key"] != "macro"]   # per atlas only, no mean
    ax = control_panel(5, ctrl["cellwhisperer"], rows=cw_rows, ylabel="1 \u2212 perplexity quantile")   # f
    center_ylabel(ax, Cb.YL_LADDER[1])   # one line is longer than the tick range: centre it on the
    # whole axes height
    for ax in axes:
        ax.tick_params(axis="x", labelsize=5.5)
        for t in ax.get_xticklabels():
            t.set_rotation(60)
    align_xlabels(fig, axes)

    # one legend per panel, naming what was shuffled there (as in fig_rq1_perturbation)
    y_leg = top + AXH + 0.98
    for i, handles in sorted(legends.items()):
        for h in handles:                                # long ChatNT, Prot2Text-V2 and CellWhisperer labels onto two lines
            h.set_label(h.get_label().replace("transcriptome count-shuffled", "transcriptome\ncount-shuffled"))
        xc = xs[i] + widths[i] / 2 + {2: -0.06, 3: 0.10}.get(i, 0.0)   # nudge c and d apart
        if i == len(widths) - 1:
            fig.legend(handles=handles, loc="upper right", ncol=1, handlelength=1.1,
                       bbox_to_anchor=((W - 0.02) / W, 1 - y_leg / H), **LEG)
        else:
            fig.legend(handles=handles, loc="upper center", ncol=1, handlelength=1.1,
                       bbox_to_anchor=(xc / W, 1 - y_leg / H), **LEG)
    return emit(fig, "fig_rq1_destroyed", out_dir)


# ------------------------------------------------------------------------------------------------
# figure 2: swapped -- one row: BioReason | BioReason-Pro, Prot2Text | C2S-Scale, CellWhisperer
# ------------------------------------------------------------------------------------------------
def pct_axis(ax, ylabel):
    finish(ax, ylabel, ylim=S.YL, yticks=S.YT, spine_top=1.0)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    center_ylabel(ax, 1.0)


def draw_swapped(specs, controls, out_dir):
    HEAD, LET, TIT, BOT, AXH_SW = 0.18, 0.04, 0.30, 0.74, 1.15
    H = HEAD + LET + TIT + AXH_SW + BOT
    fig = plt.figure(figsize=(W, H))
    unit = 1.20 / max(len(v) for _, _, v, _, _, _ in specs)   # one bar-group width
    single = 0.58                                             # width of a one-group panel (slots = single / unit)
    widths = [1.20, 1.20, single, single, single]
    xs = positions(0.66, widths, 0.77)          # left margin just wide enough for the percent ticks + y label
    top = HEAD + LET + TIT
    headings(fig, H, [(xs[0], xs[0] + widths[0]), (xs[1], xs[2] + widths[2]), (xs[3], xs[4] + widths[4])], top_in=0.04)
    letters = iter("abcde")
    ctrl = {c["csv_figure"]: c for c in controls if c["sw"]}
    axes = []

    def new_ax(i):
        ax = ax_in(fig, xs[i], top, widths[i], AXH_SW)
        panel_letter(fig, xs[i] - 0.42, top - TIT + 0.01, next(letters))   # top edge level with the title's first line
        axes.append(ax)
        return ax

    def bar_ticks(ax, names):
        """One tick per bar of a single-group panel (bars sit at -bw, 0, +bw in S.bars)."""
        ax.set_xticks([-S.BW, 0.0, S.BW])
        ax.set_xticklabels(names, linespacing=1.1, rotation=45, ha="right", rotation_mode="anchor")
        ax.tick_params(axis="x", labelsize=4.5)

    def main_panel(i, spec, xlabel, ttl, ylabel, bar_names=None):
        _, pal, vals, ticks, _, _ = spec
        ax = new_ax(i)
        pct_axis(ax, ylabel)
        S.bars(ax, pal, vals, slots=widths[i] / unit, fs=4.5)
        if bar_names:
            bar_ticks(ax, bar_names)
        else:
            ax.set_xticklabels(ticks, linespacing=1.1, rotation=45, ha="right", rotation_mode="anchor")
            ax.tick_params(axis="x", labelsize=4.5)
        ax.set_xlabel("")
        n = vals[0]["n"] if len(vals) == 1 else max(v["n"] for v in vals)
        title(ax, f"{ttl}\nN = {n:,}")

    def control_panel(i, c, bar_names):
        ax = new_ax(i)
        c["module"].panel_swapped(ax, c["sw"], slots=widths[i] / unit, fs=4.5)
        bar_ticks(ax, bar_names)
        ax.set_xlabel("")
        title(ax, f"{c['module'].MODEL}\nN = {c['sw']['n']:,}")

    # y labels name what is predicted; the percent axis counts the conflicts whose prediction
    # matches A / B
    main_panel(0, specs[0], "DNA source", "BioReason (RL)", "Predicted\ndisease")                    # a
    main_panel(1, specs[1], "Protein source", "BioReason-Pro (RL)", "Predicted functional\nproperty")  # b c-e hold one bar group each, so every bar gets its own
    # tick: A's input, B's text, neither
    control_panel(2, ctrl["prot2text"], ["ESM2 embedding", "Protein name", "Neither"])                # c
    main_panel(3, specs[2], "Cell source", "C2S-Scale 27B", "Cell type\nannotation",
               bar_names=["Cell sentence", "Description", "Neither"])                                 # d
    control_panel(4, ctrl["cellwhisperer"], ["Embedding", "Gene text (k=1000)", "Neither"])           # e

    # the matching A / B / neither legend applies to a and b only (c-e label every bar), so it sits
    # under those two panels and shows their two hues
    pals = (DNA, PROTEIN)
    handles = [tuple(rect(S.shade(pal, which), "") for pal in pals) for which, _ in S.WHICH]
    fig.legend(handles=handles, labels=[lab for _, lab in S.WHICH], loc="lower center",
               bbox_to_anchor=(((xs[0] + xs[1] + widths[1]) / 2) / W, 0.06 / H), ncol=3, frameon=False,
               fontsize=LEGEND_FS, handlelength=1.8, handleheight=0.8, handletextpad=0.5, columnspacing=1.2,
               handler_map={tuple: HandlerTuple(ndivide=2, pad=0)})
    return emit(fig, "fig_rq1_swapped", out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=OUT_DEFAULT)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    dna, prot_ladder, c2s = P.data_dna_rl(), M.data_ladder(), P.data_c2s_atlas()
    specs = S.panel_specs(S.data_dna_swap(), S.data_protein_abn(M.data_conflict()), S.data_sc_swap())
    controls = []
    for col, (name, mod, loader_name, spec) in enumerate((
            ("ChatNT", Cb.N, "data_chatnt", dict(xlabel_destroyed="Native tasks", xlabel_swapped="DNA source",
                                                  unit_destroyed="queries", unit_swapped="gated pairs")),
            ("Prot2Text-V2", Cb.T, "data_prot2text", dict(xlabel_destroyed="Text with ESM2", xlabel_swapped="Protein source",
                                                          unit_destroyed="proteins", unit_swapped="conflicts")),
            ("CellWhisperer", Cb.CW, "data_cellwhisperer", dict(xlabel_destroyed="Atlas", xlabel_swapped="Cell source",
                                                                unit_destroyed="cells", unit_swapped="conflicts")))):
        if mod is None:
            raise SystemExit(f"{name}: module missing")
        data = getattr(mod, loader_name)()
        if data is None:
            raise SystemExit(f"{name}: loader returned None (numbers missing)")
        rows, sw = data
        controls.append(dict(module=mod, rows=rows, sw=sw, modality_col=col, csv_figure=name.lower().replace("-", "").replace("prot2textv2", "prot2text"), **spec))

    init_print_style()
    draw_destroyed(dna, prot_ladder, c2s, controls, a.out_dir)

    keys = ["figure", "panel", "model", "checkpoint", "group", "series", "x", "value", "n", "k",
            "ci_low", "ci_high", "sd_over_folds", "source", "key"]
    maps = {"fig_rq1_destroyed": {("perturbation", "a"): "a", ("chatnt", "a"): "b", ("protein", "a"): "c",
                                  ("prot2text", "a"): "d", ("perturbation", "c"): "e", ("cellwhisperer", "a"): "f"}}
    for stem, panel_of in maps.items():
        rows = [{**{k: r.get(k, "") for k in keys}, "figure": stem, "panel": panel_of[(r["figure"], r["panel"])]}
                for r in M.ROWS if (r["figure"], r["panel"]) in panel_of
                and not (stem == "fig_rq1_destroyed" and r["figure"] == "cellwhisperer" and r.get("group") == "mean of atlases")]
        p = os.path.join(a.out_dir, f"{stem}_numbers.csv")
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {p} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

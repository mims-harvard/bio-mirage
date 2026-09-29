#!/usr/bin/env python
"""Draws the RQ4 appendix figure "Predicted reference-to-variant base pairs for the 257 bp auxiliary
condition": three confusion matrices of the base pair BioReason writes against the true pair, with
original DNA (a), shuffled DNA (b), and an evidence conflict that replaces both windows with those of
another held-out query (c, rows are the donor's pair).

Each row is normalised by its own total, and the three training seeds are pooled (435 generations
per matrix). The matrices and the counts come from auxiliary_supervision_panels.py, where they are
panels c-e. Writes fig_rq4_auxiliary_pt2 (pdf, png, svg), its numbers CSV with the panels renamed
a-c, and a LaTeX caption to outputs/figures, or to --out_dir. The canvas is 5.50 in wide, the paper
text width.

    python figures/auxiliary_supervision_matrices.py
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import auxiliary_supervision_panels as aux  # noqa: E402
from figure_style import init_print_style, ax_in, panel_letter, emit  # noqa: E402

STEM = "fig_rq4_auxiliary_pt2"
FIGURE = "rq4_pt2"
PANELS = {"c", "d", "e"}                  # the rows of the panel module's CSV this figure carries
PANEL_MAP = {"c": "a", "d": "b", "e": "c"}
TICK_FS_PAIRS = 6.0                       # pair tick labels, colour bar ticks and header
FIG_DIR = aux.FIG_DIR


W = 5.50                       # == \linewidth, so nothing is rescaled on the page
# Width budget, left to right, summing to W. The colour bar and its header get their own column, so
# the header cannot overlap panel c; draw() asserts the clearance.
L = 0.54                       # panel a's y tick labels + "True pair (ref>var)" + the "a" letter
GAP_AB = 0.20                  # b has no y ticks; this only has to hold the "b" letter
GAP_BC = 0.48                  # c's y tick labels + "Donor's pair (ref>var)" + the "c" letter
CB_GAP = 0.34                  # panel c's right edge -> the bar; wider than the centred header's
#                                overhang to the left of the bar
CB_W = 0.08                    # the bar itself
CB_TICKS = 0.16                # "0.0" / "0.5" / "1.0" at 6 pt, to the right of the bar
RIGHT = 0.10                   # canvas margin
CB_COLUMN = CB_GAP + CB_W + CB_TICKS + RIGHT
M_W = (W - L - GAP_AB - GAP_BC - CB_COLUMN) / 3      # 1.20 in per matrix


def draw(data, ref, out_dir):
    TOP, TIT, BOT = 0.10, 0.30, 0.45
    H = TOP + TIT + M_W + BOT
    fig = plt.figure(figsize=(W, H))
    row_top = TOP + TIT
    a_x = L
    b_x = a_x + M_W + GAP_AB
    c_x = b_x + M_W + GAP_BC
    cb_x = c_x + M_W + CB_GAP
    assert abs((cb_x + CB_W + CB_TICKS + RIGHT) - W) < 1e-9, cb_x + CB_W + CB_TICKS + RIGHT
    assert M_W > 1.0, f"matrices too small at W={W}: {M_W:.3f} in"

    per_seed = data["aux257"]
    n_tot = aux.N * len(per_seed)
    cmap = aux.dna_cmap()
    axes_m = []
    # `src` is the panel letter recorded in the rows (c, d, e of the panel module) and `letter` the
    # printed one; PANEL_MAP renames them when the CSV is written.
    for (cond, head, ylab, ticks), x0, src, letter in zip(aux.MATRIX_PANELS, (a_x, b_x, c_x),
                                                          "cde", "abc"):
        ax = ax_in(fig, x0, row_top, M_W, M_W)
        im = aux.draw_matrix_panel(ax, per_seed, cond, head, ylab, ticks, panel=src,
                                   figure=FIGURE, cmap=cmap, n_tot=n_tot, tick_fs=TICK_FS_PAIRS)
        panel_letter(fig, x0 - (0.48 if ticks else 0.11), row_top - TIT, letter)
        axes_m.append(ax)
    cax = aux.draw_fraction_colorbar(fig, im, cb_x, row_top + 0.25 * M_W, CB_W, 0.5 * M_W,
                                     fs=TICK_FS_PAIRS)
    for ax in axes_m:
        ax.xaxis.labelpad = 1.5

    # clearance between every pair of panels and the colour bar, element by element, in inches
    ax_a, ax_b, ax_c = axes_m
    aux.assert_clear(fig, [
        ("colour bar (axes, ticks, header)", cax, "panel c (matrix, ticks, y label)", ax_c, True),
        ("panel c's y label", ax_c.yaxis.label, "panel b (matrix, ticks)", ax_b, True),
        ("panel a, every element", ax_a, "panel b, every element", ax_b, True),
        ("panel b, every element", ax_b, "panel c, every element", ax_c, True),
        ("panel a, every element", ax_a, "panel c, every element", ax_c, True),
    ], min_gap_in=0.02, name=STEM)
    aux.assert_inside_canvas(fig, margin_in=0.02, name=STEM)
    return emit(fig, STEM, out_dir)


def write_tex(data, ref, out_dir):
    """Writes the LaTeX figure with the appendix caption."""
    n_seeds = len(data["aux257"])
    cap = rf"""% Built by auxiliary_supervision_matrices.py. Include at \linewidth = 5.5 in; the PDF canvas is
% 5.50 in wide, so it is placed at scale 1.000 and the 7 pt / 6 pt type prints at 7 pt / 6 pt.
% Numbers: {STEM}_numbers.csv (every drawn cell). The {n_seeds} training seeds are pooled,
% {aux.N}x{n_seeds}={aux.N * n_seeds} generations per matrix.
\begin{{figure}}[h]
    \centering
    \includegraphics[width=\linewidth]{{figures/{STEM}.pdf}}
    \caption{{
    \textbf{{a--c}} Predicted reference-to-variant base pairs for the 257 bp auxiliary condition with
    original DNA, shuffled DNA, or, in evidence conflicts, genomic windows from another held-out
    query.
    % Each row is normalized by its own total, so a
    % diagonal cell is the accuracy for that pair.
    }}
    \label{{fig:rq4_auxiliary_matrices}}
\end{{figure}}
"""
    open(os.path.join(out_dir, f"{STEM}.tex"), "w").write(cap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=FIG_DIR)
    a = ap.parse_args()
    init_print_style()
    data, ref = aux.load()
    paths = draw(data, ref, a.out_dir)
    write_tex(data, ref, a.out_dir)
    n = aux.write_numbers(aux.ROWS, a.out_dir, STEM, figure=FIGURE, panels=PANELS,
                          panel_map=PANEL_MAP)
    print("wrote", paths, f"{STEM}_numbers.csv ({n} rows)", f"{STEM}.tex")


if __name__ == "__main__":
    main()

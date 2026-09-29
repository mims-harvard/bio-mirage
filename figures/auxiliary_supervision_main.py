#!/usr/bin/env python
"""Draws Figure 7 (RQ4): disease prediction accuracy (a) and accuracy on the edited base pair (b)
for BioReason trained with and without auxiliary sequence supervision, on 145 held-out queries with
Z_Evo2 from intact or shuffled DNA.

Bars are the mean over three training seeds, error bars the standard error of that mean, open
circles the individual seeds and dotted lines the text-only references. The panels and the data
come from auxiliary_supervision_panels.py. Writes fig_rq4_auxiliary_pt1 (pdf, png, svg), its
numbers CSV and a LaTeX caption to outputs/figures, or to --out_dir. The canvas is 3.00 x 2.40 in and
the paper includes it at 0.38 of the 5.50 in text width.

    python figures/auxiliary_supervision_main.py
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import auxiliary_supervision_panels as aux  # noqa: E402
from figure_style import init_print_style, ax_in, align_xlabels, panel_letter, emit  # noqa: E402

STEM = "fig_rq4_auxiliary_pt1"
FIGURE = "rq4_pt1"
PANELS = {"reference", "a", "b"}      # the rows of the panel module's CSV this figure carries
FIG_DIR = aux.FIG_DIR


W, H = 3.00, 2.40                 # canvas, in inches
TITLE_PAD = 2.0                   # points between the axes top and the title anchor
LINEWIDTH_IN = 5.50               # the paper's \linewidth
WRAP_FRAC = 0.38                  # the paper wraps this figure at 0.38\linewidth
INCLUDE_IN = WRAP_FRAC * LINEWIDTH_IN   # 2.09 in -- the box the PDF is squeezed into


def level_titles_and_letters(fig, axes, letters, letter_x, texts, base_pad):
    """Puts the first title line of every panel on one level, then the panel letters on that level.

    matplotlib anchors a title by its last line, so a one-line and a two-line title share a bottom.
    Adding the height difference to the shorter title's pad makes the first lines share a top."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    heights = [ax.title.get_window_extent(r).height for ax in axes]
    tallest = max(heights)
    for ax, txt, h in zip(axes, texts, heights):
        aux.panel_title(ax, txt, pad=base_pad + (tallest - h) * 72 / fig.dpi)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    H_in = fig.get_size_inches()[1]
    tops = [ax.title.get_window_extent(r).y1 / fig.dpi for ax in axes]
    assert max(tops) - min(tops) < 0.002, ("title tops did not level up", tops)
    top_in = H_in - max(tops)
    for x, l in zip(letter_x, letters):
        panel_letter(fig, x, top_in, l)
    return top_in


def draw(data, ref, out_dir):
    # Width budget, left to right, summing to W: L and gap_ab hold the y tick labels, the y label and
    # the panel letter, and the rest is split between the two axes. Height budget, top to bottom,
    # summing to H: BOT holds the rotated two-line arm labels, the x label and the two-row legend.
    TOP, TIT, BOT = 0.06, 0.26, 0.95
    AXH = H - TOP - TIT - BOT                                  # 1.13
    # headroom above 1.0 for the rotated value labels, which have a fixed printed height
    YLIM = (0, 1.34)
    # RIGHT keeps panel b's x label, which is wider than the panel, inside the canvas; gap_ab
    # holds panel a's x label and panel b's y tick labels and y label.
    L, gap_ab, RIGHT = 0.45, 0.58, 0.15
    a_w = b_w = (W - L - gap_ab - RIGHT) / 2                   # 0.91
    assert a_w > 0.9, f"axes too narrow at W={W}: {a_w:.3f} in"
    fig = plt.figure(figsize=(W, H))
    row_top = TOP + TIT
    a_x = L
    b_x = a_x + a_w + gap_ab
    assert abs((b_x + b_w + RIGHT) - W) < 1e-9, b_x + b_w + RIGHT

    ax_a = ax_in(fig, a_x, row_top, a_w, AXH)
    aux.draw_bars_panel(ax_a, data, aux.ARM_KEYS, "disease", ref["disease"], aux.BAR_PANELS[0][2],
                        panel="a", figure=FIGURE,
                        ylim=YLIM, title_pad=TITLE_PAD)

    ax_b = ax_in(fig, b_x, row_top, b_w, AXH)
    aux.draw_bars_panel(ax_b, data, aux.ARM_KEYS, "pair", ref["pair"], aux.BAR_PANELS[1][2],
                        panel="b", figure=FIGURE,
                        ylim=YLIM, title_pad=TITLE_PAD)

    level_titles_and_letters(fig, (ax_a, ax_b), "ab", (0.035, b_x - 0.40),
                             [aux.BAR_PANELS[0][2], aux.BAR_PANELS[1][2]], TITLE_PAD)
    align_xlabels(fig, [ax_a, ax_b], pad_in=-0.07)
    # four legend entries in one row do not fit a 3.00 in canvas, so the legend is 2 x 2
    aux.place_bar_legend(fig, aux.bar_legend_handles(), (ax_a, ax_b), W / 2, ncol=2)
    # panel b's y gutter must clear panel a's labels, and neither panel may reach the canvas edge
    aux.assert_clear(fig, [("panel a, every element", ax_a, "panel b, every element", ax_b, True)],
                     min_gap_in=0.02, name=STEM)
    aux.assert_inside_canvas(fig, margin_in=0.02, name=STEM)
    return emit(fig, STEM, out_dir)


def write_tex(data, ref, out_dir):
    """Writes the LaTeX wrapfigure with the Figure 7 caption, and a header comment listing the
    printed type sizes at the scale the wrapfigure applies."""
    n_seeds = len(data["aux257"])
    seed_word = {2: "two", 3: "three", 4: "four"}[n_seeds]
    scale = INCLUDE_IN / W
    floor = 5.0                                    # print_style.py's stated floor for printed text
    drawn = [(7.0, "y tick labels, x and y axis labels, panel titles"),
             (6.0, "arm tick labels, value labels, legend"),
             (8.0, "panel letters (bold)")]
    sizes = "\n".join(
        f"%   {d:>4.1f} pt  ->  {d * scale:>4.1f} pt   {'BELOW the %.1f pt floor' % floor if d * scale < floor else 'above the floor':<26s} {what}"
        for d, what in drawn)
    cap = rf"""% Built by auxiliary_supervision_main.py. The PDF canvas is {W:.2f} x {H:.2f} in.
% Included at {WRAP_FRAC}\linewidth = {INCLUDE_IN:.2f} in, so the file is scaled by {scale:.3f}.
% Drawn size -> printed size at that scale:
{sizes}
% Numbers: {STEM}_numbers.csv (every drawn value and every seed).
\begin{{wrapfigure}}{{r}}{{{WRAP_FRAC}\linewidth}}
    \centering
    \vspace{{-2mm}}
    \includegraphics[width=\linewidth]{{figures/{STEM}.pdf}}
    \vspace{{-5mm}}
    \caption{{
    \textbf{{Auxiliary supervision enables sequence-sensitive use of $\mathbf{{Z}}_{{\mathrm{{Evo2}}}}$ with little transfer to disease prediction.}}
    \textbf{{a}} Disease prediction accuracy and \textbf{{b}} edited base pair accuracy on {aux.N} held-out
    queries with $\mathbf{{Z}}_{{\mathrm{{Evo2}}}}$ derived from original or shuffled DNA, for the model
    trained without the auxiliary task and for the two window lengths trained with it. Dotted lines
    are text-only references. Bars are the mean over {seed_word} training seeds, error bars the
    standard error of that mean.
    }}
    \vspace{{-3mm}}
    \label{{fig:rq4_auxiliary}}
\end{{wrapfigure}}
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
    n = aux.write_numbers(aux.ROWS, a.out_dir, STEM, figure=FIGURE, panels=PANELS)
    print("wrote", paths, f"{STEM}_numbers.csv ({n} rows)", f"{STEM}.tex")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Draws the RQ3 appendix figure "Genes referenced in C2S-Scale rationales, and how that set changes
when DEGs are removed" (fig_rq3_rationale_supp).

    python figures/rationale_genes.py [--out_dir <dir>]

Panel a is the composition of the genes referenced by C2S-Scale 27B rationales on the unmodified
cell sentence, per atlas and pooled, beside the composition of the input cell sentence (hatched).
Panel b is DEG precision among the first k referenced genes, k = 5, 10, 20, 40 and all. Panel c is
the Jaccard similarity of the referenced gene set to that of the unmodified rationale after removing
50% or 100% of the strongest DEGs, beside removing the same number of non-DEGs with similar
expression.

Every mark is drawn by rationale_panels.py (panel_a, panel_b and panel_d, with their legends), from
the same data loaders. This script places the three panels on a 5.50 in canvas, raises any text
drawn below 5 pt to 5 pt, and checks that no text or panel overlaps another. Reads the inputs of
rationale_panels.py under INPUT_USE_RESULTS_DIR. Writes fig_rq3_rationale_supp (pdf, png, svg) and
fig_rq3_rationale_supp_numbers.csv to outputs/figures, or to --out_dir.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rationale_panels as RP                  # noqa: E402

import faulthandler                            # noqa: E402
faulthandler.cancel_dump_traceback_later()     # rationale_panels arms a repeating timer at import

import matplotlib                              # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                # noqa: E402
from figure_style import (init_print_style, ax_in, panel_letter, align_xlabels,  # noqa: E402
                          emit, check_overflow, check_text_collisions)

STEM = "fig_rq3_rationale_supp"
OUT_DEFAULT = RP.OUT_DEFAULT

# ---- geometry, all in inches ----
W = 5.50                 # = \linewidth; the figure is included at scale 1.000
A_LEFT = 0.50            # panel a's y label + its "0 25 50 75 100" tick labels
B_LEFT = 0.04            # where row 2 starts (panel b's letter sits here)
B_LABEL, D_LABEL = 0.36, 0.47    # the label gutters of panels b and d in rationale_panels.py
AXH = 1.35               # axes height, as in rationale_panels.py, so the panels print at that size
TOP = 0.18               # canvas top to row-1 axis top: the panel-letter band
LET_TOP = 0.02           # canvas top to the top of a panel letter
ROW_GAP = 0.20           # row-1 deepest ink to row-2 axis top (holds row 2's letters)
LEG_GAP = 0.06           # axis right edge to its legend
PANEL_GAP = 0.20         # panel b's legend to panel c's label gutter
RIGHT = 0.05             # last legend to the canvas edge
BOTPAD = 0.05            # deepest ink to the canvas bottom
B_SHARE = 0.55           # how the leftover width of row 2 is split between b and c
FONT_FLOOR = 5.0         # smallest type size allowed anywhere in the figure
H_PROVISIONAL = 6.0      # first pass only: any height larger than the content can need


def measure_legend(fig, handles, extra):
    """Width of a legend in inches, before deciding where it goes. A legend's size does not depend
    on its anchor, so it is drawn at a throwaway anchor, measured and removed."""
    leg = fig.legend(handles=handles, bbox_to_anchor=(0.0, 0.5), **RP.LEGEND_KW, **extra)
    fig.canvas.draw()
    bb = leg.get_window_extent(fig.canvas.get_renderer())
    leg.remove()
    return bb.width / fig.dpi


def raise_tiny_text(ax, floor=FONT_FLOOR):
    """Lift any text this panel drew below the floor up to it, and return what was lifted."""
    lifted = []
    for t in ax.texts:
        if t.get_text() and t.get_fontsize() < floor:
            lifted.append((t.get_text(), t.get_fontsize()))
            t.set_fontsize(floor)
    return lifted


def ink_bottom(fig, artists):
    """Lowest drawn pixel of a set of artists, as inches from the TOP of the canvas."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    H = fig.get_size_inches()[1]
    lo = min(a.get_window_extent(r).y0 for a in artists)
    return H - lo / fig.dpi


def build(per, R, H, verbose=True):
    """One complete figure at height H. Returns the figure, the blocks for the QA checks, and the
    height the content actually needs."""
    fig = plt.figure(figsize=(W, H))

    def y_centre(top_in):
        return 1 - (top_in + AXH / 2) / H

    def place_legend(handles, x_in, top_in, extra):
        return fig.legend(handles=handles, bbox_to_anchor=(x_in / W, y_centre(top_in)),
                          **RP.LEGEND_KW, **extra)

    # ---------------- row 1: panel a ----------------
    leg_a_w = measure_legend(fig, RP.handles_a(), RP.LEGEND_EXTRA["a"])
    A_W = W - RIGHT - leg_a_w - LEG_GAP - A_LEFT
    ax_a = ax_in(fig, A_LEFT, TOP, A_W, AXH)
    RP.panel_a(ax_a, per, R)
    lifted = raise_tiny_text(ax_a)
    panel_letter(fig, 0.04, LET_TOP, "a")
    leg_a = place_legend(RP.handles_a(), A_LEFT + A_W + LEG_GAP, TOP, RP.LEGEND_EXTRA["a"])

    row1 = [ax_a] + list(ax_a.texts) + [t for t in ax_a.get_xticklabels() if t.get_text()] + [leg_a]
    d1 = ink_bottom(fig, row1) - (TOP + AXH)          # depth of row 1 below its axis

    # ---------------- row 2: panels b and c ----------------
    row2_top = TOP + AXH + d1 + ROW_GAP
    leg_b_w = measure_legend(fig, RP.handles_b(), RP.LEGEND_EXTRA["b"])
    leg_d_w = measure_legend(fig, RP.handles_d(), RP.LEGEND_EXTRA["d"])
    slack = (W - RIGHT - B_LEFT - B_LABEL - LEG_GAP - leg_b_w
             - PANEL_GAP - D_LABEL - LEG_GAP - leg_d_w)
    B_W, D_W = slack * B_SHARE, slack * (1 - B_SHARE)
    assert B_W > 0.5 and D_W > 0.5, (B_W, D_W)

    ax_b = ax_in(fig, B_LEFT + B_LABEL, row2_top, B_W, AXH)
    RP.panel_b(ax_b, per, R)
    panel_letter(fig, B_LEFT, row2_top - (TOP - LET_TOP), "b")
    leg_b = place_legend(RP.handles_b(), B_LEFT + B_LABEL + B_W + LEG_GAP, row2_top,
                         RP.LEGEND_EXTRA["b"])

    d_left = B_LEFT + B_LABEL + B_W + LEG_GAP + leg_b_w + PANEL_GAP
    ax_d = ax_in(fig, d_left + D_LABEL, row2_top, D_W, AXH)
    # rot=0 follows rationale_panels.layout: value labels are written flat when the panel d axis
    # is at least 0.95 in wide.
    RP.panel_d(ax_d, R, rot=0 if D_W >= 0.95 else 90)
    panel_letter(fig, d_left, row2_top - (TOP - LET_TOP), "c")
    leg_d = place_legend(RP.handles_d(), d_left + D_LABEL + D_W + LEG_GAP, row2_top,
                         RP.LEGEND_EXTRA["d"])

    align_xlabels(fig, [ax_b, ax_d])             # one level for both x labels of the row

    row2 = ([ax_b, ax_d, leg_b, leg_d]
            + [t for ax in (ax_b, ax_d) for t in ax.get_xticklabels() if t.get_text()]
            + [ax.xaxis.label for ax in (ax_b, ax_d)]
            + list(ax_b.texts) + list(ax_d.texts))
    d2 = ink_bottom(fig, row2) - (row2_top + AXH)
    h_need = row2_top + AXH + d2 + BOTPAD

    blocks = [("a axes", ax_a), ("a legend", leg_a), ("b axes", ax_b), ("b legend", leg_b),
              ("c axes", ax_d), ("c legend", leg_d)]
    if verbose:
        print(f"[layout] H {H:.2f} -> content needs {h_need:.2f} in | a axis {A_W:.2f} in at "
              f"{A_LEFT:.2f}, legend {leg_a_w:.2f}; row 2 top {row2_top:.2f}; b axis {B_W:.2f} at "
              f"{B_LEFT + B_LABEL:.2f}, legend {leg_b_w:.2f}; c axis {D_W:.2f} at "
              f"{d_left + D_LABEL:.2f}, legend {leg_d_w:.2f}; depths {d1:.2f} / {d2:.2f}",
              flush=True)
        if lifted:
            print(f"[layout] raised {len(lifted)} in-bar labels from "
                  f"{sorted({s for _, s in lifted})} pt to {FONT_FLOOR:.1f} pt", flush=True)
    return fig, blocks, h_need


# ------------------------------------------------------------------------------------------------
# QA
# ------------------------------------------------------------------------------------------------
def text_artists(fig):
    """Every text the figure will actually draw, with a name for the report."""
    out = []
    for ax in fig.axes:
        out += [(f"ax text {t.get_text()[:18]!r}", t) for t in ax.texts if t.get_text()]
        out += [(f"xtick {t.get_text()[:18]!r}", t) for t in ax.get_xticklabels() if t.get_text()]
        out += [(f"ytick {t.get_text()[:18]!r}", t) for t in ax.get_yticklabels() if t.get_text()]
        for lab in (ax.xaxis.label, ax.yaxis.label, ax.title):
            if lab.get_text():
                out.append((f"label {lab.get_text()[:18]!r}", lab))
    for leg in fig.legends:
        out += [(f"legend {t.get_text()[:18]!r}", t) for t in leg.get_texts() if t.get_text()]
    out += [(f"figure text {t.get_text()[:18]!r}", t) for t in fig.texts if t.get_text()]
    return out


def check_min_fontsize(fig, floor=FONT_FLOOR):
    """No drawn text below the floor. Reports the smallest sizes actually used."""
    sizes = {}
    for name, t in text_artists(fig):
        sizes.setdefault(round(t.get_fontsize(), 2), []).append(name)
    bad = {s: n for s, n in sizes.items() if s < floor}
    print(f"[qa] type sizes drawn: " + ", ".join(
        f"{s:g} pt x{len(n)}" for s, n in sorted(sizes.items())))
    for s, names in sorted(bad.items()):
        print(f"  BELOW FLOOR: {len(names)} items at {s} pt: {names[:4]}")
    return sum(len(n) for n in bad.values()), min(sizes)


def _boxes(name, obj, r, dpi):
    """The individual bounding boxes of one block, in inches, each with a name.

    An axes contributes its own frame AND every tick label, axis label and value label separately,
    so a number that hangs past the frame is measured where it is drawn rather than being hidden
    inside a single tight box."""
    out = []
    if hasattr(obj, "get_xticklabels"):                     # an Axes
        out.append((f"{name} frame", obj.get_window_extent(r)))
        for t in list(obj.get_xticklabels()) + list(obj.get_yticklabels()):
            if t.get_text() and t.get_visible():
                out.append((f"{name} tick {t.get_text()[:12]!r}", t.get_window_extent(r)))
        for lab in (obj.xaxis.label, obj.yaxis.label):
            if lab.get_text():
                out.append((f"{name} axis label", lab.get_window_extent(r)))
        for t in obj.texts:
            if t.get_text():
                out.append((f"{name} value {t.get_text()[:12]!r}", t.get_window_extent(r)))
    else:                                                   # a Legend
        out.append((f"{name} box", obj.get_window_extent(r)))
        for t in obj.get_texts():
            out.append((f"{name} {t.get_text()[:14]!r}", t.get_window_extent(r)))
    return [(n, [b.x0 / dpi, b.y0 / dpi, b.x1 / dpi, b.y1 / dpi]) for n, b in out]


def _clearance(a, b):
    """Distance in inches between two rectangles; negative = they overlap, by the smaller of the
    two penetration depths."""
    dx = max(b[0] - a[2], a[0] - b[2], 0.0)
    dy = max(b[1] - a[3], a[1] - b[3], 0.0)
    if dx == 0.0 and dy == 0.0:
        return -min(min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1]))
    return math.hypot(dx, dy)


def check_separation(fig, blocks, floor=0.01):
    """Element-by-element clearance between every pair of panel/legend blocks.

    check_text_collisions compares text against text only, so a legend box or a colour-bar header
    sitting on a neighbouring axes passes it. This compares every element of one block against
    every element of another (frames, ticks, axis labels, value labels, legend box, legend
    entries) and reports the closest pair of each block pair."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    dpi = fig.dpi
    elems = [(name, _boxes(name, obj, r, dpi)) for name, obj in blocks]
    pairs = []
    for i in range(len(elems)):
        for j in range(i + 1, len(elems)):
            best = min(((_clearance(ba, bb), na, nb)
                        for na, ba in elems[i][1] for nb, bb in elems[j][1]),
                       key=lambda t: t[0])
            pairs.append((best[0], elems[i][0], elems[j][0], best[1], best[2]))
    pairs.sort()
    print("[qa] block clearances, closest first (in):")
    for gap, bi, bj, ni, nj in pairs:
        flag = "  <-- OVERLAP" if gap < 0 else ("  <-- TIGHT" if gap < floor else "")
        print(f"      {gap:+7.3f}  {bi:9s} x {bj:9s}   {ni} x {nj}{flag}")
    n_bad = sum(1 for p in pairs if p[0] < floor)
    print(f"[qa] {n_bad} block pairs closer than {floor:.3f} in; "
          f"closest {pairs[0][0]:.3f} in ({pairs[0][1]} x {pairs[0][2]})")
    return n_bad, pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=OUT_DEFAULT)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    R = RP.load_artifact()
    per = RP.compute(RP.RATIONALES)
    init_print_style()
    matplotlib.rcParams["hatch.linewidth"] = 0.45      # the hatch weight of rationale_panels.py

    # the rows these panels record go out under this figure's name, with d renamed to c
    RP.FIGURE = "rq3_rationale_supp"
    RP.PANEL_ALIAS = {"a": "a", "b": "b", "d": "c"}

    # pass 1 measures the two tick blocks at a height that cannot constrain them; pass 2 is the
    # figure, built at the height they need.
    RP.ROWS.clear()
    fig, _, h_need = build(per, R, H_PROVISIONAL)
    plt.close(fig)
    RP.ROWS.clear()
    fig, blocks, h2 = build(per, R, h_need)
    assert abs(h2 - h_need) <= 0.01, (h_need, h2)

    n_small, smallest = check_min_fontsize(fig)
    assert n_small == 0, f"{n_small} text items below {FONT_FLOOR} pt"
    n_sep, _ = check_separation(fig, blocks)
    assert n_sep == 0, f"{n_sep} block pairs overlap or nearly touch"
    n_over = len(check_overflow(fig, STEM))
    n_coll = check_text_collisions(fig, STEM)
    print(f"[qa] check_overflow {n_over}, check_text_collisions {n_coll}, "
          f"smallest type {smallest:g} pt, figure {W:.2f} x {h_need:.2f} in "
          f"(scale at 5.50 in \\linewidth: {5.50 / W:.3f})")
    assert n_over == 0 and n_coll == 0

    emit(fig, STEM, a.out_dir)
    RP.write_numbers(RP.ROWS, os.path.join(a.out_dir, f"{STEM}_numbers.csv"))


if __name__ == "__main__":
    main()

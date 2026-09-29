"""Shared plotting style for all figures: the palette (one hue per modality, with lighter and darker
variants for the model, perturbed, probe and baseline bars), axes placement in inches, bar labels,
panel letters, text overlap checks, and emit(), which saves a figure as pdf, png and svg.

Re-exports the font sizes and line widths of print_style.py.
"""
import os
import sys

import numpy as np
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from print_style import (init_print_style, save_exact, center_ylabel, margins,  # noqa: E402,F401
                         check_overflow, TICK_FS, AXIS_FS, ANNOT_FS, LEGEND_FS,
                         EDGE_LW, ERR_LW, SPINE_LW)

# ================================ palette ================================ hue says which modality
# is being read; saturation and lightness say what role the mark plays.
import colorsys
from typing import NamedTuple


class Palette(NamedTuple):
    model: str
    model_dark: str
    perturbed: str
    probe: str
    baseline: str
    perturbed2: str


def _hex2rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def _rgb2hex(c):
    return "#%02X%02X%02X" % tuple(round(max(0.0, min(1.0, v)) * 255) for v in c)


def _mix(h, target, t):
    a, b = _hex2rgb(h), _hex2rgb(target)
    return _rgb2hex(tuple(a[i] * (1 - t) + b[i] * t for i in range(3)))


def _drain(h, saturation, lightness):
    hue, _, _ = colorsys.rgb_to_hls(*_hex2rgb(h))
    return _rgb2hex(colorsys.hls_to_rgb(hue, lightness, saturation))


def derive(base):
    """The four roles of one modality hue. Add a fourth modality by calling this on its base hex; do
    not hand-pick a shade, or the roles stop meaning the same thing across hues.
    """
    return Palette(model=base, model_dark=_mix(base, "#000000", 0.28),
                   perturbed=_drain(base, 0.20, 0.64), probe=_mix(base, "#FFFFFF", 0.55),
                   baseline=_mix(base, "#FFFFFF", 0.35), perturbed2=_drain(base, 0.12, 0.76))


DNA = derive("#D86ECC")
PROTEIN = derive("#0F9ED5")
CELL = derive("#4EA72E")
MODALITY = {"dna": DNA, "protein": PROTEIN, "cell": CELL}

# Resolved values, so the file (and skill.md) states them without being run. Asserted against the
# derivation so the table and the code cannot drift apart.
_RESOLVED = {
    "dna":     Palette("#D86ECC", "#9C4F93", "#B691B1", "#EDBEE8", "#E6A1DE", "#C9BAC7"),
    "protein": Palette("#0F9ED5", "#0B7299", "#91ABB6", "#93D3EC", "#63C0E4", "#BAC5C9"),
    "cell":    Palette("#4EA72E", "#387821", "#9BB691", "#AFD7A1", "#8CC677", "#BEC9BA"),
}
for _k, _v in _RESOLVED.items():
    assert MODALITY[_k] == _v, (_k, MODALITY[_k], _v)

GREY = "#B0B0B0"                              # a non-model reference: chance, text-only, zero
CTRL = ["#C9C9C9", "#9E9E9E", "#707070"]      # graded non-model control arms, light -> dark (the lightest
                                              # must still read as a 0.9 pt line)

# Axis headroom above the last tick, in axis units, sized to what the value labels occupy. Carrying
# a blanket 1.34 everywhere leaves a band of empty axis under the panel letters.
HEADROOM_ROTATED = 0.20      # value labels written at 90 degrees
HEADROOM_HORIZONTAL = 0.10   # value labels written flat


# ================================ layout ================================
def ax_in(fig, left, top, width, height):
    """Axes placed by inches from the left and top edges. Fractions change meaning when the figure size
    changes, which is how a 1.2 in panel ends up with a 0.9 in gutter.
    """
    W, H = fig.get_size_inches()
    return fig.add_axes([left / W, 1 - (top + height) / H, width / W, height / H])


def finish(ax, ylabel=None, ylim=(0, 1.05), yticks=None, spine_top=None):
    """Despine, drop the x label, set the scale, and stop the left spine where the ticks stop."""
    sns.despine(ax=ax)
    ax.set_xlabel("")
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    if yticks is not None:
        ax.set_yticks(yticks)
    if spine_top is not None:
        ax.spines["left"].set_bounds(ylim[0], spine_top)
    ax.tick_params(axis="x", length=0)


def bar_label(ax, x, y, text, rot=90, pad=0.012, fs=ANNOT_FS, mask=False):
    """Value label above a bar. Rotate when the bar is narrower than the number; write it flat when
    there is room, and drop the unit if the axis already carries it.
    """
    lo, hi = ax.get_ylim()
    ax.text(x, y + pad * (hi - lo), text, ha="center", va="bottom", fontsize=fs, rotation=rot,
            zorder=6, bbox=dict(facecolor="white", edgecolor="none", pad=0.3) if mask else None)


def facet_title(ax, text):
    """Bold facet name above the axes. Use for a facet (a checkpoint, a model size), not as a panel
    title restating the caption.
    """
    ax.set_title(text, fontsize=TICK_FS, fontweight="bold", pad=2.5)


def legend_upper(ax, handles, ncol, leg_y, loc="lower left", anchor_x=0.0):
    """Legend on the band above the axes. At print size `loc="best"` is not available: there is no
    empty corner left inside a 1-2 in panel.
    """
    ax.legend(handles=handles, loc=loc, bbox_to_anchor=(anchor_x, leg_y), ncol=ncol,
              frameon=False, fontsize=LEGEND_FS, borderaxespad=0.0, handlelength=1.0,
              handletextpad=0.4, columnspacing=0.9)


def rotate_ticks(ax, labels, rotation=45, size=6.0):
    """45-degree x tick labels that end at their tick."""
    ax.set_xticklabels(labels, rotation=rotation, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", labelsize=size)


def align_xlabels(fig, axes, pad_in=0.05):
    """Put the x labels of several panels on one measured level.

    matplotlib drops each x label just under that axes' own tick block, so a panel with deeper tick
    labels (two lines, or rotated) gets a lower label than its neighbour and the row looks broken.
    This finds the deepest tick block across `axes` and puts every label just below it, so the level
    survives a label change."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    lowest = min(min((t.get_window_extent(r).y0 for t in ax.get_xticklabels() if t.get_text()),
                     default=float("inf")) for ax in axes)
    for ax in axes:
        bb = ax.get_window_extent(r)
        ax.xaxis.set_label_coords(0.5, (lowest - pad_in * fig.dpi - bb.y0) / bb.height)


def panel_letter(fig, x_in, top_in, letter, fontsize=8):
    """Bold panel letter placed by inches from the left and top edges."""
    W, H = fig.get_size_inches()
    fig.text(x_in / W, 1 - top_in / H, letter, fontsize=fontsize, fontweight="bold",
             ha="left", va="top")


# ================================ qa ================================
def check_text_collisions(fig, name=""):
    """Measured overlap between any two text items, plus clipping of figure-level text.

    Two things `print_style.check_overflow` cannot catch, both found the hard way:
      * it walks fig.axes and fig.legends but not fig.texts, so a figure-level column header or
        panel letter can be cut by the canvas edge with no warning;
      * an axes turned off with set_axis_off() still carries tick-label artists, which it reports as
        overflowing. Call ax.set_xticks([]) / set_yticks([]) on such an axes.

    Rotated text is skipped in the pairwise test: a diagonal label's bounding box is the axis-aligned
    rectangle around it, so two neighbouring 45-degree labels overlap as boxes long before their
    glyphs do. Judge those by eye on the rendered png."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    n_bad, items = 0, []
    for ax in fig.axes:
        items += [t for t in ax.texts if t.get_text() and t.get_visible()]
        if not ax.axison:
            continue

        def in_view(axis, t, ax=ax):
            lo, hi = sorted(axis.get_view_interval())
            pos = t.get_position()[1 if axis is ax.yaxis else 0]
            return lo - 1e-9 <= pos <= hi + 1e-9
        items += [t for t in ax.xaxis.get_ticklabels(which="both")
                  if t.get_text() and t.get_visible() and in_view(ax.xaxis, t)]
        items += [t for t in ax.yaxis.get_ticklabels()
                  if t.get_text() and t.get_visible() and in_view(ax.yaxis, t)]
        if ax.get_legend():
            items += list(ax.get_legend().get_texts())
    for ax in fig.axes:
        if ax.title.get_text():
            items.append(ax.title)
        items += [lab for lab in (ax.xaxis.label, ax.yaxis.label) if lab.get_text() and ax.axison]
    for leg in fig.legends:
        items += list(leg.get_texts())
    items += [t for t in fig.texts if t.get_text()]

    Wpx, Hpx = (d * r.dpi for d in fig.get_size_inches())
    for t in fig.texts:
        if not t.get_text():
            continue
        b = t.get_window_extent(r)
        off = [d for d, v in (("left", -b.x0), ("bottom", -b.y0),
                              ("right", b.x1 - Wpx), ("top", b.y1 - Hpx)) if v > 0.5]
        if off:
            n_bad += 1
            short = max(-b.x0, -b.y0, b.x1 - Wpx, b.y1 - Hpx) / r.dpi
            print(f"  CLIPPED {name}: figure text {t.get_text()[:34]!r} leaves the "
                  f"{'+'.join(off)} edge by {short:.3f} in")
    boxes = [t.get_window_extent(r) for t in items]
    rot = [abs(float(t.get_rotation())) % 180 for t in items]
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if rot[i] not in (0.0, 90.0) or rot[j] not in (0.0, 90.0):
                continue
            b1, b2 = boxes[i], boxes[j]
            ox = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
            oy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
            if ox > 1.0 and oy > 1.0:
                n_bad += 1
                print(f"  COLLISION {name}: {items[i].get_text()[:28]!r} x "
                      f"{items[j].get_text()[:28]!r} ({ox / r.dpi:.3f} x {oy / r.dpi:.3f} in)")
    return n_bad


def emit(fig, stem, out_dir, formats=("pdf", "png", "svg")):
    """Check, save at exactly figsize, and refuse to report success on an empty file."""
    import matplotlib.pyplot as plt
    n = check_text_collisions(fig, stem)
    print(f"[layout] {stem}: {n} text collisions")
    paths = save_exact(fig, stem, out_dir, formats=formats)
    w, h = fig.get_size_inches()
    plt.close(fig)
    for p in paths:
        if os.path.getsize(p) == 0:
            raise RuntimeError(f"empty output {p}")
        print(f"  {p}  {os.path.getsize(p) / 1024:.1f} KB")
    print(f"  {stem}: {w:.2f} x {h:.2f} in")
    return paths

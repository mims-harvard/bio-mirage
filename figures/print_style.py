"""Font sizes, line widths and matplotlib settings for figure panels printed about 2 in wide, plus
helpers that set margins, centre y labels, check for text outside the figure, and save at exact size.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

TICK_FS = 7.0      # x and y tick labels
AXIS_FS = 7.0      # axis labels
ANNOT_FS = 6.0     # numbers printed on the bars, direction labels, side columns
LEGEND_FS = 6.0

EDGE_LW = 0.5      # bar outlines
ERR_LW = 0.6       # error-bar lines
SPINE_LW = 0.6


def init_print_style():
    """rcParams for print-size panels. Call once, before any figure is made."""
    mpl.rcParams.update({
        "font.size": TICK_FS,
        "axes.labelsize": AXIS_FS,
        "axes.titlesize": AXIS_FS,
        "xtick.labelsize": TICK_FS,
        "ytick.labelsize": TICK_FS,
        "legend.fontsize": LEGEND_FS,
        "axes.linewidth": SPINE_LW,
        "xtick.major.width": SPINE_LW,
        "ytick.major.width": SPINE_LW,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "xtick.major.pad": 1.5,
        "ytick.major.pad": 1.5,
        "axes.labelpad": 3.0,
        "lines.linewidth": 0.8,
        "hatch.linewidth": 0.4,
        "legend.handlelength": 1.1,
        "legend.handletextpad": 0.4,
        "legend.columnspacing": 0.8,
        "legend.labelspacing": 0.4,
        "svg.fonttype": "none",     # keep text as text so the typesetter can restyle it
        "pdf.fonttype": 42,
    })


def margins(fig, left, right, bottom, top):
    """Place the axes by margins given in inches rather than figure fractions."""
    w, h = fig.get_size_inches()
    fig.subplots_adjust(left=left / w, right=1 - right / w,
                        bottom=bottom / h, top=1 - top / h)


def check_overflow(fig, name=""):
    """Print any text that runs outside the canvas; return the offending items."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = (d * fig.dpi for d in fig.get_size_inches())
    bad = []
    def in_view(axis, label):
        """Tick labels exist for ticks outside the view; those are never drawn."""
        lo, hi = sorted(axis.get_view_interval())
        pos = label.get_position()[1 if axis is axis.axes.yaxis else 0]
        return lo - 1e-9 <= pos <= hi + 1e-9

    for ax in fig.axes:
        items = list(ax.texts) + [ax.title, ax.xaxis.label, ax.yaxis.label]
        items += [t for t in ax.get_xticklabels() if in_view(ax.xaxis, t)]
        items += [t for t in ax.get_yticklabels() if in_view(ax.yaxis, t)]
        if ax.get_legend():
            items += ax.get_legend().get_texts()
        for t in items:
            if not t.get_text() or not t.get_visible():
                continue
            b = t.get_window_extent(r)
            off = [d for d, v in (("left", -b.x0), ("bottom", -b.y0),
                                  ("right", b.x1 - W), ("top", b.y1 - H)) if v > 0.5]
            if off:
                short = max(max(-b.x0, -b.y0, b.x1 - W, b.y1 - H), 0) / fig.dpi
                bad.append((t.get_text().replace("\n", " "), "+".join(off), short))
    for leg in getattr(fig, "legends", []):
        for t in leg.get_texts():
            b = t.get_window_extent(r)
            if b.x0 < -0.5 or b.y0 < -0.5 or b.x1 > W + 0.5 or b.y1 > H + 0.5:
                bad.append((t.get_text(), "figure legend", 0.0))
    for text, where, short in bad:
        print(f"  WARNING {name}: {text[:36]!r} runs off the {where} edge "
              f"by {short:.3f} in -- widen that gutter")

    # A value label that grows past the top of the axes lands on the title. Nothing leaves the
    # canvas, so the check above cannot see it; this compares the two boxes directly.
    for ax in fig.axes:
        if not ax.title.get_text() or not ax.title.get_visible():
            continue
        tb = ax.title.get_window_extent(r)
        for t in ax.texts:
            if not t.get_text() or not t.get_visible():
                continue
            b = t.get_window_extent(r)
            if b.overlaps(tb):
                over = (min(b.y1, tb.y1) - max(b.y0, tb.y0)) / fig.dpi
                bad.append((t.get_text(), "title", over))
                print(f"  WARNING {name}: {t.get_text()[:36]!r} overlaps the title by "
                      f"{over:.3f} in -- add headroom above the last tick")
    return bad


def center_ylabel(ax, tick_top, tick_bottom=0.0):
    """Centre the y label on the tick range, not on the full ylim."""
    lo, hi = ax.get_ylim()
    if hi <= lo:
        return
    # set_y, not set_label_coords: the y label's y is in axes coordinates and is preserved across
    # draws, while its x is recomputed each draw from the tick-label extents.
    ax.yaxis.label.set_y(((tick_bottom + tick_top) / 2 - lo) / (hi - lo))


def grow_bottom_to_fit(fig, pad_in=0.05):
    """Grow the canvas downward until nothing hangs off the bottom, axes size unchanged."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    lowest = 0.0
    for ax in fig.axes:
        for t in list(ax.texts) + ax.get_xticklabels() + [ax.xaxis.label]:
            if t.get_text() and t.get_visible():
                lowest = min(lowest, t.get_window_extent(r).y0)
    add = max(0.0, -lowest / fig.dpi + pad_in) if lowest < 0 else 0.0
    if add <= 0:
        return 0.0
    w, h = fig.get_size_inches()
    new_h = h + add
    boxes = [(ax, ax.get_position()) for ax in fig.axes]
    fig.set_size_inches(w, new_h, forward=True)
    for ax, b in boxes:
        ax.set_position([b.x0, (b.y0 * h + add) / new_h, b.width, b.height * h / new_h])
    return add


def save_exact(fig, path_no_ext, fig_dir, formats=("svg", "pdf", "png"),
               registry=None):
    """Save at exactly `figsize` -- no bbox_inches='tight'."""
    import os
    check_overflow(fig, os.path.basename(path_no_ext))
    os.makedirs(fig_dir, exist_ok=True)
    out = []
    for ext in formats:
        p = os.path.join(fig_dir, f"{path_no_ext}.{ext}")
        fig.savefig(p, format=ext, dpi=600 if ext == "png" else None,
                    bbox_inches=None, facecolor="white")
        out.append(p)
        if registry is not None:
            registry.append(p)
    return out


def bar_width_in(n_models, n_groups, axes_w_in, group_span=0.90):
    """Printed width of one bar, inches. Below ~0.03 in it is thinner than its outline."""
    return group_span / n_models * axes_w_in / n_groups

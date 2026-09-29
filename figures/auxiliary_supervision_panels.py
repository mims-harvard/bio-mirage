#!/usr/bin/env python
"""Panels and data for Figure 7 and the appendix figure of predicted base pairs (RQ4): BioReason
trained with and without auxiliary sequence supervision, three training seeds per condition.

load() scores the stored generations of three conditions on the 145 held-out queries: no auxiliary
target with 2,048 bp windows, and the auxiliary target with 257 bp or 2,048 bp windows, each window
centred on the edit. Disease prediction accuracy is read from the mapped answer. The base pair is
read with one parser for all three conditions, which takes the first reference-to-variant base pair
stated in the reasoning trace, and is scored against the pair at the first index where the two
windows differ, so all 145 queries are in the denominator. For the two auxiliary conditions the
script asserts that this parser returns the same score as reading the "Edit at" line. The
confusion matrices pool the three seeds of the 257 bp condition.

auxiliary_supervision_main.py (Figure 7) and auxiliary_supervision_matrices.py (appendix) import
the panel functions and load() from here. Run on its own, the module draws all five panels on one
canvas as fig_rq4_auxiliary_panels, which is not in the paper. Reads the generations and the
training exports in dna/bioreason/auxiliary_supervision under INPUT_USE_RESULTS_DIR and
kegg/ood_test_network_split.csv under INPUT_USE_DATA_DIR. Writes the
figure (pdf, png, svg), its numbers CSV and a LaTeX caption to outputs/figures, or to --out_dir.

    python figures/auxiliary_supervision_panels.py
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from input_use.core import paths as RD

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figure_style import (init_print_style, center_ylabel, ax_in, finish, bar_label,  # noqa: E402
                          align_xlabels, panel_letter, emit, DNA, GREY,
                          TICK_FS, LEGEND_FS, EDGE_LW, ERR_LW)

DATA_DIR = os.environ.get("INPUT_USE_DATA_DIR", "data")
FIG_DIR = os.path.join(os.environ.get("INPUT_USE_OUTPUT_DIR", "outputs"), "figures")
STEM = "fig_rq4_auxiliary_panels"
SRC = f"{DATA_DIR}/kegg"
GENERATIONS = RD.AUX_GENERATIONS
N = 145
SEEDS = [23, 24, 25]
PAT = re.compile(r"Edit at (\d+): ref ([ACGT]) -> var ([ACGT])")
PAIRS = [f"{x}>{y}" for x in "ACGT" for y in "ACGT" if x != y]      # 12, grouped by reference base
ARMS = [
    dict(key="noaux2048c", label="no auxiliary,\n2,048 bp", short="no auxiliary, 2,048 bp centered",
         gen="no_aux_2048bp", data=RD.AUX_NO_TARGET_2048BP, pair=True, aux_format=False),
    dict(key="aux257", label="auxiliary,\n257 bp", short="auxiliary, 257 bp",
         gen="aux_257bp", data=RD.AUX_TARGET_257BP, pair=True, aux_format=True),
    dict(key="aux2048c", label="auxiliary,\n2,048 bp", short="auxiliary, 2,048 bp centered",
         gen="aux_2048bp", data=RD.AUX_TARGET_2048BP, pair=True, aux_format=True),
]
ROWS: list[dict] = []

# format-agnostic base-pair parser: the first "X>Y", "ref X -> var Y", "X->Y", "X to/into Y" or
# spelled-out base pair in the text
BASENAME = {"adenine": "A", "thymine": "T", "cytosine": "C", "guanine": "G"}
P_UNION = [re.compile(r"\b([ACGT])>([ACGT])\b"),
           re.compile(r"ref\s*([ACGT])\s*->\s*var\s*([ACGT])"),
           re.compile(r"(?<![A-Za-z0-9])([ACGT])\s*(?:->|-->|→|>)\s*([ACGT])(?![A-Za-z0-9])"),
           re.compile(r"(?<![A-Za-z0-9])([ACGT])\s+(?:to|into)\s+([ACGT])(?![A-Za-z0-9])"),
           re.compile(r"\b(adenine|thymine|cytosine|guanine)\s*(?:>|->|→|\s+to\s+|\s+into\s+)\s*"
                      r"(adenine|thymine|cytosine|guanine)\b", re.I)]
THINK = re.compile(r"<think>(.*?)</think>", re.S)


def trace_of(g):
    """The reasoning part of a generation: inside <think> if it closed, else everything before
    'Answer:'.  Keeps a claim in the answer sentence from being read as the trace's claim."""
    m = THINK.search(g or "")
    if m:
        return m.group(1)
    i = (g or "").find("Answer:")
    return (g or "")[:i] if i >= 0 else (g or "")


def claim_union(text):
    """First stated reference>variant base pair in `text`, as 'C>T', else None."""
    best = None
    for pat in P_UNION:
        m = pat.search(text or "")
        if m and (best is None or m.start() < best[0]):
            a, b = m.group(1), m.group(2)
            if a.lower() in BASENAME:
                a, b = BASENAME[a.lower()], BASENAME[b.lower()]
            best = (m.start(), a.upper() + ">" + b.upper())
    return best[1] if best else None


def variant_classes():
    """Class of each held-out query (single-base or multi-base substitution, insertion, deletion)
    from the full source sequences, by the rule export_257bp_windows.py uses for the class tag of the
    auxiliary target. Generation row i is source row perm[i], because evaluate_generation.py
    shuffles the queries with Random(11)."""
    import random
    csv.field_size_limit(10 ** 9)
    te = list(csv.DictReader(open(f"{SRC}/ood_test_network_split.csv", newline="")))
    assert len(te) == N, len(te)

    def klass(a, b):
        if len(a) == len(b):
            d = sum(1 for x, y in zip(a, b) if x != y)
            return "single-base substitution" if d == 1 else "multi-base substitution"
        return "insertion" if len(b) > len(a) else "deletion"
    cl = [klass(r["reference_sequence"], r["variant_sequence"]) for r in te]
    perm = list(range(N))
    random.Random(11).shuffle(perm)
    return [cl[i] for i in perm], te, perm      # indexed by GENERATION row


def rec(**kw):
    ROWS.append(kw)


# ================================ scoring (as in experiments/auxiliary_supervision/score_seeds.py)
def norm_label(text):
    if not text:
        return ""
    t = text.strip().lower().replace("\n", " ")
    t = re.sub(r"[^\w\s\-'/]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def first_diff(a, b):
    return next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), None)


def score_one(path, want_pair, aux_format, gen_class):
    d = json.load(open(path))
    R, s = d["rows"], d["summary"]
    assert len(R) == N == s["n_rows"], (path, len(R))
    assert s["generation_identical_wt_vs_same"] == 1.0, ("determinism control failed", path)
    out = dict(path=path, window_bp=len(R[0]["ref_wt"]),
               unmapped={c: int(s[f"unmapped_{c}"]) for c in ("wt", "shuffle", "swap")})
    for c in ("wt", "shuffle", "swap"):
        hits = [norm_label(r[f"pred_{c}"]) == norm_label(r["truth"]) for r in R]
        out[f"disease_{c}"] = sum(hits) / N
        out[f"gen_changed_{c}"] = int(sum(r[f"gen_{c}"] != r["gen_wt"] for r in R))
        out[f"answer_changed_{c}"] = int(sum(norm_label(r[f"pred_{c}"]) != norm_label(r["pred_wt"]) for r in R))
    if not want_pair:
        return out
    tp, dp, em, em_aux = [], [], {}, {}
    for r in R:
        j = first_diff(r["ref_wt"], r["var_wt"]); tp.append(r["ref_wt"][j] + ">" + r["var_wt"][j])
        k = first_diff(r["ref_swap"], r["var_swap"]); dp.append(r["ref_swap"][k] + ">" + r["var_swap"][k])
    sub = np.array([c == "single-base substitution" for c in gen_class])
    assert len(gen_class) == N and sub.sum() == 111, (path, int(sub.sum()))
    for c in ("wt", "shuffle", "swap"):
        # the drawn score: one format-agnostic parser for every arm
        em[c] = [claim_union(trace_of(r[f"gen_{c}"])) for r in R]
        # the "Edit at" parser, kept so the two auxiliary arms can be checked against it
        m = [PAT.search(r[f"gen_{c}"] or "") for r in R]
        em_aux[c] = [(x.group(2) + ">" + x.group(3)) if x else None for x in m]
        out[f"unparseable_{c}"] = int(sum(x is None for x in em[c]))
        out[f"unparseable_auxformat_{c}"] = int(sum(x is None for x in m))
        hit = np.array([e == t for e, t in zip(em[c], tp)])
        out[f"pair_{c}"] = float(hit.mean())
        out[f"pair_k_{c}"] = int(hit.sum())
        out[f"pair_substitution_rows_{c}"] = float(hit[sub].mean())
        out[f"pair_substitution_rows_k_{c}"] = int(hit[sub].sum())
        out[f"pair_auxformat_{c}"] = sum(e == t for e, t in zip(em_aux[c], tp)) / N
        if aux_format:
            # on the auxiliary arms the drawn score must equal the "Edit at" score
            assert abs(out[f"pair_{c}"] - out[f"pair_auxformat_{c}"]) < 1e-12, (path, c)
    out["n_substitution_rows"] = int(sub.sum())
    out["donor_matching"] = sum(e == p for e, p in zip(em["swap"], dp)) / N
    out["recipient_matching_swap"] = sum(e == t for e, t in zip(em["swap"], tp)) / N
    out["ref_base_wt"] = sum(e is not None and e[0] == t[0] for e, t in zip(em["wt"], tp)) / N
    if not aux_format:
        return out
    # confusion counts (panels c-e, 257 bp arm only): rows = the recipient's pair (wt, shuffle) or
    # the donor's pair (swap). Read with the "Edit at" parser.
    out["conf"] = {c: collections.Counter(zip(tp if c != "swap" else dp, em_aux[c]))
                   for c in ("wt", "shuffle", "swap")}
    # expected diagonal if the written pair were independent of the pair defining the rows
    out["chance"] = {}
    for c in ("wt", "shuffle", "swap"):
        ref = tp if c != "swap" else dp
        out["chance"][c] = sum(sum(1 for x in ref if x == q) * sum(1 for x in em_aux[c] if x == q)
                               for q in set(ref)) / N
    return out


def references():
    """Text-only references: a majority-vote lookup from the gene named in the query (disease),
    and always writing the most frequent training pair (base pair)."""
    csv.field_size_limit(10 ** 9)
    tr = list(csv.DictReader(open(f"{RD.AUX_TARGET_257BP}/train.csv", newline="")))
    te = list(csv.DictReader(open(f"{RD.AUX_TARGET_257BP}/test.csv", newline="")))
    pair = lambda r: "%s>%s" % re.search(r"ref ([ACGT]) -> var ([ACGT])", r["reasoning"]).groups()
    top = collections.Counter(pair(r) for r in tr).most_common(1)[0][0]
    gene = lambda r: (re.search(r"of this (\S+) allele", r["question"]) or [None, None])[1]
    lut = collections.defaultdict(collections.Counter)
    for r in tr:
        lut[gene(r)][r["answer"]] += 1
    hit = sum(1 for r in te if gene(r) in lut and lut[gene(r)].most_common(1)[0][0] == r["answer"])
    return dict(disease=hit / len(te), disease_k=hit, pair=sum(pair(r) == top for r in te) / len(te),
                pair_k=sum(pair(r) == top for r in te), top_pair=top)


def load():
    ref = references()
    for k, v in ref.items():
        rec(figure="rq4_auxiliary_panels", panel="reference", arm="text-only", seed="", condition="", series=k,
            value=v, n=N, drawn=(k in ("disease", "pair")),
            source=f"{RD.AUX_TARGET_257BP}/{{train,test}}.csv", key="references() in this script")
    gen_class, _te, _perm = variant_classes()
    for c, n in sorted(collections.Counter(gen_class).items()):
        rec(figure="rq4_auxiliary_panels", panel="b", arm="all arms", seed="", condition="",
            series=f"held-out queries whose variant is a {c}", value=round(n / N, 4), n=N, k=n,
            drawn=False, source=f"{SRC}/ood_test_network_split.csv",
            key="export_257bp_windows.py classify() on the full sequences")
    data = {}
    for a in ARMS:
        per_seed = {}
        for s in SEEDS:
            fn = f"{a['gen']}_seed{s}.json"
            p = os.path.join(GENERATIONS, fn)
            if not os.path.exists(p):
                print(f"  MISSING {fn}"); continue
            per_seed[s] = score_one(p, a["pair"], a["aux_format"], gen_class)
            e = per_seed[s]
            for c in ("wt", "shuffle", "swap"):
                rec(figure="rq4_auxiliary_panels", panel="a", arm=a["short"], seed=s, condition=c,
                    series="disease accuracy", value=e[f"disease_{c}"], n=N,
                    k=round(e[f"disease_{c}"] * N), drawn=(c in ("wt", "shuffle")), source=fn,
                    key=f"rows[*].pred_{c} vs truth")
                rec(figure="rq4_auxiliary_panels", panel="a", arm=a["short"], seed=s, condition=c,
                    series="generation differs from wt", value=e[f"gen_changed_{c}"] / N, n=N,
                    k=e[f"gen_changed_{c}"], drawn=False, source=fn, key="gen_changed")
                rec(figure="rq4_auxiliary_panels", panel="a", arm=a["short"], seed=s, condition=c,
                    series="mapped answer differs from wt", value=e[f"answer_changed_{c}"] / N, n=N,
                    k=e[f"answer_changed_{c}"], drawn=False, source=fn, key="answer_changed")
                if a["pair"]:
                    rec(figure="rq4_auxiliary_panels", panel="b", arm=a["short"], seed=s, condition=c,
                        series="stated ref>var equals the variant's, all 145 rows (DRAWN)",
                        value=round(e[f"pair_{c}"], 4), n=N, k=e[f"pair_k_{c}"],
                        drawn=(c in ("wt", "shuffle")), source=fn,
                        key=f"format-agnostic parse of rows[*].gen_{c} vs ref_wt/var_wt at first_diff")
                    rec(figure="rq4_auxiliary_panels", panel="b", arm=a["short"], seed=s, condition=c,
                        series="stated ref>var equals the variant's, single-base substitution rows only",
                        value=round(e[f"pair_substitution_rows_{c}"], 4), n=e["n_substitution_rows"],
                        k=e[f"pair_substitution_rows_k_{c}"], drawn=False, source=fn,
                        key="same parse, restricted to the 111 substitution rows")
                    rec(figure="rq4_auxiliary_panels", panel="b", arm=a["short"], seed=s, condition=c,
                        series="same score read with the published 'Edit at' parser",
                        value=round(e[f"pair_auxformat_{c}"], 4), n=N,
                        k=round(e[f"pair_auxformat_{c}"] * N), drawn=False, source=fn,
                        key="score_seeds.py PAT; equals the drawn value for the two auxiliary arms")
                    rec(figure="rq4_auxiliary_panels", panel="b", arm=a["short"], seed=s, condition=c,
                        series="rows stating no base pair at all", value=e[f"unparseable_{c}"] / N,
                        n=N, k=e[f"unparseable_{c}"], drawn=False, source=fn, key="unparseable")
            if a["pair"]:
                for name, key in (("stated ref>var equals the donor's pair", "donor_matching"),
                                  ("stated ref>var equals the recipient's pair under swap", "recipient_matching_swap"),
                                  ("stated ref base equals the variant's", "ref_base_wt")):
                    rec(figure="rq4_auxiliary_panels", panel="b", arm=a["short"], seed=s, condition="swap" if "swap" in name or "donor" in name else "wt",
                        series=name, value=round(e[key], 4), n=N, k=round(e[key] * N), drawn=False,
                        source=fn, key=key)
        data[a["key"]] = per_seed
    return data, ref


def agg(per_seed, key):
    """mean over seeds, standard error of that mean, and the individual seed values."""
    v = [per_seed[s][key] for s in sorted(per_seed)]
    sem = float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
    return float(np.mean(v)), sem, v


# ================================ figure
# The five panels are drawn by module-level functions (panel_title, draw_bars_panel,
# draw_matrix_panel, draw_fraction_colorbar, bar_legend_handles, place_bar_legend). draw() places
# all five on one canvas; auxiliary_supervision_main.py and auxiliary_supervision_matrices.py call
# the same functions and place subsets of them.
def dna_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("dna_seq", ["#FFFFFF", DNA.probe, DNA.model, DNA.model_dark])


# ---------------- geometric checks in inches. check_text_collisions compares text with text, and
# check_overflow compares text with the canvas edge. These also compare text with the plotting area
# of a neighbouring axes, and raise instead of printing.
def boxes_in(fig, r, item, with_labels=False):
    """[(name, box)] for one item, each box in INCHES from the figure's lower-left corner.

    An Axes contributes its plotting area (the patch) and, with `with_labels`, one box per visible
    tick label, per axis label and per text it carries.  Any other artist contributes one box.
    Element by element, not one union: panel a's x label and panel b's y label share a column of
    the canvas and are 0.4 in apart vertically, so a union of each panel's decorations would
    report an overlap that does not exist."""
    from matplotlib.transforms import Bbox

    def inch(b):
        return Bbox.from_extents(b.x0 / fig.dpi, b.y0 / fig.dpi, b.x1 / fig.dpi, b.y1 / fig.dpi)
    if not hasattr(item, "xaxis"):
        return [(getattr(item, "get_text", lambda: type(item).__name__)() or
                 type(item).__name__, inch(item.get_window_extent(r)))]
    out = [("plotting area", inch(item.get_window_extent(r)))]
    if with_labels:
        for t in item.get_xticklabels() + item.get_yticklabels():
            if t.get_text() and t.get_visible():
                out.append(("tick %r" % t.get_text().replace("\n", "|"),
                            inch(t.get_window_extent(r))))
        for nm, l in (("x label", item.xaxis.label), ("y label", item.yaxis.label)):
            if l.get_text():
                out.append((nm, inch(l.get_window_extent(r))))
        for t in item.texts:
            if t.get_text() and t.get_visible():
                out.append(("text %r" % t.get_text(), inch(t.get_window_extent(r))))
        if item.title.get_text():
            out.append(("title", inch(item.title.get_window_extent(r))))
    return out


def separation_in(b1, b2):
    """Signed clearance between two boxes, in inches.

    Positive = the boxes are apart, and the value is the gap along whichever axis separates them.
    Zero or negative = they intersect, and the value is minus the smaller overlap."""
    dx = max(b1.x0 - b2.x1, b2.x0 - b1.x1)
    dy = max(b1.y0 - b2.y1, b2.y0 - b1.y1)
    return max(dx, dy)


def assert_inside_canvas(fig, margin_in=0.02, name=""):
    """Every visible text must end at least `margin_in` inches inside the canvas.

    check_overflow only warns past 0.005 in and does not raise, so an axis label that ends on the
    edge passes it. Raises."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.get_size_inches()
    items = list(fig.texts)
    for ax in fig.axes:
        items += [t for t in list(ax.texts) + [ax.title, ax.xaxis.label, ax.yaxis.label]
                  if t.get_text() and t.get_visible()]
        items += [t for t in ax.get_xticklabels() + ax.get_yticklabels()
                  if t.get_text() and t.get_visible()]
        if ax.get_legend():
            items += list(ax.get_legend().get_texts())
    for leg in fig.legends:
        items += list(leg.get_texts())
    worst, who, bad = 9.9, None, []
    for t in items:
        if not t.get_text():
            continue
        b = t.get_window_extent(r)
        m = min(b.x0, b.y0, W * fig.dpi - b.x1, H * fig.dpi - b.y1) / fig.dpi
        if m < worst:
            worst, who = m, t.get_text().replace("\n", "|")
        if m < margin_in:
            bad.append(f"{t.get_text().replace(chr(10), '|')[:30]!r} {m:+.3f} in from the edge")
    print(f"  [canvas] {name}: tightest margin {worst:+.3f} in ({who!r}), want >= {margin_in:.3f}")
    assert not bad, f"{name}: text too near the canvas edge: " + "; ".join(bad)
    return worst


def assert_clear(fig, pairs, min_gap_in=0.0, name=""):
    """`pairs` is [(label_a, items_a, label_b, items_b, with_labels), ...].

    For each pair, every box of side A is tested against every box of side B; the printed number is
    the smallest clearance found and the two elements that produced it.  Raises AssertionError if
    any pair is closer than `min_gap_in`.  Returns [(label_a, label_b, gap_in, culprit)]."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()

    def expand(items, with_labels):
        items = items if isinstance(items, (list, tuple)) else [items]
        out = []
        for i in items:
            out += boxes_in(fig, r, i, with_labels)
        return out
    out, bad = [], []
    for la, ia, lb, ib, wl in pairs:
        A, B = expand(ia, wl), expand(ib, wl)
        g, culprit = min(((separation_in(x[1], y[1]), f"{x[0]} / {y[0]}") for x in A for y in B),
                         key=lambda t: t[0])
        out.append((la, lb, g, culprit))
        print(f"  [clearance] {name}: {la} <-> {lb}: {g:+.3f} in  (closest: {culprit})")
        if g < min_gap_in:
            bad.append(f"{la} <-> {lb} = {g:+.3f} in at [{culprit}] (want >= {min_gap_in:.3f})")
    assert not bad, f"{name}: overlapping or too-close elements: " + "; ".join(bad)
    return out


def panel_title(ax, text, pad=3.0):
    """Plain-weight panel title (the house style forbids bold ones).

    `pad` is the gap in POINTS between the axes top and the title's baseline anchor.  It is a
    parameter because a small panel wants a tighter gap, and because a one-line title needs a
    LARGER pad than a two-line one if the two are to share a top: matplotlib anchors a title by
    its last line, so titles of different line counts share a bottom, not a top."""
    ax.set_title(text, fontsize=TICK_FS, fontweight="normal", pad=pad, linespacing=1.15)


def draw_bars_panel(ax, data, keys, prefix, reference, title_text, panel, figure,
                    mask=True, tick_fs=6.0, ylim=(0, 1.34), title_pad=3.0):
    """Panels a / b: one group of two bars per arm (Evo2 intact, Evo2 shuffled).

    Bar height is the mean over training seeds, the black line the standard error of that mean, the
    open circles the individual seeds, and the dotted horizontal line the text-only reference.
    `panel` and `figure` are the values written to those two columns of the numbers CSV; every other
    recorded column is a property of the data, not of which figure the panel is placed in.

    `ylim` sets the headroom above the 1.0 tick that the rotated value labels are written into.  It
    has to grow when the axes gets shorter: a "0.984" at 6 pt is 0.24 in tall whatever the panel
    size, so at 0.76 in of axes height the default 1.34 puts it on top of the title.
    """
    xs = np.arange(len(keys))
    w = 0.38
    finish(ax, "Test Accuracy", ylim=ylim, yticks=np.arange(0, 1.01, 0.25), spine_top=1.0)
    center_ylabel(ax, 1.0)
    ax.axhline(reference, color=GREY, ls=(0, (1.2, 1.2)), lw=0.9, zorder=1)
    for i, k in enumerate(keys):
        per_seed = data[k]
        for j, c in enumerate(("wt", "shuffle")):
            mean, sem, vals = agg(per_seed, f"{prefix}_{c}")
            xb = xs[i] + (j - 0.5) * w
            ax.bar(xb, mean, w, color=DNA.model if c == "wt" else DNA.perturbed, edgecolor="black",
                   linewidth=EDGE_LW, zorder=3)
            ax.errorbar([xb], [mean], yerr=[[sem], [sem]], fmt="none", ecolor="black",
                        elinewidth=ERR_LW, capsize=0, zorder=5)
            # the individual seeds, jittered so equal values stay visible
            jit = np.linspace(-0.09, 0.09, len(vals))
            ax.scatter(xb + jit, vals, s=4.0, facecolors="white", edgecolors="black",
                       linewidths=0.4, zorder=6, clip_on=False)
            bar_label(ax, xb, max(mean + sem, max(vals)), f"{mean:.3f}", mask=mask)
            arm = next(a for a in ARMS if a["key"] == k)
            rec(figure=figure, panel=panel, arm=arm["short"],
                seed="mean of 3", condition=c, series=f"{prefix} accuracy, mean over seeds",
                value=round(mean, 4), n=N, drawn=True, source="seeds " + ",".join(map(str, sorted(data[k]))),
                key="bar height")
            rec(figure=figure, panel=panel, arm=arm["short"],
                seed="sem of 3", condition=c, series=f"{prefix} accuracy, standard error of the mean",
                value=round(sem, 4), n=len(vals), drawn=True, source="sd(ddof=1)/sqrt(n_seeds)",
                key="error bar half-length")
    ax.set_xticks(xs)
    ax.set_xticklabels([next(a for a in ARMS if a["key"] == k)["label"] for k in keys],
                       rotation=45, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", labelsize=tick_fs)
    ax.set_xlim(-0.62, len(keys) - 0.38)
    ax.set_xlabel("Training target, window", fontsize=TICK_FS)
    panel_title(ax, title_text, pad=title_pad)
    return ax


def bar_legend_handles():
    """The four keys shared by the two bar panels."""
    handles = [Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="black", lw=EDGE_LW, label=l)
               for l, c in (("Evo2 intact", DNA.model), ("Evo2 shuffled", DNA.perturbed))]
    handles += [Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
                       markeredgecolor="black", markeredgewidth=0.4, markersize=2.2,
                       label="training seed"),
                Line2D([0], [0], color=GREY, ls=(0, (1.2, 1.2)), lw=0.9, label="text-only reference")]
    return handles


def place_bar_legend(fig, handles, axes, center_x_in, ncol=4, pad_in=0.05):
    """Legend centred at `center_x_in` inches, on the level just under the deepest x label of
    `axes`.  Measured rather than fixed: the x labels are themselves placed by align_xlabels."""
    W, H = fig.get_size_inches()
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    y_top = (min(ax.xaxis.label.get_window_extent(r).y0 for ax in axes) / fig.dpi - pad_in) / H
    return fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(center_x_in / W, y_top),
                      ncol=ncol, frameon=False, fontsize=LEGEND_FS, handlelength=1.1,
                      handletextpad=0.4, columnspacing=1.0)


def matrix_counts(per_seed, cond):
    """12 x 12 written-pair counts for one condition, the training seeds pooled."""
    M = np.zeros((len(PAIRS), len(PAIRS)), dtype=int)
    for s in sorted(per_seed):
        for (t, e), cnt in per_seed[s]["conf"][cond].items():
            if t in PAIRS and e in PAIRS:
                M[PAIRS.index(t), PAIRS.index(e)] += cnt
    return M


def draw_matrix_panel(ax, per_seed, cond, head, ylab, ticks, panel, figure, cmap,
                      n_tot=None, tick_fs=5.0):
    """One confusion matrix (panels c-e of fig_rq4_auxiliary_panels).

    Rows are the pair defining the query (or the donor's pair, under swap), columns the pair the
    model wrote, and each row is divided by its own total, so a diagonal cell is that pair's
    accuracy.  The second title line is the diagonal total over all `n_tot` generations.  Returns
    the QuadMesh, which the colourbar is built from.
    """
    n_tot = n_tot or N * len(per_seed)
    M = matrix_counts(per_seed, cond)
    tot = M.sum(axis=1, keepdims=True)
    F = np.divide(M, tot, out=np.zeros(M.shape, dtype=float), where=tot > 0)
    n_ = len(PAIRS)
    edges = np.arange(-0.5, n_ + 0.5, 1.0)
    im = ax.pcolormesh(edges, edges, F, cmap=cmap, vmin=0, vmax=1, shading="flat",
                       edgecolors="white", linewidth=0.4, antialiased=False, rasterized=False)
    ax.set_xlim(-0.5, n_ - 0.5); ax.set_ylim(n_ - 0.5, -0.5)
    ax.set_aspect("equal")
    for i in range(n_):
        for j in range(n_):
            if M[i, j]:
                rec(figure=figure, panel=panel, arm="auxiliary, 257 bp", seed="3 seeds pooled",
                    condition=cond, series=f"drawn cell: {PAIRS[i]} row -> {PAIRS[j]} predicted, "
                                           f"fraction of that row's {int(tot[i, 0])} generations",
                    value=round(float(F[i, j]), 4), n=int(tot[i, 0]), k=int(M[i, j]), drawn=True,
                    source="generations/aux_257bp_seed*, 3 seeds", key="conf / row total")
    ax.set_xticks(range(n_)); ax.set_yticks(range(n_))
    ax.set_xticklabels(PAIRS, rotation=90, fontsize=tick_fs)
    if ticks:
        ax.set_yticklabels(PAIRS, fontsize=tick_fs)
        ax.set_ylabel(ylab, fontsize=TICK_FS)
    else:
        ax.set_yticklabels([])
    ax.tick_params(length=1.5, pad=1.5)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    diag = int(np.trace(M))
    panel_title(ax, f"{head}\nAccuracy {diag / n_tot:.3f}")
    rec(figure=figure, panel=panel, arm="auxiliary, 257 bp", seed="3 seeds pooled",
        condition=cond, series="title: diagonal total / generations", value=round(diag / n_tot, 4),
        n=n_tot, k=diag, drawn=True, source="generations/aux_257bp_seed*, 3 seeds", key="trace(conf)")
    ax.set_xlabel("Predicted pair", fontsize=TICK_FS)
    return im


def draw_fraction_colorbar(fig, im, x, top, w, h, fs=5.0):
    """Row-fraction scale beside the last matrix, with its header centred over the bar."""
    cax = ax_in(fig, x, top, w, h)
    cb = fig.colorbar(im, cax=cax, ticks=[0, 0.5, 1])
    cax.text(0.5, 1.12, "Fraction\nof row", transform=cax.transAxes, ha="center", va="bottom",
             fontsize=fs, linespacing=1.0)
    cb.ax.tick_params(labelsize=fs, length=1.5, pad=1)
    cb.outline.set_linewidth(0.5)
    return cax


# the three matrix panels, in order: condition, heading, y label, y ticks
MATRIX_PANELS = [("wt", "Evo2 intact", "True pair (ref>var)", True),
                 ("shuffle", "Evo2 shuffled", None, False),
                 ("swap", "Another query's windows", "Donor's pair (ref>var)", True)]
BAR_PANELS = [("disease", "disease", "Disease prediction task"),
              ("pair", "pair", "Base pair\nat the edit")]
ARM_KEYS = ["noaux2048c", "aux257", "aux2048c"]


def draw(data, ref, out_dir):
    W = 8.0
    TOP, TIT, AXH, BOT = 0.10, 0.30, 1.10, 0.88
    H = TOP + TIT + AXH + BOT
    fig = plt.figure(figsize=(W, H))
    row_top = TOP + TIT
    # panels a and b have the same width, so their bars print at the same width
    L = 0.42
    a_w, b_w, M_W = 1.20, 1.20, AXH
    gap_ab, gap_bc, gap_cd, gap_de = 0.48, 0.48, 0.10, 0.48
    a_x = L
    b_x = a_x + a_w + gap_ab
    c_x = b_x + b_w + gap_bc
    d_x = c_x + M_W + gap_cd
    e_x = d_x + M_W + gap_de
    CB_GAP, CB_W = 0.08, 0.06
    assert e_x + M_W + CB_GAP + CB_W + 0.20 <= W, e_x + M_W + CB_GAP + CB_W

    ax_a = ax_in(fig, a_x, row_top, a_w, AXH)
    draw_bars_panel(ax_a, data, ARM_KEYS, "disease", ref["disease"], BAR_PANELS[0][2],
                    panel="a", figure="rq4_auxiliary_panels")
    panel_letter(fig, 0.02, TOP, "a")

    ax_b = ax_in(fig, b_x, row_top, b_w, AXH)
    draw_bars_panel(ax_b, data, ARM_KEYS, "pair", ref["pair"], BAR_PANELS[1][2],
                    panel="b", figure="rq4_auxiliary_panels")
    panel_letter(fig, b_x - 0.44, TOP, "b")

    handles = bar_legend_handles()

    # ---------------- c, d, e: confusion matrices, 257 bp arm, the three seeds pooled
    per_seed = data["aux257"]
    n_tot = N * len(per_seed)
    cmap = dna_cmap()
    axes_m = []
    for (cond, head, ylab, ticks), x0, letter in zip(MATRIX_PANELS, (c_x, d_x, e_x), "cde"):
        ax = ax_in(fig, x0, row_top, M_W, M_W)
        im = draw_matrix_panel(ax, per_seed, cond, head, ylab, ticks, panel=letter,
                               figure="rq4_auxiliary_panels", cmap=cmap, n_tot=n_tot)
        panel_letter(fig, x0 - (0.46 if ticks else 0.10), row_top - TIT, letter)
        axes_m.append(ax)
    draw_fraction_colorbar(fig, im, e_x + M_W + CB_GAP, row_top + 0.25 * M_W, CB_W, 0.5 * M_W)
    align_xlabels(fig, [ax_a, ax_b], pad_in=-0.07)
    for ax in axes_m:
        ax.xaxis.labelpad = 1.5
    place_bar_legend(fig, handles, (ax_a, ax_b), (a_x + b_x + b_w) / 2)
    return emit(fig, STEM, out_dir)


def write_tex(data, ref, out_dir):
    f = lambda v: f"{v:.3f}"
    d = {k: {q: agg(data[k], q) for q in
             (["disease_wt", "disease_shuffle"] + (["pair_wt", "pair_shuffle", "donor_matching"]
                                                   if next(a for a in ARMS if a["key"] == k)["pair"] else []))}
         for k in data}
    n_seeds = len(data["aux257"])
    cap = rf"""% Built by auxiliary_supervision_panels.py. Include at \textwidth.
% Numbers: {STEM}_numbers.csv (every drawn value, every seed, every matrix cell).
% Generations: experiments/auxiliary_supervision/evaluate_generation.py with --truncate-per-side 0.
% The base pair is read with one format-agnostic parser for all three conditions, which returns the
% "Edit at" values on the two auxiliary conditions (asserted in the script). Denominator: all 145
% held-out queries (111 single-base substitutions, 5 multi-base substitutions, 9 insertions,
% 20 deletions). Values on the substitutions alone are in the CSV.
\begin{{figure}}[t]
\centering
\includegraphics[width=\textwidth]{{figures/{STEM}.pdf}}
\caption{{\textbf{{A sequence-determined auxiliary target makes BioReason use
$\mathbf{{Z}}_{{\mathrm{{Evo2}}}}$ for the supervised sequence prediction, with a small effect on disease
prediction accuracy.}} Three supervised fine-tuning runs from the same Qwen3-4B continued-pretraining base on the
public KEGG benchmark (network split, $1{{,}}159$ training queries), differing only in the training
target and the window length, each repeated with ${n_seeds}$ training seeds. The target is either the
original reasoning trace followed by the disease, or a single line, \texttt{{Edit at $i$: ref X -> var
Y (class)}}, followed by the disease, where $i$ is the offset of the edit in the window and X and Y
are the reference and variant bases there. Windows are cut around the edit before training, so
training and evaluation encode the same bases. Magenta is the model with its real genome,
grey-magenta the same model with both windows base-shuffled, composition preserved, before Evo2
encodes them. Everything is greedy free generation on the ${N}$ held-out pathway queries
(\texttt{{ood\_test}}, none of whose prompts occurs in training).
\textbf{{(a, b)}} Bars are the mean over the ${n_seeds}$ seeds, black lines the standard error of that
mean, and open circles the individual seeds. Dotted lines are text-only references: a majority-vote
lookup from the gene named in the prompt for the disease (${f(ref['disease'])}$), and always writing the
most frequent training pair for the base pair (${f(ref['pair'])}$).
\textbf{{(a)}} Disease accuracy is ${f(d['noaux2048c']['disease_wt'][0])}$ intact and
${f(d['noaux2048c']['disease_shuffle'][0])}$ shuffled without the auxiliary,
${f(d['aux257']['disease_wt'][0])}$ and ${f(d['aux257']['disease_shuffle'][0])}$ with it on $257$~bp
windows, and ${f(d['aux2048c']['disease_wt'][0])}$ and ${f(d['aux2048c']['disease_shuffle'][0])}$ with it
on $2{{,}}048$~bp windows.
\textbf{{(b)}} The base pair the model states equals the variant's on
${f(d['noaux2048c']['pair_wt'][0])}$ of queries intact and ${f(d['noaux2048c']['pair_shuffle'][0])}$
shuffled without the auxiliary, ${f(d['aux257']['pair_wt'][0])}$ and
${f(d['aux257']['pair_shuffle'][0])}$ with it on $257$~bp windows, and
${f(d['aux2048c']['pair_wt'][0])}$ and ${f(d['aux2048c']['pair_shuffle'][0])}$ with it on
$2{{,}}048$~bp windows. The condition without the auxiliary target writes no \texttt{{Edit at}} line, so all three arms are read
with one format-agnostic parser that takes the first reference-to-variant base pair stated anywhere
in the trace; on the two auxiliary arms it returns exactly the \texttt{{Edit at}} values. The pair
is the one at the first index where the two windows differ, which is defined for insertions and
deletions as well as substitutions, and is the rule the auxiliary target itself was built with, so
all ${N}$ queries are in the denominator.
\textbf{{(c--e)}} For the $257$~bp arm, the pair defining each row against the written pair, over the
${N}\times{n_seeds}={N * n_seeds}$ generations of the three seeds pooled. Each matrix row is normalised by
its own total, so a diagonal cell is the accuracy for that pair, and the second title line is the
diagonal total over all ${N * n_seeds}$ generations. (c) Evo2 intact, rows are the query's own pair.
(d) Both windows shuffled, rows are still the query's own pair. (e) Both windows taken from another
held-out query, with the prompt unchanged, rows are the donor's pair; the written pair matches the
donor's on ${f(d['aux257']['donor_matching'][0])}$ of queries.}}
\label{{fig:rq4_auxiliary}}
\end{{figure}}
"""
    open(os.path.join(out_dir, f"{STEM}.tex"), "w").write(cap)


CSV_COLS = ["figure", "panel", "arm", "seed", "condition", "series", "value", "n", "k", "drawn",
            "source", "key"]


def write_numbers(rows, out_dir, stem, figure=None, panel_map=None, panels=None):
    """The numbers CSV.  `panels` keeps only those panel values, `panel_map` renames them and
    `figure` overrides the figure column -- the only two columns a split may change."""
    n = 0
    with open(os.path.join(out_dir, f"{stem}_numbers.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            if panels is not None and r["panel"] not in panels:
                continue
            r = dict(r)
            if figure is not None:
                r["figure"] = figure
            if panel_map is not None:
                r["panel"] = panel_map.get(r["panel"], r["panel"])
            w.writerow(r)
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=FIG_DIR)
    a = ap.parse_args()
    init_print_style()
    data, ref = load()
    paths = draw(data, ref, a.out_dir)
    write_tex(data, ref, a.out_dir)
    write_numbers(ROWS, a.out_dir, STEM)
    print("wrote", paths, f"{STEM}_numbers.csv", f"{STEM}.tex")
    for k in data:
        for q in ("disease_wt", "disease_shuffle", "pair_wt", "pair_shuffle",
                  "pair_substitution_rows_wt", "pair_substitution_rows_shuffle",
                  "pair_auxformat_wt", "donor_matching"):
            if q in data[k][23]:
                m, sem, v = agg(data[k], q)
                print(f"  {k:12s} {q:28s} mean {m:.4f}  sem {sem:.4f}  seeds {[round(x, 4) for x in v]}")


if __name__ == "__main__":
    main()

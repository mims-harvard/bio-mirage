#!/usr/bin/env python
"""Data loaders and panels for the C2S-Scale rationale figures (RQ3): which genes the C2S-Scale 27B
rationales reference, and how that set changes when DEGs are removed from the cell sentence.

panel_a is the composition of referenced genes (DEG or not, ribosomal or not) across five atlases,
panel_b DEG precision among the first k referenced genes, panel_c the ribosomal share and DEG
precision with the cell sentence in expression order or random order, and panel_d the Jaccard
similarity to the unmodified set after removing the strongest DEGs or non-DEGs with similar
expression. handles_a to handles_d return each panel's legend. reasoning_traces.py draws panel_c as
Figure 6d, and rationale_genes.py draws panel_a, panel_b and panel_d as the RQ3 appendix figure.
Those callers set FIGURE and PANEL_ALIAS before drawing, which relabel the rows record() collects,
and write them with write_numbers().

Reads rationale_removal/rationale_removal.json, rationale_order/rationale_order.json and the stored
generations in rationales/<atlas> under single_cell/c2s_scale of INPUT_USE_RESULTS_DIR. Run on its
own, the module draws all four panels as fig_rq3_rationale_panels (not in the paper) and writes it
with its numbers CSV to outputs/figures, or to --out_dir; reasoning_traces.py reads that CSV.
"""
from __future__ import annotations

import faulthandler
import sys
faulthandler.dump_traceback_later(180, repeat=True, file=sys.stderr)  # where it is, if it stalls

import argparse
import csv
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figure_style import (init_print_style, center_ylabel, ax_in, finish, bar_label,  # noqa: E402
                          align_xlabels, panel_letter, emit, CELL, GREY, CTRL,
                          ANNOT_FS, LEGEND_FS, EDGE_LW, ERR_LW, rotate_ticks)
from input_use.core.analyze_rationale_removal import (load_dataset, cited_genes,  # noqa: E402
                                                       DATASETS)
from input_use.core import config as _cfg  # noqa: E402
from input_use.core import paths as RD  # noqa: E402

RATIONALES = RD.C2S_RATIONALES
SRC = f"{RD.C2S_RATIONALE_REMOVAL}/rationale_removal.json"
SRC_POS = f"{RD.C2S_RATIONALE_ORDER}/rationale_order.json"
RIBO = os.path.join(os.path.dirname(os.path.dirname(_cfg.__file__)), "data", "hgnc_ribosomal_proteins.json")
FIG_DIR = os.path.join(os.environ.get("INPUT_USE_OUTPUT_DIR", "outputs"), "figures")
OUT_DEFAULT = FIG_DIR

ATLAS = [("immune1", "o"), ("immune2", "v"), ("immune3", "s"), ("pancreas", "D"), ("lung", "^")]
MARK = dict(ATLAS)
PREFIX = [5, 10, 20, 40, None]
PREFIX_LAB = ["5", "10", "20", "40", "all"]
TOP100, MT100 = "top_deg_dropout_p100", "matched_nondeg_dropout_p100"

# colours: green = the annotated type's DEGs (as genes in a, as the removed arm in c); the drained
# green marks DEGs that are ribosomal-protein genes; greys = ribosomal non-DEGs (a) and the matched
# control arm (c); pale green-grey = other input genes
CAT = [("deg_nr", CELL.model, "DEG, not ribosomal"),
       ("deg_r", CELL.perturbed, "DEG, ribosomal"),
       ("ribo", CTRL[1], "not DEG, ribosomal"),
       ("other", CELL.perturbed2, "other, in sentence")]
C_ARM = {"deg": CELL.model, "matched": CTRL[2]}
ORDER = [("wt", CELL.model, "expression\norder"), ("scramble_rank", CELL.perturbed, "random\norder")]
DOT = "#404040"
HATCH = "////"
N_BOOT, SEED = 4000, 0
FILL_LEG = "#E6E6E6"   # neutral swatch for the fill-pattern legend entries

ROWS = []

# Stamped by record() on every row. A figure that reuses these panels sets both before drawing:
# FIGURE fills the `figure` column and PANEL_ALIAS renames panel letters. An empty alias keeps the
# letters of fig_rq3_rationale_panels.
FIGURE = "rq3_rationale_panels"
PANEL_ALIAS: dict = {}

CSV_KEYS = ["figure", "panel", "group", "series", "x", "value", "n", "ci_low", "ci_high",
            "source", "key"]


def record(panel, **kw):
    ROWS.append({"figure": FIGURE, "panel": PANEL_ALIAS.get(panel, panel), **kw})


def write_numbers(rows, path):
    """Writes rows to a numbers CSV with the CSV_KEYS columns. Used by fig_rq3_rationale_panels and
    by every figure that reuses its panels."""
    with open(path, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=CSV_KEYS)
        wtr.writeheader()
        for r in rows:
            wtr.writerow({k: r.get(k, "") for k in CSV_KEYS})
    print(f"wrote {path} ({len(rows)} rows)")
    return path


# ------------------------------------------------------------------------------------------------
# recompute per-cell quantities from the stored generations
# ------------------------------------------------------------------------------------------------
def compute(root):
    ribo = set(json.load(open(RIBO))["symbols"])
    per = {}     # dataset -> list of per-cell dicts (wt condition)
    for ds in DATASETS:
        meta, deg_by_type, vocab, sent, salient, control, gt, recs, cells = \
            load_dataset(root, ds, "27b")
        rows = []
        for c in cells:
            D = deg_by_type[gt[c]]
            S = set(sent[(c, "wt")])
            G = cited_genes(recs[(c, "wt")], vocab)
            rel = D & S
            cat = {"absent": 0, "absent_ribo": 0, "deg_nr": 0, "deg_r": 0, "ribo": 0, "other": 0}
            for g in G:
                if g not in S:
                    cat["absent"] += 1
                    cat["absent_ribo"] += g in ribo
                elif g in rel:
                    cat["deg_r" if g in ribo else "deg_nr"] += 1
                elif g in ribo:
                    cat["ribo"] += 1
                else:
                    cat["other"] += 1
            n = len(G)
            nS = len(S)
            s = {"deg_nr": len(rel - ribo) / nS, "deg_r": len(rel & ribo) / nS,
                 "ribo": len((S & ribo) - D) / nS}
            s["other"] = 1.0 - sum(s.values())
            row = {"n_ref": n, "cat": cat,
                   "ribo_any": sum(1 for g in G if g in ribo),
                   "in_sent": sum(1 for g in G if g in S),
                   # input-sentence composition, weighted later by n_ref (the chance convention)
                   "sent": {k: n * v for k, v in s.items()},
                   "prefix": {}}
            for k in PREFIX:
                gk = G[:k] if k else G
                row["prefix"][k] = {"n": len(gk), "hits": sum(1 for g in gk if g in rel),
                                    "hits_nr": sum(1 for g in gk if g in rel and g not in ribo),
                                    "chance": len(gk) * len(rel) / nS,
                                    "chance_nr": len(gk) * len(rel - ribo) / nS}
            rows.append(row)
        per[ds] = rows
        print(f"[v7] {ds}: {len(rows)} cells, mean referenced "
              f"{np.mean([r['n_ref'] for r in rows]):.2f}", flush=True)
    per["pooled"] = [r for ds in DATASETS for r in per[ds]]
    return per


def pooled(rows, num, den):
    return 100.0 * sum(num(r) for r in rows) / sum(den(r) for r in rows)


def load_artifact():
    d = json.load(open(SRC))
    R = d["results"]
    for ds in DATASETS:
        assert R[ds]["n_cells"] == 400, (ds, R[ds]["n_cells"])
    assert R["pooled"]["n_cells"] == 2000
    return R


def close(a, b, tol=0.06):
    assert abs(a - b) <= tol, (a, b)


# ------------------------------------------------------------------------------------------------
# panel a: composition
# ------------------------------------------------------------------------------------------------
def panel_a(ax, per, R):
    groups = DATASETS + ["pooled"]
    w, gap = 0.44, 0.02

    for i, g in enumerate(groups):
        rows = per[g]
        n_sym = sum(r["n_ref"] for r in rows)
        wt = R[g]["conditions"]["wt"]
        nref = lambda r: r["n_ref"]
        # rationale composition, pooled over referenced symbols
        rat = {k: pooled(rows, lambda r, k=k: r["cat"][k], nref)
               for k in ("deg_nr", "deg_r", "ribo", "other", "absent", "absent_ribo")}
        close(rat["deg_nr"] + rat["deg_r"], 100 * wt["remaining_deg_precision"]["value"])
        close(100 - rat["absent"], 100 * wt["in_sentence_share"]["value"])
        close(rat["deg_r"] + rat["ribo"] + rat["absent_ribo"], 100 * wt["ribosomal_share"]["value"])
        # input-sentence composition, weighted by each response's citation count
        inp = {k: pooled(rows, lambda r, k=k: r["sent"][k], nref)
               for k in ("deg_nr", "deg_r", "ribo", "other")}
        inp["absent"] = 0.0
        close(inp["deg_nr"] + inp["deg_r"], 100 * wt["remaining_deg_chance"]["value"])
        # precision@5 split the same way, for the CSV and the caption
        p5 = pooled(rows, lambda r: r["prefix"][5]["hits"], lambda r: r["prefix"][5]["n"])
        p5_nr = pooled(rows, lambda r: r["prefix"][5]["hits_nr"], lambda r: r["prefix"][5]["n"])
        record("a", group=g, series="precision@5 (%)", x=g, value=p5, n=sum(r["prefix"][5]["n"] for r in rows),
               source=SRC, key=f"results.{g}.conditions.wt.remaining_deg_precision5")
        record("a", group=g, series="precision@5, non-ribosomal DEGs only (%)", x=g, value=p5_nr,
               n=sum(r["prefix"][5]["n"] for r in rows), source="recomputed", key="")
        print(f"[v7] {g:8s} rationale: DEG non-ribo {rat['deg_nr']:5.2f}  DEG ribo {rat['deg_r']:5.2f}  "
              f"ribo non-DEG {rat['ribo']:5.2f}  other {rat['other']:5.2f}  absent {rat['absent']:4.2f} | "
              f"input: DEG non-ribo {inp['deg_nr']:5.2f}  DEG ribo {inp['deg_r']:4.2f}  "
              f"ribo {inp['ribo']:4.2f} | p@5 {p5:5.2f} of which non-ribo {p5_nr:5.2f}", flush=True)
        for j, (name, comp) in enumerate((("R", rat), ("I", inp))):
            x = i + (j - 0.5) * (w + gap)

            bottom = 0.0
            hatch = HATCH if name == "I" else None
            for key, col, _ in CAT:
                ax.bar(x, comp[key], w, bottom=bottom, color=col, edgecolor="black",
                       linewidth=EDGE_LW, hatch=hatch, zorder=3)
                if key == "ribo" and comp[key] >= 12:
                    ax.text(x, bottom + comp[key] / 2, f"{comp[key]:.0f}", ha="center",
                            va="center", fontsize=4.3, color="white", zorder=5)
                bottom += comp[key]
            if comp["absent"] > 0:
                ax.bar(x, comp["absent"], w, bottom=bottom, facecolor="white", edgecolor="black",
                       linewidth=EDGE_LW, zorder=3)
            for key in ("deg_nr", "deg_r", "ribo", "other", "absent"):
                src = ""
                if name == "R" and key in ("deg_nr", "deg_r"):
                    src = f"results.{g}.conditions.wt.remaining_deg_precision (deg_nr + deg_r)"
                if name == "I" and key in ("deg_nr", "deg_r"):
                    src = f"results.{g}.conditions.wt.remaining_deg_chance (deg_nr + deg_r)"
                record("a", group=g, series=f"{'rationale' if name == 'R' else 'input'}: {key}",
                       x=g, value=comp[key], n=n_sym, source=SRC if src else "recomputed", key=src)
        record("a", group=g, series="rationale: ribosomal-protein genes, any", x=g,
               value=rat["deg_r"] + rat["ribo"] + rat["absent_ribo"], n=n_sym, source=SRC,
               key=f"results.{g}.conditions.wt.ribosomal_share")
    finish(ax, "Proportion of genes (%)", (0, 100), np.arange(0, 101, 25), spine_top=100)
    center_ylabel(ax, 100)
    ax.set_xticks(range(len(groups)))
    rotate_ticks(ax, groups, rotation=45)
    ax.tick_params(axis="x", which="major", pad=2, length=0)
    ax.set_xlim(-0.6, len(groups) - 0.4)


# ------------------------------------------------------------------------------------------------
# panel b: DEG precision over the first k referenced genes, relative to chance
# ------------------------------------------------------------------------------------------------
def handles_a():
    """Legend of panel a: the four gene categories and genes not in the sentence, a blank row, then
    the two fill patterns (plain for rationale genes, hatched for cell-sentence genes)."""
    h = [Patch(facecolor=col, edgecolor="black", linewidth=EDGE_LW, label=lab)
         for _, col, lab in CAT]
    h.append(Patch(facecolor="white", edgecolor="black", linewidth=EDGE_LW,
                   label="not in sentence"))
    h.append(Line2D([], [], ls="none", label=""))          # one blank row as the separator
    h += [Patch(facecolor=FILL_LEG, edgecolor="black", linewidth=EDGE_LW,
                label="rationale genes"),
          Patch(facecolor=FILL_LEG, edgecolor="black", linewidth=EDGE_LW, hatch=HATCH,
                label="cell-sentence genes")]
    return h


def boot_weights(n, b=N_BOOT, seed=SEED):
    """Paired cell bootstrap weights, the scheme of analyze_rationale_removal.Boot: b multinomial draws
    of n cells with replacement, one weight row per draw.
    """
    rng = np.random.default_rng(seed)
    return rng.multinomial(n, np.full(n, 1.0 / n), size=b).astype(float)


def boot_ci(w, num, den):
    bs = (w @ np.asarray(num, float)) / np.maximum(w @ np.asarray(den, float), 1e-12)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def panel_b(ax, per, R):
    xs = np.arange(len(PREFIX))
    for g in DATASETS + ["pooled"]:
        rows = per[g]
        wt = R[g]["conditions"]["wt"]
        prec_l, ch_l, lo_b, hi_b = [], [], [], []
        w = boot_weights(len(rows)) if g == "pooled" else None
        for k, lab in zip(PREFIX, PREFIX_LAB):
            nk = lambda r, k=k: r["prefix"][k]["n"]
            prec = pooled(rows, lambda r, k=k: r["prefix"][k]["hits"], nk)
            chance = pooled(rows, lambda r, k=k: r["prefix"][k]["chance"], nk)
            prec_nr = pooled(rows, lambda r, k=k: r["prefix"][k]["hits_nr"], nk)
            chance_nr = pooled(rows, lambda r, k=k: r["prefix"][k]["chance_nr"], nk)
            n_sym = sum(r["prefix"][k]["n"] for r in rows)
            if k == 5:
                close(prec, 100 * wt["remaining_deg_precision5"]["value"])
                close(chance, 100 * wt["remaining_deg_chance5"]["value"])
            if k is None:
                close(prec, 100 * wt["remaining_deg_precision"]["value"])
                close(chance, 100 * wt["remaining_deg_chance"]["value"])
            prec_l.append(prec)
            ch_l.append(chance)
            ci_p, ci_r = {}, {}
            if w is not None:
                hits = [r["prefix"][k]["hits"] for r in rows]
                ch = [r["prefix"][k]["chance"] for r in rows]
                nn = [r["prefix"][k]["n"] for r in rows]
                plo, phi = boot_ci(w, hits, nn)          # precision, in the same cell bootstrap
                lo_b.append(100 * plo)
                hi_b.append(100 * phi)
                ci_p = {"ci_low": 100 * plo, "ci_high": 100 * phi}
                # the ratio of two item-pooled shares over the same denominator is the ratio of the
                # summed numerators, so one bootstrap of sum(hits)/sum(chance) is its CI
                rlo, rhi = boot_ci(w, hits, ch)
                ci_r = {"ci_low": rlo, "ci_high": rhi}
                # the same weights reproduce the stored precision interval to within Monte Carlo
                # error (independent draws, 4,000 each)
                stored = (wt["remaining_deg_precision5"] if k == 5 else
                          wt["remaining_deg_precision"] if k is None else None)
                if stored is not None:
                    close(100 * plo, 100 * stored["ci95"][0], tol=0.6)
                    close(100 * phi, 100 * stored["ci95"][1], tol=0.6)
                print(f"[v7] pooled k={lab}: precision {prec:.2f} [{100 * plo:.2f}, {100 * phi:.2f}]"
                      f"{'' if stored is None else ' stored [%.2f, %.2f]' % tuple(100 * c for c in stored['ci95'])}"
                      f"  chance {chance:.2f}  ratio {prec / chance:.3f} [{rlo:.3f}, {rhi:.3f}]",
                      flush=True)
            src_key = (f"results.{g}.conditions.wt.remaining_deg_precision5" if k == 5 else
                       f"results.{g}.conditions.wt.remaining_deg_precision" if k is None else "")
            record("b", group=g, series="DEG precision (%)", x=f"k={lab}", value=prec, n=n_sym,
                   source=SRC if src_key else "recomputed", key=src_key, **ci_p)
            record("b", group=g, series="chance precision (%)", x=f"k={lab}", value=chance,
                   n=n_sym, source="recomputed", key=src_key.replace("precision", "chance"))
            record("b", group=g, series="precision / chance", x=f"k={lab}", value=prec / chance,
                   n=n_sym, source="recomputed", key="", **ci_r)
            record("b", group=g, series="non-ribosomal DEG precision / chance", x=f"k={lab}",
                   value=prec_nr / chance_nr if chance_nr else "", n=n_sym, source="recomputed",
                   key="")
        if g == "pooled":
            ax.plot(xs, ch_l, color=GREY, lw=0.8, ls="--", zorder=1)
            ax.fill_between(xs, lo_b, hi_b, color=CELL.model, alpha=0.3, lw=0, zorder=2)
            ax.plot(xs, prec_l, color=CELL.model, lw=1.2, marker="o", ms=3.4, mec="black",
                    mew=EDGE_LW, zorder=4)
        else:
            ax.plot(xs, prec_l, color=CTRL[1], lw=0.7, marker=MARK[g], ms=2.8, mfc=DOT,
                    mec="white", mew=0.3, zorder=3)
    finish(ax, "DEG Precision@$k$ (%)", (0, 53), [0, 10, 20, 30, 40, 50], spine_top=50)
    center_ylabel(ax, 50)
    ax.set_xticks(xs)
    ax.set_xticklabels(PREFIX_LAB)
    ax.set_xlim(-0.4, len(PREFIX) - 0.6)
    ax.set_xlabel("First $k$ genes\nin rationale")


# ------------------------------------------------------------------------------------------------
# panel c: the positional control.
def handles_b():
    """Legend of panel b: the pooled mean, one line per atlas, and the random reference."""
    h = [Line2D([], [], color=CELL.model, lw=1.2, marker="o", ms=3.4, mec="black", mew=EDGE_LW,
                label="mean")]
    h += [Line2D([], [], color=CTRL[1], lw=0.7, marker=mk, ms=2.8, mfc=DOT, mec="white",
                 mew=0.3, label=ds) for ds, mk in ATLAS]
    h.append(Line2D([], [], color=GREY, lw=0.8, ls="--", label="Random\nreference"))
    return h


def load_position():
    d = json.load(open(SRC_POS))
    P = d["results"]
    assert P["pooled"]["n_cells"] == 2000 and d["conditions"] == ["wt", "scramble_rank"]
    for ds in DATASETS:
        assert P[ds]["n_cells"] == 400
    return P


def panel_c(ax, P, rot=0):
    w, gap = 0.26, 0.03
    groups = [("Ribosomal", "ribosomal_share", "ribosomal_chance"),
              ("DEG", "deg_precision", "deg_chance")]
    for gi, (glab, key, ckey) in enumerate(groups):
        bars = [("wt", key, CELL.model, None), ("scramble_rank", key, CELL.perturbed, None),
                ("wt", ckey, "white", HATCH)]
        for bi, (k, kk, col, hatch) in enumerate(bars):
            x = gi + (bi - 1) * (w + gap)
            e = P["pooled"]["conditions"][k][kk]
            v, lo, hi = 100 * e["value"], 100 * e["ci95"][0], 100 * e["ci95"][1]
            ax.bar(x, v, w, color=col, edgecolor="black", linewidth=EDGE_LW, hatch=hatch, zorder=3)
            ax.errorbar([x], [v], yerr=[[v - lo], [hi - v]], fmt="none", ecolor="black",
                        elinewidth=ERR_LW, capsize=0, zorder=4)
            series = f"{kk} ({'input share' if kk == ckey else k}) (%)"
            record("c", group="pooled", series=series, x=glab, value=v, n=2000, ci_low=lo,
                   ci_high=hi, source=SRC_POS, key=f"results.pooled.conditions.{k}.{kk}")
            per = []
            for ds, mk in ATLAS:
                pv = 100 * P[ds]["conditions"][k][kk]["value"]
                per.append(pv)
                record("c", group=ds, series=series + " (per-atlas dot)", x=glab, value=pv, n=400,
                       source=SRC_POS, key=f"results.{ds}.conditions.{k}.{kk}")
            off = np.linspace(-0.3 * w, 0.3 * w, len(per))
            for (ds, mk), o, pv in zip(ATLAS, off, per):
                ax.plot([x + o], [pv], marker=mk, ms=2.2, mfc=DOT, mec="white", mew=0.3,
                        ls="none", zorder=5)
            bar_label(ax, x, max(v, hi, max(per)) + 0.5, f"{v:.0f}", rot=rot, pad=0.012)
            print(f"[v7] c {glab:10s} {k:14s} {kk:18s} {v:5.1f} [{lo:.1f}, {hi:.1f}]  "
                  f"atlases {' '.join(f'{p:.1f}' for p in per)}", flush=True)
        e2 = P["pooled"]["conditions"]["scramble_rank"][ckey]
        record("c", group="pooled", series=f"{ckey} (input share, weighted by random-order citations) (%)",
               x=glab, value=100 * e2["value"], n=2000, ci_low=100 * e2["ci95"][0],
               ci_high=100 * e2["ci95"][1], source=SRC_POS,
               key=f"results.pooled.conditions.scramble_rank.{ckey}")
    for name, d in P["pooled"]["paired_wt_minus_scramble"].items():
        record("c", group="pooled", series=f"paired difference expression - random order: {name} (pp)",
               x="", value=100 * d["diff"], n=2000, ci_low=100 * d["diff_ci95"][0],
               ci_high=100 * d["diff_ci95"][1], source=SRC_POS,
               key=f"results.pooled.paired_wt_minus_scramble.{name}.diff")
    for k, _, _ in ORDER:
        for name in ("deg_vs_nondeg_within_nonribo", "ribo_vs_nonribo"):
            q = P["pooled"]["conditions"][k]["rate_ratio"][name]
            record("c", group="pooled", series=f"citation rate ratio {name}, {k}", x="",
                   value=q["value"], n=2000, ci_low=q["ci95"][0], ci_high=q["ci95"][1],
                   source=SRC_POS, key=f"results.pooled.conditions.{k}.rate_ratio.{name}")
        pr = P["pooled"]["conditions"][k]["position_ratio_dec1_over_dec10"]["all"]
        record("c", group="pooled", series=f"P(cited) decile 1 / decile 10, all genes, {k}", x="",
               value=pr["value"], n=2000, ci_low=pr["ci95"][0], ci_high=pr["ci95"][1],
               source=SRC_POS, key=f"results.pooled.conditions.{k}.position_ratio_dec1_over_dec10.all")
    finish(ax, "Share of referenced\ngenes (%)", (0, 100 if rot == 0 else 108), [0, 25, 50, 75, 100],
           spine_top=100)
    center_ylabel(ax, 100)
    ax.set_xticks(range(len(groups)))
    rotate_ticks(ax, [g for g, _, _ in groups], rotation=45)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.tick_params(axis="x", which="major", pad=2, length=0)


def handles_c():
    """Legend of panel c: the two gene orders, then the hatched input share."""
    h = [Patch(facecolor=col, edgecolor="black", linewidth=EDGE_LW, label=lab)
         for _, col, lab in ORDER]
    h.append(Patch(facecolor="white", edgecolor="black", linewidth=EDGE_LW, hatch=HATCH,
                   label="input\nshare"))
    return h


# ------------------------------------------------------------------------------------------------
# panel d: Jaccard similarity of the named-gene set to the unmodified rationale's set, as DEGs are
# removed, against matched removal.
DOSES = [("50", "top_deg_dropout_p50", "matched_nondeg_dropout_p50"),
         ("100", "top_deg_dropout_p100", "matched_nondeg_dropout_p100")]


def panel_d(ax, R, rot=0):
    name = "set_change_jd"
    w, gap = 0.32, 0.05
    tops = []
    for di, (dose, top, mt) in enumerate(DOSES):
        for ai, (arm, k) in enumerate((("deg", top), ("matched", mt))):
            x = di + (ai - 0.5) * (w + gap)
            e = R["pooled"]["conditions"][k][name]
            v = 1.0 - e["value"]
            lo, hi = 1.0 - e["ci95"][1], 1.0 - e["ci95"][0]
            ax.bar(x, v, w, color=C_ARM[arm], edgecolor="black", linewidth=EDGE_LW, zorder=3)
            ax.errorbar([x], [v], yerr=[[v - lo], [hi - v]], fmt="none", ecolor="black",
                        elinewidth=ERR_LW, capsize=0, zorder=4)
            record("d", group="pooled", series=f"jaccard similarity to unmodified: {arm}",
                   x=f"{dose}%", value=v, n=2000, ci_low=lo, ci_high=hi, source=SRC,
                   key=f"1 - results.pooled.conditions.{k}.{name}")
            per_atlas = []
            for ds, mk in ATLAS:
                pv = 1.0 - R[ds]["conditions"][k][name]["value"]
                per_atlas.append(pv)
                record("d", group=ds, series=f"jaccard similarity to unmodified: {arm} (per-atlas dot)",
                       x=f"{dose}%", value=pv, n=400, source=SRC,
                       key=f"1 - results.{ds}.conditions.{k}.{name}")
            off = np.linspace(-0.3 * w, 0.3 * w, len(per_atlas))
            for (ds, mk), o, pv in zip(ATLAS, off, per_atlas):
                ax.plot([x + o], [pv], marker=mk, ms=2.4, mfc=DOT, mec="white", mew=0.3,
                        ls="none", zorder=5)
            top_v = max(v, hi, max(per_atlas))
            tops.append(top_v)
            bar_label(ax, x, top_v + 0.005, f"{v:.2f}", rot=rot, pad=0.02)
        p = R["pooled"]["paired"][dose][name]
        d, (dlo, dhi) = -p["diff"], (-p["diff_ci95"][1], -p["diff_ci95"][0])
        record("d", group="pooled", series="jaccard similarity: paired difference deg - matched",
               x=f"{dose}%", value=d, n=2000, ci_low=dlo, ci_high=dhi, source=SRC,
               key=f"-results.pooled.paired.{dose}.{name}.diff")
    assert max(tops) <= 1.0
    finish(ax, "Jaccard similarity to rationale\nof unmodified cell sentence",
           (0, 1.12 if rot == 0 else 1.18), [0, 0.25, 0.5, 0.75, 1.0], spine_top=1.0)
    center_ylabel(ax, 1.0)
    ax.set_xticks(range(len(DOSES)))
    ax.set_xticklabels([f"{d}%" for d, _, _ in DOSES])
    ax.set_xlim(-0.6, len(DOSES) - 0.4)
    ax.set_xlabel("Strongest\nDEGs removed")


# ------------------------------------------------------------------------------------------------
def handles_d():
    """Legend of panel d: which set of genes was removed from the cell sentence."""
    return [Patch(facecolor=C_ARM["deg"], edgecolor="black", linewidth=EDGE_LW, label="DEGs"),
            Patch(facecolor=C_ARM["matched"], edgecolor="black", linewidth=EDGE_LW,
                  label="non-DEGs")]


# Legend style shared by every figure that draws these panels, and the handle length per panel.
LEGEND_KW = dict(loc="center left", frameon=False, fontsize=LEGEND_FS, handletextpad=0.5,
                 borderaxespad=0.0, borderpad=0.3, labelspacing=0.35)
LEGEND_EXTRA = {"a": dict(ncol=1, handlelength=1.2), "b": dict(ncol=1, handlelength=1.6),
                "c": dict(ncol=1, handlelength=1.2), "d": dict(ncol=1, handlelength=1.4)}


def layout(per, R, P, W):
    """One pass at width W."""
    AXH, TOP, BOT = 1.35, 0.18, 0.48
    A_W, B_W, C_W, D_W = 1.00, 0.66, 0.62, 0.60   # axis widths, in inches
    B_LABEL, C_LABEL, D_LABEL = 0.36, 0.42, 0.47  # room for the y label and tick labels
    LEG_GAP, PANEL_GAP, RIGHT = 0.06, 0.10, 0.05
    GAP_BC = 0.16                             # gutter between b's legend and c
    H = TOP + AXH + BOT
    y_c = 1 - (TOP + AXH / 2) / H
    fig = plt.figure(figsize=(W, H))
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    kw = dict(LEGEND_KW)

    def legend(handles, x_in, **extra):
        leg = fig.legend(handles=handles, bbox_to_anchor=(x_in / W, y_c), **kw, **extra)
        fig.canvas.draw()
        return leg.get_window_extent(r).transformed(inv)

    # a
    x = 0.50
    ax_a = ax_in(fig, x, TOP, A_W, AXH)
    panel_a(ax_a, per, R)
    panel_letter(fig, 0.04, 0.02, "a")
    bb = legend(handles_a(), x + A_W + LEG_GAP, **LEGEND_EXTRA["a"])
    # b
    x_b0 = bb.x1 * W + PANEL_GAP
    ax_b = ax_in(fig, x_b0 + B_LABEL, TOP, B_W, AXH)
    panel_b(ax_b, per, R)
    panel_letter(fig, x_b0, 0.02, "b")
    bb = legend(handles_b(), x_b0 + B_LABEL + B_W + LEG_GAP, **LEGEND_EXTRA["b"])
    # c: positional control (atlas markers are those of b's legend)
    x_c0 = bb.x1 * W + GAP_BC
    ax_c = ax_in(fig, x_c0 + C_LABEL, TOP, C_W, AXH)
    panel_c(ax_c, P, rot=90)
    panel_letter(fig, x_c0, 0.02, "c")
    bb = legend(handles_c(), x_c0 + C_LABEL + C_W + LEG_GAP, **LEGEND_EXTRA["c"])
    # d
    x_d0 = bb.x1 * W + PANEL_GAP
    ax_d = ax_in(fig, x_d0 + D_LABEL, TOP, D_W, AXH)
    panel_d(ax_d, R, rot=0 if D_W >= 0.95 else 90)
    panel_letter(fig, x_d0, 0.02, "d")
    bb = legend(handles_d(), x_d0 + D_LABEL + D_W + LEG_GAP, **LEGEND_EXTRA["d"])
    x_end = bb.x1 * W + RIGHT
    align_xlabels(fig, [ax_b, ax_d])   # c has rotated ticks and no x label, like a

    # nothing may leave the canvas: axis labels are not covered by check_text_collisions
    fig.canvas.draw()
    texts = [t for ax in (ax_a, ax_b, ax_c, ax_d) for t in ax.get_xticklabels() if t.get_text()]
    texts += [lab for ax in (ax_a, ax_b, ax_c, ax_d) for lab in (ax.xaxis.label, ax.yaxis.label)
              if lab.get_text()]
    low = min(t.get_window_extent(r).transformed(inv).y0 for t in texts) * H
    print(f"[layout] W {W:.2f} -> content needs {x_end:.2f} in; b axis at {x_b0 + B_LABEL:.2f}, "
          f"c axis at {x_c0 + C_LABEL:.2f}, d axis at {x_d0 + D_LABEL:.2f}; lowest text {low:.2f} in above the bottom", flush=True)
    assert low >= 0.02, "x labels run off the bottom; raise BOT"
    return fig, x_end


def draw(per, R, P, out_dir):
    matplotlib.rcParams["hatch.linewidth"] = 0.45
    W = 5.5
    for _ in range(3):
        ROWS.clear()
        fig, x_end = layout(per, R, P, W)
        if abs(x_end - W) <= 0.01:
            break
        plt.close(fig)
        W = x_end
    else:
        raise RuntimeError(f"layout did not converge: W {W:.2f}, needs {x_end:.2f}")
    if W > 5.51:
        print(f"[layout] NOTE: the content needs {W:.2f} in, wider than the 5.5 in text width",
              flush=True)
    return emit(fig, "fig_rq3_rationale_panels", out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=OUT_DEFAULT)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    R = load_artifact()
    P = load_position()
    per = compute(RATIONALES)
    init_print_style()
    draw(per, R, P, a.out_dir)
    write_numbers(ROWS, os.path.join(a.out_dir, "fig_rq3_rationale_panels_numbers.csv"))


if __name__ == "__main__":
    main()

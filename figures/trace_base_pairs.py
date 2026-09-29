#!/usr/bin/env python
"""Draws fig_rq3_trace_base_pair (not in the paper): the reference-to-variant base pair BioReason
states in its reasoning trace against the true pair, and provides the data and matrix drawing for
panels a and b of Figure 6 (figures/reasoning_traces.py).

    python figures/trace_base_pairs.py [--ckpt rl|sft|both] [--out_dir <dir>]

Queries are the 145 ood_test queries of the pathway network split, restricted to the 111 whose
reference and variant windows (dna/bioreason/perturbations/examples.jsonl) differ at one position.
Rows of each matrix are the true pair, columns the first pair the trace states (rule in subs()),
plus a "none" column. Rows are normalised by their totals. Panel a uses Z_Evo2 from the intact DNA,
panel b from both windows shuffled with composition kept, with rows still the query's own pair.
Reads the records of the released RL and SFT checkpoints in dna/bioreason/perturbations and the
split in dna/bioreason/network_split.json under INPUT_USE_RESULTS_DIR, and writes
fig_rq3_trace_base_pair (RL) and fig_rq3_trace_base_pair_sft (SFT) with numbers CSVs and LaTeX
captions to outputs/figures, or to --out_dir.
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
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figure_style import (init_print_style, ax_in, panel_letter, emit,  # noqa: E402
                          DNA, GREY, TICK_FS, LEGEND_FS)
from input_use.core import paths as RD  # noqa: E402

FIG_DIR = os.path.join(os.environ.get("INPUT_USE_OUTPUT_DIR", "outputs"), "figures")
KEGG = RD.DNA_PERTURBATIONS
SPLIT_JSON = RD.DNA_NETWORK_SPLIT
EXAMPLES = os.path.join(KEGG, "examples.jsonl")
CKPTS = {
    "rl":  dict(file="records_rl.jsonl", stem="fig_rq3_trace_base_pair",
                name="BioReason RL (released)"),
    "sft": dict(file="records_sft.jsonl", stem="fig_rq3_trace_base_pair_sft",
                name="BioReason SFT (released)"),
}
CONDS = [("wt", "a", "Evo2 intact"), ("scramble", "b", "Evo2 shuffled")]
PAIRS = [f"{x}>{y}" for x in "ACGT" for y in "ACGT" if x != y]      # 12, grouped by reference base
NONE = "none"                                                      # 13th column: no pair stated
VARIANT_IDX = 1024

# Counts the parse must reproduce before anything is drawn: queries whose trace states a pair, of
# those the single-substitution queries, and the queries where the stated pair is the true pair.
EXPECT = {
    ("rl", "wt"):        dict(states=145, n_sub=111, correct=8),
    ("rl", "scramble"):  dict(states=145, n_sub=111, correct=8),
    ("sft", "wt"):       dict(states=66,  n_sub=54,  correct=4),
    ("sft", "scramble"): dict(states=52,  n_sub=42,  correct=4),
}

ROWS: list[dict] = []


def rec(**kw):
    ROWS.append(kw)


# ============================ extraction rule for a stated base pair
RE_SUB_SYM = re.compile(r'(?<![A-Za-z0-9])([ACGT])\s*(?:>|-+>|→)\s*([ACGT])(?![A-Za-z0-9])')
RE_SUB_TO = re.compile(r'(?<![A-Za-z0-9])([ACGT])\s+(?:to|into)\s+([ACGT])(?![A-Za-z0-9])')
BASENAME = {"adenine": "A", "thymine": "T", "cytosine": "C", "guanine": "G"}
RE_SUB_WORD = re.compile(r'\b(adenine|thymine|cytosine|guanine)\s*(?:>|-+>|→|\s+to\s+|\s+into\s+)\s*'
                         r'(adenine|thymine|cytosine|guanine)\b', re.I)


def subs(t):
    """Every ref>var pair stated in t: symbol form (A>G, A->G), "A to G", then base names.
    The first element is the one scored."""
    out = [(a, b) for a, b in RE_SUB_SYM.findall(t)]
    out += [(a, b) for a, b in RE_SUB_TO.findall(t)]
    out += [(BASENAME[a.lower()], BASENAME[b.lower()]) for a, b in RE_SUB_WORD.findall(t)]
    return out


# ================================================================ data
def load():
    assign = json.load(open(SPLIT_JSON))
    ood = {e for e, s in assign.items() if s == "ood_test"}
    n_split = collections.Counter(assign.values())
    assert len(ood) == 145, (len(ood), n_split)

    truth, shape = {}, collections.Counter()
    for line in open(EXAMPLES):
        e = json.loads(line)
        if e.get("condition") != "wt" or e["example_id"] not in ood:
            continue
        ref, var = e["payload"]["dna_sequences"]
        if len(ref) != len(var):
            shape["length_mismatch"] += 1
            continue
        d = [i for i, (a, b) in enumerate(zip(ref, var)) if a != b]
        shape[{0: "identical", 1: "single_substitution"}.get(len(d), "multi_base")] += 1
        if len(d) == 1:
            truth[e["example_id"]] = (ref[d[0]], var[d[0]], d[0])
    assert len(truth) == 111, (len(truth), dict(shape))
    assert all(i == VARIANT_IDX for _, _, i in truth.values())
    print(f"  ood_test {len(ood)} queries; window-difference classes {dict(shape)}; "
          f"single-substitution {len(truth)}")

    data, states_all = {}, {}
    for ck, cfg in CKPTS.items():
        states_all[ck] = {}
        by = collections.defaultdict(dict)
        for line in open(os.path.join(KEGG, cfg["file"])):
            r = json.loads(line)
            if r["example_id"] in ood and r["condition"] in ("wt", "scramble"):
                by[r["condition"]][r["example_id"]] = r["output"].get("raw", "")
        per_cond = {}
        for cond, _, _ in CONDS:
            assert len(by[cond]) == 145, (ck, cond, len(by[cond]))
            stated = {e: (subs(by[cond][e]) or [None])[0] for e in sorted(ood)}
            stated = {e: (f"{p[0]}>{p[1]}" if p else None) for e, p in stated.items()}
            n_states = sum(v is not None for v in stated.values())
            sub_ids = sorted(truth)
            n_sub_stated = sum(stated[e] is not None for e in sub_ids)
            correct = sum(stated[e] == f"{truth[e][0]}>{truth[e][1]}" for e in sub_ids)
            exp = EXPECT[(ck, cond)]
            assert n_states == exp["states"], (ck, cond, "states", n_states, exp)
            assert n_sub_stated == exp["n_sub"], (ck, cond, "n_sub", n_sub_stated, exp)
            assert correct == exp["correct"], (ck, cond, "correct", correct, exp)
            print(f"  ASSERT ok  {ck:3s} {cond:8s}  states {n_states}/145  "
                  f"states a pair on a single-substitution query {n_sub_stated}/111  "
                  f"correct {correct}/111 = {correct / 111:.4f}")
            per_cond[cond] = stated
            states_all[ck][cond] = n_states
        data[ck] = per_cond
    return truth, data, states_all


def matrix(truth, stated):
    """(12 x 13) counts: rows = true pair, columns = stated pair then 'none'."""
    cols = PAIRS + [NONE]
    M = np.zeros((len(PAIRS), len(cols)), dtype=int)
    for e, (a, b, _) in truth.items():
        i = PAIRS.index(f"{a}>{b}")
        s = stated[e]
        M[i, cols.index(s if s in PAIRS else NONE)] += 1
    return M, cols


def chance_diagonal(M):
    """Expected diagonal count if the stated pair were independent of the row's true pair, with the
    same denominator: sum_q (#rows with true pair q) x (#queries stating q) / N.  A query that
    states no pair is never on the diagonal and is counted in N."""
    n = int(M.sum())
    rows = M.sum(axis=1)
    cols = M.sum(axis=0)
    return float(sum(rows[i] * cols[i] for i in range(len(PAIRS))) / max(1, n))


# ================================================================ figure
def dna_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("dna_seq", ["#FFFFFF", DNA.probe, DNA.model, DNA.model_dark])


GAPCOL = 0.35                       # blank column-units between the 12 pairs and 'none'
NCOL_U = len(PAIRS) + 1 + GAPCOL    # column-units spanned by the matrix (width / cell-height)


def draw_matrix_panel(fig, x0, y0, M_W, M_H, M, *, cmap, show_yticklabels=True,
                      title=None, tick_fs=5.0, title_fs=TICK_FS, empty_fs=4.6,
                      xlabel="Pair stated in the trace", ylabel="True pair (ref>var), n queries",
                      label_fs=TICK_FS, title_pad=3.0,
                      empty_text="no held-out query with this pair"):
    """ONE row-normalised confusion matrix of stated-against-true base pair, placed by inches.

    `M` is the (12 x 13) count matrix from matrix(); everything drawn here is derived from it, so a
    caller cannot draw a panel from one matrix and label it with another's totals.  Returns
    (ax, mesh, empty_row_indices). Used by the two-panel figure below and by panels a and b of
    fig_rq3_traces; the defaults reproduce the two-panel figure.
    """
    # 14 edges = the 12 pair columns, a blank spacer column, then the 'none' column
    xedges = np.concatenate([np.arange(-0.5, len(PAIRS) + 0.5, 1.0),
                             [len(PAIRS) - 0.5 + GAPCOL, len(PAIRS) + 0.5 + GAPCOL]])
    yedges = np.arange(-0.5, len(PAIRS) + 0.5, 1.0)
    xticks = list(range(len(PAIRS))) + [len(PAIRS) + GAPCOL]
    tot = M.sum(axis=1, keepdims=True)
    F = np.divide(M, tot, out=np.zeros(M.shape, dtype=float), where=tot > 0)

    Fdraw = np.ma.masked_invalid(np.insert(F, len(PAIRS), np.nan, axis=1))   # spacer column
    ax = ax_in(fig, x0, y0, M_W, M_H)
    im = ax.pcolormesh(xedges, yedges, Fdraw, cmap=cmap, vmin=0, vmax=1, shading="flat",
                       edgecolors="white", linewidth=0.4, antialiased=False, rasterized=False)
    ax.set_xlim(xedges[0], xedges[-1])
    ax.set_ylim(yedges[-1], yedges[0])
    # a row with no held-out query is marked, never left as an ambiguous blank
    empty = [i for i in range(len(PAIRS)) if tot[i, 0] == 0]
    for i in empty:
        ax.add_patch(Rectangle((xedges[0], i - 0.5), xedges[-1] - xedges[0], 1.0,
                               facecolor="#F2F2F2", edgecolor="white", linewidth=0.4,
                               hatch="////", zorder=4))
        ax.text(np.mean([xedges[0], xedges[-1]]), i, empty_text,
                ha="center", va="center", fontsize=empty_fs, color=GREY, zorder=5)
    # The diagonal is outlined because these matrices are sparse: without it a reader cannot see
    # that the one dark column is not the diagonal.
    for i in range(len(PAIRS)):
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1.0, 1.0, facecolor="none", edgecolor=GREY,
                               linewidth=0.45, linestyle=(0, (1.2, 1.2)), zorder=6))
    ax.set_xticks(xticks)
    ax.set_yticks(range(len(PAIRS)))
    ax.set_xticklabels(PAIRS + [NONE], rotation=90, fontsize=tick_fs)
    if show_yticklabels:
        ax.set_yticklabels([f"{p} ({int(tot[i, 0])})" for i, p in enumerate(PAIRS)],
                           fontsize=tick_fs)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=label_fs)
    else:
        ax.set_yticklabels([])
    ax.tick_params(length=1.5, pad=1.5)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=label_fs)
        ax.xaxis.labelpad = 1.5
    if title:
        ax.set_title(title, fontsize=title_fs, fontweight="normal", pad=title_pad,
                     linespacing=1.15)
    return ax, im, empty


def draw(ck, truth, stated_by_cond, out_dir):
    cfg = CKPTS[ck]
    W = 5.5
    TOP, TIT, M_H, BOT = 0.10, 0.36, 1.85, 0.58
    H = TOP + TIT + M_H + BOT
    cell = M_H / len(PAIRS)
    M_W = NCOL_U * cell
    L, GAP_AB = 0.52, 0.34
    CB_GAP, CB_W = 0.10, 0.07
    a_x = L
    b_x = a_x + M_W + GAP_AB
    assert b_x + M_W + CB_GAP + CB_W + 0.26 <= W, b_x + M_W + CB_GAP + CB_W

    fig = plt.figure(figsize=(W, H))
    row_top = TOP + TIT
    cmap = dna_cmap()
    cmap.set_bad(color="white", alpha=0.0)
    empty_rows_any = []
    im = None

    for (cond, letter, head), x0 in zip(CONDS, (a_x, b_x)):
        M, cols = matrix(truth, stated_by_cond[cond])
        tot = M.sum(axis=1, keepdims=True)
        F = np.divide(M, tot, out=np.zeros(M.shape, dtype=float), where=tot > 0)
        n_tot = int(M.sum())
        diag = int(sum(M[i, i] for i in range(len(PAIRS))))
        ch = chance_diagonal(M)

        ax, im, empty = draw_matrix_panel(
            fig, x0, row_top, M_W, M_H, M, cmap=cmap, show_yticklabels=(letter == "a"),
            title=f"{head}\nAccuracy {diag}/{n_tot} = {diag / n_tot:.3f}")
        empty_rows_any += [(cond, PAIRS[i]) for i in empty]
        panel_letter(fig, x0 - (0.46 if letter == "a" else 0.12), TOP, letter)

        # ---- numbers
        for i in range(len(PAIRS)):
            for j, cj in enumerate(cols):
                if M[i, j]:
                    rec(figure="rq3_trace_base_pair", panel=letter, arm=cfg["name"], checkpoint=ck,
                        condition=cond,
                        series=f"drawn cell: true {PAIRS[i]} -> stated {cj}, fraction of that row's "
                               f"{int(tot[i, 0])} held-out queries",
                        value=round(float(F[i, j]), 4), n=int(tot[i, 0]), k=int(M[i, j]), drawn=True,
                        source=cfg["file"], key="count / row total")
            rec(figure="rq3_trace_base_pair", panel=letter, arm=cfg["name"], checkpoint=ck,
                condition=cond, series=f"row total: held-out single-substitution queries whose true "
                                       f"pair is {PAIRS[i]}",
                value=int(tot[i, 0]), n=n_tot, k=int(tot[i, 0]), drawn=True, source=EXAMPLES,
                key="row denominator")
        rec(figure="rq3_trace_base_pair", panel=letter, arm=cfg["name"], checkpoint=ck,
            condition=cond, series="title: diagonal total / queries (stated pair equals the true pair)",
            value=round(diag / n_tot, 4), n=n_tot, k=diag, drawn=True, source=cfg["file"],
            key="trace(M[:12,:12])")
        rec(figure="rq3_trace_base_pair", panel=letter, arm=cfg["name"], checkpoint=ck,
            condition=cond, series="expected diagonal if the stated pair were independent of the true pair",
            value=round(ch / n_tot, 4), n=n_tot, k=round(ch, 2), drawn=False, source=cfg["file"],
            key="chance_diagonal()")
        rec(figure="rq3_trace_base_pair", panel=letter, arm=cfg["name"], checkpoint=ck,
            condition=cond, series="queries stating no pair ('none' column total)",
            value=round(int(M[:, -1].sum()) / n_tot, 4), n=n_tot, k=int(M[:, -1].sum()), drawn=True,
            source=cfg["file"], key="M[:, 'none'].sum()")
        rec(figure="rq3_trace_base_pair", panel=letter, arm=cfg["name"], checkpoint=ck,
            condition=cond, series="most frequent stated pair, share of all queries",
            value=round(max(M[:, :len(PAIRS)].sum(axis=0)) / n_tot, 4), n=n_tot,
            k=int(max(M[:, :len(PAIRS)].sum(axis=0))), drawn=False, source=cfg["file"],
            key=PAIRS[int(np.argmax(M[:, :len(PAIRS)].sum(axis=0)))])
        print(f"  {ck} {cond:8s} diagonal {diag}/{n_tot} = {diag / n_tot:.4f} "
              f"(chance {ch / n_tot:.4f}); 'none' {int(M[:, -1].sum())}; "
              f"empty rows {len(empty)}")

    cax = ax_in(fig, b_x + M_W + CB_GAP, row_top + 0.25 * M_H, CB_W, 0.5 * M_H)
    cb = fig.colorbar(im, cax=cax, ticks=[0, 0.5, 1])
    cax.text(0.0, 1.10, "Fraction\nof row", transform=cax.transAxes, ha="left", va="bottom",
             fontsize=5.0, linespacing=1.0)
    cb.ax.tick_params(labelsize=5.0, length=1.5, pad=1)
    cb.outline.set_linewidth(0.5)
    note = "dotted outline: the diagonal, where the stated pair equals the true pair"
    if empty_rows_any:
        note += ";  hatched row: no held-out query has that true pair"
    fig.text(a_x / W, 0.07 / H, note, ha="left", va="bottom", fontsize=LEGEND_FS, color="black")
    return emit(fig, cfg["stem"], out_dir), empty_rows_any


def write_tex(ck, truth, stated_by_cond, out_dir, states_all):
    cfg = CKPTS[ck]
    v = {}
    for cond, letter, _ in CONDS:
        M, _ = matrix(truth, stated_by_cond[cond])
        n_tot = int(M.sum())
        diag = int(sum(M[i, i] for i in range(len(PAIRS))))
        top_j = int(np.argmax(M[:, :len(PAIRS)].sum(axis=0)))
        v[cond] = dict(n=n_tot, diag=diag, acc=diag / n_tot, chance=chance_diagonal(M) / n_tot,
                       none=int(M[:, -1].sum()), top=PAIRS[top_j],
                       top_k=int(M[:, :len(PAIRS)].sum(axis=0)[top_j]))
    rowN = [int(x) for x in matrix(truth, stated_by_cond["wt"])[0].sum(axis=1)]
    f = lambda x: f"{x:.3f}"
    nz = f"${min(rowN)}$--${max(rowN)}$"
    sft_note = ("" if ck == "rl" else
                rf" A query whose trace states no pair at all is counted in the row total and shown in "
                rf"the \texttt{{none}} column, so the denominators stay at the true-pair counts: "
                rf"${v['wt']['none']}$ of ${v['wt']['n']}$ queries intact and ${v['scramble']['none']}$ "
                rf"shuffled state no pair.")
    degen = ("" if v["wt"]["acc"] != v["wt"]["chance"] else
             r" The two are equal because a single pair is stated on every query, so the diagonal is "
             r"exactly the number of queries whose true pair is that one.")
    all_note = (rf" Over all $145$ held-out queries, including the $34$ without a single pair, a pair is "
                rf"stated on ${states_all['wt']}$ intact and ${states_all['scramble']}$ shuffled.")
    cap = rf"""% Built by trace_base_pairs.py --ckpt {ck}. Include at \linewidth.
% Numbers: {cfg['stem']}_numbers.csv (every cell count, every row total, every drawn fraction).
% Queries: ood_test of the pathway network split, restricted to single substitutions.
% Traces: dna/bioreason/perturbations/{cfg['file']}; pair extraction = subs() in trace_base_pairs.py.
\begin{{figure}}[t]
\centering
\includegraphics[width=\linewidth]{{figures/{cfg['stem']}.pdf}}
\caption{{\textbf{{The base change {'the released BioReason RL checkpoint' if ck == 'rl' else 'the released BioReason SFT checkpoint'} states in its reasoning
trace is {'the same one regardless of the sequence it is given' if ck == 'rl' else 'almost never the query'"'"'s own'}.}} Rows are the true reference-to-variant pair,
read off the reference and variant windows of each query; columns are the pair the trace states.
The queries are the ${len(truth)}$ held-out queries (\texttt{{ood\_test}} of the pathway-network split,
$145$ queries) whose two windows differ at exactly one position; the other $34$ are multi-base or
length-changing edits and have no single pair. Each matrix row is normalised by its own total, so a
diagonal cell is the accuracy for that true pair and an off-diagonal cell is the fraction of that
row's queries the trace assigned to another pair. Row totals are in the $y$ tick labels and range
from {nz}; the dotted outline marks the diagonal. \textbf{{(a)}} The original DNA. \textbf{{(b)}} The same queries with both windows
base-shuffled, composition preserved, before Evo2 encodes them; rows are still the query's own pair,
since the shuffled windows no longer carry it. The stated pair equals the true pair on
${v['wt']['diag']}/{v['wt']['n']}={f(v['wt']['acc'])}$ of queries with the DNA intact and
${v['scramble']['diag']}/{v['scramble']['n']}={f(v['scramble']['acc'])}$ with it shuffled, against
${f(v['wt']['chance'])}$ and ${f(v['scramble']['chance'])}$ expected if the stated pair were
independent of the true pair.{degen} The trace states \texttt{{{v['wt']['top']}}} on
${v['wt']['top_k']}$ of the ${v['wt']['n']}$ queries intact and \texttt{{{v['scramble']['top']}}} on
${v['scramble']['top_k']}$ shuffled.{all_note}{sft_note}}}
\label{{fig:rq3_trace_base_pair{'' if ck == 'rl' else '_sft'}}}
\end{{figure}}
"""
    open(os.path.join(out_dir, f"{cfg['stem']}.tex"), "w").write(cap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="both", choices=["rl", "sft", "both"])
    ap.add_argument("--out_dir", default=FIG_DIR)
    a = ap.parse_args()
    init_print_style()
    truth, data, states_all = load()
    todo = ["rl", "sft"] if a.ckpt == "both" else [a.ckpt]
    for ck in todo:
        del ROWS[:]
        print(f"\n== {CKPTS[ck]['name']}")
        paths, empty = draw(ck, truth, data[ck], a.out_dir)
        write_tex(ck, truth, data[ck], a.out_dir, states_all[ck])
        cols = ["figure", "panel", "arm", "checkpoint", "condition", "series", "value", "n", "k",
                "drawn", "source", "key"]
        out_csv = os.path.join(a.out_dir, f"{CKPTS[ck]['stem']}_numbers.csv")
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in ROWS:
                w.writerow(r)
        print(f"  wrote {out_csv} ({len(ROWS)} rows) and {CKPTS[ck]['stem']}.tex")
        print(f"  empty rows marked: {len(empty)} {empty}")


if __name__ == "__main__":
    main()

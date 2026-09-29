#!/usr/bin/env python
"""Draws the ChatNT panel of Figure 2: accuracy with intact or shuffled DNA on three of its own binary
tasks (splice donors, splice acceptors, TATA promoters), with 95% intervals.

perturbations_and_conflicts.py calls data_chatnt() and panel_destroyed(). Run on its own, the module
reads dna/chatnt/results/chatnt_summary.csv under INPUT_USE_RESULTS_DIR (CHATNT_SUMMARY overrides
the path) and writes fig_rq1_chatnt and its numbers CSV to outputs/figures, or to --out_dir. The
evidence conflict panel is drawn only when CHATNT_DRAW_SWAP=1.
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
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figure_style import (init_print_style, center_ylabel, ax_in, finish, bar_label,  # noqa: E402
                          align_xlabels, panel_letter, emit, DNA, HEADROOM_ROTATED,
                          TICK_FS, LEGEND_FS, EDGE_LW, ERR_LW)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rq1_data as M  # noqa: E402
import evidence_swap_panels as S  # noqa: E402

CHATNT = f"{M.RD.CHATNT}/results"
SUMMARY = os.environ.get("CHATNT_SUMMARY", f"{CHATNT}/chatnt_summary.csv")   # env override: layout tests on a partial run
STEM = "fig_rq1_chatnt"
MODEL = "ChatNT"
TASK_ORDER = ["NT_splice_sites_donors", "NT_splice_sites_acceptors", "NT_promoter_tata"]
TASK_LABEL = {"NT_splice_sites_donors": "splice\ndonors", "NT_splice_sites_acceptors": "splice\nacceptors",
              "NT_promoter_tata": "TATA\npromoters"}
YL = (0, 1.18)
DRAW_SWAP = os.environ.get("CHATNT_DRAW_SWAP", "0") == "1"
YT = np.arange(0, 1.01, 0.25)


def _ci(r, col):
    return (float(r[f"{col}_ci_low"]), float(r[f"{col}_ci_high"]))


def data_chatnt(per_task=True, with_mean=False):
    """None if the ChatNT summary has not been written yet. Every row carries n_total = all questions."""
    if not os.path.exists(SUMMARY):
        print(f"[chatnt] summary missing ({SUMMARY}); panels NOT DRAWN")
        return None
    with open(SUMMARY) as fh:
        table = {r["task"]: r for r in csv.DictReader(fh)}
    tasks = [t for t in TASK_ORDER if t in table]
    assert "macro_mean" in table and (len(tasks) == 3 or "CHATNT_SUMMARY" in os.environ), table.keys()
    n_total = sum(int(table[t]["n"]) for t in tasks)
    assert int(table["macro_mean"]["n"]) == n_total

    def group(key, label, r, n, drawn):
        row = {"key": key, "label": label, "n": n, "intact": float(r["acc_intact"]),
               "shuffled": float(r["acc_shuffle_mean"]), "intact_ci": _ci(r, "acc_intact"),
               "shuffled_ci": _ci(r, "acc_shuffle_mean"), "delta": float(r["delta_shuffle"]),
               "delta_ci": _ci(r, "delta_shuffle")}
        tag = "" if drawn else " (not drawn)"
        for k, series in (("intact", "DNA intact"), ("shuffled", "DNA shuffled")):
            M.record("chatnt", "a", model=MODEL, checkpoint="released", group=label.replace("\n", " "),
                     series=series + tag, x=key, value=row[k], n=n, k=3 if key == "macro_mean" else "",
                     ci_low=row[f"{k}_ci"][0], ci_high=row[f"{k}_ci"][1], source=SUMMARY,
                     key=f"{key}.{'acc_intact' if k == 'intact' else 'acc_shuffle_mean'}")
        M.record("chatnt", "a", model=MODEL, checkpoint="released", group=label.replace("\n", " "),
                 series="intact minus shuffled (caption only)", x=key, value=row["delta"], n=n,
                 k=3 if key == "macro_mean" else "", ci_low=row["delta_ci"][0], ci_high=row["delta_ci"][1],
                 source=SUMMARY, key=f"{key}.delta_shuffle")
        return row

    rows = [group(t, TASK_LABEL[t], table[t], int(table[t]["n"]), per_task) for t in tasks]
    macro = group("macro_mean", "3 tasks\nmean", table["macro_mean"], n_total, with_mean or not per_task)
    rows = (rows + ([macro] if with_mean else [])) if per_task else [macro]
    for r in rows:
        r["n_total"] = n_total

    m = table["macro_mean"]
    # gated_switch_rate = P(answer == label of the DNA donor) is therefore "matching A" here
    f, (flo, fhi) = float(m["gated_switch_rate"]), _ci(m, "gated_switch_rate")
    sw = {"A": f, "B": 1.0 - f, "neither": 0.0, "n": int(m["gated_n"]),
          "ci": {"A": (flo, fhi), "B": (1.0 - fhi, 1.0 - flo), "neither": (0.0, 0.0)},
          "label": "question(B)\nDNA(A)", "epsilon": None,
          "per_task": {t: (float(table[t]["gated_switch_rate"]), int(table[t]["gated_n"])) for t in tasks}}
    for which, val, ci, what in (("A", sw["A"], sw["ci"]["A"], "answer follows the DNA (spec: gated_switch_rate)"),
                                 ("B", sw["B"], sw["ci"]["B"], "answer follows the question (spec: 1 - gated_switch_rate)")):
        M.record("chatnt", "b", model=MODEL, checkpoint="released", group="gated pairs, 3 tasks mean",
                 series=f"matching {which}: {what}", x=sw["label"].replace("\n", " "), value=val, n=sw["n"], k=3,
                 ci_low=ci[0], ci_high=ci[1], source=SUMMARY, key="macro_mean.gated_switch_rate")
    for t in tasks:
        M.record("chatnt", "b", model=MODEL, checkpoint="released", group=TASK_LABEL[t].replace("\n", " "),
                 series="matching A: answer follows the DNA (per task, not drawn)", x=sw["label"].replace("\n", " "),
                 value=sw["per_task"][t][0], n=sw["per_task"][t][1], k="",
                 ci_low=_ci(table[t], "gated_switch_rate")[0], ci_high=_ci(table[t], "gated_switch_rate")[1],
                 source=SUMMARY, key=f"{t}.gated_switch_rate")
    for col in ("donor_follow_rate", "donor_follow_rate_given_intact_correct", "acc_shuffle_dinuc",
                "delta_margin_shuffle", "mean_signed_margin_intact", "mean_signed_margin_shuffle"):
        M.record("chatnt", "b", model=MODEL, checkpoint="released", group="3 tasks mean",
                 series=f"{col} (caption only)", x=col, value=float(m[col]), n=n_total, k=3,
                 ci_low=_ci(m, col)[0] if f"{col}_ci_low" in m else "",
                 ci_high=_ci(m, col)[1] if f"{col}_ci_high" in m else "", source=SUMMARY, key=f"macro_mean.{col}")
    print(f"[chatnt] " + "; ".join(f"{r['label'].replace(chr(10), ' ')}: {r['intact']:.3f}/{r['shuffled']:.3f}"
                                   for r in rows)
          + f" (intact/shuffled) | swap gated n={sw['n']}: follows DNA (A) {sw['A']:.3f}, question (B) {sw['B']:.3f}"
          + " | per task A " + " ".join(f"{v[0]:.3f}({v[1]})" for v in sw["per_task"].values()))
    return rows, (sw if DRAW_SWAP else None)


def err(ax, x, v, ci):
    if not ci:
        return v
    ax.errorbar([x], [v], yerr=[[max(0.0, v - ci[0])], [max(0.0, ci[1] - v)]], fmt="none",
                ecolor="black", elinewidth=ERR_LW, capsize=0, zorder=5)
    return max(v, ci[1])


def panel_destroyed(ax, rows, fs=4.5, ylabel="Accuracy", tick_fs=6.0):
    """One group per row, DNA intact vs shuffled, drawn like the combined figure's ladder panels."""
    finish(ax, ylabel, ylim=YL, yticks=YT, spine_top=1.0)
    center_ylabel(ax, 1.0)
    xs = np.arange(len(rows))
    w = 0.38
    for i, r in enumerate(rows):
        for j, (k, col) in enumerate((("intact", DNA.model), ("shuffled", DNA.perturbed))):
            x = xs[i] + (j - 0.5) * w
            ax.bar(x, r[k], w, color=col, edgecolor="black", linewidth=EDGE_LW)
            bar_label(ax, x, err(ax, x, r[k], r.get(f"{k}_ci")), f"{r[k]:.3f}", fs=fs)
    ax.set_xticks(xs)
    ax.set_xticklabels([r["label"] for r in rows], rotation=45, ha="right", rotation_mode="anchor",
                       linespacing=1.1)
    ax.tick_params(axis="x", labelsize=tick_fs)
    ax.set_xlim(-0.62, len(rows) - 0.38)
    return [Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="black", lw=EDGE_LW, label=l)
            for l, c in (("DNA intact", DNA.model), ("DNA shuffled", DNA.perturbed))]


def panel_swapped(ax, sw, slots=None, fs=4.5, ylabel="Answers", tick_fs=4.5):
    S.pct_axis(ax, ylabel)
    S.bars(ax, DNA, [sw], slots=slots, fs=fs)
    ax.set_xticklabels([sw["label"]], linespacing=1.1, rotation=45, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", labelsize=tick_fs)
    # no "matching neither" entry: that bar is 0 by construction (binary answer) and the grey is
    # already defined by the d-f legend above; a third entry collided with the neighbouring legend
    return [Rectangle((0, 0), 1, 1, facecolor=S.shade(DNA, w), edgecolor="black", lw=EDGE_LW, label=l)
            for w, l in (("A", "matching A"), ("B", "matching B"))]


def title(ax, text):
    ax.set_title(text, fontsize=TICK_FS, fontweight="normal", pad=3.0, linespacing=1.15)


def draw(rows, sw, out_dir):
    W = 3.4 if sw is not None else 0.40 + 0.30 * len(rows) + 1.0
    TOP, TIT, AXH, BOT = 0.30, 0.30, 1.25, 1.30
    H = TOP + TIT + AXH + BOT
    fig = plt.figure(figsize=(W, H))
    row_top = TOP + TIT
    wa = 0.40 + 0.30 * len(rows)
    xa, xb, wb = 0.52, 2.50, 0.62
    ax = ax_in(fig, xa, row_top, wa, AXH)
    panel_letter(fig, xa - 0.42, TOP - 0.14, "a")
    a_handles = panel_destroyed(ax, rows)
    ax.set_xlabel("Task", fontsize=TICK_FS)
    title(ax, f"DNA: {MODEL}\n{rows[0]['n_total']:,} questions")
    axes = [ax]
    y_top = 0.62 / H
    leg = dict(frameon=False, fontsize=LEGEND_FS, handletextpad=0.4, labelspacing=0.3)
    fig.legend(handles=a_handles, loc="upper center", ncol=1, handlelength=1.1,
               bbox_to_anchor=((xa + wa / 2) / W, y_top), **leg)
    if sw is not None:
        ax = ax_in(fig, xb, row_top, wb, AXH)
        panel_letter(fig, xb - 0.40, TOP - 0.14, "b")
        b_handles = panel_swapped(ax, sw, slots=1.0)
        ax.set_xlabel("DNA source", fontsize=TICK_FS)
        title(ax, f"DNA: {MODEL}\n{sw['n']:,} gated pairs")
        axes.append(ax)
        fig.legend(handles=b_handles, loc="upper center", ncol=1, handlelength=1.1,
                   bbox_to_anchor=((xb + wb / 2) / W, y_top), **leg)
    align_xlabels(fig, axes)
    return emit(fig, STEM, out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=M.OUT_DEFAULT)
    ap.add_argument("--mean_only", action="store_true", help="draw the 3-task mean instead of the tasks")
    ap.add_argument("--with_mean", action="store_true", help="append the 3-task mean as a fourth group")
    a = ap.parse_args()
    init_print_style()
    d = data_chatnt(per_task=not a.mean_only, with_mean=a.with_mean)
    if d is None:
        sys.exit(1)
    rows, sw = d
    draw(rows, sw, a.out_dir)
    keys = ["figure", "panel", "model", "checkpoint", "group", "series", "x", "value", "n", "k",
            "ci_low", "ci_high", "sd_over_folds", "source", "key"]
    p = os.path.join(a.out_dir, f"{STEM}_numbers.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in keys} for r in M.ROWS if r["figure"] == "chatnt"])
    print(f"wrote {p}")


if __name__ == "__main__":
    main()

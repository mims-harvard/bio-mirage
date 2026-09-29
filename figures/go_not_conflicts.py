#!/usr/bin/env python
"""Draws the appendix figure "Evidence conflicts for BioReason-Pro (RL) on GO NOT pairs".

    python figures/go_not_conflicts.py [--out_dir <dir>]

Repeats the BioReason-Pro evidence conflicts of Figure 3b on the GO NOT pairs only: pairs whose
negative member carries an experimental NOT annotation for the readout GO term, rather than lacking
the term. The ESM3 representation comes from protein A and the text inputs from protein B, as in
Figure 3b. Reads protein/bioreason_pro/evidence_conflicts_rl/metrics.json under
INPUT_USE_RESULTS_DIR. Intervals are the
conflict-level bootstrap of analysis/bootstrap_evidence_swap.py, 10,000 resamples. Writes
fig_si_gonot_conflict.{pdf,png,svg} and fig_si_gonot_conflict_numbers.csv to outputs/figures.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "analysis"))
import rq1_data as M                 # noqa: E402
import evidence_swap_panels as S     # noqa: E402
import perturbations_and_conflicts as BM  # noqa: E402
from bootstrap_evidence_swap import boot  # noqa: E402
from figure_style import init_print_style, ax_in, emit, PROTEIN, LEGEND_FS  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
import matplotlib.pyplot as plt      # noqa: E402

STEM = "fig_si_gonot_conflict"
CAT = "go_not"
RUN = f"{M.RD.PROTEIN_EVIDENCE_CONFLICTS['rl']}/metrics.json"
RATES = (("B", "context_override_rate"), ("A", "protein_retention_rate"), ("neither", "other_rate"))


def data():
    m = M.load(RUN)["by_category"][CAT]
    swapAB = str.maketrans("AB", "BA")          # roles as in Figure 3b: ESM3 = A, text inputs = B
    rng = np.random.default_rng(20260918)
    out, rows = [], []
    for key, label, _prov in M.CONFLICT_ARMS:
        fam = m["by_conflict_family"][key]
        n = fam["context_override_rate"]["n"]
        ks = {}
        for which, rk in RATES:
            k = fam[rk]["rate"] * n
            assert abs(k - round(k)) < 1e-6 and fam[rk]["n"] == n, (key, rk, k, n)
            ks[which] = int(round(k))
        assert sum(ks.values()) == n, (key, ks, n)
        ci = boot([(f"{key}:{w}:{i}", w) for w, k in ks.items() for i in range(k)], 10000, rng)
        tick = label.translate(swapAB)
        out.append({"key": key, "label": tick, "n": n, **{w: ks[w] / n for w in ks},
                    "ci": {w: list(ci[w]) for w in ci}})
        for which, rk in RATES:
            rows.append({"panel": "a", "model": "BioReason-Pro RL", "group": key,
                         "x": tick.replace("\n", " "), "series": which, "value": ks[which] / n, "n": n,
                         "k": ks[which], "ci_low": ci[which][0], "ci_high": ci[which][1], "source": RUN,
                         "key": f"by_category.{CAT}.by_conflict_family.{key}.{rk}"})
        print(f"[{key}] n {n}: " + "  ".join(
            f"{w} {100 * ks[w] / n:.1f}% [{100 * ci[w][0]:.1f}, {100 * ci[w][1]:.1f}]" for w in ("A", "B", "neither")))
    print(f"gated pairs {m['n_passing_gate']} of {m['n_pairs']}")
    return out, rows, m["n_passing_gate"], m["n_pairs"]


def draw(vals, n_gated, n_pairs, out_dir):
    unit = 0.40                                  # one bar group width, as in Figure 3b
    L, TOP, TIT, AXH, BOT = 0.66, 0.06, 0.30, 1.15, 0.74
    Wf = 2.60                                   # wide enough for the one-row legend
    H = TOP + TIT + AXH + BOT
    fig = plt.figure(figsize=(Wf, H))
    ax = ax_in(fig, L, TOP + TIT, unit * len(vals), AXH)
    BM.pct_axis(ax, "Predicted functional\nproperty")
    S.bars(ax, PROTEIN, vals, slots=len(vals), fs=4.5)
    ax.set_xticklabels([v["label"] for v in vals], linespacing=1.1, rotation=45, ha="right",
                       rotation_mode="anchor")
    ax.tick_params(axis="x", labelsize=4.5)
    ax.set_xlabel("")
    BM.title(ax, f"BioReason-Pro (RL), GO NOT\nN = {vals[0]['n']:,}")
    handles = [BM.rect(S.shade(PROTEIN, which), "") for which, _ in S.WHICH]
    fig.legend(handles=handles, labels=[lab for _, lab in S.WHICH], loc="lower center",
               bbox_to_anchor=((L + unit * len(vals) / 2) / Wf, 0.02), ncol=3, frameon=False, fontsize=LEGEND_FS,
               handlelength=1.2, handleheight=0.8, handletextpad=0.4, columnspacing=0.8)
    return emit(fig, STEM, out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=M.OUT_DEFAULT)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    vals, rows, n_gated, n_pairs = data()
    init_print_style()
    draw(vals, n_gated, n_pairs, a.out_dir)
    p = os.path.join(a.out_dir, f"{STEM}_numbers.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {p} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

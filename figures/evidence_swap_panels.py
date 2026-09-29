#!/usr/bin/env python
"""Evidence conflict data loaders and bar panels for Figure 3, and a standalone figure fig_rq1_swap
with one panel per modality: BioReason, BioReason-Pro and C2S-Scale 27B given the foundation model
representation of entity A with the text of entity B.

Each panel shows the share of answers that match A, match B, or match neither.
perturbations_and_conflicts.py uses the loaders, panel_specs() and bars(). Bootstrap intervals come
from ci_swap.json, written by analysis/bootstrap_evidence_swap.py, and are omitted when it is absent. Writes fig_rq1_swap (and
fig_rq1_swap_vertical with --vertical) and a numbers CSV to outputs/figures, or to --out_dir.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerTuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figure_style import (init_print_style, center_ylabel, ax_in, finish, bar_label,  # noqa: E402
                          align_xlabels, panel_letter, emit, DNA, PROTEIN, CELL, GREY, _mix,
                          TICK_FS, LEGEND_FS, EDGE_LW, ERR_LW, HEADROOM_ROTATED)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rq1_data as M  # noqa: E402

REPO_DIR = os.environ.get("INPUT_USE_HOME", ".")
sys.path.insert(0, REPO_DIR)
from input_use.metrics.dna import label_of, normalize_label_text, canonical_gold  # noqa: E402


OUT_DEFAULT = M.OUT_DEFAULT
STEM = "fig_rq1_swap"
# 95% bootstrap intervals for every bar, from analysis/bootstrap_evidence_swap.py; None until it has been run,
# and the figure then draws no intervals.
CI = M.ci_json("ci_swap.json")


def ci_of(panel, key):
    """{"A": [lo, hi], ...} for one group of bars, or {} if the artifact has not been computed."""
    cell = ((CI or {}).get(panel) or {}).get(key)
    return {w: cell[w]["ci95"] for w in ("A", "B", "neither")} if cell else {}


def read_jsonl(path):
    with open(path) as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


# ------------------------------------------------------------------------------------------------
# a.
SWAPBOTH = M.RD.DNA_EVIDENCE_CONFLICTS
# One group per evidence-source combination, labelled like panel b: which entity each channel comes
# from. Evo2 (both DNA blocks) is always the donor A's; the two text channels are each A's or B's.
PANEL_ARMS = [("swap_both_donor", "pathway(B)\ngene(B)\nEvo2(A)"),
              ("swap_both_gene_donor", "pathway(B)\ngene(A)\nEvo2(A)"),
              ("swap_both_pathway_donor", "pathway(A)\ngene(B)\nEvo2(A)")]
# swap_variant_donor keeps only the variant block from A, so its Evo2 channel is a mix; it is the
# arm the stored artifact asserts against and it stays in the CSV, and it is what panel a falls back
# to while the three combinations above have no records yet.
FALLBACK_ARM = ("swap_variant_donor", "variant block(A)\nreference block(B)\ntext(B)")
SWAP_ARMS = [FALLBACK_ARM] + PANEL_ARMS
# (key, label for the CSV / a one-arm axis, short label for the four-group axis)
ROWSETS = [("all", "all 1,449 rows", "all rows"), ("dep", "165 genome-dependent rows", "165 dep.")]


def data_dna_swap():
    """Same label mapping as metrics.dna.donor_following_rate: a prediction matches the donor (A, the
    source of the swapped DNA) when its normalised text equals the donor's canonical label, matches
    B (the row whose text is shown) when it equals the row's own gold, and is 'neither' otherwise;
    rows whose prediction maps to no label are dropped.
    """
    D = M.KEGG
    pairs = json.load(open(f"{D}/pairs.json"))
    donors = pairs["donors"]
    sens = set(pairs["strata"]["sensitivity"]["rows"])
    assert len(sens) == 165, len(sens)
    out = {}
    for ck in M.DNA_CKPTS:
        f = M.RD.dna_records(D, ck)
        recs = list(read_jsonl(f))
        labels = sorted({(r.get("ground_truth") or {}).get("answer", "") for r in recs} - {""})
        wt = {r["example_id"]: r for r in recs if r["condition"] == "wt"}
        arms = {"swap_variant_donor":
                (f, {r["example_id"]: r for r in recs if r["condition"] == "swap_variant_donor"})}
        assert len(wt) == len(arms["swap_variant_donor"][1]) == 1449, len(wt)
        f2 = M.RD.dna_records(SWAPBOTH, ck)
        if os.path.exists(f2):
            new = list(read_jsonl(f2))
            for arm, _ in PANEL_ARMS:
                sb = {r["example_id"]: r for r in new if r["condition"] == arm}
                if not sb:
                    continue
                assert set(sb) <= set(wt), len(set(sb) - set(wt))
                moved = sum(sb[e]["input"]["dna_embed_sha1"] != wt[e]["input"]["dna_embed_sha1"] for e in sb)
                same_text = sum(sb[e]["input"]["text_token_sha1"] == wt[e]["input"]["text_token_sha1"]
                                for e in sb)
                # Evo2 comes from A in every arm; the text is B's own only in the arm that swaps no
                # text channel, and must differ from wt on every row in the two that do
                want_same = len(sb) if arm == "swap_both_donor" else 0
                assert moved == len(sb) and same_text == want_same, (arm, moved, same_text, len(sb))
                arms[arm] = (f2, sb)
                print(f"[a] {ck} {arm}: {len(sb)} rows, Evo2 embedding differs from wt on all, text tokens "
                      f"{'identical to' if want_same else 'differ from'} wt on all")

        def cls(sw, eid):
            r = sw[eid]
            pred = label_of(r["output"].get("raw", ""), labels)
            d, g = donors.get(eid, {}).get("donor_answer"), r["ground_truth"].get("answer")
            if d is None or g is None or pred is None:
                return None
            p = normalize_label_text(pred)
            return "A" if p == canonical_gold(d) else ("B" if p == canonical_gold(g) else "neither")

        def wt_correct(eid):
            r = wt[eid]
            pred = label_of(r["output"].get("raw", ""), labels)
            return pred is not None and \
                normalize_label_text(pred) == canonical_gold(r["ground_truth"].get("answer"))

        stored = json.load(open(f"{D}/metrics_{ck.lower()}.json"))["experiment3"]
        out[ck] = {}
        for arm, arm_label in SWAP_ARMS:
            if arm not in arms:
                continue
            src, sw = arms[arm]
            out[ck][arm] = {}
            for key, label, short in ROWSETS:
                rows = set(sw) if key == "all" else set(sw) & sens
                if arm == "swap_variant_donor":      # assert the unconditional counts against the artifact
                    skey = "donor_swap" if key == "all" else "donor_swap_sensitivity_only"
                    c = Counter(cls(sw, e) for e in rows)
                    c.pop(None, None)
                    s = stored[skey]
                    assert (sum(c.values()), c["B"], c["A"], c["neither"]) == \
                        (s["n"], s["follows_original"], s["follows_donor"], s["neither"]), (ck, key, c, s)
                    M.record("swap", "a", model="BioReason", checkpoint=ck,
                             group=f"{arm.replace('swap_', '').replace('_donor', '')} from A, {label}",
                             series="matches B (own text; unconditional, stored)", x=arm,
                             value=s["follows_original_rate"], n=s["n"], k=s["follows_original"],
                             source=f"{D}/metrics_{ck.lower()}.json",
                             key=f"experiment3.{skey}.follows_original_rate")
                cond = [e for e in rows if wt_correct(e)]
                cc = Counter(cls(sw, e) for e in cond)
                cc.pop(None, None)
                n = sum(cc.values())
                out[ck][arm][key] = {"label": f"{arm_label}\n{short}", "rows_label": label,
                                     "arm_label": arm_label, "n": n, "n_wt_correct": len(cond),
                                     "A": cc["A"] / n, "B": cc["B"] / n, "neither": cc["neither"] / n,
                                     "ci": ci_of("a", f"{arm}|{key}") if ck == "RL" else {}}
                ci = out[ck][arm][key]["ci"]
                for which in ("A", "B", "neither"):
                    M.record("swap", "a", model="BioReason", checkpoint=ck,
                             group=f"{arm.replace('swap_', '').replace('_donor', '')} from A, {label}",
                             series={"A": "matches A (donor, swapped DNA)", "B": "matches B (own text)",
                                     "neither": "neither"}[which],
                             x=f"{arm} | wt correct", value=cc[which] / n, n=n, k=cc[which],
                             ci_low=ci.get(which, ["", ""])[0], ci_high=ci.get(which, ["", ""])[1],
                             source=src,
                             key="conditional on wt answer == gold; label_of/normalize_label_text/"
                                 "canonical_gold as metrics.dna.donor_following_rate")
                print(f"[a] {ck} {arm} {label}: wt-correct {len(cond)}, scorable {n}: "
                      f"A (donor DNA) {cc['A']} B (own text) {cc['B']} ({100 * cc['B'] / n:.1f}%) "
                      f"neither {cc['neither']}")
    return out


# ------------------------------------------------------------------------------------------------
# b.
def data_protein_abn(prot):
    """Adds the three roles to each arm of M.data_conflict(): A = the ESM3 protein
    (protein_retention_rate), B = the text protein (context_override_rate, = data_conflict's
    "rate"), neither = other_rate; each a conflict-weighted mean over the same categories.
    """
    f = f"{M.RD.PROTEIN_EVIDENCE_CONFLICTS['rl']}/metrics.json"
    m = M.load(f)
    cats = [c for c, _ in M.FAM_CATS]
    for o in prot:
        fam = {c: m["by_category"][c]["by_conflict_family"][o["key"]] for c in cats}
        n = sum(fam[c]["context_override_rate"]["n"] for c in cats)
        assert n == o["n"], (o["key"], n, o["n"])
        vals = {}
        for which, rk in (("B", "context_override_rate"), ("A", "protein_retention_rate"),
                          ("neither", "other_rate")):
            assert all(fam[c][rk]["n"] == fam[c]["context_override_rate"]["n"] for c in cats), (o["key"], rk)
            vals[which] = sum(fam[c][rk]["rate"] * fam[c][rk]["n"] for c in cats) / n
        assert abs(vals["B"] - o["rate"]) < 1e-9, (o["key"], vals["B"], o["rate"])
        assert abs(sum(vals.values()) - 1.0) < 1e-6, (o["key"], vals)
        o.update(vals)
        o["ci"] = ci_of("b", o["key"])
        for which in ("A", "neither"):
            M.record("swap", "b", model="BioReason-Pro RL", group=o["key"],
                     series="matches A (ESM3 protein)" if which == "A" else "neither", x=o["provenance"],
                     value=vals[which], n=n, k=round(vals[which] * n),
                     ci_low=o["ci"].get(which, ["", ""])[0], ci_high=o["ci"].get(which, ["", ""])[1],
                     source=f,
                     key=f"conflict-weighted mean over {cats} of by_category.*.by_conflict_family."
                         f"{o['key']}.{'protein_retention_rate' if which == 'A' else 'other_rate'}")
        print(f"[b] {o['label'].replace(chr(10), ' ')}: n {n}: A {100 * vals['A']:.1f}% "
              f"B {100 * vals['B']:.1f}% neither {100 * vals['neither']:.1f}%")
    return prot


# ------------------------------------------------------------------------------------------------
# c.
SC_RUN = M.RD.C2S_EVIDENCE_CONFLICTS
# One arm, labelled by where each channel comes from, as in panels a and b. The two-way variant
SC_ARMS = [("tc_xA_tB", "cell sentence(A)\ndescription(B)"), ("tc_xA_tB_2way", "only A or B\noffered")]
SC_PANEL_ARMS = {"tc_xA_tB"}


def data_sc_swap():
    f = f"{SC_RUN}/c2s_scale_27b/records.jsonl"
    by = defaultdict(dict)
    for r in read_jsonl(f):
        by[r["example_id"]][r["condition"]] = r
    need = {"tc_xA", "tc_xA_tB", "tc_xA_tB_2way"}
    cells = [c for c, v in by.items() if need <= set(v)]
    pred = lambda c, k: by[c][k]["output"]["parsed"]["cell_type"]  # noqa: E731
    gt = lambda c: by[c]["tc_xA"]["ground_truth"]["cell_type"]  # noqa: E731
    B = lambda c: by[c]["tc_xA"]["provenance"]["B"]  # noqa: E731
    cond = [c for c in cells if pred(c, "tc_xA") == gt(c)]
    n_pairs = len({by[c]["tc_xA"]["provenance"]["pair"] for c in cells})
    stored = json.load(open(f"{SC_RUN}/summary.json"))
    # keys are "<run directory>/c2s_scale_27b" and "<run directory>/c2s_scale_2b"; match the model
    stored = next(v for k, v in stored.items() if k.endswith("/c2s_scale_27b"))
    assert len(cond) == stored["n_conditioning_set"] and len(cells) == stored["n_cells"]
    out = {"n": len(cond), "n_cells": len(cells), "n_pairs": n_pairs, "arms": []}
    for arm, label in SC_ARMS:
        c = Counter("A" if pred(x, arm) == gt(x) else ("B" if pred(x, arm) == B(x) else "neither")
                    for x in cond)
        n = len(cond)
        tor_key = "TOR_cond" if arm == "tc_xA_tB" else "TOR_2way_cond"
        assert abs(c["B"] / n - stored[tor_key]) < 5e-5, (arm, c["B"] / n, stored[tor_key])
        ci = ci_of("c", arm)
        out["arms"].append({"arm": arm, "label": label, "n": n, "A": c["A"] / n, "B": c["B"] / n,
                            "neither": c["neither"] / n, "ci": ci})
        for which in ("A", "B", "neither"):
            M.record("swap", "c", model="C2S-Scale 27B", group=label.replace("\n", " "),
                     series=f"matches {which}" if which != "neither" else "neither", x=arm,
                     value=c[which] / n, n=n, k=c[which],
                     ci_low=ci.get(which, ["", ""])[0], ci_high=ci.get(which, ["", ""])[1], source=f,
                     key="conditional on parsed cell_type under tc_xA == ground truth (A)")
        print(f"[c] {label.replace(chr(10), ' ')}: n {n}: A {c['A']} ({100 * c['A'] / n:.1f}%) "
              f"B {c['B']} ({100 * c['B'] / n:.1f}%) neither {c['neither']}")
    return out


# ------------------------------------------------------------------------------------------------
# the figure
# ------------------------------------------------------------------------------------------------
WHICH = [("A", "matching A"), ("B", "matching B"), ("neither", "matching neither")]


def shade(pal, which):
    """A is the entity whose measured input the model gets, B the one whose text it gets. A is a light
    tint of the modality hue (45% white) and B the dark shade, so the two read apart at bar width."""
    return {"A": _mix(pal.model, "#FFFFFF", 0.45), "B": pal.model_dark, "neither": GREY}[which]


YL = (0, 1.0 + HEADROOM_ROTATED)   # rotated value labels sit above the interval, near 100 %
YT = np.arange(0, 1.01, 0.25)
BW = 0.26                          # three bars per group, 0.78 of the group pitch


def pct_axis(ax, ylabel, ylim=YL):
    finish(ax, ylabel, ylim=ylim, yticks=YT, spine_top=1.0)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    if ylabel:
        center_ylabel(ax, 1.0)
    else:
        ax.tick_params(axis="y", labelleft=False)


def panel_title(ax, text):
    ax.set_title(text, fontsize=TICK_FS, fontweight="normal", pad=3.0, linespacing=1.15)


def bars(ax, pal, vals, slots=None, fs=None):
    """vals: list of dicts with A / B / neither and an optional "ci"; one group of three bars each. The
    95% interval is a plain black line without caps, and the value label clears its top.
    """
    xs = np.arange(len(vals)) - (len(vals) - 1) / 2 if slots else np.arange(len(vals))
    for i, v in enumerate(vals):
        for j, (which, _) in enumerate(WHICH):
            x = xs[i] + (j - 1) * BW
            y = v[which]
            ax.bar(x, y, BW, color=shade(pal, which), edgecolor="black", linewidth=EDGE_LW)
            ci = (v.get("ci") or {}).get(which)
            top = y
            if ci:
                ax.errorbar([x], [y], yerr=[[max(0.0, y - ci[0])], [max(0.0, ci[1] - y)]],
                            fmt="none", ecolor="black", elinewidth=ERR_LW, capsize=0, zorder=5)
                top = max(y, ci[1])
            bar_label(ax, x, top, f"{100 * y:.1f}", rot=90, **({"fs": fs} if fs else {}))
    ax.set_xticks(xs)
    ax.set_xlim(*((-slots / 2, slots / 2) if slots else (-0.6, len(vals) - 0.4)))
    return xs


def panel_specs(dna, prot, sc):
    """(letter, palette, values, tick labels, x label, title) for each panel, in order."""
    a_arms = [arm for arm, _ in PANEL_ARMS if arm in dna["RL"]]
    a_vals = ([dna["RL"][arm]["all"] for arm in a_arms] if a_arms else
              [dna["RL"][FALLBACK_ARM[0]][k] for k, _, _ in ROWSETS])
    a_ticks = [v["arm_label"] if a_arms else v["rows_label"].replace(
               "genome-dependent", "genome-\ndependent").replace("1,449 ", "1,449\n") for v in a_vals]
    ns = [v["n"] for v in a_vals]
    swapAB = str.maketrans("AB", "BA")                    # roles: ESM3 = A, text channels = B
    c_vals = [o for o in sc["arms"] if o["arm"] in SC_PANEL_ARMS]
    n_conf = prot[0]["n"]
    return [("a", DNA, a_vals, a_ticks,
             "Evidence source, DNA (A/B)" if a_arms else "KEGG rows scored",
             f"DNA: BioReason (RL)\n{min(ns):,}-{max(ns):,} conflicts" if a_arms else
             "DNA: BioReason (RL)\nA's variant DNA, B's text"),
            ("b", PROTEIN, prot, [o["label"].translate(swapAB) for o in prot],
             "Evidence source protein (A/B)",
             f"Protein: BioReason-Pro (RL)\n{n_conf} conflicts, {n_conf // 2} pairs"),
            ("c", CELL, c_vals, [o["label"] for o in c_vals], "Evidence source, cell (A/B)",
             f"Single cell: C2S-Scale 27B\n{c_vals[0]['n']:,} conflicts")]


def ab_legend(fig, y, ncol=3):
    """One row of A / B / neither, each handle showing the three modality hues side by side."""
    handles = [tuple(Rectangle((0, 0), 1, 1, facecolor=shade(pal, which), edgecolor="black", lw=EDGE_LW)
                     for pal in (DNA, PROTEIN, CELL)) for which, _ in WHICH]
    return fig.legend(handles=handles, labels=[lab for _, lab in WHICH], loc="lower center",
                      bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False, fontsize=LEGEND_FS,
                      handlelength=2.4, handleheight=0.8, handletextpad=0.5, columnspacing=1.2,
                      handler_map={tuple: HandlerTuple(ndivide=3, pad=0)})


def draw_vertical(dna, prot, sc, out_dir):
    """The same three panels stacked, one per row, at single-column width. Include it at its natural
    size (\includegraphics[width=3.4in]) so the type prints at the same 7 pt as the wide version.
    """
    specs = panel_specs(dna, prot, sc)
    W, TOP, L_AX, TIT, AXH, XLAB, GAP, LEG = 3.4, 0.10, 0.52, 0.34, 1.02, 0.20, 0.08, 0.34
    slots = max(len(v) for _, _, v, _, _, _ in specs)      # same bar width in every panel a value label rotated 90
    # degrees is ~0.20 in tall whatever the axes height, so a
    # shorter panel needs more headroom in axis units than the
    # wide layout's HEADROOM_ROTATED or it hits the next title
    ylim = (0, 1.0 + 0.20 / AXH * 1.35)
    p_w = W - L_AX - 0.10
    ticks_h = [0.10 + 0.096 * max(t.count("\n") + 1 for t in tk) for _, _, _, tk, _, _ in specs]
    H = TOP + sum(TIT + AXH + t + XLAB for t in ticks_h) + GAP * (len(specs) - 1) + LEG
    fig = plt.figure(figsize=(W, H))
    top = TOP
    axes = []
    for (letter, pal, vals, tk, xlab, ttl), th in zip(specs, ticks_h):
        ax = ax_in(fig, L_AX, top + TIT, p_w, AXH)
        pct_axis(ax, "Exact-match\nanswer", ylim)
        bars(ax, pal, vals, slots=slots)
        ax.set_xticklabels(tk, linespacing=1.15)
        ax.tick_params(axis="x", labelsize=6.0)
        ax.set_xlabel(xlab, fontsize=TICK_FS)
        panel_title(ax, ttl)
        panel_letter(fig, 0.02, top, letter)
        axes.append(ax)
        top += TIT + AXH + th + XLAB + GAP
    ab_legend(fig, 0.03 / H)
    return emit(fig, STEM + "_vertical", out_dir)


def draw(dna, prot, sc, out_dir):
    # panel a holds one group per (swap arm, row set); the second arm appears only once the RL
    # records for it exist (dna/bioreason/evidence_conflicts), so the figure still builds without
    # them
    a_arms = [arm for arm, _ in PANEL_ARMS if arm in dna["RL"]]
    a_vals = ([dna["RL"][arm]["all"] for arm in a_arms] if a_arms else
              [dna["RL"][FALLBACK_ARM[0]][k] for k, _, _ in ROWSETS])
    W = 5.5
    TOP = 0.10
    TIT = 0.32
    AXH = 1.25
    BOT = 0.80                   # three-line ticks + x labels on one level + the A/B/neither legend
    H = TOP + TIT + AXH + BOT
    fig = plt.figure(figsize=(W, H))
    L_AX = 0.49
    row_top = TOP + TIT
    if len(a_vals) > 2:                              # three combinations: a takes width from b and c
        a_w, b_w, c_w, gap = 1.68, 1.50, 1.22, 0.22   # c holds one group but a two-line title, so it cannot be
        # narrower than that title
    else:
        a_w, b_w, c_w, gap = 1.45, 1.70, 1.21, 0.25
    a_x = L_AX
    b_x = a_x + a_w + gap
    c_x = b_x + b_w + gap
    assert c_x + c_w <= W - 0.10, c_x + c_w
    # ---- a: DNA, RL only; (variant block | both blocks) x (all rows | genome-dependent rows)
    ax = ax_in(fig, a_x, row_top, a_w, AXH)
    pct_axis(ax, "Exact-match\nanswer")
    bars(ax, DNA, a_vals)
    ax.set_xticklabels([v["arm_label"] if a_arms else v["rows_label"].replace(
                        "genome-dependent", "genome-\ndependent").replace("1,449 ", "1,449\n")
                        for v in a_vals], linespacing=1.15)
    ax.tick_params(axis="x", labelsize=6.0)
    ax.set_xlabel("Evidence source, DNA (A/B)" if a_arms else "KEGG rows scored", fontsize=TICK_FS)
    ns = [v["n"] for v in a_vals]        # the arms differ slightly: rows the model got right with no
    panel_title(ax, f"DNA: BioReason (RL)\n{min(ns):,}-{max(ns):,} conflicts" if a_arms else   # conflict, and
          "DNA: BioReason (RL)\nA's variant DNA, B's text")                              # that map to a label
    ax_a = ax
    panel_letter(fig, 0.02, TOP, "a")

    # ---- b: protein, three conflict arms
    ax = ax_in(fig, b_x, row_top, b_w, AXH)
    pct_axis(ax, None)
    bars(ax, PROTEIN, prot)
    swapAB = str.maketrans("AB", "BA")                    # roles: ESM3 = A, text channels = B
    ax.set_xticklabels([o["label"].translate(swapAB) for o in prot], linespacing=1.15)
    ax.tick_params(axis="x", labelsize=6.0)
    ax.set_xlabel("Evidence source protein (A/B)", fontsize=TICK_FS)
    n_conf = prot[0]["n"]
    panel_title(ax, f"Protein: BioReason-Pro (RL)\n{n_conf} conflicts, {n_conf // 2} pairs")
    ax_b = ax
    panel_letter(fig, b_x - 0.22, TOP, "b")

    # ---- c: single cell, the all-31-types arm
    ax = ax_in(fig, c_x, row_top, c_w, AXH)
    pct_axis(ax, None)
    c_vals = [o for o in sc["arms"] if o["arm"] in SC_PANEL_ARMS]
    bars(ax, CELL, c_vals)
    ax.set_xticklabels([o["label"] for o in c_vals], linespacing=1.15)
    ax.tick_params(axis="x", labelsize=6.0)
    ax.set_xlabel("Evidence source, cell (A/B)", fontsize=TICK_FS)
    panel_title(ax, f"Single cell: C2S-Scale 27B\n{c_vals[0]['n']:,} conflicts")
    ax_c = ax
    panel_letter(fig, c_x - 0.22, TOP, "c")
    align_xlabels(fig, [ax_a, ax_b, ax_c])

    ab_legend(fig, 0.03 / H)
    return emit(fig, STEM, out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=OUT_DEFAULT)
    ap.add_argument("--vertical", action="store_true",
                    help="also write fig_rq1_swap_vertical.*: the same panels stacked, 3.4 in wide")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    dna = data_dna_swap()                 # records ("swap", "a")
    prot = data_protein_abn(M.data_conflict())   # records ("protein", "c") for A, ("swap", "b") for B /
    # neither
    sc = data_sc_swap()                   # records ("swap", "c")

    init_print_style()
    draw(dna, prot, sc, a.out_dir)
    if a.vertical:                      # same numbers, stacked; the CSV below covers both
        draw_vertical(dna, prot, sc, a.out_dir)

    panel_of = {("swap", "a"): "a", ("protein", "c"): "b", ("swap", "b"): "b", ("swap", "c"): "c"}
    keys = ["figure", "panel", "model", "checkpoint", "group", "series", "x", "value", "n", "k",
            "ci_low", "ci_high", "sd_over_folds", "source", "key"]
    rows = [{**{k: r.get(k, "") for k in keys}, "figure": "swap",
             "panel": panel_of[(r["figure"], r["panel"])]}
            for r in M.ROWS if (r["figure"], r["panel"]) in panel_of]
    csv_path = os.path.join(a.out_dir, f"{STEM}_numbers.csv")
    with open(csv_path, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=keys)
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"wrote {csv_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

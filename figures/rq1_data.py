#!/usr/bin/env python
"""Shared RQ1 data loaders and CSV recorder for the figure scripts, and Figure 4 right ("DEGs contribute
to C2S-Scale performance"): C2S-Scale 2B and 27B cell type accuracy as increasing fractions of the
strongest DEGs, weakest DEGs, random genes, or non-DEGs with similar expression are removed.

record() collects every plotted number. Run on its own, the module also draws fig_rq1_protein and
fig_rq1_dna, which are not in the paper. Reads INPUT_USE_RESULTS_DIR and writes fig_rq1_singlecell,
the two other figures and fig_rq1_numbers.csv to outputs/figures, or to --out_dir.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics as st
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# The house style lives in the input-use figure-style skill: the role-based palette, the layout and
# qa helpers, and skill.md explaining why each rule exists. It re-exports the print-size half
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figure_style import (init_print_style, center_ylabel, ax_in, finish, bar_label,  # noqa: E402
                          facet_title, legend_upper, align_xlabels, panel_letter,
                          check_text_collisions, emit, DNA, PROTEIN, CELL, GREY, CTRL,
                          TICK_FS, ANNOT_FS, LEGEND_FS, EDGE_LW)
from input_use.core import paths as RD  # noqa: E402

REPO_DIR = os.environ.get("INPUT_USE_HOME", ".")

FIG_DIR = os.path.join(os.environ.get("INPUT_USE_OUTPUT_DIR", "outputs"), "figures")
OUT_DEFAULT = FIG_DIR
ANALYSIS_DIR = os.path.join(os.environ.get("INPUT_USE_OUTPUT_DIR", "outputs"), "analysis")

ROWS = []                          # every plotted number, one row each


def record(figure, panel, **kw):
    ROWS.append({"figure": figure, "panel": panel, **kw})


def load(path):
    with open(path) as fh:
        return json.load(fh)


# ------------------------------------------------------------------------------------------------
# protein a.
LADDER = [
    ("GO-GPT &\nInterPro", "wt", ("shuffle_esm3_only", "shuffle_esm3_only_s1", "shuffle_esm3_only_s2")),
    ("GO-GPT", "gogpt_only_wt_esm3", ("gogpt_only",)),
    ("InterPro", "interpro_only_wt_esm3", ("interpro_only",)),
    ("None", "no_symbolic", ("no_modality",)),
]
PROTEOME = {"RL": f"{RD.PROTEIN_PERTURBATIONS['rl']}/metrics.json",
            "SFT": f"{RD.PROTEIN_PERTURBATIONS['sft']}/metrics.json"}


CI_DIR = ANALYSIS_DIR


def ci_json(name):
    """A bootstrap-CI artifact from outputs/analysis/, or None if it has not been computed (figures then draw no interval)."""
    p = f"{CI_DIR}/{name}"
    return load(p) if os.path.exists(p) else None


def data_ladder():
    out = {}
    meanperm = ci_json("ci_protein_meanperm.json")
    for ck, f in PROTEOME.items():
        m = load(f)
        fw, bt = m["go_fmax_weighted"], m["go_f1_bootstrap"]
        n_prot = bt["_bootstrap"]["n_proteins"]
        rows = []
        for label, intact, shuffled in LADDER:
            vi = fw[intact]
            assert abs(vi - bt[intact]["f"]) < 1e-9
            vs_each = [fw[c] for c in shuffled]
            vs = float(np.mean(vs_each))
            if len(shuffled) == 1:
                vs_ci = bt[shuffled[0]]["ci"]
            else:                                   # mean of permutations: paired protein bootstrap of the mean
                e = (meanperm or {}).get("runs", {}).get(ck)
                vs_ci = None
                if e:
                    assert abs(e["mean_perm"]["value"] - vs) < 1e-4, (ck, e["mean_perm"]["value"], vs)
                    vs_ci = e["mean_perm"]["ci95"]
            rows.append({"label": label, "intact": vi, "shuffled": vs, "intact_ci": bt[intact]["ci"], "shuffled_ci": vs_ci})
            record("protein", "a", model="BioReason-Pro", checkpoint=ck,
                   group=label.replace("\n", " "), series="ESM3 intact", x=intact, value=vi,
                   n=n_prot, source=f, key=f"go_fmax_weighted.{intact}",
                   ci_low=bt[intact]["ci"][0], ci_high=bt[intact]["ci"][1])
            for c, v in zip(shuffled, vs_each):
                record("protein", "a", model="BioReason-Pro", checkpoint=ck,
                       group=label.replace("\n", " "), series="ESM3 shuffled (single permutation)",
                       x=c, value=v, n=bt[c]["n_scored"]["MF"], source=f,
                       key=f"go_fmax_weighted.{c}", ci_low=bt[c]["ci"][0], ci_high=bt[c]["ci"][1])
            if len(shuffled) > 1:
                record("protein", "a", model="BioReason-Pro", checkpoint=ck,
                       group=label.replace("\n", " "),
                       series="ESM3 shuffled (plotted: mean of permutations)", x="+".join(shuffled),
                       value=vs, n=n_prot, source=f, key="mean of go_fmax_weighted over seeds",
                       ci_low=vs_ci[0] if vs_ci else "", ci_high=vs_ci[1] if vs_ci else "")
        out[ck] = rows
    return out


# ------------------------------------------------------------------------------------------------
# protein b.
FAMILY_JSON = f"{ANALYSIS_DIR}/one_metric_with_shuffled.json"
FAM_CATS = [("pseudoenzyme", "enzyme\nactivity"), ("organelle_targeted", "organelle\ntargeting"),
            ("dna_binding", "DNA\nbinding")]
FAM_ARMS = [("ESM3 linear probe", "ESM3 linear probe", PROTEIN.probe),
            ("BioReason-Pro RL", "BioReason-Pro", PROTEIN.model),
            ("BioReason-Pro RL, shuffled ESM3", "BioReason-Pro, shuffled ESM3", PROTEIN.perturbed)]


def data_family_auroc():
    j = load(FAMILY_JSON)
    out = []
    for cat, label in FAM_CATS:
        arms = j[cat]["arms"]
        missing = [k for k, _, _ in FAM_ARMS if k not in arms]
        if missing:
            raise SystemExit(f"{FAMILY_JSON} lacks {missing} for {cat} -- run "
                             f"protein_family_table.py after the shuffled-ESM3 records land")
        row = {"cat": cat, "label": label, "n_families": j[cat]["n_families"],
               "n_proteins": j[cat]["n_proteins"], "vals": []}
        for key, name, col in FAM_ARMS:
            a = arms[key]
            row["vals"].append((name, a["within_family_auroc"], a["ci95_family"], col))
            record("protein", "b", model=name, group=cat, series="within-family AUROC", x=cat,
                   value=a["within_family_auroc"], n=j[cat]["n_proteins"], source=FAMILY_JSON,
                   key=f"{cat}.arms.{key}.within_family_auroc",
                   ci_low=a["ci95_family"][0], ci_high=a["ci95_family"][1])
        out.append(row)
    print("[protein b] " + " | ".join(
        f"{r['label'].replace(chr(10),' ')}: " + " ".join(f"{v:.3f}" for _, v, _, _ in r["vals"])
        for r in out))
    return out


# ------------------------------------------------------------------------------------------------
# protein c.
CONFLICT_ARMS = [("both channels swapped", "GO-GPT(A)\nInterPro(A)\nESM3(B)", "InterPro A, GO-GPT A"),
                 ("GO-GPT only", "GO-GPT(A)\nInterPro(B)\nESM3(B)", "InterPro B, GO-GPT A"),
                 ("InterPro text only", "GO-GPT(B)\nInterPro(A)\nESM3(B)", "InterPro A, GO-GPT B")]


def data_conflict():
    f = f"{RD.PROTEIN_EVIDENCE_CONFLICTS['rl']}/metrics.json"
    m = load(f)
    cats = [c for c, _ in FAM_CATS]
    out = []
    for key, label, prov in CONFLICT_ARMS:
        num = den = 0.0
        for c in cats:
            r = m["by_category"][c]["by_conflict_family"][key]["context_override_rate"]
            num += r["rate"] * r["n"]; den += r["n"]
        out.append({"key": key, "label": label, "provenance": prov, "rate": num / den,
                    "n": int(den)})
        record("protein", "c", model="BioReason-Pro RL", group=key,
               series="conflicts answered with protein A's property", x=prov, value=num / den,
               n=int(den), source=f,
               key=f"conflict-weighted mean over {cats} of "
                   f"by_category.*.by_conflict_family.{key}.context_override_rate")
    print("[protein c] " + " | ".join(f"{o['label'].replace(chr(10),' ')} {o['rate']:.4f} "
                                      f"(n={o['n']})" for o in out))
    return out


# ------------------------------------------------------------------------------------------------
# independent and the counts overstate the data). Stored in dna/bioreason/genome_dependent.
KEGG = RD.DNA_PERTURBATIONS
DNA_ARMS = [("wt", "real genome", DNA.model), ("scramble", "genome content destroyed", DNA.perturbed),
            ("no_dna", "no genome", GREY)]
DNA_CKPTS = ["SFT", "RL"]
# (condition key, x label) for panel a, ordered by what the model still receives. Values are
# experiment1_2_per_arm.accuracy: averaged within genome then across genomes, all 1,449 rows.
DNA_LADDER = [("wt", "all text\n+ Evo2"),
              ("no_dna", "all text,\nEvo2 removed"),
              ("no_gene", "pathway fields with\ngene symbols masked\n+ Evo2"),
              ("no_pathway", "question stem\nnaming the gene\n+ Evo2"),
              ("no_textkey", "question stem with\ngene symbol masked\n+ Evo2")]


def data_dna_ladder():
    out = {}
    for ck in DNA_CKPTS:
        f = f"{KEGG}/metrics_{ck.lower()}.json"
        m = load(f)["experiment1_2_per_arm"]
        out[ck] = []
        for arm, label in DNA_LADDER:
            v = m[arm]
            assert v["n_rows"] == 1449 and v["n_clusters"] == 708, (arm, v["n_rows"], v["n_clusters"])
            out[ck].append({"arm": arm, "label": label, "value": v["accuracy"],
                            "raw": v["accuracy_unclustered"]})
            record("dna", "a", model="BioReason", checkpoint=ck, group="all 1,449 rows",
                   series=label.replace("\n", " "), x=arm, value=v["accuracy"], n=v["n_rows"],
                   k=v["n_clusters"], source=f, key=f"experiment1_2_per_arm.{arm}.accuracy")
        print(f"[dna a] {ck} " + " ".join(f"{r['arm']}={r['value']:.3f}" for r in out[ck]))
    return out


def data_dna_rows():
    """Row-level strata and accuracy from genome_dependent_queries.json (written by
    analysis/dna_genome_dependent_rows.py), the strict 41-row minority split from
    minority_label_queries.json, and the Evo2 row probe from evo2_probe.json.
    """
    f = RD.DNA_GENOME_DEPENDENT_QUERIES
    d = load(f)
    fm = RD.DNA_MINORITY_LABEL_QUERIES
    mn = load(fm)
    fp = RD.DNA_GENOME_DEPENDENT_PROBE
    probe = load(fp)["evo2 only C=1"]
    assert d["n_dep"] + d["n_inv"] == d["n_rows"] and mn["n_dep"] == d["n_dep"]
    assert mn["n"]["minor"] == 41 and mn["n"]["major"] == 118 and mn["n"]["tied"] == 6

    out = {"n_dep": d["n_dep"], "n_inv": d["n_inv"], "texts_dep": d["n_dep_texts"],
           "genomes_dep": d["n_dep_genomes"], "ceiling_dep": d["ceiling_dep"],
           "n_genes_dep": len({g["gene"] for g in d["dep_groups"]}),
           "n_minor": mn["n"]["minor"], "acc": {}, "probe": probe}
    for ck in DNA_CKPTS:
        k = ck.lower()
        out["acc"][ck] = {}
        for arm, label, _ in DNA_ARMS:
            a_all = d[k]["acc"][arm]["dep_raw"]
            a_min = mn["acc"][k][arm]["minor"]
            assert abs(mn["acc"][k][arm]["dep"] - a_all) < 1e-9, (ck, arm)
            out["acc"][ck][arm] = {"dep": a_all, "minor": a_min}
            record("dna", "b", model="BioReason", checkpoint=ck, group="all 165 genome-dependent rows",
                   series=label, x=arm, value=a_all, n=d["n_dep"], source=f,
                   key=f"{k}.acc.{arm}.dep_raw")
            record("dna", "b", model="BioReason", checkpoint=ck, group="41 minority rows",
                   series=label, x=arm, value=a_min, n=mn["n"]["minor"],
                   k=mn["acc"][k][arm]["n_correct_minor"], source=fm, key=f"acc.{k}.{arm}.minor")
    for nm, val, n, key in (
            ("Evo2 probe, all genome-dependent rows", probe["acc"], d["n_dep"], "acc"),
            ("text-only predictor (training folds), all rows", probe["text_baseline"], d["n_dep"],
             "text_baseline"),
            ("Evo2 probe, labels permuted within text, all rows", probe["null_acc"][0], d["n_dep"],
             "null_acc[0]"),
            ("Evo2 probe, minority rows", probe["minor"], mn["n"]["minor"], "minor"),
            ("text-only predictor (training folds), minority rows", probe["text_baseline_minor"],
             mn["n"]["minor"], "text_baseline_minor"),
            ("Evo2 probe, labels permuted within text, minority rows", probe["null_minor"][0],
             mn["n"]["minor"], "null_minor[0]")):
        record("dna", "b", model="Evo2 row probe", series=nm, value=val, n=n,
               sd_over_folds=probe["acc_sd"] if key == "acc" else (probe["minor_sd"] if key == "minor" else ""),
               source=fp, key=f"evo2 only C=1.{key}")
    print(f"[dna b] dep {d['n_dep']} rows / {d['n_dep_texts']} texts / {d['n_dep_genomes']} genomes / "
          f"{out['n_genes_dep']} genes, ceiling {d['ceiling_dep']:.3f}; minority {mn['n']['minor']} rows")
    for ck in DNA_CKPTS:
        print(f"[dna b] {ck} " + " ".join(f"{a}: all={out['acc'][ck][a]['dep']:.3f} "
                                          f"minor={out['acc'][ck][a]['minor']:.3f}" for a, _, _ in DNA_ARMS))
    print(f"[dna b] probe {probe['acc']:.3f}+-{probe['acc_sd']:.3f} vs text {probe['text_baseline']:.3f}, "
          f"null {probe['null_acc'][0]:.3f}; minority {probe['minor']:.3f}+-{probe['minor_sd']:.3f} "
          f"vs text {probe['text_baseline_minor']:.3f}, null {probe['null_minor'][0]:.3f}")
    return out


# ------------------------------------------------------------------------------------------------
# singlecell a.
C2S_DS = ["immune1", "immune2", "immune3", "pancreas", "lung"]
C2S_MODELS = [("2b", "C2S-Scale 2B"), ("27b", "C2S-Scale 27B")]
DOSES = [("p10", 10), ("p25", 25), ("p50", 50), ("p75", 75), ("p100", 100)]
C2S_ARMS = [("top", "strongest DEGs", CELL.model, "o"),
            ("bottom", "weakest DEGs", CTRL[0], "v"),
            ("random", "random genes", CTRL[1], "s"),
            ("matched", "non-DEGs with similar expression", CTRL[2], "D")]


def data_c2s():
    f = RD.C2S_DEG_REMOVAL_SUMMARY
    d = load(f)
    cis = ci_json("ci_c2s.json")
    out = {}
    for m, name in C2S_MODELS:
        wt = {ds: d[f"{ds}|{m}"]["wt"] for ds in C2S_DS}
        ns = {ds: d[f"{ds}|{m}"]["n_cells"] for ds in C2S_DS}
        wt_ci = cis["singlecell_macro_drop"][m]["wt_macro"]["ci95"] if cis else None
        out[m] = {"wt_mean": st.mean(wt.values()), "wt_ci": wt_ci, "n_cells": sum(ns.values()), "series": {}}
        record("singlecell", "a", model=name,
               series="unmodified cell sentence (macro mean over 5 atlases)",
               value=out[m]["wt_mean"], n=sum(ns.values()), source=f, key="wt",
               ci_low=wt_ci[0] if wt_ci else "", ci_high=wt_ci[1] if wt_ci else "")
        for arm, label, _, _ in C2S_ARMS:
            pts = []
            pts.append((0, out[m]["wt_mean"], wt_ci))      # dose 0 = the unmodified cell sentence
            for dose, pct in DOSES:
                vals = [d[f"{ds}|{m}"]["doses"][dose].get(arm) for ds in C2S_DS]
                if any(v is None for v in vals):
                    continue                      # weakest-DEG arm stops at 75%
                acc = st.mean(vals)
                delta = st.mean(wt[ds] - v for ds, v in zip(C2S_DS, vals))
                ci = None
                if cis:
                    e = next(q for q in cis["singlecell_macro_acc"][m][arm] if q["dose"] == dose)
                    assert abs(e["plotted_value"] - acc) < 1e-9, (m, arm, dose, e["plotted_value"], acc)
                    ci = e["ci95"]
                pts.append((pct, acc, ci))
                record("singlecell", "a", model=name, series=label, x=pct, value=acc,
                       n=sum(ns.values()), source=f, key=f"mean over atlases of doses.{dose}.{arm}",
                       ci_low=ci[0] if ci else "", ci_high=ci[1] if ci else "")
                record("singlecell", "a (reference, not drawn)", model=name, series=label, x=pct,
                       value=delta, n=sum(ns.values()), source=f,
                       key=f"mean over atlases of wt - doses.{dose}.{arm}")
            out[m]["series"][arm] = pts
    return out


# ------------------------------------------------------------------------------------------------
# figure 1: BioReason-Pro -- RL checkpoint throughout (SFT is in the CSV and in the appendix tables)
# ------------------------------------------------------------------------------------------------
def draw_protein(ladder, fam, conflict, out_dir, stem):
    W = 5.5
    AXH = 1.25
    L_AX = 0.46
    TOP = 0.10
    TIT = 0.12                   # band for the panel letters; the panels carry no titles
    BOT = 0.94                   # 45-degree x ticks + x axis label + the shared legend
    H = TOP + TIT + AXH + BOT
    fig = plt.figure(figsize=(W, H))
    row_top = TOP + TIT
    a_w, b_w, c_w = 1.12, 0.98, 1.48
    gap_ab, gap_bc = 0.60, 0.76
    a_x = L_AX
    b_x = a_x + a_w + gap_ab
    c_x = b_x + b_w + gap_bc

    # ---- a: ESM3 intact vs shuffled as the text sources are removed
    ax = ax_in(fig, a_x, row_top, a_w, AXH)
    rows = ladder["RL"]
    xs = np.arange(len(rows))
    w = 0.38
    finish(ax, r"$F_{\max}$", ylim=(0, 1.18), yticks=np.arange(0, 1.01, 0.25), spine_top=1.0)
    center_ylabel(ax, 1.0)
    for i, r in enumerate(rows):
        ax.bar(xs[i] - w / 2, r["intact"], w, color=PROTEIN.model, edgecolor="black",
               linewidth=EDGE_LW)
        ax.bar(xs[i] + w / 2, r["shuffled"], w, color=PROTEIN.perturbed, edgecolor="black",
               linewidth=EDGE_LW)
        bar_label(ax, xs[i] - w / 2, r["intact"], f"{r['intact']:.3f}")
        bar_label(ax, xs[i] + w / 2, r["shuffled"], f"{r['shuffled']:.3f}")
    ax.set_xticks(xs)
    ax.set_xticklabels([r["label"] for r in rows], rotation=45, ha="right",
                       rotation_mode="anchor")
    ax.tick_params(axis="x", labelsize=6.0)
    ax.set_xlim(-0.62, len(rows) - 0.38)
    ax.set_xlabel("Text evidence sources", fontsize=TICK_FS)
    ax_a = ax
    panel_letter(fig, 0.02, TOP, "a")

    # ---- b: within-family AUROC, three properties, three arms
    ax = ax_in(fig, b_x, row_top, b_w, AXH)
    xs = np.arange(len(fam))
    w = 0.27
    finish(ax, "Mean per-family AUROC", ylim=(0, 1.20), yticks=np.arange(0, 1.01, 0.25),
           spine_top=1.0)
    center_ylabel(ax, 1.0)
    for i, r in enumerate(fam):
        for k, (name, v, ci, col) in enumerate(r["vals"]):
            xb = xs[i] + (k - 1) * w
            ax.bar(xb, v, w, color=col, edgecolor="black", linewidth=EDGE_LW)
            bar_label(ax, xb, v, f"{v:.3f}")
    ax.set_xticks(xs)
    ax.set_xticklabels([r["label"] for r in fam], rotation=45, ha="right",
                       rotation_mode="anchor")
    ax.tick_params(axis="x", labelsize=6.0)
    ax.set_xlim(-0.58, len(fam) - 0.42)  # 3 groups
    ax.set_xlabel("Functional property", fontsize=TICK_FS)
    ax_b = ax
    panel_letter(fig, b_x - 0.46, TOP, "b")

    # ---- c: evidence conflict
    ax = ax_in(fig, c_x, row_top, c_w, AXH)
    xs = np.arange(len(conflict))
    w = 0.6
    finish(ax, "Predictions matching\nprotein A's property", ylim=(0, 1.12),
           yticks=np.arange(0, 1.01, 0.25), spine_top=1.0)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    center_ylabel(ax, 1.0)
    for i, o in enumerate(conflict):
        ax.bar(xs[i], o["rate"], w, color=PROTEIN.model, edgecolor="black", linewidth=EDGE_LW)
        bar_label(ax, xs[i], o["rate"], f"{100 * o['rate']:.1f}", rot=0)
    ax.set_xticks(xs)
    ax.set_xticklabels([o["label"] for o in conflict], linespacing=1.15)
    ax.tick_params(axis="x", labelsize=6.0)
    ax.set_xlim(-0.55, len(conflict) - 0.45)
    ax.set_xlabel("Evidence source protein (A/B)", fontsize=TICK_FS)
    ax_c = ax
    panel_letter(fig, c_x - 0.52, TOP, "c")
    align_xlabels(fig, [ax_a, ax_b, ax_c])

    # one legend for the whole figure: the colour means the same thing in (a) and (b) -- light blue
    # the probe, blue the model with its real embedding, grey-blue the model with it destroyed
    fig.legend(handles=[Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="black", lw=EDGE_LW, label=l)
                        for l, c in (("ESM3 linear probe", PROTEIN.probe),
                                     ("BioReason-Pro, ESM3 intact", PROTEIN.model),
                                     ("BioReason-Pro, ESM3 shuffled", PROTEIN.perturbed))],
               loc="lower center", bbox_to_anchor=(0.5, 0.03 / H), ncol=3, frameon=False,
               fontsize=LEGEND_FS, handlelength=1.1, handletextpad=0.4, columnspacing=1.4)
    return emit(fig, stem, out_dir)


# ------------------------------------------------------------------------------------------------
# figure 2: BioReason
# ------------------------------------------------------------------------------------------------
def draw_dna(ladder, dna, out_dir, stem):
    W = 5.5
    A_H, B_H = 1.05, 1.15
    L_AX = 0.46
    TOP = 0.12
    TIT = 0.26
    GAP = 0.78                   # three-line x labels of row 1 + panel letter of row 2
    BOT = 0.50
    H = TOP + TIT + A_H + GAP + TIT + B_H + BOT
    fig = plt.figure(figsize=(W, H))
    left_x = L_AX
    row1_top = TOP + TIT
    row2_top = TOP + TIT + A_H + GAP + TIT
    LEG_A = 1 + 0.115 / A_H
    LEG_B = 1 + 0.115 / B_H
    ckpts = DNA_CKPTS

    # ---- a: accuracy on all 1,449 rows as the supplied sources are removed
    ax = ax_in(fig, left_x, row1_top, W - left_x - 0.16, A_H)
    xs = np.arange(len(DNA_LADDER))
    bw = 0.34
    finish(ax, "Accuracy, 1,449 rows", ylim=(0, 1.32), yticks=np.arange(0, 1.01, 0.25),
           spine_top=1.0)
    for ci, ck in enumerate(ckpts):
        col = DNA.model if ck == "SFT" else DNA.model_dark
        off = (-bw / 2) if ci == 0 else (bw / 2)
        for i, r in enumerate(ladder[ck]):
            ax.bar(xs[i] + off, r["value"], bw, color=col, edgecolor="black", linewidth=EDGE_LW)
            bar_label(ax, xs[i] + off, r["value"], f"{r['value']:.3f}")
    ax.set_xticks(xs)
    ax.set_xticklabels([lab for _, lab in DNA_LADDER])
    ax.tick_params(axis="x", which="major", pad=10, labelsize=6.3)
    ax.set_xticks(np.concatenate([xs - bw / 2, xs + bw / 2]), minor=True)
    ax.set_xticklabels([ckpts[0]] * len(xs) + [ckpts[1]] * len(xs), minor=True)
    ax.tick_params(axis="x", which="minor", length=0, labelsize=ANNOT_FS, pad=1.5)
    ax.set_xlim(-0.6, len(xs) - 0.4)
    facet_title(ax, "KEGG disease prediction, all 1,449 questions, by input sources supplied")
    legend_upper(ax, [Rectangle((0, 0), 1, 1, facecolor=DNA.model, edgecolor="black", lw=EDGE_LW,
                                label="SFT"),
                      Rectangle((0, 0), 1, 1, facecolor=DNA.model_dark, edgecolor="black", lw=EDGE_LW,
                                label="RL")], 2, LEG_A, loc="lower right", anchor_x=1.0)
    panel_letter(fig, 0.02, TOP, "a")

    # ---- b: availability and use on one accuracy axis, two facets sharing y
    pr = dna["probe"]
    gap = 0.30
    w_each = (W - left_x - 0.16 - gap) / 2
    gx = np.array([0.0, 1.05, 2.10])          # probe, SFT, RL
    bw = 0.27
    HATCH = "////"
    facets = [("all", f"all {dna['n_dep']} genome-dependent rows", dna["ceiling_dep"],
               (pr["acc"], pr["null_acc"][0], pr["text_baseline"]), "dep"),
              ("minor", f"the {dna['n_minor']} minority rows", 0.0,
               (pr["minor"], pr["null_minor"][0], pr["text_baseline_minor"]), "minor")]
    for fi, (fk, title, ceiling, probe_vals, mkey) in enumerate(facets):
        x0 = left_x + fi * (w_each + gap)
        ax = ax_in(fig, x0, row2_top, w_each, B_H)
        finish(ax, "Row accuracy" if fi == 0 else None, ylim=(0, 1.34),
               yticks=np.arange(0, 1.01, 0.25), spine_top=1.0)
        # probe group, hatched: real labels / labels permuted / text-only predictor
        for bi, ((_, _, col), v) in enumerate(zip(DNA_ARMS, probe_vals)):
            xb = gx[0] + (bi - 1) * bw
            ax.bar(xb, v, bw, color=col, edgecolor="black", linewidth=EDGE_LW, hatch=HATCH)
            bar_label(ax, xb, v, f"{v:.3f}", mask=ceiling > 0)
        # model groups, solid: original / scrambled / removed genome
        for gi, ck in enumerate(ckpts):
            for bi, (arm, _, col) in enumerate(DNA_ARMS):
                v = dna["acc"][ck][arm][mkey]
                xb = gx[gi + 1] + (bi - 1) * bw
                ax.bar(xb, v, bw, color=col, edgecolor="black", linewidth=EDGE_LW)
                bar_label(ax, xb, v, f"{v:.3f}", mask=ceiling > 0)
        if ceiling > 0:
            ax.plot([gx[0] - 1.6 * bw, gx[-1] + 1.6 * bw], [ceiling, ceiling], color="black",
                    lw=0.7, ls=":", zorder=4)
        ax.set_xticks(gx)
        ax.set_xticklabels(["Evo2 linear\nprobe", "BioReason\nSFT", "BioReason\nRL"])
        ax.tick_params(axis="x", labelsize=6.3)
        ax.set_xlim(gx[0] - 0.62, gx[-1] + 0.62)
        if fi == 1:
            ax.tick_params(axis="y", labelleft=False)
        facet_title(ax, title)
        panel_letter(fig, 0.02 if fi == 0 else x0 - 0.20, row2_top - TIT, "b" if fi == 0 else "")
    # one legend for both facets: colour = genome information, hatch = probe
    handles = [Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="black", lw=EDGE_LW, label=l)
               for _, l, c in DNA_ARMS]
    handles.append(Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", lw=EDGE_LW,
                             hatch=HATCH, label="hatched: linear probe on Evo2; solid: BioReason"))
    handles.append(Line2D([], [], color="black", lw=0.7, ls=":",
                          label=f"text-only ceiling ({dna['ceiling_dep']:.3f})"))
    ax0 = fig.axes[1]
    ax0.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, LEG_B), ncol=3,
               frameon=False, fontsize=LEGEND_FS, borderaxespad=0.0, handlelength=1.0,
               handletextpad=0.4, columnspacing=0.9, labelspacing=0.3)
    return emit(fig, stem, out_dir)


# ------------------------------------------------------------------------------------------------
# figure 3: C2S-Scale
# ------------------------------------------------------------------------------------------------
def draw_c2s(c2s, out_dir, stem):
    """Figure 4 right: C2S-Scale 2B (a) and 27B (b) accuracy against the fraction of genes removed, one
    line per removal set, with the unmodified cell sentence at dose 0 and 95% bootstrap bands."""
    # right-hand legend layout)
    AXH = 1.45
    L_AX = 0.62
    TOP = 0.10
    TIT = 0.20
    BOT = 0.60 + 0.52           # x ticks + a two-line x label, then a three-row legend
    e_gap = 0.26
    e_w = 1.45
    W = L_AX + 2 * e_w + e_gap + 0.12
    H = TOP + TIT + AXH + BOT
    fig = plt.figure(figsize=(W, H))
    left_x = L_AX
    row_top = TOP + TIT
    ymax = max((ci[1] if ci else v) for m in c2s.values() for pts in m["series"].values() for _, v, ci in pts)
    ytop = 0.5 if ymax <= 0.5 else np.ceil(ymax * 10) / 10
    handles = []
    for j, (m, name) in enumerate(C2S_MODELS):
        ax = ax_in(fig, left_x + j * (e_w + e_gap), row_top, e_w, AXH)
        for arm, label, color, marker in C2S_ARMS:
            pts = c2s[m]["series"][arm]
            if all(ci for _, _, ci in pts):          # 95% bootstrap band (cells resampled within atlas)
                ax.fill_between([p for p, _, _ in pts], [ci[0] for _, _, ci in pts], [ci[1] for _, _, ci in pts],
                                color=color, alpha=0.25, lw=0, zorder=1)
            ax.plot([p for p, _, _ in pts], [v for _, v, _ in pts], color=color, lw=0.9, marker=marker,
                    ms=3.4, mec="black", mew=EDGE_LW, zorder=3 if arm == "top" else 2)
            if j == 0:
                handles.append(Line2D([], [], color=color, lw=0.9, marker=marker, ms=3.4,
                                      mec="black", mew=EDGE_LW, label=label))
            if arm == "top":
                for p, v, ci in pts:
                    if p:                            # the dose-0 point is labelled once, below
                        ax.text(p, (ci[1] if ci else v) + 0.012, f"{v:.3f}", ha="center", va="bottom",
                                fontsize=ANNOT_FS)
        wt, wt_ci = c2s[m]["wt_mean"], c2s[m]["wt_ci"]
        ax.plot([0], [wt], marker="o", ms=4.4, mfc="white", mec="black", mew=0.7, ls="none", zorder=6)
        ax.text(0, (wt_ci[1] if wt_ci else wt) + 0.012, f"{wt:.3f}", ha="left", va="bottom", fontsize=ANNOT_FS)
        if j == 0:
            handles.insert(0, Line2D([], [], marker="o", ms=4.4, mfc="white", mec="black", mew=0.7, ls="none",
                                     label="unmodified cell sentence"))
        ax.set_xticks([0] + [p for _, p in DOSES])
        ax.set_xticklabels([str(p) for p in [0] + [p for _, p in DOSES]])   # no '%'; the x label carries it
        ax.set_xlim(0, 104)
        sns.despine(ax=ax)
        ax.set_ylim(0, ytop * 1.14)
        ax.set_yticks(np.arange(0, ytop + 1e-9, 0.1))
        ax.spines["left"].set_bounds(0, ytop)        # the two spines meet at the origin
        ax.spines["bottom"].set_bounds(0, 100)
        ax.set_xlabel("Genes removed\n(% of the cell's DEG set)", linespacing=1.15)
        if j == 0:
            ax.set_ylabel("Cell type annotation accuracy")
            center_ylabel(ax, ytop)              # centre on the tick range (0-ytop), not the headroom-padded
            # ylim
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_title(name, fontsize=TICK_FS, fontweight="normal", pad=2.5)   # unbold
        panel_letter(fig, left_x + j * (e_w + e_gap) - (0.52 if j == 0 else 0.20), TOP, "ab"[j])
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=((left_x + e_w + e_gap / 2) / W, 1 - (row_top + AXH + 0.58) / H),
               ncol=2, frameon=False, fontsize=LEGEND_FS, handlelength=1.6, handletextpad=0.5, labelspacing=0.3,
               columnspacing=1.2)
    return emit(fig, stem, out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=OUT_DEFAULT)
    ap.add_argument("--csv", default=None, help="defaults to <out_dir>/fig_rq1_numbers.csv")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    csv_path = a.csv or os.path.join(a.out_dir, "fig_rq1_numbers.csv")

    ladder = data_ladder()
    fam = data_family_auroc()
    conflict = data_conflict()
    dna_ladder = data_dna_ladder()
    dna = data_dna_rows()
    c2s = data_c2s()
    for ck, rows in ladder.items():
        print(f"[protein a] {ck}", [(r["label"].replace(chr(10), " "), round(r["intact"], 3),
                                    round(r["shuffled"], 3)) for r in rows])
    for m, v in c2s.items():
        print(f"[singlecell a] {m} wt={v['wt_mean']:.3f}",
              {k: [(p, round(x, 3)) for p, x, _ in pts] for k, pts in v["series"].items()})

    init_print_style()
    draw_protein(ladder, fam, conflict, a.out_dir, "fig_rq1_protein")
    draw_dna(dna_ladder, dna, a.out_dir, "fig_rq1_dna")
    draw_c2s(c2s, a.out_dir, "fig_rq1_singlecell")

    keys = ["figure", "panel", "model", "checkpoint", "group", "series", "x", "value", "n", "k",
            "ci_low", "ci_high", "sd_over_folds", "source", "key"]
    with open(csv_path, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=keys)
        wtr.writeheader()
        for r in ROWS:
            wtr.writerow({k: r.get(k, "") for k in keys})
    print(f"wrote {csv_path} ({len(ROWS)} rows)")


if __name__ == "__main__":
    main()

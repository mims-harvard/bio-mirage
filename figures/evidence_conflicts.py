#!/usr/bin/env python
"""Draws Figure 3 (fig_rq1_swapped, "Evidence conflicts from unmodified queries") for all six models.

    python figures/evidence_conflicts.py [--out_dir <dir>]

Each evidence conflict pairs the foundation model representation of entity A with the text of
entity B, and each panel shows the share of answers that match A, match B, or match neither. Panels
a (BioReason RL), b (BioReason-Pro RL) and d (C2S-Scale 27B) come from the same loaders and drawing
code as perturbations_and_conflicts.py. Panels c (Prot2Text-V2) and e (CellWhisperer) keep only the
pairs in which each member, given its own complete input, is closer to its own reference:

  c  Prot2Text-V2: the description generated from the full name, taxon and sequence of each member
     is closer to its own reference function (the `aligned_gate_pass|biobert` block of
     prot2text_analysis.json, 2,000 bootstrap draws over pairs, seed 0).
  e  CellWhisperer: with its own transcriptome embedding and its own 1,000-gene text, each member
     prefers its own cell type by per-token log-likelihood. Rates and 95% intervals are computed
     here from cellwhisperer_conflict_predictions.parquet and cellwhisperer_conflict_per_pair.parquet
     (multinomial bootstrap over pairs, 2,000 draws, seed 0, one pair carrying both directions).

Every value of panels c and e is asserted to 1e-6 against evidence_conflict_filters.csv (TABLE2)
before anything is drawn. Reads these files (protein/prot2text_v2, single_cell/cellwhisperer and
evidence_conflict_filters.csv) and the inputs of perturbations_and_conflicts.py under
INPUT_USE_RESULTS_DIR. Writes fig_rq1_swapped (pdf, png, svg) and fig_rq1_swapped_numbers.csv to
outputs/figures, or to --out_dir.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figure_style import init_print_style, emit as _emit  # noqa: E402

import rq1_data as M                       # noqa: E402
import perturbation_panels as P            # noqa: E402
import evidence_swap_panels as S           # noqa: E402
import combined_panels as Cb               # noqa: E402
import perturbations_and_conflicts as BM   # noqa: E402

STEM = "fig_rq1_swapped"
OUT_DEFAULT = M.OUT_DEFAULT
P2T_ANALYSIS = f"{M.RD.PROT2TEXT}/results/prot2text_analysis.json"
CWPC = f"{M.RD.CELLWHISPERER}/analysis_default"
CW_CONF = f"{CWPC}/cellwhisperer_conflict_predictions.parquet"
CW_PER = f"{CWPC}/cellwhisperer_conflict_per_pair.parquet"
TABLE2 = M.RD.EVIDENCE_CONFLICT_FILTERS

BOOT_N, BOOT_SEED, SWAP_K = 2000, 0, 1000
TOL = 1e-6


# ------------------------------------------------------------------------------------------------
# the filter table the two redrawn panels must agree with
# ------------------------------------------------------------------------------------------------
def expected():
    """{model: row} from evidence_conflict_filters.csv, the values panels c and e must reproduce."""
    with open(TABLE2) as fh:
        return {r["model"]: r for r in csv.DictReader(fh)}


def check(label, got, want, n_got=None, n_want=None):
    assert abs(got - want) <= TOL, f"{label}: {got!r} != {want!r} (|d| = {abs(got - want):.3g} > {TOL})"
    if n_got is not None:
        assert int(n_got) == int(n_want), f"{label}: N {n_got} != {n_want}"
    print(f"  OK  {label:52s} {got!r:22s} (filter table {want!r})")


# ------------------------------------------------------------------------------------------------
# panel c: Prot2Text-V2, pairs passing aligned_gate_pass (read, not recomputed)
# ------------------------------------------------------------------------------------------------
def prot2text_common(exp):
    ana = M.load(P2T_ANALYSIS)
    blk = ana["experiment_B"]["aligned_gate_pass|biobert"]
    eps = ana["epsilon"]
    e = blk[f"eps_{eps}"]
    assert blk["gate"] == "aligned_gate_pass" and blk["metric"] == "biobert", blk["gate"]
    row = exp["Prot2Text-V2 (released)"]
    print("[assert] panel c, Prot2Text-V2, experiment_B['aligned_gate_pass|biobert']"
          f"['eps_{eps}'] vs evidence_conflict_filters.csv")
    check("follows A (sequence)", e["sequence_follow_rate"]["mean"], float(row["follows_A"]),
          blk["n_conflicts"], row["n_kept"])
    check("follows B (name)", e["text_follow_rate"]["mean"], float(row["follows_B"]))
    check("neither (|d| < eps)", e["ambiguous_rate"]["mean"], float(row["other"]))

    sw = {"A": e["sequence_follow_rate"]["mean"], "B": e["text_follow_rate"]["mean"],
          "neither": e["ambiguous_rate"]["mean"],
          "ci": {"A": e["sequence_follow_rate"]["ci95"], "B": e["text_follow_rate"]["ci95"],
                 "neither": e["ambiguous_rate"]["ci95"]},
          "n": blk["n_conflicts"], "n_pairs": blk["n_pairs"], "epsilon": eps,
          "label": "Full Name(B)\nESM2(A)",
          "shift": blk["mean_preference_shift_from_aligned"],
          "toward": blk["mean_preference_toward_sequence"]}
    grp = ("ESM2(A) + Full Name(B), both directions, aligned-gate pairs "
           "(full-prompt caption of each member closer to its own function)")
    base = f"experiment_B.aligned_gate_pass|biobert.eps_{eps}"
    for which, name in (("A", "sequence"), ("B", "text"), ("neither", "ambiguous")):
        M.record("prot2text", "b", model=T_MODEL, checkpoint="released", group=grp,
                 series=f"matching {which} ({name})", x="share of conflicts", value=sw[which],
                 n=sw["n"], ci_low=sw["ci"][which][0], ci_high=sw["ci"][which][1],
                 source=P2T_ANALYSIS,
                 key=f"{base}.{name}_follow_rate" if which != "neither" else f"{base}.ambiguous_rate")
    M.record("prot2text", "b", model=T_MODEL, checkpoint="released", group=grp,
             series="mean preference shift from aligned prompt, toward sequence source",
             x="BioBERT F1 difference", value=sw["shift"]["mean"], n=sw["n"],
             ci_low=sw["shift"]["ci95"][0], ci_high=sw["shift"]["ci95"][1], source=P2T_ANALYSIS,
             key="experiment_B.aligned_gate_pass|biobert.mean_preference_shift_from_aligned")
    M.record("prot2text", "b", model=T_MODEL, checkpoint="released", group=grp,
             series="pairs kept by the aligned gate (of 700 proposed)", x="pairs",
             value=blk["n_pairs"], n=sw["n"], source=P2T_ANALYSIS,
             key="experiment_B.aligned_gate_pass|biobert.n_pairs")
    return sw


# ------------------------------------------------------------------------------------------------
# panel e: CellWhisperer, pairs passing the aligned gate (computed here: results.json has no such block)
# ------------------------------------------------------------------------------------------------
class Boot:
    """Pooled bootstrap estimator of the CellWhisperer analysis: multinomial weights over the pooled
    pairs."""

    def __init__(self, n, b, rng):
        self.n = n
        self.w = rng.multinomial(n, np.full(n, 1.0 / n), size=b).astype(float)

    def mean(self, x):
        x = np.asarray(x, float)
        return float(x.mean()), (self.w @ x) / self.n


def ci95(bs):
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def cellwhisperer_common(exp, k=SWAP_K):
    conf = pd.read_parquet(CW_CONF)
    per = pd.read_parquet(CW_PER)
    ck = conf[conf["k"] == k]
    piv = ck.pivot_table(index="pair_id", columns="condition", values="preference_A_minus_B_ppl")
    gate = (piv["aligned_A"] > 0) & (piv["aligned_B"] < 0)
    ids = sorted(piv.index[gate])
    sub = per[per["k"] == k].set_index("pair_id").reindex(ids)
    assert sub.notna()["emb_ppl"].all(), "a gated pair has no per-pair conflict row"
    n_pairs_all = int(len(piv))

    boot = Boot(len(ids), BOOT_N, np.random.RandomState(BOOT_SEED))
    pt, cis = {}, {}
    for m in ("emb_ppl", "text_ppl", "amb_ppl", "pref_text_ppl",
              "embedding_effect_ppl", "text_effect_ppl"):
        v, bs = boot.mean(sub[m].to_numpy())
        pt[m], cis[m] = v, ci95(bs)

    row = exp["CellWhisperer (default)"]
    print("[assert] panel e, CellWhisperer, aligned gate recomputed from the conflict parquets "
          "vs evidence_conflict_filters.csv")
    check("follows A (embedding)", pt["emb_ppl"], float(row["follows_A"]), 2 * len(ids), row["n_kept"])
    check("follows B (gene text)", pt["text_ppl"], float(row["follows_B"]))
    check("neither (exact perplexity tie)", pt["amb_ppl"], float(row["other"]))
    assert 2 * n_pairs_all == int(row["n_total"]), f"{2 * n_pairs_all} != {row['n_total']}"

    sw = {"A": pt["emb_ppl"], "B": pt["text_ppl"], "neither": pt["amb_ppl"],
          "ci": {"A": cis["emb_ppl"], "B": cis["text_ppl"], "neither": cis["amb_ppl"]},
          "n": 2 * len(ids), "n_pairs": len(ids), "k": k,
          "label": f"Gene text k={k} (B)\nEmbedding (A)",
          "toward": pt["pref_text_ppl"], "toward_ci": cis["pref_text_ppl"],
          "emb_effect": pt["embedding_effect_ppl"], "emb_effect_ci": cis["embedding_effect_ppl"],
          "text_effect": pt["text_effect_ppl"], "text_effect_ci": cis["text_effect_ppl"]}
    grp = (f"Embedding(A) + {k}-gene text(B), both directions, aligned-gate pairs "
           "(each member's own embedding + own gene text prefers its own type, per-token log-likelihood)")
    src = f"{CW_CONF} + {CW_PER}"
    for which, name, m in (("A", "embedding", "emb_ppl"), ("B", "gene text", "text_ppl"),
                           ("neither", "exact perplexity tie", "amb_ppl")):
        M.record("cellwhisperer", "b", model=CW_MODEL, checkpoint="default (released chat model)",
                 group=grp, series=f"matching {which} ({name}), lower perplexity",
                 x="share of conflicts", value=sw[which], n=sw["n"], k=k,
                 ci_low=sw["ci"][which][0], ci_high=sw["ci"][which][1], source=src,
                 key=f"RECOMPUTED aligned_gate_ppl_k{k}.{m}")
    for series, vk, ck_, m in (
            ("mean per-token log-likelihood preference toward the gene-text source", "toward", "toward_ci", "pref_text_ppl"),
            ("2x2: per-token shift of log p(A)-log p(B) from swapping the transcriptome", "emb_effect", "emb_effect_ci", "embedding_effect_ppl"),
            ("2x2: per-token shift of log p(A)-log p(B) from swapping the gene text", "text_effect", "text_effect_ci", "text_effect_ppl")):
        M.record("cellwhisperer", "b", model=CW_MODEL, checkpoint="default (released chat model)",
                 group=grp, series=series, x="nats per token", value=sw[vk], n=sw["n"], k=k,
                 ci_low=sw[ck_][0], ci_high=sw[ck_][1], source=src,
                 key=f"RECOMPUTED aligned_gate_ppl_k{k}.{m}")
    M.record("cellwhisperer", "b", model=CW_MODEL, checkpoint="default (released chat model)",
             group=grp, series=f"pairs kept by the aligned gate (of {n_pairs_all:,} frozen pairs)",
             x="pairs", value=len(ids), n=sw["n"], k=k, source=src,
             key=f"RECOMPUTED aligned_gate_ppl_k{k}.n_pairs")
    return sw


T_MODEL = "Prot2Text-V2"
CW_MODEL = "CellWhisperer"


def replace_swapped(figure, i0, new_loader, *a):
    """Drops the panel b rows the model module just recorded for <figure> (they use its own pair
    filter) and records the rows of the aligned filter in their place, so the CSV keeps its panel
    order."""
    kept = [r for r in M.ROWS[i0:] if not (r["figure"] == figure and r["panel"] == "b")]
    n_dropped = len(M.ROWS) - i0 - len(kept)
    del M.ROWS[i0:]
    M.ROWS.extend(kept)
    sw = new_loader(*a)
    print(f"[csv] {figure}: replaced {n_dropped} published panel-b rows with "
          f"{len([r for r in M.ROWS if r['figure'] == figure and r['panel'] == 'b'])}")
    return sw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=OUT_DEFAULT)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    exp = expected()

    # the same call sequence as perturbations_and_conflicts.main, so panels a, b and d and the rows
    # in M.ROWS are the same as there
    dna, prot_ladder, c2s = P.data_dna_rl(), M.data_ladder(), P.data_c2s_atlas()
    specs = S.panel_specs(S.data_dna_swap(), S.data_protein_abn(M.data_conflict()), S.data_sc_swap())
    controls = []
    for col, (name, mod, loader_name, spec) in enumerate((
            ("ChatNT", Cb.N, "data_chatnt", dict(xlabel_destroyed="Native tasks", xlabel_swapped="DNA source",
                                                 unit_destroyed="queries", unit_swapped="gated pairs")),
            ("Prot2Text-V2", Cb.T, "data_prot2text", dict(xlabel_destroyed="Text with ESM2", xlabel_swapped="Protein source",
                                                          unit_destroyed="proteins", unit_swapped="conflicts")),
            ("CellWhisperer", Cb.CW, "data_cellwhisperer", dict(xlabel_destroyed="Atlas", xlabel_swapped="Cell source",
                                                                unit_destroyed="cells", unit_swapped="conflicts")))):
        if mod is None:
            raise SystemExit(f"{name}: module missing")
        i0 = len(M.ROWS)
        data = getattr(mod, loader_name)()
        if data is None:
            raise SystemExit(f"{name}: loader returned None (numbers missing)")
        rows, sw = data
        if name == "Prot2Text-V2":
            sw = replace_swapped("prot2text", i0, prot2text_common, exp)
        elif name == "CellWhisperer":
            sw = replace_swapped("cellwhisperer", i0, cellwhisperer_common, exp)
        controls.append(dict(module=mod, rows=rows, sw=sw, modality_col=col,
                             csv_figure=name.lower().replace("-", "").replace("prot2textv2", "prot2text"), **spec))

    init_print_style()
    BM.emit = lambda fig, stem, out_dir: _emit(fig, STEM, out_dir)   # same drawing code, new stem
    BM.draw_swapped(specs, controls, a.out_dir)

    keys = ["figure", "panel", "model", "checkpoint", "group", "series", "x", "value", "n", "k",
            "ci_low", "ci_high", "sd_over_folds", "source", "key"]
    panel_of = {("swap", "a"): "a", ("protein", "c"): "b", ("swap", "b"): "b", ("prot2text", "b"): "c",
                ("swap", "c"): "d", ("cellwhisperer", "b"): "e"}
    rows = [{**{k: r.get(k, "") for k in keys}, "figure": STEM, "panel": panel_of[(r["figure"], r["panel"])]}
            for r in M.ROWS if (r["figure"], r["panel"]) in panel_of]
    p = os.path.join(a.out_dir, f"{STEM}_numbers.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {p} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

"""Computes the CellWhisperer results per atlas, as the mean over atlases and pooled over pairs,
with bootstrap intervals. Last step of four (build_cells, embed_cells, score_llm, analyze). Runs on
CPU in CellWhisperer's pixi environment.

Reads per atlas under --root (default single_cell/cellwhisperer under $INPUT_USE_RESULTS_DIR):
cells.parquet, pairs.parquet, cells_qc.json, embed_qc.json, <model>__expA.jsonl, <model>__expB.jsonl
and base__textgate.jsonl. Writes to --out_dir (default <root>/analysis_<model>): results.json, which
figures/cellwhisperer.py reads for Fig. 2f and Fig. 3e, and the tables cellwhisperer_cells,
cellwhisperer_embedding_conditions, cellwhisperer_ppl_quantile, cellwhisperer_pairs,
cellwhisperer_conflict_predictions and cellwhisperer_conflict_per_pair (.parquet),
cellwhisperer_summary.csv and cellwhisperer_qc.json.

Definitions:
  1 - perplexity quantile  Fig. 2f. For cell i with label L_i, the share of cells j in the atlas with
                           L_j != L_i for which the per-token log-likelihood of "This cell is a L_i"
                           under Z_i (intact or shuffled) exceeds that under the intact Z_j (chance
                           0.5). Stored as auc_intact, auc_shuffle and auc_delta_shuffle.
  evidence conflicts       Fig. 3e. Pairs whose two cells each give their own answer a higher
                           per-token log-likelihood than the partner's answer under their own intact
                           Z_CW (set embgate_ppl). Z_CW of one cell is combined with the top-1,000
                           gene text of the other, and the answer with the lower perplexity is
                           counted as following Z_CW (emb_ppl), the gene text (text_ppl) or neither
                           on an exact tie (amb_ppl).
  pairwise preference      logp_sum(answer A | prompt) - logp_sum(answer B | prompt), summed over
                           answer tokens. Suffix _ppl: the same with per-token means.
  multi-class top-1        argmax over the atlas's answers of the per-token log-likelihood
                           (acc_ppl); the summed version is acc_sum.
  embedding gate           pref_A(Z_A) > 0 and pref_A(Z_B) < 0 on the representation-only prompt.
  text gate (k)            base Mistral with no Z_CW: pref_A(gene text of A) > 0 and pref_A(gene
                           text of B) < 0. Set `common` requires both gates at every k.
  follow_class             sign of logp(answer of the gene text source) - logp(answer of the Z_CW
                           source) in a conflict prompt: > 0 text, < 0 embedding, 0 ambiguous.
  bootstrap                2,000 resamples, seed 0. The unit is the cell for the representation-only
                           prompt and the pair, with both directions, for conflicts. The mean over
                           atlases is unweighted.

    python input_use/models/cellwhisperer_driver/analyze.py [--model default] [--out_dir DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    ATLASES, BOOT_N, BOOT_SEED, KS, OUT, SHUFFLE_SEEDS, dump_json, read_jsonl)

SHUF = [f"shuffle_s{s}" for s in SHUFFLE_SEEDS]


class Boot:
    def __init__(self, n, b, rng):
        self.n = n
        self.w = rng.multinomial(n, np.full(n, 1.0 / n), size=b).astype(float) if n > 0 else np.zeros((b, 0))

    def mean(self, x):
        x = np.asarray(x, float)
        if self.n == 0:
            return float("nan"), np.full(self.w.shape[0], np.nan)
        return float(x.mean()), (self.w @ x) / self.n


def ci(bs):
    bs = np.asarray(bs, float)
    bs = bs[np.isfinite(bs)]
    if len(bs) == 0:
        return [float("nan"), float("nan")]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def load_atlas(atlas, model, root):
    d = root / atlas
    cells = pd.read_parquet(d / "cells.parquet")
    pairs = pd.read_parquet(d / "pairs.parquet")
    A = pd.DataFrame(read_jsonl(d / f"{model}__expA.jsonl"))
    B = pd.DataFrame(read_jsonl(d / f"{model}__expB.jsonl"))
    T = pd.DataFrame(read_jsonl(d / "base__textgate.jsonl"))
    eq = json.load(open(d / "embed_qc.json"))
    return cells, pairs, A, B, T, eq


def exp_a(atlas, cells, A, rng):
    """Multi-class scores per (cell, condition) + the swap rows derived from intact."""
    labels = sorted(A["candidate"].unique())
    A = A.copy()
    key = ["cell_id", "condition"]
    # top-1 by per-token mean (perplexity) and by sequence sum; margin of correct vs best other
    def per_group(g):
        g = g.set_index("candidate")
        corr = g.index[g["is_correct"]].tolist()
        assert len(corr) == 1, corr
        c = corr[0]
        others = g.drop(index=c)
        return pd.Series({
            "correct_label": c,
            "top1_ppl": g["logp_mean"].idxmax(), "top1_sum": g["logp_sum"].idxmax(),
            "correct_logp_sum": g.loc[c, "logp_sum"], "correct_logp_mean": g.loc[c, "logp_mean"],
            "margin_sum": g.loc[c, "logp_sum"] - others["logp_sum"].max(),
            "margin_mean": g.loc[c, "logp_mean"] - others["logp_mean"].max(),
            "emb_hash": g["emb_hash"].iloc[0], "n_candidates": len(g),
            "within_released_cap": bool(g["within_released_cap"].all()),
        })
    G = A.groupby(key).apply(per_group).reset_index()
    G["acc_ppl"] = (G["top1_ppl"] == G["correct_label"]).astype(float)
    G["acc_sum"] = (G["top1_sum"] == G["correct_label"]).astype(float)
    # swap rows: the cell's prompt with the swap partner's intact embedding == partner's intact row
    intact = G[G["condition"] == "intact"].set_index("cell_id")
    sw = cells[["cell_id", "swap_source_cell_id", "canonical_label"]].copy()
    src = intact.loc[sw["swap_source_cell_id"].to_numpy()]
    swap_rows = pd.DataFrame({
        "cell_id": sw["cell_id"].to_numpy(), "condition": "swap",
        "correct_label": sw["canonical_label"].to_numpy(),            # the ORIGINAL cell's label
        "top1_ppl": src["top1_ppl"].to_numpy(), "top1_sum": src["top1_sum"].to_numpy(),
        "correct_logp_sum": np.nan, "correct_logp_mean": np.nan, "margin_sum": np.nan, "margin_mean": np.nan,
        "emb_hash": src["emb_hash"].to_numpy(), "n_candidates": src["n_candidates"].to_numpy(),
        "within_released_cap": src["within_released_cap"].to_numpy(),
        "source_cell_id": sw["swap_source_cell_id"].to_numpy(), "source_label": src["correct_label"].to_numpy(),
    })
    swap_rows["acc_ppl"] = (swap_rows["top1_ppl"] == swap_rows["source_label"]).astype(float)  # pred == label_B
    swap_rows["acc_sum"] = (swap_rows["top1_sum"] == swap_rows["source_label"]).astype(float)
    G["source_cell_id"] = G["cell_id"]
    G["source_label"] = G["correct_label"]
    G = pd.concat([G, swap_rows], ignore_index=True)
    G.insert(0, "atlas", atlas)
    G["shuffle_seed"] = G["condition"].map(lambda c: int(c.split("_s")[1]) if "_s" in c else None)

    # atlas-level statistics with cell bootstrap
    cells_order = list(cells["cell_id"])
    piv = {m: G[G["condition"] != "swap"].pivot(index="cell_id", columns="condition", values=m).loc[cells_order]
           for m in ["acc_ppl", "acc_sum", "margin_sum", "margin_mean"]}
    boot = Boot(len(cells_order), BOOT_N, rng)
    st = {"n_cells": len(cells_order), "n_labels": len(labels), "chance_uniform": 1.0 / len(labels)}
    bsd = {}
    for m, P in piv.items():
        pt, bs = boot.mean(P["intact"]); st[f"{m}_intact"] = pt; st[f"{m}_intact_ci"] = ci(bs); bsd[f"{m}_intact"] = bs
        shuf = P[SHUF].mean(1)
        pt, bs2 = boot.mean(shuf); st[f"{m}_shuffle"] = pt; st[f"{m}_shuffle_ci"] = ci(bs2); bsd[f"{m}_shuffle"] = bs2
        st[f"{m}_shuffle_per_seed"] = [float(P[c].mean()) for c in SHUF]
        st[f"{m}_delta_shuffle"] = float(pt and (st[f"{m}_intact"] - pt)); st[f"{m}_delta_shuffle_ci"] = ci(bs - bs2)
        bsd[f"{m}_delta_shuffle"] = bs - bs2
        pt, bs3 = boot.mean(P["rankonly_s0"]); st[f"{m}_rankonly"] = pt; st[f"{m}_rankonly_ci"] = ci(bs3)
        st[f"{m}_delta_rankonly"] = st[f"{m}_intact"] - pt; st[f"{m}_delta_rankonly_ci"] = ci(bs - bs3)
        bsd[f"{m}_rankonly"] = bs3
    sw = swap_rows.set_index("cell_id").loc[cells_order]
    pt, bs = boot.mean(sw["acc_ppl"]); st["swap_follow_rate"] = pt; st["swap_follow_rate_ci"] = ci(bs); bsd["swap_follow_rate"] = bs
    st["frac_within_released_cap"] = float(G["within_released_cap"].mean())
    return G, st, bsd, intact


def ppl_quantile_stats(cells, A, rng):
    """1 - perplexity quantile (Fig. 2f) and the perplexity ratio, per cell and Z_CW condition.

    For cell i with canonical label L_i, the correct answer's per-token log-likelihood under its own
    embedding (condition c) is compared with the same answer under the intact embedding of every
    other cell j whose label differs (CellWhisperer's own evaluation draws 30 such cells; here all
    are used):
        auc_c(i)      = share of j with logp_mean(L_i | Z_i^c) > logp_mean(L_i | Z_j)   (chance 0.5)
        quantile_c(i) = 1 - auc_c(i)                                                    (lower is better)       
        log2 ratio    = log2( mean_j ppl(L_i | Z_j) / ppl(L_i | Z_i^c) )                (CellWhisperer's log_mean_perplexity.ratio)
    This is invariant to the answer string's length, unlike a top-1 argmax over labels of very
    different lengths."""
    labels = sorted(A["candidate"].unique())
    order = list(cells["cell_id"])
    Mi = A[A["condition"] == "intact"].pivot(index="cell_id", columns="candidate", values="logp_mean").loc[order, labels]
    lab = cells.set_index("cell_id").loc[order, "canonical_label"].to_numpy()
    out = {}
    for cond in ["intact"] + SHUF + ["rankonly_s0"]:
        Mc = A[A["condition"] == cond].pivot(index="cell_id", columns="candidate", values="logp_mean").loc[order, labels]
        auc, l2 = np.zeros(len(order)), np.zeros(len(order))
        for i, (cid, L) in enumerate(zip(order, lab)):
            own = Mc.at[cid, L]
            mis = Mi.loc[lab != L, L].to_numpy()
            auc[i] = float(np.mean(own > mis))
            l2[i] = float(np.log2(np.mean(np.exp(-mis)) / np.exp(-own)))
        out[cond] = {"auc": auc, "log2_ratio": l2}
    boot = Boot(len(order), BOOT_N, rng)
    st, bsd = {}, {}
    for m in ["auc", "log2_ratio"]:
        pt, bs = boot.mean(out["intact"][m]); st[f"{m}_intact"] = pt; st[f"{m}_intact_ci"] = ci(bs); bsd[f"{m}_intact"] = bs
        shuf = np.mean([out[c][m] for c in SHUF], axis=0)
        pt2, bs2 = boot.mean(shuf); st[f"{m}_shuffle"] = pt2; st[f"{m}_shuffle_ci"] = ci(bs2); bsd[f"{m}_shuffle"] = bs2
        st[f"{m}_shuffle_per_seed"] = [float(out[c][m].mean()) for c in SHUF]
        st[f"{m}_delta_shuffle"] = pt - pt2; st[f"{m}_delta_shuffle_ci"] = ci(bs - bs2); bsd[f"{m}_delta_shuffle"] = bs - bs2
        pt3, bs3 = boot.mean(out["rankonly_s0"][m]); st[f"{m}_rankonly"] = pt3; st[f"{m}_rankonly_ci"] = ci(bs3); bsd[f"{m}_rankonly"] = bs3
        st[f"{m}_delta_rankonly"] = pt - pt3; st[f"{m}_delta_rankonly_ci"] = ci(bs - bs3)
    st["frac_cells_auc_above_half_intact"] = float(np.mean(out["intact"]["auc"] > 0.5))
    st["frac_cells_auc_above_half_shuffle"] = float(np.mean(np.mean([out[c]["auc"] for c in SHUF], axis=0) > 0.5))
    st["median_quantile_intact"] = float(np.median(1 - out["intact"]["auc"]))
    st["median_quantile_shuffle"] = float(np.median(1 - np.mean([out[c]["auc"] for c in SHUF], axis=0)))
    percell = pd.DataFrame({"cell_id": order, **{f"auc_{c}": out[c]["auc"] for c in out},
                            **{f"log2_ratio_{c}": out[c]["log2_ratio"] for c in out}})
    return st, bsd, percell


def decomp_2x2(C):
    """Per pair and k, from the four prompts (Z_A|G_A, Z_B|G_B, Z_B|G_A, Z_A|G_B) and the pairwise
    preference pref_A = logp(T_A) - logp(T_B):
        embedding_effect = mean of pref_A(Z_A, G) - pref_A(Z_B, G) over the two texts   (nats)
        text_effect      = mean of pref_A(Z, G_A) - pref_A(Z, G_B) over the two embeddings
    Differences of the same pair's preference across inputs, so the answer-length bias cancels;
    the source with the larger effect is the one that controls the answer for that pair at that k."""
    out = {}
    for val, suf in (("preference_A_minus_B", ""), ("preference_A_minus_B_ppl", "_ppl")):
        P = C.pivot(index=["pair_id", "k"], columns="condition", values=val)
        E = 0.5 * ((P["aligned_A"] - P["Bembed_Agenes"]) + (P["Aembed_Bgenes"] - P["aligned_B"]))
        T = 0.5 * ((P["aligned_A"] - P["Aembed_Bgenes"]) + (P["Bembed_Agenes"] - P["aligned_B"]))
        out[f"embedding_effect{suf}"] = E; out[f"text_effect{suf}"] = T
        out[f"embedding_dominates{suf}"] = (E > T).astype(float)
    return pd.DataFrame(out).reset_index()


def pair_prefs(pairs, intact_scores_A):
    """Embedding-gate quantities from the intact multi-class scores (sum log-likelihood)."""
    S = intact_scores_A[intact_scores_A["condition"] == "intact"].set_index(["cell_id", "candidate"])["logp_sum"]
    Sm = intact_scores_A[intact_scores_A["condition"] == "intact"].set_index(["cell_id", "candidate"])["logp_mean"]
    out = []
    for _, p in pairs.iterrows():
        a, b, la, lb = p["cell_A"], p["cell_B"], p["canonical_A"], p["canonical_B"]
        pa_za = S[(a, la)] - S[(a, lb)]
        pa_zb = S[(b, la)] - S[(b, lb)]
        out.append({"pair_id": p["pair_id"], "prefA_given_ZA": float(pa_za), "prefA_given_ZB": float(pa_zb),
                    "prefA_given_ZA_mean": float(Sm[(a, la)] - Sm[(a, lb)]),
                    "prefA_given_ZB_mean": float(Sm[(b, la)] - Sm[(b, lb)]),
                    "embedding_gate_pass": bool(pa_za > 0 and pa_zb < 0),
                    "embedding_gate_pass_meanLL": bool((Sm[(a, la)] - Sm[(a, lb)]) > 0 and (Sm[(b, la)] - Sm[(b, lb)]) < 0),
                    # secondary, length-invariant: the embedding moves the preference toward its own type
                    "embedding_gate_shift": bool(pa_za > pa_zb),
                    "flip_A_to_B": bool(pa_za > 0 and pa_zb < 0)})
    return pd.DataFrame(out).set_index("pair_id")


def text_gate(T):
    """Per pair and k: base-Mistral preferences for each gene-text source."""
    P = T.pivot_table(index=["pair_id", "k", "gene_text_source"], columns="candidate", values="logp_sum")
    P["prefA"] = P["A"] - P["B"]
    W = P["prefA"].unstack("gene_text_source")
    W["text_gate"] = (W["A"] > 0) & (W["B"] < 0)
    W["text_gate_shift"] = W["A"] > W["B"]       # secondary, length-invariant
    tok = T.groupby(["pair_id", "k"])["prompt_token_count"].max()
    W = W.join(tok)
    return W  # index (pair_id, k); columns A, B, text_gate, text_gate_shift, prompt_token_count


def conflict_table(B, pairs):
    P = B.pivot_table(index=["pair_id", "k", "condition"],
                      columns="candidate", values=["logp_sum", "logp_mean"])
    meta = B.groupby(["pair_id", "k", "condition"]).agg(
        embedding_source=("embedding_source", "first"), gene_text_source=("gene_text_source", "first"),
        prompt_token_count=("prompt_token_count", "max"), spliced_len=("spliced_len", "max"),
        within_released_cap=("within_released_cap", "all"), emb_hash=("emb_hash", "first"))
    C = pd.DataFrame({"logp_answer_A": P[("logp_sum", "A")], "logp_answer_B": P[("logp_sum", "B")],
                      "logp_mean_A": P[("logp_mean", "A")], "logp_mean_B": P[("logp_mean", "B")]}).join(meta)
    C["preference_A_minus_B"] = C["logp_answer_A"] - C["logp_answer_B"]
    C = C.reset_index()
    C["direction"] = C["condition"].map({"aligned_A": "A", "aligned_B": "B", "Bembed_Agenes": "B3", "Aembed_Bgenes": "B4"})
    # preference toward the gene-text source (conflicts) / toward the shared source (aligned)
    sign = np.where(C["gene_text_source"] == "A", 1.0, -1.0)
    C["pref_text_source"] = C["preference_A_minus_B"] * sign
    esign = np.where(C["embedding_source"] == "A", 1.0, -1.0)
    C["pref_embedding_source"] = C["preference_A_minus_B"] * esign
    is_conf = C["condition"].isin(["Bembed_Agenes", "Aembed_Bgenes"])
    C["follow_class"] = np.where(~is_conf, "aligned",
                                 np.where(C["pref_text_source"] > 0, "text",
                                          np.where(C["pref_text_source"] < 0, "embedding", "ambiguous")))
    # per-token (perplexity) version: the answer with the lower perplexity wins, as in CellWhisperer's
    # response ranking; independent of the two answer strings' token counts
    C["preference_A_minus_B_ppl"] = C["logp_mean_A"] - C["logp_mean_B"]
    C["pref_text_source_ppl"] = C["preference_A_minus_B_ppl"] * sign
    C["follow_class_ppl"] = np.where(~is_conf, "aligned",
                                     np.where(C["pref_text_source_ppl"] > 0, "text",
                                              np.where(C["pref_text_source_ppl"] < 0, "embedding", "ambiguous")))
    C.insert(0, "atlas", pairs["atlas"].iloc[0])
    return C


def exp_b(atlas, pairs, C, PP, TG, rng):
    """Rates per k on the gated sets, pair bootstrap. Sets: `common` (embedding gate and text
    gate at every k), `common_shift` (secondary, length-invariant gates), `all` (every frozen pair),
    and the k-specific sets `k<k>`."""
    pairs = pairs.set_index("pair_id")
    gate = pd.DataFrame(index=pairs.index)
    gate["embedding_gate_pass"] = PP["embedding_gate_pass"]
    gate["embedding_gate_shift"] = PP["embedding_gate_shift"]
    gate["embedding_gate_ppl"] = PP["embedding_gate_pass_meanLL"]     # per-token (perplexity) gate
    for k in KS:
        gate[f"text_gate_{k}"] = TG["text_gate"].xs(k, level="k").reindex(pairs.index).fillna(False).astype(bool)
        gate[f"text_gate_shift_{k}"] = TG["text_gate_shift"].xs(k, level="k").reindex(pairs.index).fillna(False).astype(bool)
    gate["common_k_gate_pass"] = gate["embedding_gate_pass"] & gate[[f"text_gate_{k}" for k in KS]].all(1)
    gate["common_shift_gate_pass"] = gate["embedding_gate_shift"] & gate[[f"text_gate_shift_{k}" for k in KS]].all(1)
    conf = C[C["follow_class"] != "aligned"].copy()
    per = conf.groupby(["pair_id", "k"]).agg(text=("follow_class", lambda s: float((s == "text").mean())),
                                             emb=("follow_class", lambda s: float((s == "embedding").mean())),
                                             amb=("follow_class", lambda s: float((s == "ambiguous").mean())),
                                             pref_text=("pref_text_source", "mean"),
                                             text_ppl=("follow_class_ppl", lambda s: float((s == "text").mean())),
                                             emb_ppl=("follow_class_ppl", lambda s: float((s == "embedding").mean())),
                                             amb_ppl=("follow_class_ppl", lambda s: float((s == "ambiguous").mean())),
                                             pref_text_ppl=("pref_text_source_ppl", "mean"),
                                             n_dir=("follow_class", "size"),
                                             within_cap=("within_released_cap", "all"),
                                             spliced_max=("spliced_len", "max")).reset_index()
    assert (per["n_dir"] == 2).all()
    per = per.merge(decomp_2x2(C), on=["pair_id", "k"], how="left")
    METRICS = ["text", "emb", "amb", "pref_text", "embedding_effect", "text_effect", "embedding_dominates",
               "text_ppl", "emb_ppl", "amb_ppl", "pref_text_ppl", "embedding_effect_ppl", "text_effect_ppl", "embedding_dominates_ppl"]
    res = {"n_pairs": int(len(pairs)), "n_embedding_gate": int(gate["embedding_gate_pass"].sum()),
           "n_embedding_gate_shift": int(gate["embedding_gate_shift"].sum()),
           "n_embedding_gate_ppl": int(gate["embedding_gate_ppl"].sum()),
           "n_text_gate": {int(k): int(gate[f"text_gate_{k}"].sum()) for k in KS},
           "n_text_gate_shift": {int(k): int(gate[f"text_gate_shift_{k}"].sum()) for k in KS},
           "n_common": int(gate["common_k_gate_pass"].sum()),
           "n_common_shift": int(gate["common_shift_gate_pass"].sum()),
           "flip_rate_A_to_B_all_pairs": float(PP["flip_A_to_B"].mean()),
           "embedding_shift_rate_all_pairs": float(PP["embedding_gate_shift"].mean()),
           "embedding_gate_pass_meanLL_n": int(PP["embedding_gate_pass_meanLL"].sum())}
    common = gate.index[gate["common_k_gate_pass"]].tolist()
    common_shift = gate.index[gate["common_shift_gate_pass"]].tolist()
    embgate = gate.index[gate["embedding_gate_pass"]].tolist()
    embgate_ppl = gate.index[gate["embedding_gate_ppl"]].tolist()
    sets = [("common", common), ("common_shift", common_shift), ("embgate", embgate), ("embgate_ppl", embgate_ppl),
            ("all", list(pairs.index))] + \
           [(f"k{k}", gate.index[gate["embedding_gate_pass"] & gate[f"text_gate_{k}"]].tolist()) for k in KS]
    bsd = {}
    for setname, members in sets:
        boot = Boot(len(members), BOOT_N, rng)
        res[f"{setname}_n"] = len(members)
        for k in KS:
            if setname.startswith("k") and int(setname[1:]) != k:
                continue
            sub = per[(per["k"] == k) & per["pair_id"].isin(members)].set_index("pair_id").reindex(members)
            r = {"n": len(members)}
            for m in METRICS:
                pt, bs = boot.mean(sub[m].to_numpy())
                r[m] = pt; r[f"{m}_ci"] = ci(bs)
                if setname in ("common", "common_shift", "embgate", "embgate_ppl", "all"):
                    bsd[(setname, k, m)] = bs
            r["frac_within_released_cap"] = float(sub["within_cap"].mean()) if len(sub) else float("nan")
            r["spliced_len_max"] = int(sub["spliced_max"].max()) if len(sub) else None
            res[f"{setname}_k{k}"] = r
    # descriptive regression of source preference on log10(k), pair-cluster bootstrap
    for setname, members in (("common", common), ("common_shift", common_shift), ("embgate", embgate),
                             ("embgate_ppl", embgate_ppl), ("all", list(pairs.index))):
        if len(members) < 2:
            continue
        sub = per[per["pair_id"].isin(members)].copy()
        x = np.log10(sub["k"].to_numpy(float)); pid = sub["pair_id"].to_numpy()
        boot = Boot(len(members), BOOT_N, rng)
        idx = {p: i for i, p in enumerate(members)}
        wpair = boot.w[:, [idx[p] for p in pid]]
        for m in ("pref_text", "text", "embedding_effect", "text_effect", "embedding_dominates",
                  "pref_text_ppl", "text_ppl", "embedding_effect_ppl", "text_effect_ppl", "embedding_dominates_ppl"):
            y = sub[m].to_numpy(float)
            slopes = []
            for b in range(BOOT_N):
                w = wpair[b]
                if w.sum() == 0:
                    slopes.append(np.nan); continue
                xm = np.average(x, weights=w); ym = np.average(y, weights=w)
                den = np.sum(w * (x - xm) ** 2)
                slopes.append(np.sum(w * (x - xm) * (y - ym)) / den if den > 0 else np.nan)
            res[f"slope_{m}_per_log10k_{setname}"] = float(np.polyfit(x, y, 1)[0])
            res[f"slope_{m}_per_log10k_{setname}_ci"] = ci(slopes)
    return gate.reset_index(), per, res, bsd


SETS = ["common", "common_shift", "embgate", "embgate_ppl", "all"]
B_METRICS = ["text", "emb", "amb", "pref_text", "embedding_effect", "text_effect", "embedding_dominates",
             "text_ppl", "emb_ppl", "amb_ppl", "pref_text_ppl", "embedding_effect_ppl", "text_effect_ppl", "embedding_dominates_ppl"]
A_KEYS = ["acc_ppl_intact", "acc_ppl_shuffle", "acc_ppl_delta_shuffle", "acc_ppl_rankonly",
          "acc_sum_intact", "acc_sum_shuffle", "acc_sum_delta_shuffle",
          "margin_sum_intact", "margin_sum_shuffle", "margin_sum_delta_shuffle",
          "margin_mean_intact", "margin_mean_shuffle", "margin_mean_delta_shuffle", "swap_follow_rate",
          "auc_intact", "auc_shuffle", "auc_delta_shuffle", "auc_rankonly",
          "log2_ratio_intact", "log2_ratio_shuffle", "log2_ratio_delta_shuffle", "log2_ratio_rankonly"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="default")
    ap.add_argument("--atlases", nargs="+", default=ATLASES)
    ap.add_argument("--root", default=str(OUT))
    ap.add_argument("--out_dir", default=None, help="default: <root>/analysis_<model>")
    a = ap.parse_args()
    root = Path(a.root)
    outd = Path(a.out_dir) if a.out_dir else root / f"analysis_{a.model}"
    outd.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(BOOT_SEED)

    all_cells, all_cond, all_pairs, all_conf, all_per, all_pq, per_atlas = [], [], [], [], [], [], {}
    bootA, bootB, qc = {}, {}, {"model": a.model, "atlases": {}}
    for atlas in a.atlases:
        cells, pairs, A, B, T, eq = load_atlas(atlas, a.model, root)
        G, stA, bsA, _ = exp_a(atlas, cells, A, rng)
        stQ, bsQ, pq = ppl_quantile_stats(cells, A, rng)
        stA.update(stQ); bsA.update(bsQ)
        pq.insert(0, "atlas", atlas)
        PP = pair_prefs(pairs, A)
        TG = text_gate(T)
        C = conflict_table(B, pairs)
        gate, per, stB, bsB = exp_b(atlas, pairs, C, PP, TG, rng)
        per.insert(0, "atlas", atlas)
        # embedding-invariance check: same embedding hash across k for a given (pair, embedding source)
        inv = C.groupby(["pair_id", "embedding_source"])["emb_hash"].nunique()
        qc["atlases"][atlas] = {
            "cells_qc": json.load(open(root / atlas / "cells_qc.json")),
            "embed_qc": {k: v for k, v in eq.items() if k != "embedding_hash"},
            "embedding_hash_constant_across_k": bool((inv == 1).all()),
            "expA_rows": int(len(A)), "expB_rows": int(len(B)), "textgate_rows": int(len(T)),
            "expB_frac_within_released_cap_by_k": {int(k): float(C[C["k"] == k]["within_released_cap"].mean()) for k in KS},
            "expB_spliced_len_by_k": {int(k): {"min": int(C[C["k"] == k]["spliced_len"].min()),
                                               "median": float(C[C["k"] == k]["spliced_len"].median()),
                                               "max": int(C[C["k"] == k]["spliced_len"].max())} for k in KS},
            "textgate_prompt_tokens_by_k": {int(k): int(T[T["k"] == k]["prompt_token_count"].max()) for k in KS},
            "gates": {k: v for k, v in stB.items() if isinstance(v, (int, float, dict)) and not k.endswith("_ci")
                      and "_k" not in k},
            "top1_prediction_distribution_intact": G[G["condition"] == "intact"]["top1_ppl"].value_counts().head(5).to_dict(),
        }
        per_atlas[atlas] = {"expA": stA, "expB": stB}
        bootA[atlas] = bsA; bootB[atlas] = bsB
        gate = gate.merge(pairs[["pair_id", "atlas", "cell_A", "cell_B", "label_A", "label_B", "canonical_A", "canonical_B",
                                 "same_tissue", "same_donor"]], on="pair_id")
        gate = gate.join(PP[["prefA_given_ZA", "prefA_given_ZB"]], on="pair_id")
        for k in KS:
            gate[f"text_prefA_given_GkA_{k}"] = TG["A"].xs(k, level="k").reindex(gate["pair_id"]).to_numpy()
            gate[f"text_prefA_given_GkB_{k}"] = TG["B"].xs(k, level="k").reindex(gate["pair_id"]).to_numpy()
        all_cells.append(cells); all_cond.append(G); all_pairs.append(gate); all_conf.append(C); all_per.append(per); all_pq.append(pq)
        print(f"[analyze] {atlas}: auc_intact={stA['auc_intact']:.3f} auc_shuffle={stA['auc_shuffle']:.3f} "
              f"auc_rankonly={stA['auc_rankonly']:.3f} acc_intact={stA['acc_ppl_intact']:.3f} acc_shuffle={stA['acc_ppl_shuffle']:.3f} "
              f"gate_emb={stB['n_embedding_gate']} gate_emb_shift={stB['n_embedding_gate_shift']} "
              f"common={stB['n_common']} common_shift={stB['n_common_shift']}", flush=True)

    # macro averages over atlases (unweighted), CIs from the per-atlas bootstrap draws
    macro = {"expA": {}, "expB": {}}
    for key in A_KEYS:
        pts = [per_atlas[x]["expA"][key] for x in a.atlases]
        bs = np.mean([bootA[x][key] for x in a.atlases], axis=0)
        macro["expA"][key] = float(np.mean(pts)); macro["expA"][key + "_ci"] = ci(bs)
    for setname in SETS:
        for k in KS:
            for m in B_METRICS:
                pts = [per_atlas[x]["expB"][f"{setname}_k{k}"][m] for x in a.atlases]
                draws = [bootB[x][(setname, k, m)] for x in a.atlases if (setname, k, m) in bootB[x]]
                bs = np.nanmean(draws, axis=0) if draws else np.array([np.nan])
                macro["expB"][f"{setname}_k{k}_{m}"] = float(np.nanmean(pts)); macro["expB"][f"{setname}_k{k}_{m}_ci"] = ci(bs)
        macro["expB"][f"{setname}_n"] = int(sum(per_atlas[x]["expB"][f"{setname}_n"] for x in a.atlases))
    # pooled over pairs (each pair weight 1) for each set
    confall = pd.concat(all_conf, ignore_index=True)
    pairsall = pd.concat(all_pairs, ignore_index=True)
    perall = pd.concat(all_per, ignore_index=True)
    pooled = {}
    for setname, col in (("common", "common_k_gate_pass"), ("common_shift", "common_shift_gate_pass"),
                         ("embgate", "embedding_gate_pass"), ("embgate_ppl", "embedding_gate_ppl"), ("all", None)):
        ids = sorted(pairsall["pair_id"] if col is None else pairsall.loc[pairsall[col], "pair_id"])
        pooled[f"n_{setname}"] = len(ids)
        boot = Boot(len(ids), BOOT_N, rng)
        for k in KS:
            sub = perall[perall["k"] == k].set_index("pair_id").reindex(ids)
            for m in B_METRICS:
                pt, bs = boot.mean(sub[m].to_numpy()); pooled[f"{setname}_k{k}_{m}"] = pt; pooled[f"{setname}_k{k}_{m}_ci"] = ci(bs)
    pooled["n_common"] = pooled["n_common"]
    for k in KS:  # short pooled keys for the common set
        for m in ["text", "emb", "pref_text"]:
            pooled[f"k{k}_{m}"] = pooled[f"common_k{k}_{m}"]; pooled[f"k{k}_{m}_ci"] = pooled[f"common_k{k}_{m}_ci"]

    # ---- tables ----------------------------------------------------------------------------
    cells_df = pd.concat(all_cells, ignore_index=True)
    cells_df.to_parquet(outd / "cellwhisperer_cells.parquet", index=False)
    cond = pd.concat(all_cond, ignore_index=True).rename(columns={"emb_hash": "embedding_hash",
                                                                  "correct_logp_sum": "correct_answer_logprob",
                                                                  "correct_logp_mean": "correct_answer_logprob_per_token",
                                                                  "top1_ppl": "top1_predicted_label",
                                                                  "top1_sum": "top1_predicted_label_by_sum"})
    cond["raw_generation_optional"] = None
    cond.to_parquet(outd / "cellwhisperer_embedding_conditions.parquet", index=False)
    pd.concat(all_pq, ignore_index=True).to_parquet(outd / "cellwhisperer_ppl_quantile.parquet", index=False)
    pairsall.to_parquet(outd / "cellwhisperer_pairs.parquet", index=False)
    confall.to_parquet(outd / "cellwhisperer_conflict_predictions.parquet", index=False)
    perall.to_parquet(outd / "cellwhisperer_conflict_per_pair.parquet", index=False)

    def a_cols(sa):
        return {"acc_intact": sa["acc_ppl_intact"], "acc_intact_ci_lo": sa["acc_ppl_intact_ci"][0], "acc_intact_ci_hi": sa["acc_ppl_intact_ci"][1],
                "acc_shuffle_mean": sa["acc_ppl_shuffle"], "acc_shuffle_ci_lo": sa["acc_ppl_shuffle_ci"][0], "acc_shuffle_ci_hi": sa["acc_ppl_shuffle_ci"][1],
                "delta_shuffle": sa["acc_ppl_delta_shuffle"], "delta_shuffle_ci_lo": sa["acc_ppl_delta_shuffle_ci"][0], "delta_shuffle_ci_hi": sa["acc_ppl_delta_shuffle_ci"][1],
                "acc_rankonly": sa["acc_ppl_rankonly"], "acc_intact_by_sum": sa["acc_sum_intact"], "acc_shuffle_by_sum": sa["acc_sum_shuffle"],
                "margin_intact": sa["margin_sum_intact"], "margin_shuffle": sa["margin_sum_shuffle"], "delta_margin": sa["margin_sum_delta_shuffle"],
                "auc_intact": sa["auc_intact"], "auc_intact_ci_lo": sa["auc_intact_ci"][0], "auc_intact_ci_hi": sa["auc_intact_ci"][1],
                "auc_shuffle": sa["auc_shuffle"], "auc_shuffle_ci_lo": sa["auc_shuffle_ci"][0], "auc_shuffle_ci_hi": sa["auc_shuffle_ci"][1],
                "auc_delta_shuffle": sa["auc_delta_shuffle"], "auc_delta_ci_lo": sa["auc_delta_shuffle_ci"][0], "auc_delta_ci_hi": sa["auc_delta_shuffle_ci"][1],
                "auc_rankonly": sa["auc_rankonly"],
                "log2_ppl_ratio_intact": sa["log2_ratio_intact"], "log2_ppl_ratio_shuffle": sa["log2_ratio_shuffle"],
                "swap_follow_rate": sa["swap_follow_rate"]}

    def b_cols(r, prefix):
        return {f"{prefix}embedding_follow_rate": r["emb"], f"{prefix}embedding_follow_ci_lo": r["emb_ci"][0], f"{prefix}embedding_follow_ci_hi": r["emb_ci"][1],
                f"{prefix}text_follow_rate": r["text"], f"{prefix}text_follow_ci_lo": r["text_ci"][0], f"{prefix}text_follow_ci_hi": r["text_ci"][1],
                f"{prefix}ambiguous_rate": r["amb"], f"{prefix}mean_pref_text": r["pref_text"], f"{prefix}mean_pref_text_ci_lo": r["pref_text_ci"][0], f"{prefix}mean_pref_text_ci_hi": r["pref_text_ci"][1],
                f"{prefix}embedding_effect_nats": r["embedding_effect"], f"{prefix}embedding_effect_ci_lo": r["embedding_effect_ci"][0], f"{prefix}embedding_effect_ci_hi": r["embedding_effect_ci"][1],
                f"{prefix}text_effect_nats": r["text_effect"], f"{prefix}text_effect_ci_lo": r["text_effect_ci"][0], f"{prefix}text_effect_ci_hi": r["text_effect_ci"][1],
                f"{prefix}embedding_dominates_rate": r["embedding_dominates"], f"{prefix}embedding_dominates_ci_lo": r["embedding_dominates_ci"][0], f"{prefix}embedding_dominates_ci_hi": r["embedding_dominates_ci"][1],
                f"{prefix}n": r["n"],
                **{f"{prefix}{m}{sfx}": (r[m] if sfx == "" else r[m + "_ci"][0 if sfx == "_ci_lo" else 1])
                   for m in ("emb_ppl", "text_ppl", "amb_ppl", "pref_text_ppl", "embedding_effect_ppl", "text_effect_ppl", "embedding_dominates_ppl")
                   for sfx in ("", "_ci_lo", "_ci_hi")}}

    rows = []
    for atlas in a.atlases + ["macro", "pooled"]:
        for k in KS:
            row = {"atlas": atlas, "k": k}
            if atlas in per_atlas:
                sa, sb = per_atlas[atlas]["expA"], per_atlas[atlas]["expB"]
                row.update({"n_cells": sa["n_cells"], "n_labels": sa["n_labels"], "chance_uniform": sa["chance_uniform"]})
                row.update(a_cols(sa))
                row.update({"flip_rate_A_to_B": sb["flip_rate_A_to_B_all_pairs"], "n_pairs": sb["n_pairs"],
                            "n_embedding_gate": sb["n_embedding_gate"], "n_embedding_gate_shift": sb["n_embedding_gate_shift"],
                            "n_text_gate_k": sb["n_text_gate"][k], "n_text_gate_shift_k": sb["n_text_gate_shift"][k],
                            "common_pair_n": sb["n_common"], "common_shift_pair_n": sb["n_common_shift"]})
                row.update(b_cols(sb[f"common_k{k}"], ""))
                row.update(b_cols(sb[f"common_shift_k{k}"], "shiftgate_"))
                row.update(b_cols(sb[f"embgate_k{k}"], "embgate_"))
                row.update(b_cols(sb[f"embgate_ppl_k{k}"], "embgateppl_"))
                row.update(b_cols(sb[f"all_k{k}"], "allpairs_"))
                row.update({"frac_within_released_cap": sb[f"all_k{k}"]["frac_within_released_cap"], "spliced_len_max": sb[f"all_k{k}"]["spliced_len_max"],
                            "kspecific_n": sb[f"k{k}_n"], "kspecific_embedding_follow_rate": sb[f"k{k}_k{k}"]["emb"], "kspecific_text_follow_rate": sb[f"k{k}_k{k}"]["text"]})
            elif atlas == "macro":
                ma, mb = macro["expA"], macro["expB"]
                row.update(a_cols({**{kk: ma[kk] for kk in A_KEYS}, **{kk + "_ci": ma[kk + "_ci"] for kk in A_KEYS}}))
                for setname, prefix in (("common", ""), ("common_shift", "shiftgate_"), ("embgate", "embgate_"), ("embgate_ppl", "embgateppl_"), ("all", "allpairs_")):
                    r = {m: mb[f"{setname}_k{k}_{m}"] for m in B_METRICS}
                    r.update({m + "_ci": mb[f"{setname}_k{k}_{m}_ci"] for m in B_METRICS}); r["n"] = mb[f"{setname}_n"]
                    row.update(b_cols(r, prefix))
                row["common_pair_n"] = mb["common_n"]; row["common_shift_pair_n"] = mb["common_shift_n"]
            else:
                for setname, prefix in (("common", ""), ("common_shift", "shiftgate_"), ("embgate", "embgate_"), ("embgate_ppl", "embgateppl_"), ("all", "allpairs_")):
                    r = {m: pooled[f"{setname}_k{k}_{m}"] for m in B_METRICS}
                    r.update({m + "_ci": pooled[f"{setname}_k{k}_{m}_ci"] for m in B_METRICS}); r["n"] = pooled[f"n_{setname}"]
                    row.update(b_cols(r, prefix))
                row["common_pair_n"] = pooled["n_common"]; row["common_shift_pair_n"] = pooled["n_common_shift"]
            rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(outd / "cellwhisperer_summary.csv", index=False)
    qc["bootstrap"] = {"n_resamples": BOOT_N, "seed": BOOT_SEED}
    qc["definitions"] = __doc__
    dump_json(outd / "cellwhisperer_qc.json", qc)
    dump_json(outd / "results.json", {"per_atlas": per_atlas, "macro": macro, "pooled": pooled})
    show = ["atlas", "k", "auc_intact", "auc_shuffle", "auc_rankonly", "acc_intact", "acc_shuffle_mean", "common_pair_n",
            "embedding_follow_rate", "text_follow_rate", "shiftgate_n", "shiftgate_embedding_follow_rate", "shiftgate_text_follow_rate",
            "allpairs_embedding_effect_nats", "allpairs_text_effect_nats", "allpairs_embedding_dominates_rate"]
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(summary[[c for c in show if c in summary.columns]].to_string())


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Computes the Prot2Text-V2 results for the shuffled sequence conditions and the evidence conflicts,
with bootstrap intervals over proteins and over pairs.

For the shuffle conditions it reports BERTScore F1 (BioBERT and RoBERTa) and ROUGE-L against each
protein's own function text, for the sequence alone and for the released prompt, intact and
shuffled. For the evidence conflicts it reports the rate at which a generation is closer to the
function of the sequence's protein than to that of the named protein, over pairs whose sequence-only
generations are each closer to their own protein, and also over the other pair sets. Writes
prot2text_analysis.json and prot2text_figure.json (read by figures/prot2text_v2.py for Figs. 2 and
3), prot2text_summary.csv, prot2text_qc.json and run_manifest.json to
protein/prot2text_v2/results under $INPUT_USE_RESULTS_DIR.

    python input_use/models/prot2text_driver/analyze.py
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

METRICS = {"biobert": "biobert_score_to_own_function", "roberta": "roberta_score_to_own_function", "rougeL": "rougeL"}


def boot_mean(arr, idx):
    return arr[idx].mean(axis=1)


def ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def experiment_a(pred, rng, R):
    out = {}
    within = pred[pred.condition != "native_full"].protein_id.unique()
    conds = {"seq_only_intact": [-1], "seq_only_shuffle": [0, 1, 2], "native_full": [-1], "native_shuffle": [0, 1, 2]}
    wide = {}
    for cond, seeds in conds.items():
        for s in seeds:
            sub = pred[(pred.condition == cond) & (pred.shuffle_seed == s)].set_index("protein_id")
            wide[(cond, s)] = sub
    have = set(within)
    for k, sub in wide.items():
        have &= set(sub.index)
    ids = sorted(have)
    out["n_proteins"] = len(ids)
    out["n_within_1021_with_any_generation"] = int(len(within))
    idx = rng.integers(0, len(ids), size=(R, len(ids)))
    for mname, col in METRICS.items():
        val = {k: sub.loc[ids, col].to_numpy(dtype=float) for k, sub in wide.items()}
        intact = val[("seq_only_intact", -1)]
        shuf_seeds = np.stack([val[("seq_only_shuffle", s)] for s in (0, 1, 2)])
        shuf = shuf_seeds.mean(0)
        nat = val[("native_full", -1)]
        natshuf_seeds = np.stack([val[("native_shuffle", s)] for s in (0, 1, 2)])
        natshuf = natshuf_seeds.mean(0)
        res = {}
        for name, arr in (("seq_only_intact", intact), ("seq_only_shuffle", shuf), ("seq_only_delta", intact - shuf),
                          ("native_intact", nat), ("native_shuffle", natshuf), ("native_delta", nat - natshuf),
                          ("name_effect_intact", nat - intact)):
            res[name] = {"mean": float(arr.mean()), "ci95": ci(boot_mean(arr, idx)), "n": len(ids)}
        for s in (0, 1, 2):
            res[f"seq_only_shuffle_seed{s}"] = {"mean": float(shuf_seeds[s].mean()), "ci95": ci(boot_mean(shuf_seeds[s], idx))}
            res[f"seq_only_delta_seed{s}"] = {"mean": float((intact - shuf_seeds[s]).mean()),
                                              "ci95": ci(boot_mean(intact - shuf_seeds[s], idx))}
            res[f"native_shuffle_seed{s}"] = {"mean": float(natshuf_seeds[s].mean()), "ci95": ci(boot_mean(natshuf_seeds[s], idx))}
            res[f"native_delta_seed{s}"] = {"mean": float((nat - natshuf_seeds[s]).mean()),
                                            "ci95": ci(boot_mean(nat - natshuf_seeds[s], idx))}
        res["frac_proteins_intact_gt_shuffle_seq_only"] = float((intact > shuf).mean())
        res["frac_proteins_intact_gt_shuffle_native"] = float((nat > natshuf).mean())
        out[mname] = res
    # intervention verification: did the caption text change?
    g = {k: sub.loc[ids, "generation"] for k, sub in wide.items()}
    out["caption_changed"] = {
        "seq_only_shuffle_vs_intact": {f"seed{s}": float((g[("seq_only_shuffle", s)].to_numpy() != g[("seq_only_intact", -1)].to_numpy()).mean()) for s in (0, 1, 2)},
        "native_shuffle_vs_native": {f"seed{s}": float((g[("native_shuffle", s)].to_numpy() != g[("native_full", -1)].to_numpy()).mean()) for s in (0, 1, 2)},
        "seq_only_intact_vs_native": float((g[("seq_only_intact", -1)].to_numpy() != g[("native_full", -1)].to_numpy()).mean()),
    }
    out["mean_new_tokens"] = {f"{k[0]}|{k[1]}": float(sub.loc[ids, "n_new_tokens"].mean()) for k, sub in wide.items()}
    out["n_hit_max_new_tokens"] = {f"{k[0]}|{k[1]}": int((~sub.loc[ids, "ended_with_eos"]).sum()) for k, sub in wide.items()}
    return out


def experiment_b(pairs, conf, rng, R, eps, gate_col="sequence_gate_pass", metric="biobert"):
    pcol = "preference_A_minus_B" if metric == "biobert" else "roberta_preference_A_minus_B"
    keep = pairs[pairs[gate_col].fillna(False).astype(bool) & pairs.has_both_conflicts.fillna(False).astype(bool)].pair_id.tolist()
    c = conf[conf.pair_id.isin(keep) & conf.condition.isin(["Bseq_Aname", "Aseq_Bname"])]
    pv = c.pivot(index="pair_id", columns="condition", values=pcol).loc[keep]
    ali = conf[conf.pair_id.isin(keep)].pivot(index="pair_id", columns="condition", values=pcol).loc[keep]
    # oriented: positive = towards the sequence source's function
    tow_seq = np.stack([-pv["Bseq_Aname"].to_numpy(float), pv["Aseq_Bname"].to_numpy(float)], axis=1)   # (n_pairs, 2) shift from the name source's aligned prompt
    # when only the sequence is swapped (positive = moved away
    # from the name source's function, i.e. towards the sequence
    # source)
    shift = np.stack([ali["aligned_A"].to_numpy(float) - pv["Bseq_Aname"].to_numpy(float),
                      -(ali["aligned_B"].to_numpy(float) - pv["Aseq_Bname"].to_numpy(float))], axis=1)
    n = len(keep)
    out = {"n_pairs": n, "n_conflicts": 2 * n, "gate": gate_col, "metric": metric}
    if n == 0:
        return out
    idx = rng.integers(0, n, size=(R, n))

    def rates(e):
        seq = (tow_seq > e).astype(float)
        txt = (tow_seq < -e).astype(float)
        amb = 1.0 - seq - txt
        res = {}
        for name, m in (("sequence_follow_rate", seq), ("text_follow_rate", txt), ("ambiguous_rate", amb)):
            per_pair = m.mean(axis=1)
            res[name] = {"mean": float(per_pair.mean()), "ci95": ci(boot_mean(per_pair, idx))}
        res["per_direction"] = {"seqB_nameA": {"sequence": float(seq[:, 0].mean()), "text": float(txt[:, 0].mean()), "ambiguous": float(amb[:, 0].mean())},
                                "seqA_nameB": {"sequence": float(seq[:, 1].mean()), "text": float(txt[:, 1].mean()), "ambiguous": float(amb[:, 1].mean())}}
        return res
    out[f"eps_{eps}"] = rates(eps)
    out["eps_0"] = rates(0.0)
    pp = tow_seq.mean(axis=1)
    out["mean_preference_toward_sequence"] = {"mean": float(pp.mean()), "ci95": ci(boot_mean(pp, idx))}
    sp = shift.mean(axis=1)
    out["mean_preference_shift_from_aligned"] = {"mean": float(sp.mean()), "ci95": ci(boot_mean(sp, idx))}
    out["aligned_own_preference"] = {"aligned_A_mean_prefA": float(ali["aligned_A"].mean()), "aligned_B_mean_prefA": float(ali["aligned_B"].mean()),
                                     "frac_aligned_A_prefers_own": float((ali["aligned_A"] > 0).mean()),
                                     "frac_aligned_B_prefers_own": float((ali["aligned_B"] < 0).mean())}
    if "seqonly_A" in ali.columns:
        out["seqonly_own_preference"] = {"frac_seqonly_A_prefers_own": float((ali["seqonly_A"] > 0).mean()),
                                         "frac_seqonly_B_prefers_own": float((ali["seqonly_B"] < 0).mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epsilon", type=float, default=0.005)
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ex = pd.read_parquet(os.path.join(C.RESULTS, "prot2text_examples.parquet"))
    pred = pd.read_parquet(os.path.join(C.RESULTS, "prot2text_shuffle_predictions.parquet"))
    have_b = os.path.exists(os.path.join(C.RESULTS, "prot2text_conflict_predictions.parquet"))
    pairs = pd.read_parquet(os.path.join(C.RESULTS, "prot2text_pairs.parquet")) if have_b else None
    conf = pd.read_parquet(os.path.join(C.RESULTS, "prot2text_conflict_predictions.parquet")) if have_b else None
    rep = json.load(open(os.path.join(C.RESULTS, "prot2text_reproduction.json")))
    integ = json.load(open(os.path.join(C.RESULTS, "prot2text_shuffle_integrity.json")))["summary"]
    smoke_p = os.path.join(C.RUNS, "smoke", "smoke_summary.json")
    smoke = json.load(open(smoke_p)) if os.path.exists(smoke_p) else {}
    sel_p = os.path.join(C.RESULTS, "prot2text_pair_selection.json")
    sel = json.load(open(sel_p)) if os.path.exists(sel_p) else {}

    A = experiment_a(pred, np.random.default_rng(a.seed), a.resamples)
    res = {"experiment_A": A, "epsilon": a.epsilon, "resamples": a.resamples, "bootstrap_seed": a.seed}
    if have_b:
        B = {}
        for gate in ("sequence_gate_pass", "aligned_gate_pass"):
            for metric in ("biobert", "roberta"):
                B[f"{gate}|{metric}"] = experiment_b(pairs, conf, np.random.default_rng(a.seed), a.resamples, a.epsilon, gate, metric)
        pairs_all = pairs.copy()
        pairs_all["all"] = True
        B["ungated|biobert"] = experiment_b(pairs_all, conf, np.random.default_rng(a.seed), a.resamples, a.epsilon, "all", "biobert")
        res["experiment_B"] = B
        res["experiment_B_primary"] = "sequence_gate_pass|biobert"
    C.write_json(res, os.path.join(C.RESULTS, "prot2text_analysis.json"))

    # ---- summary csv (long) -------------------------------------------------------------------
    rows = []

    def add(block, name, d, n=None, note=""):
        rows.append({"block": block, "metric": name, "value": d["mean"], "ci_low": d["ci95"][0], "ci_high": d["ci95"][1],
                     "n": n if n is not None else d.get("n", ""), "note": note})
    rows.append({"block": "counts", "metric": "n_test", "value": len(ex), "ci_low": "", "ci_high": "", "n": "", "note": "official test split"})
    rows.append({"block": "counts", "metric": "n_within_1021", "value": int(ex.within_1021.sum()), "ci_low": "", "ci_high": "", "n": "", "note": ""})
    rows.append({"block": "counts", "metric": "n_proteins_experiment_A", "value": A["n_proteins"], "ci_low": "", "ci_high": "", "n": "", "note": "all 8 conditions generated"})
    for m in ("biobert", "roberta", "rougeL"):
        for k in ("seq_only_intact", "seq_only_shuffle", "seq_only_delta", "native_intact", "native_shuffle", "native_delta", "name_effect_intact"):
            add(f"A_{m}", k, A[m][k], A["n_proteins"], "shuffle = mean of seeds 0,1,2")
    for k, v in rep.items():
        if isinstance(v, dict) and "f1" in v:
            rows.append({"block": "reproduction", "metric": f"{k}_f1", "value": v["f1"], "ci_low": "", "ci_high": "", "n": rep["n"], "note": "native prompt, all test proteins, official truncation"})
    if have_b:
        rows.append({"block": "counts", "metric": "n_pairs_proposed", "value": len(pairs), "ci_low": "", "ci_high": "", "n": "", "note": ""})
        for key, B1 in res["experiment_B"].items():
            rows.append({"block": f"B_{key}", "metric": "n_pairs_gated", "value": B1["n_pairs"], "ci_low": "", "ci_high": "", "n": "", "note": ""})
            if B1["n_pairs"] == 0:
                continue
            for e in (f"eps_{a.epsilon}", "eps_0"):
                for k in ("sequence_follow_rate", "text_follow_rate", "ambiguous_rate"):
                    add(f"B_{key}", f"{k}@{e}", B1[e][k], B1["n_conflicts"], "both directions pooled, bootstrap over pairs")
            add(f"B_{key}", "mean_preference_toward_sequence", B1["mean_preference_toward_sequence"], B1["n_conflicts"])
            add(f"B_{key}", "mean_preference_shift_from_aligned", B1["mean_preference_shift_from_aligned"], B1["n_conflicts"])
    with open(os.path.join(C.RESULTS, "prot2text_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["block", "metric", "value", "ci_low", "ci_high", "n", "note"])
        w.writeheader()
        w.writerows(rows)

    # ---- qc + manifest ------------------------------------------------------------------------
    manifests = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(C.RUNS, "gen*", "manifest_*.json")))]
    prompt_hash = pred.groupby(["condition", "shuffle_seed"]).protein_id.count().to_dict()
    qc = {"reproduction": rep, "smoke_invariances": smoke.get("invariances"), "smoke_templates": {k: v for k, v in smoke.items() if k in C.LLAMA_TOK},
          "shuffle_integrity": integ, "generations_per_condition": {f"{k[0]}|{k[1]}": int(v) for k, v in prompt_hash.items()},
          "caption_changed": A["caption_changed"], "n_hit_max_new_tokens": A["n_hit_max_new_tokens"], "mean_new_tokens": A["mean_new_tokens"],
          "n_empty_generations": int((pred.generation.str.strip() == "").sum()),
          "exclusions": {"over_1021_aa": int((~ex.within_1021).sum()), "full_name_missing": int(ex.full_name_missing.sum()),
                         "taxon_missing": int(ex.taxon_missing.sum()), "function_missing": int(ex.function_missing.sum()),
                         "nonstandard_aa": int(ex.nonstandard_aa.sum())},
          "pair_selection": sel, "epsilon": a.epsilon}
    if have_b:
        qc["pairs"] = {"n_proposed": int(len(pairs)), "n_sequence_gate": int(pairs.sequence_gate_pass.fillna(False).astype(bool).sum()),
                       "n_aligned_gate": int(pairs.aligned_gate_pass.fillna(False).astype(bool).sum()),
                       "n_both_conflicts_generated": int(pairs.has_both_conflicts.fillna(False).astype(bool).sum()),
                       "taxon_identical_within_pair": True, "names_differ": bool(pairs.names_differ.all())}
    C.write_json(qc, os.path.join(C.RESULTS, "prot2text_qc.json"))
    man = {"repo": C.REPO_URL, "repo_commit": C.REPO_COMMIT, "model": C.MODEL_ID, "model_revision": C.MODEL_REV,
           "esm_tokenizer": [C.ESM_ID, C.ESM_REV], "llama_tokenizer_candidates": C.LLAMA_TOK,
           "llama_tokenizer_used": manifests[0]["llama_tokenizer"] if manifests else None,
           "dataset": [C.DATASET_ID, C.DATASET_REV], "bertscore_models": C.BERTSCORE_MODELS, "bertscore_truncation": C.BERTSCORE_TRUNC,
           "generation": C.GEN_KWARGS, "generation_seed": C.GEN_SEED, "vendored_prompt_sha256_of_dataset_py": C.VENDORED_PROMPT_SHA,
           "system_message": C.SYSTEM_MESSAGE, "placeholder": [C.PLACEHOLDER, C.PLACEHOLDER_ID], "pad": [C.PAD_TOKEN, C.PAD_ID],
           "versions": C.versions(), "job_manifests": manifests, "bootstrap": {"resamples": a.resamples, "seed": a.seed},
           "epsilon": a.epsilon}
    C.write_json(man, os.path.join(C.RESULTS, "run_manifest.json"))

    # ---- what the figure reads ------------------------------------------------------------------
    fig = {"n_proteins": A["n_proteins"], "metric": "BioBERT BERTScore F1 to the reference function",
           "destroyed": {k: A["biobert"][k] for k in ("seq_only_intact", "seq_only_shuffle", "seq_only_delta", "native_intact", "native_shuffle", "native_delta")}}
    if have_b:
        Bp = res["experiment_B"][res["experiment_B_primary"]]
        fig["swapped"] = Bp
    C.write_json(fig, os.path.join(C.RESULTS, "prot2text_figure.json"))
    print(json.dumps(fig, indent=1))


if __name__ == "__main__":
    main()

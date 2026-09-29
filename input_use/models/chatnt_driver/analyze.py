#!/usr/bin/env python
"""Computes the ChatNT results per task and as the mean over the three tasks, with bootstrap intervals
over queries.

Reports accuracy with intact and shuffled sequences (from the Yes/No likelihoods and from the
generations) and, for the evidence conflict condition, the rate at which the answer follows the
paired query's sequence, over all queries and over pairs whose two queries are both answered
correctly when intact. Reads chatnt_examples.parquet and the prediction files under
dna/chatnt/results under $INPUT_USE_RESULTS_DIR and writes chatnt_summary.csv (read by
figures/chatnt.py for Figs. 2 and 3), chatnt_predictions.parquet, chatnt_qc.json,
figure_numbers.json and run_manifest.json there.

    python input_use/models/chatnt_driver/analyze.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chatnt_common import (RESULTS, MODEL_ID, DATA_ID, TASKS, TASK_ORDER, SHUFFLE_SEEDS, DINUC_SEED, BOOT_N, BOOT_SEED,  # noqa: E402
                           sha)

ALL_TASKS = list(TASK_ORDER)
SPEC_COLS = ["example_id", "task", "condition", "shuffle_seed", "question_hash", "sequence_hash", "L_yes",
             "L_no", "margin", "predicted_label", "gold_source_label", "donor_label", "raw_generation",
             "parsed_generation_label"]
CI_METRICS = ["acc_intact", "acc_shuffle_mean", "delta_shuffle", "mean_signed_margin_intact",
              "mean_signed_margin_shuffle", "delta_margin_shuffle", "donor_follow_rate",
              "donor_follow_rate_given_intact_correct", "gated_switch_rate", "acc_shuffle_dinuc",
              "delta_shuffle_dinuc", "gen_acc_intact", "gen_acc_shuffle_mean", "gen_delta_shuffle",
              "gen_donor_follow_rate"]


def task_arrays(pred, ex, task):
    """Per-example aligned arrays for one task."""
    e = ex[ex.task == task].set_index("example_id")
    p = pred[pred.task == task]
    ids = e.index.to_numpy()
    n = len(ids)
    pos = {k: i for i, k in enumerate(ids)}

    def take(cond, seed=None, col="predicted_label"):
        q = p[(p.condition == cond)]
        q = q[q.shuffle_seed.isna()] if seed is None else q[q.shuffle_seed == seed]
        assert len(q) == n and set(q.example_id) == set(ids), (cond, seed, len(q), n)
        arr = np.empty(n, dtype=object)
        arr[[pos[k] for k in q.example_id]] = q[col].to_numpy()
        return arr

    gold = e.gold_A.to_numpy()
    gold_b = e.gold_B.to_numpy()
    donor_pos = np.array([pos[d] for d in e.donor_id])
    sign = np.where(gold == "Yes", 1.0, -1.0)
    A = {"n": n, "gold": gold, "gold_b": gold_b, "donor_pos": donor_pos}
    A["intact_ok"] = (take("intact") == gold)
    A["intact_sm"] = sign * take("intact", col="margin").astype(float)
    A["gen_intact_ok"] = (take("intact", col="parsed_generation_label") == gold)
    A["gen_intact_unparsed"] = pd.isna(take("intact", col="parsed_generation_label"))
    A["shuf_ok"] = np.stack([take("shuffle", s) == gold for s in SHUFFLE_SEEDS], 1)
    A["shuf_sm"] = np.stack([sign * take("shuffle", s, "margin").astype(float) for s in SHUFFLE_SEEDS], 1)
    A["gen_shuf_ok"] = np.stack([take("shuffle", s, "parsed_generation_label") == gold for s in SHUFFLE_SEEDS], 1)
    A["shuf_deg"] = np.stack([take("shuffle", s, "degenerate").astype(bool) for s in SHUFFLE_SEEDS], 1)
    A["dinuc_ok"] = (take("shuffle_dinuc", DINUC_SEED) == gold)
    A["dinuc_deg"] = take("shuffle_dinuc", DINUC_SEED, "degenerate").astype(bool)
    swap = take("swap")
    A["swap_follow"] = (swap == gold_b)
    A["gen_swap_follow"] = (take("swap", col="parsed_generation_label") == gold_b)
    A["gate"] = A["intact_ok"] & A["intact_ok"][donor_pos]          # A correct, and B correct on its own question
    A["n_shuffle_degenerate"] = int(A["shuf_deg"].any(1).sum())
    A["n_dinuc_degenerate"] = int(A["dinuc_deg"].sum())
    return A


def stats(A, idx):
    """All task-level statistics on the resample `idx` (an index array into the examples)."""
    ok_s = ~A["shuf_deg"][idx].any(1)                  # rows usable for the shuffle metrics
    ok_d = ~A["dinuc_deg"][idx]
    io = A["intact_ok"][idx]
    s = {}
    s["n"] = len(idx)
    s["acc_intact"] = io.mean()
    for j, seed in enumerate(SHUFFLE_SEEDS):
        s[f"acc_shuffle_seed{seed}"] = A["shuf_ok"][idx][ok_s, j].mean()
    s["acc_shuffle_mean"] = A["shuf_ok"][idx][ok_s].mean()
    s["delta_shuffle"] = io[ok_s].mean() - s["acc_shuffle_mean"]
    s["mean_signed_margin_intact"] = A["intact_sm"][idx].mean()
    s["mean_signed_margin_shuffle"] = A["shuf_sm"][idx][ok_s].mean()
    s["delta_margin_shuffle"] = (A["intact_sm"][idx][ok_s] - A["shuf_sm"][idx][ok_s].mean(1)).mean()
    s["donor_follow_rate"] = A["swap_follow"][idx].mean()
    s["donor_follow_rate_given_intact_correct"] = A["swap_follow"][idx][io].mean() if io.any() else np.nan
    g = A["gate"][idx]
    s["gated_n"] = int(g.sum())
    s["gated_switch_rate"] = A["swap_follow"][idx][g].mean() if g.any() else np.nan
    s["acc_shuffle_dinuc"] = A["dinuc_ok"][idx][ok_d].mean()
    s["delta_shuffle_dinuc"] = io[ok_d].mean() - s["acc_shuffle_dinuc"]
    s["gen_acc_intact"] = A["gen_intact_ok"][idx].mean()
    s["gen_acc_shuffle_mean"] = A["gen_shuf_ok"][idx][ok_s].mean()
    s["gen_delta_shuffle"] = A["gen_intact_ok"][idx][ok_s].mean() - s["gen_acc_shuffle_mean"]
    s["gen_donor_follow_rate"] = A["gen_swap_follow"][idx].mean()
    s["gen_unparsed_intact"] = int(A["gen_intact_unparsed"][idx].sum())
    return s


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=TASK_ORDER,
                    help="subset for a dry run; outputs then carry a _partial suffix and the figure ignores them")
    a = ap.parse_args()
    tasks = [t for t in ALL_TASKS if t in a.tasks]
    partial = tasks != ALL_TASKS
    sfx = "_partial" if partial else ""
    ex = pd.read_parquet(os.path.join(RESULTS, "chatnt_examples.parquet"))
    cond = pd.read_parquet(os.path.join(RESULTS, "conditions.parquet"))
    ex = ex[ex.task.isin(tasks)].reset_index(drop=True)
    cond = cond[cond.task.isin(tasks)].reset_index(drop=True)
    pred = pd.concat([pd.read_parquet(os.path.join(RESULTS, "predictions", f"{t}.parquet")) for t in tasks],
                     ignore_index=True)
    for df in (cond, pred):
        df["shuffle_seed"] = df["shuffle_seed"].astype("Int64")
        df["degenerate"] = df["degenerate"].astype(bool)
    qc = json.load(open(os.path.join(RESULTS, "qc_data.json")))
    qc["inference"] = {"tasks_analyzed": tasks, "partial": partial}

    # ---- integrity of the prediction table against the pre-registered condition list
    key = ["example_id", "condition", "shuffle_seed"]
    c = cond.set_index(key).sort_index()
    p = pred.set_index(key).sort_index()
    assert len(p) == len(c) and p.index.equals(c.index), (len(p), len(c))
    assert (p.sequence_hash.to_numpy() == c.sequence_hash.to_numpy()).all()
    assert (p.question_hash.to_numpy() == c.question_hash.to_numpy()).all()
    assert set(pred.predicted_label) <= {"Yes", "No"} and set(pred.gold_source_label) <= {"Yes", "No"}
    assert (pred.gold_source_label != pred.donor_label).all()
    assert (pred.groupby("example_id").question_hash.nunique() == 1).all()
    assert (pred.groupby("example_id").task.nunique() == 1).all()
    assert (pred.groupby("example_id").n_english_tokens.nunique() == 1).all(), "prompt token count differs across conditions"
    per_ex = pred.groupby("example_id").size()
    assert (per_ex == 1 + len(SHUFFLE_SEEDS) + 1 + 1).all()
    # the swap row must carry the donor's intact sequence
    intact = pred[pred.condition == "intact"].set_index("example_id").sequence_hash
    swap = pred[pred.condition == "swap"].set_index("example_id")
    assert (swap.sequence_hash.to_numpy() == intact[swap.donor_id].to_numpy()).all()
    qc["inference"]["n_predictions"] = int(len(pred))
    qc["inference"]["n_examples"] = int(per_ex.size)
    qc["inference"]["n_bio_tokens_by_task"] = {t: sorted(pred[pred.task == t].n_bio_tokens.unique().tolist()) for t in tasks}
    qc["inference"]["n_english_tokens_range"] = [int(pred.n_english_tokens.min()), int(pred.n_english_tokens.max())]
    qc["inference"]["first_token_is_yes_or_no_rate"] = float(pred.first_token_argmax.isin(
        pred[pred.condition == "intact"].first_token_argmax.value_counts().head(2).index).mean())
    qc["inference"]["generation_unparsed_by_condition"] = pred.groupby("condition").parsed_generation_label.apply(
        lambda s: int(s.isna().sum())).to_dict()
    qc["inference"]["likelihood_generation_agreement_by_condition"] = pred.groupby("condition").apply(
        lambda d: float((d.predicted_label == d.parsed_generation_label).mean())).to_dict()
    qc["inference"]["predictions_sha256"] = sha(pred[SPEC_COLS].to_json(orient="records"))
    manifests = {}
    for t in tasks:
        mp = os.path.join(RESULTS, f"run_manifest_{t}.json")
        if os.path.exists(mp):
            manifests[t] = json.load(open(mp))
    qc["inference"]["manifests"] = {t: {k: v for k, v in m.items() if k not in ("versions",)} for t, m in manifests.items()}

    # ---- metrics + bootstrap
    rng = np.random.default_rng(BOOT_SEED)
    arrays = {t: task_arrays(pred, ex, t) for t in tasks}
    point, boot = {}, {}
    for t in tasks:
        A = arrays[t]
        point[t] = stats(A, np.arange(A["n"]))
        point[t]["n_shuffle_degenerate_excluded"] = A["n_shuffle_degenerate"]
        point[t]["n_dinuc_degenerate_excluded"] = A["n_dinuc_degenerate"]
        boot[t] = [stats(A, rng.integers(0, A["n"], A["n"])) for _ in range(BOOT_N)]
    rows = []
    for t in tasks:
        r = {"task": t, "task_name": TASKS[t], **point[t]}
        for mname in CI_METRICS:
            v = np.array([b[mname] for b in boot[t]], dtype=float)
            r[f"{mname}_ci_low"], r[f"{mname}_ci_high"] = np.nanpercentile(v, [2.5, 97.5]).tolist()
        rows.append(r)
    macro = {"task": "macro_mean", "task_name": f"macro mean of the {len(tasks)} tasks",
             "n": int(sum(point[t]["n"] for t in tasks)),
             "gated_n": int(sum(point[t]["gated_n"] for t in tasks))}
    for mname in [k for k in point[tasks[0]] if k not in ("n", "gated_n") and not k.startswith("n_") and not k.startswith("gen_unparsed")]:
        macro[mname] = float(np.mean([point[t][mname] for t in tasks]))
    for mname in CI_METRICS:
        v = np.array([[b[mname] for b in boot[t]] for t in tasks], dtype=float).mean(0)
        macro[f"{mname}_ci_low"], macro[f"{mname}_ci_high"] = np.nanpercentile(v, [2.5, 97.5]).tolist()
    rows.append(macro)
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(RESULTS, f"chatnt_summary{sfx}.csv"), index=False)

    merged = pred[SPEC_COLS + [c for c in pred.columns if c not in SPEC_COLS]]
    merged.to_parquet(os.path.join(RESULTS, f"chatnt_predictions{sfx}.parquet"), index=False)
    qc["bootstrap"] = {"n_resamples": BOOT_N, "seed": BOOT_SEED, "unit": "example within task; macro = mean of task statistics per resample"}
    qc["summary_sha256"] = sha(summary.to_json(orient="records"))
    with open(os.path.join(RESULTS, f"chatnt_qc{sfx}.json"), "w") as fh:
        json.dump(qc, fh, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

    # one run_manifest.json for the whole experiment: per-task inference manifests + the smoke tests
    def read(name):
        q = os.path.join(RESULTS, name)
        return json.load(open(q)) if os.path.exists(q) else None
    smoke_t, smoke_j, cmp = read("smoke_torch.json"), read("smoke_jax.json"), read("smoke_compare.json")
    manifest = {
        "experiment": "ChatNT evaluation",
        "model": {"id": MODEL_ID, "revision": next(iter(manifests.values()))["model"]["revision"] if manifests else None,
                  "primary_implementation": "Hugging Face custom code (chatNT.py, TorchMultiOmicsModel), bf16, released tokenizers",
                  "reference_implementation": "official repository JAX path (nucleotide_transformer.chatNT.pretrained.get_chatNT), CPU, notebook example"},
        "data": {"id": DATA_ID, "revision": next(iter(manifests.values()))["data"]["revision"] if manifests else None,
                 "tasks": {t: TASKS[t] for t in tasks}, "split": "test"},
        "official_repo": next(iter(manifests.values()))["official_repo"] if manifests else None,
        "inference": next(iter(manifests.values()))["inference"] if manifests else None,
        "per_task": {t: {k: v for k, v in m.items() if k not in ("versions", "inference", "model", "data", "official_repo")}
                     for t, m in manifests.items()},
        "versions": next(iter(manifests.values()))["versions"] if manifests else None,
        "selection": {"seed": qc.get("select_seed"), "max_per_class": qc.get("max_per_class"),
                      "shuffle_seeds": qc.get("shuffle_seeds"), "dinuc_seed": qc.get("dinuc_seed")},
        "bootstrap": qc["bootstrap"],
        "smoke_torch": None if smoke_t is None else {k: smoke_t[k] for k in ("gpu", "torch_cuda", "dtype", "loading_info",
                                                                            "answer_token_ids", "failures") if k in smoke_t}
        | ({"checks": {k: v["ok"] for k, v in smoke_t["checks"].items()},
            "notebook_generation": smoke_t["notebook_example"]["raw_generation"],
            "integrity_acc_likelihood": {t: v["acc_likelihood"] for t, v in smoke_t["integrity"].items()}} if smoke_t else {}),
        "smoke_jax": None if smoke_j is None else {k: smoke_j.get(k) for k in ("backend", "params_source", "repo_commit")}
        | ({"notebook_generation": smoke_j["notebook_example"]["generated"],
            "matches_notebook_output": smoke_j["notebook_example"]["matches_notebook_output"]} if smoke_j else {}),
        "smoke_compare": None if cmp is None else {k: cmp.get(k) for k in ("n_compared", "n_label_disagreements",
                                                                          "max_abs_margin_diff", "notebook_label_agrees")},
    }
    with open(os.path.join(RESULTS, f"run_manifest{sfx}.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    fig = {}
    for r in rows:
        fig[r["task"]] = {k: r[k] for k in r if k in ("n", "gated_n") or k.startswith(("acc_", "delta_", "donor_", "gated_", "mean_signed"))}
    with open(os.path.join(RESULTS, f"figure_numbers{sfx}.json"), "w") as fh:
        json.dump(fig, fh, indent=2)

    show = ["task_name", "n", "acc_intact", "acc_shuffle_mean", "delta_shuffle", "delta_shuffle_ci_low", "delta_shuffle_ci_high",
            "delta_margin_shuffle", "donor_follow_rate", "gated_n", "gated_switch_rate", "gated_switch_rate_ci_low",
            "gated_switch_rate_ci_high", "acc_shuffle_dinuc", "gen_acc_intact", "gen_acc_shuffle_mean"]
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.3f}".format):
        print(summary[show].to_string(index=False))
    print("wrote chatnt_summary.csv, chatnt_qc.json, chatnt_predictions.parquet, figure_numbers.json, run_manifest.json ->", RESULTS)


if __name__ == "__main__":
    main()

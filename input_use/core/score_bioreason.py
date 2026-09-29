#!/usr/bin/env python
"""Scores BioReason disease prediction runs: accuracy per condition, the paired change from `wt`, and
checks that each perturbation changed only the intended input.

Correctness is re-derived from each raw generation with the label matching of the released
checkpoints (input_use.metrics.dna). Intervals are bootstraps over variants, since several queries
share a variant. With pairs.json in the run directory it also scores query pairs whose answer should
and should not change and the donor variant condition, and with lookup.json it reports the accuracy
of a text-only gene lookup. Writes <rundir>/metrics.json.

    python -m input_use.core.score_bioreason --rundir <run> --records <run>/records_rl.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict

from input_use.core import conditions as C
from input_use.core.stats import bootstrap_ci, mean_ci, perm_test_paired_gt
from input_use.metrics.dna import (bioposttrain_correct, directional_precision,
                                   donor_following_rate, genomic_utilization, gold_named_in,
                                   label_of, pair_accuracy, pair_response_table)

SCORER_VERSION = "1"

_ANSWER_RE = re.compile(r"(?i)answer\s*:")


def load_records(path):
    recs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


LABELS = []          # closed label vocabulary, set from the records in main()


def _correct(r) -> bool:
    """Re-derived from the raw generation every time, so scoring can change without regenerating."""
    return bioposttrain_correct(r.get("output", {}).get("raw", ""),
                                (r.get("ground_truth") or {}).get("answer", ""), LABELS)


def _acc_given_answered(recs):
    """Accuracy over the subset that emitted an 'Answer:' at all, cluster-averaged over variants."""
    by_variant = defaultdict(list)
    for r in recs:
        if _ANSWER_RE.search(r.get("output", {}).get("raw", "")):
            by_variant[_variant_key(r)].append(float(_correct(r)))
    if not by_variant:
        return None
    per = [sum(v) / len(v) for v in by_variant.values()]
    return round(sum(per) / len(per), 4)


def _variant_key(r) -> str:
    """Cluster unit for the bootstrap. 1,449 rows collapse onto 708 distinct variants; treating rows as
    independent would shrink every interval by up to sqrt(1449/708) ~ 1.43.
    """
    return (r.get("input") or {}).get("variant_key") or r["example_id"]


def validity_gates(by_cond) -> dict:
    """Did the intervention actually reach the model, and did it leave the other channel alone?"""
    wt = {r["example_id"]: r.get("input", {}) for r in by_cond.get(C.WT, [])}
    gates, verdict = {}, "VALID"
    for cond, recs in sorted(by_cond.items()):
        if cond == C.WT or not wt:
            continue
        dna_moved = dna_total = text_same = text_total = 0
        for r in recs:
            ref = wt.get(r["example_id"])
            if not ref:
                continue
            inp = r.get("input", {})
            if cond in C.KEGG_GENOME_CONDS:
                text_total += 1
                text_same += inp.get("text_token_sha1") == ref.get("text_token_sha1")
            if inp.get("dna_embed_sha1") is not None and ref.get("dna_embed_sha1") is not None:
                dna_total += 1
                dna_moved += inp["dna_embed_sha1"] != ref["dna_embed_sha1"]
        g = {"n": len(recs),
             "dna_changed_rate": dna_moved / dna_total if dna_total else None,
             "text_identical_rate": text_same / text_total if text_total else None}
        # wt_repeat is the one arm that must not move: it is the decoding noise floor.
        expect_dna_change = cond in C.KEGG_GENOME_CONDS and cond not in (C.WT_REPEAT, C.NO_DNA)
        if expect_dna_change and (g["dna_changed_rate"] or 0) < 1.0:
            g["FAIL"] = "DNA embedding did not change in every example"
            verdict = "INVALID"
        if cond == C.WT_REPEAT and (g["dna_changed_rate"] or 0) > 0.0:
            g["FAIL"] = "wt_repeat changed the DNA embedding"
            verdict = "INVALID"
        if cond in C.KEGG_GENOME_CONDS and g["text_identical_rate"] is not None \
                and g["text_identical_rate"] < 1.0 and cond != C.NO_DNA:
            g["FAIL"] = "a genome arm changed the question text"
            verdict = "INVALID"
        gates[cond] = g
    return {"verdict": verdict, "per_condition": gates}


def per_arm(by_cond, splits_of) -> dict:
    """Accuracy per arm, overall and per split, with a cluster bootstrap over variants."""
    out = {}
    for cond, recs in sorted(by_cond.items()):
        by_variant = defaultdict(list)
        for r in recs:
            by_variant[_variant_key(r)].append(float(_correct(r)))
        clustered = [sum(v) / len(v) for v in by_variant.values()]
        acc, ci = mean_ci(clustered, seed=0)
        entry = {"n_rows": len(recs), "n_clusters": len(clustered),
                 "accuracy": acc, "accuracy_CI": ci,
                 "accuracy_unclustered": round(sum(_correct(r) for r in recs) / len(recs), 4),
                 "reached_gt_in_reasoning": round(
                     sum(gold_named_in(r["ground_truth"].get("answer", ""),
                                   r["output"].get("raw", "")) for r in recs) / len(recs), 4),
                 # Generations naming no disease in the 37-label vocabulary.
                 "unmapped_rate": round(
                     sum(label_of(r["output"].get("raw", ""), LABELS) is None for r in recs)
                     / len(recs), 4),
                 # Rows where the model never emitted "Answer:" at all -- it reasoned past
                 # max_new_tokens and was cut off.
                 "no_answer_emitted_rate": round(
                     sum(not _ANSWER_RE.search(r["output"].get("raw", "")) for r in recs) / len(recs), 4),
                 # Accuracy restricted to rows that emitted an answer; answer rates differ by arm.
                 "accuracy_given_answered": _acc_given_answered(recs),
                 "n_answered": sum(bool(_ANSWER_RE.search(r["output"].get("raw", ""))) for r in recs),
                 "by_split": {}}
        for split in ("train", "test", "val"):
            rs = [r for r in recs if splits_of.get(r["example_id"]) == split]
            if rs:
                entry["by_split"][split] = {
                    "n": len(rs), "accuracy": round(sum(_correct(r) for r in rs) / len(rs), 4)}
        out[cond] = entry
    return out


def deltas(by_cond) -> dict:
    """Paired change vs `wt`, per variant cluster, with a one-sided permutation test (H1: wt higher)."""
    wt_correct, wt_answer, variant_of = {}, {}, {}
    for r in by_cond.get(C.WT, []):
        wt_correct[r["example_id"]] = float(_correct(r))
        wt_answer[r["example_id"]] = label_of(r["output"].get("raw", ""), LABELS)
        variant_of[r["example_id"]] = _variant_key(r)
    out = {}
    for cond, recs in sorted(by_cond.items()):
        if cond == C.WT or not wt_correct:
            continue
        # Pair per row, then collapse rows within a variant before averaging.
        per_variant_d, per_variant_a, per_variant_b = defaultdict(list), defaultdict(list), defaultdict(list)
        changed, n = 0, 0
        for r in recs:
            eid = r["example_id"]
            if eid not in wt_correct:
                continue
            v = variant_of[eid]
            per_variant_a[v].append(wt_correct[eid])
            per_variant_b[v].append(float(_correct(r)))
            per_variant_d[v].append(wt_correct[eid] - float(_correct(r)))
            n += 1
            changed += label_of(r["output"].get("raw", ""), LABELS) != wt_answer[eid]
        if not n:
            continue
        keys = sorted(per_variant_d)
        d = [sum(per_variant_d[k]) / len(per_variant_d[k]) for k in keys]
        a = [sum(per_variant_a[k]) / len(per_variant_a[k]) for k in keys]
        b = [sum(per_variant_b[k]) / len(per_variant_b[k]) for k in keys]
        out[cond] = {"n_paired_rows": n, "n_clusters": len(keys),
                     "delta_vs_wt": round(sum(d) / len(d), 4),
                     "delta_CI": bootstrap_ci(d, seed=0),
                     "p_wt_greater": perm_test_paired_gt(a, b, seed=0),
                     "answer_changed_rate": round(changed / n, 4)}
    return out


def experiment3(by_cond, pairs) -> dict:
    """The strata from kegg_pairs.py, scored on the `wt` arm, plus the donor-swap arm."""
    wt_pred = {r["example_id"]: label_of(r["output"].get("raw", ""), LABELS)
               for r in by_cond.get(C.WT, [])}
    S, T = pairs["strata"]["sensitivity"], pairs["strata"]["stability"]
    res = {
        "sensitivity": {"texts": S["texts"], "rows": len(S["rows"]),
                        **pair_accuracy(S["pairs"], wt_pred)},
        "stability": {"texts": T["texts"], "rows": len(T["rows"]),
                      **pair_accuracy(T["pairs"], wt_pred)},
        "response_table": pair_response_table(S["pairs"] + T["pairs"], wt_pred),
    }
    # Selective genomic utilization, decomposed against a content-destroying control.
    res["genomic_utilization"] = genomic_utilization(
        S["pairs"] + T["pairs"],
        {arm: {r["example_id"]: label_of(r["output"].get("raw", ""), LABELS)
               for r in by_cond.get(arm, [])}
         for arm in (C.WT, C.SCRAMBLE, C.SCRAMBLE_VARIANT, C.NO_DNA, C.NO_MODALITY)
         if by_cond.get(arm)})
    res["directional_precision"] = directional_precision(S["pairs"], wt_pred)

    swap = by_cond.get(C.SWAP_VARIANT_DONOR, [])
    if swap:
        preds = {r["example_id"]: label_of(r["output"].get("raw", ""), LABELS) for r in swap}
        donor_ans = {eid: d["donor_answer"] for eid, d in pairs["donors"].items()}
        gts = {r["example_id"]: r["ground_truth"].get("answer", "") for r in swap}
        res["donor_swap"] = donor_following_rate(preds, donor_ans, gts)
        res["donor_swap_sensitivity_only"] = donor_following_rate(
            {k: v for k, v in preds.items() if k in set(S["rows"])}, donor_ans, gts)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rundir", required=True)
    ap.add_argument("--records", default=None, help="defaults to <rundir>/records.jsonl")
    ap.add_argument("--out", default=None, help="defaults to <rundir>/metrics.json")
    a = ap.parse_args()

    records_path = a.records or os.path.join(a.rundir, "records.jsonl")
    recs = load_records(records_path)
    assert recs, f"no records in {records_path}"

    # The label vocabulary is the set of gold answers present, exactly how the repo builds
    # `self.labels` from the dataset (train_dna_qwen.py:530).
    global LABELS
    LABELS = sorted({(r.get("ground_truth") or {}).get("answer", "") for r in recs} - {""})

    by_cond = defaultdict(list)
    for r in recs:
        by_cond[r["condition"]].append(r)
    splits_of = {r["example_id"]: ((r.get("input") or {}).get("split")
                                   or (r.get("provenance") or {}).get("split")) for r in recs}

    pairs_path = os.path.join(a.rundir, "pairs.json")
    pairs = json.load(open(pairs_path)) if os.path.exists(pairs_path) else None
    lookup_path = os.path.join(a.rundir, "lookup.json")
    lookup = json.load(open(lookup_path)) if os.path.exists(lookup_path) else None

    seen = set(by_cond)
    metrics = {
        "scorer_version": SCORER_VERSION, "records": records_path,
        "n_records": len(recs), "n_examples": len({r["example_id"] for r in recs}),
        "models": sorted({r["model"] for r in recs}),
        "conditions_seen": sorted(seen),
        "conditions_missing": sorted(set(C.KEGG_CONDS) - seen),
        "conditions_unexpected": sorted(seen - set(C.KEGG_CONDS)),
        "rows_per_split": dict(Counter(splits_of.get(r["example_id"]) for r in recs)),
        "validity": validity_gates(by_cond),
        "experiment1_2_per_arm": per_arm(by_cond, splits_of),
        "experiment1_2_deltas": deltas(by_cond),
    }
    if pairs:
        metrics["experiment3"] = experiment3(by_cond, pairs)
    if lookup:
        metrics["lookup_baseline"] = {
            k: {"test": v.get("test", {}).get("accuracy"), "val": v.get("val", {}).get("accuracy"),
                "n_keys": v.get("n_keys")}
            for k, v in lookup["keys"].items()}
        metrics["lookup_baseline"]["majority_class_rate"] = lookup["majority_class_rate"]

    out_path = a.out or os.path.join(a.rundir, "metrics.json")
    with open(out_path, "w") as fh:
        json.dump(metrics, fh, indent=2)

    v = metrics["validity"]["verdict"]
    print(f"[score-kegg] {len(recs)} records | {metrics['n_examples']} examples | "
          f"{len(seen)} arms | validity: {v}")
    if metrics["conditions_missing"]:
        print(f"  MISSING ARMS: {metrics['conditions_missing']}")
    base = (lookup or {}).get("keys", {}).get("stem_gene", {}).get("test", {}).get("accuracy")
    if base:
        print(f"  text-only gene lookup (test): {base:.3f}  <- read every arm against this")
    for cond, e in metrics["experiment1_2_per_arm"].items():
        d = metrics["experiment1_2_deltas"].get(cond, {})
        tag = "  " if cond in C.KEGG_GENOME_CONDS else "T "
        ga = e.get("accuracy_given_answered")
        print(f"  {tag}{cond:22s} acc={e['accuracy']:.3f} {e['accuracy_CI']} "
              f"(n={e['n_rows']}, {e['n_clusters']} clusters)"
              + (f" | given_answered={ga:.3f} (no_ans={e['no_answer_emitted_rate']:.3f})"
                 if ga is not None else "")
              + (f"  d_vs_wt={d['delta_vs_wt']:+.3f} changed={d['answer_changed_rate']:.3f}" if d else ""))
    if "experiment3" in metrics:
        e3 = metrics["experiment3"]
        rt = e3["response_table"]
        print(f"  Exp3 sensitivity: pair_acc={e3['sensitivity']['pair_accuracy']:.3f} "
              f"({e3['sensitivity']['n_pairs']} pairs) | stability pair_acc="
              f"{e3['stability']['pair_accuracy']:.3f} ({e3['stability']['n_pairs']} pairs)")
        print(f"  Exp3 response: sens={rt['sensitivity']:.3f} spec={rt['specificity']:.3f} "
              f"Youden J={rt['youden_j']:+.3f}")
        gu = metrics.get("experiment3", {}).get("genomic_utilization", {})
        if gu.get("content_attributable") is not None:
            print(f"  U_genome: wt={gu['u_genome_wt']:+.3f}  "
                  f"{gu['content_control']}={gu['u_genome_content_control']:+.3f}  "
                  f"-> content-attributable {gu['content_attributable']:+.3f} "
                  f"({gu['fraction_surviving_scramble']:.0%} survives the control)")
            dp = metrics["experiment3"].get("directional_precision", {})
            if dp.get("directional_precision") is not None:
                print(f"  directional precision on changed pairs: {dp['directional_precision']:.3f} "
                      f"({dp['n_correct_direction']}/{dp['n_changed']}, chance 0.5)")
        if "donor_swap" in e3:
            ds = e3["donor_swap"]
            print(f"  Exp3 donor swap: follows_donor={ds['follows_donor_rate']:.3f} "
                  f"follows_original={ds['follows_original_rate']:.3f} (n={ds['n']})")
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()

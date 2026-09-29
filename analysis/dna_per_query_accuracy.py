#!/usr/bin/env python
"""Per-query BioReason disease prediction accuracy for every condition of the KEGG-derived runs, SFT
and RL checkpoints, with 95% percentile bootstrap intervals.

Two intervals are computed with 2,000 draws and seed 0: one resampling genomes with all their
queries, one resampling queries. The genome-averaged accuracy is also stored. Point estimates are
asserted against the scorer's metrics. Writes per_query_accuracy_{rl,sft}.json into
dna/bioreason/perturbations and dna/bioreason/text_conditions under INPUT_USE_RESULTS_DIR, where
figures/perturbation_panels.py (Figure 2a) and figures/sft_vs_rl_panels.py (Figure 5) read them.
"""
import collections
import json
import os
import random
import sys
import input_use.core.score_bioreason as S  # noqa: E402
from input_use.core import paths as RD  # noqa: E402

# records per directory: the perturbation run, and the three shuffled-Evo2 text arms on their own
DIRS = [(RD.DNA_PERTURBATIONS, lambda ck: RD.dna_records(RD.DNA_PERTURBATIONS, ck)),
        (RD.DNA_TEXT_CONDITIONS, RD.dna_shuffled_evo2_arm_records)]
N_BOOT, SEED = 2000, 0


def percentile_ci(draws):
    draws = sorted(draws)
    return (round(draws[int(0.025 * len(draws))], 4),
            round(draws[min(len(draws) - 1, int(0.975 * len(draws)))], 4))


def cluster_boot(hits_by_genome, n_queries):
    """Resample genomes; the estimand stays the per-query mean over the resampled queries."""
    rng = random.Random(SEED)
    keys = sorted(hits_by_genome)
    draws = []
    for _ in range(N_BOOT):
        s = n = 0
        for _ in range(len(keys)):
            v = hits_by_genome[keys[rng.randrange(len(keys))]]
            s += sum(v); n += len(v)
        draws.append(s / n)
    return percentile_ci(draws)


def query_boot(hits):
    rng = random.Random(SEED)
    n = len(hits)
    return percentile_ci([sum(hits[rng.randrange(n)] for _ in range(n)) / n for _ in range(N_BOOT)])


def run(d, records_of):
    for ckpt in ("rl", "sft"):
        recs_path = records_of(ckpt)
        metrics_path = os.path.join(d, f"metrics_{ckpt}.json")
        if not os.path.exists(recs_path) or not os.path.exists(metrics_path):
            print(f"  {os.path.basename(d)}/{ckpt}: no records or metrics, skipped")
            continue
        recs = [json.loads(l) for l in open(recs_path)]
        S.LABELS = sorted({(r.get("ground_truth") or {}).get("answer", "") for r in recs} - {""})
        metrics = json.load(open(metrics_path))["experiment1_2_per_arm"]
        out = {}
        by_arm = collections.defaultdict(dict)
        for r in recs:
            by_arm[r["condition"]][r["example_id"]] = (S._variant_key(r), float(S._correct(r)))
        for arm, rows in sorted(by_arm.items()):
            hits = [h for _, h in rows.values()]
            per_genome = collections.defaultdict(list)
            for g, h in rows.values():
                per_genome[g].append(h)
            acc = sum(hits) / len(hits)
            if arm in metrics and metrics[arm].get("accuracy_unclustered") is not None:
                assert abs(acc - metrics[arm]["accuracy_unclustered"]) < 5e-5, (arm, acc, metrics[arm])
            out[arm] = {"n_queries": len(hits), "n_genomes": len(per_genome),
                        "accuracy_per_query": round(acc, 4),
                        "ci_cluster": list(cluster_boot(per_genome, len(hits))),
                        "ci_query": list(query_boot(hits)),
                        "accuracy_per_genome": round(
                            sum(sum(v) / len(v) for v in per_genome.values()) / len(per_genome), 4)}
        p = RD.dna_per_query(d, ckpt)
        json.dump({"source_records": os.path.basename(recs_path), "n_boot": N_BOOT, "seed": SEED,
                   "per_arm": out}, open(p, "w"), indent=1)
        print(f"  wrote {p}  ({len(out)} arms)")
        for arm in ("wt", "scramble", "no_gene", "scramble_no_gene", "no_pathway",
                    "scramble_no_pathway", "no_textkey", "scramble_no_textkey", "no_dna"):
            if arm in out:
                o = out[arm]
                print(f"    {arm:22s} per-query {o['accuracy_per_query']:.4f} "
                      f"cluster CI {o['ci_cluster']}  query CI {o['ci_query']}  "
                      f"per-genome {o['accuracy_per_genome']:.4f}")


for d, records_of in DIRS:
    print(f"== {d}")
    run(d, records_of)

#!/usr/bin/env python
"""BioReason accuracy with intact and shuffled Evo2 under the four text conditions of Figure 2a, and
the paired difference between them, for the SFT and RL checkpoints on 1,449 queries.

For each text condition it reports intact and shuffled accuracy, their difference, the number of
queries whose mapped label changes and in which direction. It also compares the checkpoints: RL
minus SFT accuracy with intact inputs, and the RL difference between intact and shuffled accuracy
minus the SFT difference. All intervals are 95% percentile bootstrap intervals that resample genomes
with all their queries (2,000 draws, seed 0), paired within query. Writes
outputs/analysis/per_query_pairs.{json,txt} and outputs/analysis/per_query_cross.{json,txt}.
"""
import collections
import json
import os
import random
import sys
import input_use.core.score_bioreason as S  # noqa: E402
from input_use.core import paths as RD  # noqa: E402

OUT_DIR = os.path.join(os.environ.get("INPUT_USE_OUTPUT_DIR", "outputs"), "analysis")
os.makedirs(OUT_DIR, exist_ok=True)

D = RD.DNA_TEXT_CONDITIONS
CONDS = [("All text", "wt", "scramble"),
         ("Pathway fields, gene symbols masked", "no_gene", "scramble_no_gene"),
         ("Query naming the gene", "no_pathway", "scramble_no_pathway"),
         ("Query, gene symbol masked", "no_textkey", "scramble_no_textkey")]
N_BOOT, SEED = 2000, 0


def ci_cluster(values_by_genome):
    """Percentile CI of the per-query mean, resampling genomes."""
    rng = random.Random(SEED)
    keys = sorted(values_by_genome)
    draws = []
    for _ in range(N_BOOT):
        s = n = 0
        for _ in range(len(keys)):
            v = values_by_genome[keys[rng.randrange(len(keys))]]
            s += sum(v); n += len(v)
        draws.append(s / n)
    draws.sort()
    return [round(draws[int(0.025 * N_BOOT)], 4), round(draws[min(N_BOOT - 1, int(0.975 * N_BOOT))], 4)]


out = {}
for ckpt in ("rl", "sft"):
    recs = [json.loads(l) for l in open(RD.dna_records(D, ckpt))]
    S.LABELS = sorted({(r.get("ground_truth") or {}).get("answer", "") for r in recs} - {""})
    hits = collections.defaultdict(dict)      # arm -> example_id -> (genome, 0/1)
    label = collections.defaultdict(dict)     # arm -> example_id -> mapped label
    for r in recs:
        hits[r["condition"]][r["example_id"]] = (S._variant_key(r), float(S._correct(r)))
        label[r["condition"]][r["example_id"]] = S.label_of(r["output"].get("raw", ""), S.LABELS)
    res = {}
    for name, a, b in CONDS:
        eids = sorted(set(hits[a]) & set(hits[b]))
        assert len(eids) == 1449, (ckpt, name, len(eids))
        g = {e: hits[a][e][0] for e in eids}
        acc = {arm: sum(hits[arm][e][1] for e in eids) / len(eids) for arm in (a, b)}
        by_gen = {arm: collections.defaultdict(list) for arm in (a, b)}
        diff_by_gen = collections.defaultdict(list)
        for e in eids:
            for arm in (a, b):
                by_gen[arm][g[e]].append(hits[arm][e][1])
            diff_by_gen[g[e]].append(hits[a][e][1] - hits[b][e][1])
        d = sum(hits[a][e][1] - hits[b][e][1] for e in eids) / len(eids)
        changed = [e for e in eids if label[a][e] != label[b][e]]
        rw = sum(1 for e in changed if hits[a][e][1] and not hits[b][e][1])
        wr = sum(1 for e in changed if not hits[a][e][1] and hits[b][e][1])
        res[name] = {"intact_arm": a, "shuffled_arm": b, "n_queries": len(eids),
                     "intact": round(acc[a], 4), "intact_ci": ci_cluster(by_gen[a]),
                     "shuffled": round(acc[b], 4), "shuffled_ci": ci_cluster(by_gen[b]),
                     "intact_minus_shuffled": round(d, 4), "diff_ci": ci_cluster(diff_by_gen),
                     "changed": len(changed), "changed_pct": round(100 * len(changed) / len(eids), 1),
                     "right_to_wrong": rw, "wrong_to_right": wr}
    out[ckpt] = res

json.dump(out, open(os.path.join(OUT_DIR, "per_query_pairs.json"), "w"), indent=1)
lines = []
for ckpt, res in out.items():
    for name, r in res.items():
        lines.append(f"{ckpt.upper():3s} {name:36s} intact {r['intact']:.4f} {r['intact_ci']}  "
                     f"shuffled {r['shuffled']:.4f} {r['shuffled_ci']}  diff {r['intact_minus_shuffled']:+.4f} "
                     f"{r['diff_ci']}  changed {r['changed']} ({r['changed_pct']}%) R->W {r['right_to_wrong']} "
                     f"W->R {r['wrong_to_right']}")
open(os.path.join(OUT_DIR, "per_query_pairs.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

# ---------------------------------------------------------------- cross-checkpoint contrasts RL -
# SFT with intact inputs, and the difference of the two intact-minus-shuffled differences, both
# paired within query and resampled over genomes.
hits = {}
for ckpt in ("rl", "sft"):
    recs = [json.loads(l) for l in open(RD.dna_records(D, ckpt))]
    S.LABELS = sorted({(r.get("ground_truth") or {}).get("answer", "") for r in recs} - {""})
    hits[ckpt] = collections.defaultdict(dict)
    for r in recs:
        hits[ckpt][r["condition"]][r["example_id"]] = (S._variant_key(r), float(S._correct(r)))
cross, lines2 = {}, []
for name, a, b in CONDS:
    eids = sorted(set(hits["rl"][a]) & set(hits["sft"][a]))
    assert len(eids) == 1449
    g = {e: hits["rl"][a][e][0] for e in eids}
    d_int = collections.defaultdict(list)     # RL - SFT, intact
    d_dd = collections.defaultdict(list)      # (RL intact - RL shuffled) - (SFT intact - SFT shuffled)
    for e in eids:
        d_int[g[e]].append(hits["rl"][a][e][1] - hits["sft"][a][e][1])
        d_dd[g[e]].append((hits["rl"][a][e][1] - hits["rl"][b][e][1])
                          - (hits["sft"][a][e][1] - hits["sft"][b][e][1]))
    m = lambda d: sum(sum(v) for v in d.values()) / sum(len(v) for v in d.values())
    cross[name] = {"rl_minus_sft_intact": round(m(d_int), 4), "ci": ci_cluster(d_int),
                   "diff_in_diff": round(m(d_dd), 4), "diff_in_diff_ci": ci_cluster(d_dd)}
    lines2.append(f"{name:36s} RL-SFT intact {cross[name]['rl_minus_sft_intact']:+.4f} {cross[name]['ci']}  "
                  f"diff-in-diff {cross[name]['diff_in_diff']:+.4f} {cross[name]['diff_in_diff_ci']}")
json.dump(cross, open(os.path.join(OUT_DIR, "per_query_cross.json"), "w"), indent=1)
open(os.path.join(OUT_DIR, "per_query_cross.txt"), "w").write("\n".join(lines2) + "\n")
print("\n".join(lines2))

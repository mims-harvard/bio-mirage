#!/usr/bin/env python
"""Selects the ChatNT test queries and writes every prompt condition to conditions.parquet, without
running the model.

For each task it draws up to 1,000 test queries per label and pairs each query with a query of the
opposite label and similar GC content, whose sequence is used in the evidence conflict condition.
Every query gets the intact sequence, three shuffles that keep base composition, one shuffle that
keeps dinucleotide counts, and the paired query's sequence, all under an identical prompt. Writes
chatnt_examples.parquet, conditions.parquet, donor_map.json and qc_data.json to
dna/chatnt/results under $INPUT_USE_RESULTS_DIR.

    python input_use/models/chatnt_driver/build_examples.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chatnt_common import (DATA_RAW, RESULTS, TASKS, TASK_ORDER, MAX_PER_CLASS, SELECT_SEED,  # noqa: E402
                           SHUFFLE_SEEDS, DINUC_SEED, GC_TOL, sha, build_prompt, parse_exchanges,
                           label_from, opposite, gc_fraction, composition, dinuc_counts, rng_for,
                           shuffle_mono, shuffle_dinuc)


def load_task(task):
    df = pd.read_parquet(os.path.join(DATA_RAW, task, "test.parquet"))
    assert (df.task == task).all()
    assert (df.task_type == "classification").all() and (df.task_modality == "DNA").all()
    q, a = zip(*(parse_exchanges(s) for s in df.exchanges))
    df = df.assign(question=list(q), answer_text=list(a), gold=[label_from(v) for v in df.label])
    df = df.sort_values("sample_id").reset_index(drop=True)
    assert df.sample_id.is_unique
    return df


def select(df, rng):
    counts = df.gold.value_counts().to_dict()
    n = min(min(counts.values()), MAX_PER_CLASS)
    keep = []
    for lab in ("Yes", "No"):
        idx = df.index[df.gold == lab].to_numpy()
        if len(idx) > n:
            idx = np.sort(rng.choice(idx, size=n, replace=False))
        keep.append(idx)
    sel = df.loc[np.sort(np.concatenate(keep))].reset_index(drop=True)
    return sel, counts, n


def donor_map(sel):
    """Symmetric one-to-one Yes<->No assignment minimising the total |GC_A - GC_B| (lengths are equal
    within a task; a length difference would cost 1,000 per base).
    """
    yes = sel.index[sel.gold == "Yes"].to_numpy()
    no = sel.index[sel.gold == "No"].to_numpy()
    assert len(yes) == len(no)
    gy, gn = sel.gc.to_numpy()[yes], sel.gc.to_numpy()[no]
    ly, ln = sel.seq_len.to_numpy()[yes], sel.seq_len.to_numpy()[no]
    cost = np.abs(gy[:, None] - gn[None, :]) + 1000.0 * np.abs(ly[:, None] - ln[None, :])
    r, c = linear_sum_assignment(cost)
    donor = np.empty(len(sel), dtype=int)
    donor[yes[r]] = no[c]
    donor[no[c]] = yes[r]
    return donor


def main():
    os.makedirs(RESULTS, exist_ok=True)
    rng = np.random.default_rng(SELECT_SEED)
    qc = {"tasks": {}, "select_seed": SELECT_SEED, "max_per_class": MAX_PER_CLASS,
          "shuffle_seeds": list(SHUFFLE_SEEDS), "dinuc_seed": DINUC_SEED, "gc_tolerance": GC_TOL}
    ex_rows, cond_rows, dmap = [], [], {}
    for task in TASK_ORDER:
        df = load_task(task)
        t = {"task_name": TASKS[task], "n_test_rows": int(len(df)),
             "class_counts_test": df.gold.value_counts().to_dict()}
        # data integrity on the whole test split
        t["n_exchanges_not_user_assistant"] = 0  # parse_exchanges asserts the structure
        t["n_label_not_yes_no"] = 0              # label_from asserts the vocabulary
        starts = [a.split(",")[0].split(" ")[0].strip().lower() for a in df.answer_text]
        t["n_answer_text_disagrees_with_label"] = int(sum(s != g.lower() for s, g in zip(starts, df.gold)))
        t["n_question_dna_placeholder_ne_1"] = int((df.question.str.count("<DNA>") != 1).sum())
        t["n_seq_non_acgt"] = int(df.sequence.str.contains("[^ACGT]", regex=True).sum())
        t["seq_lengths"] = sorted(df.sequence.str.len().unique().tolist())
        t["n_unique_questions_test"] = int(df.question.nunique())
        assert t["n_question_dna_placeholder_ne_1"] == 0
        assert t["n_answer_text_disagrees_with_label"] == 0, t
        df["seq_len"] = df.sequence.str.len()
        df["gc"] = df.sequence.map(gc_fraction)

        sel, counts, n = select(df, rng)
        t["n_per_class_selected"] = int(n)
        t["n_selected"] = int(len(sel))
        assert sel.gold.value_counts().to_dict() == {"Yes": n, "No": n}
        donor = donor_map(sel)
        assert (donor[donor] == np.arange(len(sel))).all(), "donor map is not symmetric"
        assert (sel.gold.to_numpy()[donor] != sel.gold.to_numpy()).all()
        assert (donor != np.arange(len(sel))).all()
        gcd = np.abs(sel.gc.to_numpy() - sel.gc.to_numpy()[donor])
        t["donor_abs_gc_diff"] = {"mean": float(gcd.mean()), "max": float(gcd.max()),
                                  "frac_within_tol": float((gcd <= GC_TOL).mean()),
                                  "n_over_tol": int((gcd > GC_TOL).sum())}
        t["donor_len_diff_max"] = int(np.abs(sel.seq_len.to_numpy() - sel.seq_len.to_numpy()[donor]).max())
        t["n_unique_donors"] = int(len(set(donor.tolist())))
        assert t["n_unique_donors"] == len(sel)

        shuf_stats = {"n_degenerate_mono": {str(s): 0 for s in SHUFFLE_SEEDS}, "n_degenerate_dinuc": 0,
                      "n_len_mismatch": 0, "n_composition_mismatch": 0, "n_dinuc_mismatch": 0}
        for i, r in sel.iterrows():
            eid = f"{task}:{int(r.sample_id)}"
            b = sel.iloc[int(donor[i])]
            prompt = build_prompt(r.question)
            assert prompt.count("<DNA>") == 1
            qh = sha(r.question)
            ex_rows.append({
                "example_id": eid, "task": task, "task_name": TASKS[task], "sample_id": int(r.sample_id),
                "fasta_header": r.fasta_header, "question": r.question, "question_hash": qh,
                "prompt": prompt, "prompt_hash": sha(prompt), "answer_text": r.answer_text,
                "gold_A": r.gold, "sequence_A": r.sequence, "sequence_A_hash": sha(r.sequence),
                "sequence_length_A": int(r.seq_len), "gc_A": float(r.gc),
                "donor_id": f"{task}:{int(b.sample_id)}", "gold_B": b.gold,
                "sequence_B_hash": sha(b.sequence), "sequence_length_B": int(b.seq_len),
                "gc_B": float(b.gc), "abs_gc_diff": float(abs(r.gc - b.gc)),
            })
            dmap[eid] = f"{task}:{int(b.sample_id)}"
            base = {"example_id": eid, "task": task, "question": r.question, "question_hash": qh,
                    "prompt": prompt, "gold_source_label": r.gold, "donor_label": b.gold,
                    "donor_id": f"{task}:{int(b.sample_id)}"}
            cond_rows.append({**base, "condition": "intact", "shuffle_seed": None,
                              "sequence": r.sequence, "sequence_hash": sha(r.sequence), "degenerate": False})
            comp = composition(r.sequence)
            for s in SHUFFLE_SEEDS:
                x = shuffle_mono(r.sequence, rng_for(s, r.sample_id, 0))
                if len(x) != len(r.sequence):
                    shuf_stats["n_len_mismatch"] += 1
                if composition(x) != comp:
                    shuf_stats["n_composition_mismatch"] += 1
                deg = x == r.sequence
                shuf_stats["n_degenerate_mono"][str(s)] += int(deg)
                cond_rows.append({**base, "condition": "shuffle", "shuffle_seed": s, "sequence": x,
                                  "sequence_hash": sha(x), "degenerate": deg})
            cond_rows.append({**base, "condition": "swap", "shuffle_seed": None, "sequence": b.sequence,
                              "sequence_hash": sha(b.sequence), "degenerate": False})
            x = shuffle_dinuc(r.sequence, rng_for(DINUC_SEED, r.sample_id, 1))
            if dinuc_counts(x) != dinuc_counts(r.sequence) or composition(x) != comp:
                shuf_stats["n_dinuc_mismatch"] += 1
            deg = x == r.sequence
            shuf_stats["n_degenerate_dinuc"] += int(deg)
            cond_rows.append({**base, "condition": "shuffle_dinuc", "shuffle_seed": DINUC_SEED,
                              "sequence": x, "sequence_hash": sha(x), "degenerate": deg})
        assert shuf_stats["n_len_mismatch"] == 0 and shuf_stats["n_composition_mismatch"] == 0
        assert shuf_stats["n_dinuc_mismatch"] == 0
        t["shuffle"] = shuf_stats
        qc["tasks"][task] = t
        print(f"{task}: test {len(df)} rows {counts}; selected {n}/class; donor |dGC| mean "
              f"{gcd.mean():.4f} max {gcd.max():.4f} within {GC_TOL}: {(gcd <= GC_TOL).mean():.3f}; "
              f"degenerate shuffles {shuf_stats['n_degenerate_mono']} dinuc {shuf_stats['n_degenerate_dinuc']}",
              flush=True)

    ex = pd.DataFrame(ex_rows)
    cond = pd.DataFrame(cond_rows)
    # prompt integrity across conditions of one example
    g = cond.groupby("example_id")
    assert (g.question_hash.nunique() == 1).all() and (g.prompt.nunique() == 1).all()
    assert (g.task.nunique() == 1).all()
    assert (cond.prompt.str.count("<DNA>") == 1).all()
    assert ex.example_id.is_unique and set(ex.task) == set(TASK_ORDER)
    assert (ex.gold_A != ex.gold_B).all()
    assert all(ex.set_index("example_id").task[d] == ex.set_index("example_id").task[a] for a, d in dmap.items())
    qc["n_examples"] = int(len(ex))
    qc["n_prompts"] = int(len(cond))
    qc["conditions"] = cond.groupby(["condition", "shuffle_seed"], dropna=False).size().reset_index().values.tolist()
    qc["examples_sha256"] = sha(ex.to_json(orient="records"))
    qc["donor_map_sha256"] = sha(json.dumps(dmap, sort_keys=True))
    qc["conditions_sha256"] = sha(cond.to_json(orient="records"))
    ex.to_parquet(os.path.join(RESULTS, "chatnt_examples.parquet"), index=False)
    cond.to_parquet(os.path.join(RESULTS, "conditions.parquet"), index=False)
    with open(os.path.join(RESULTS, "donor_map.json"), "w") as fh:
        json.dump(dmap, fh, indent=0, sort_keys=True)
    with open(os.path.join(RESULTS, "qc_data.json"), "w") as fh:
        json.dump(qc, fh, indent=2)
    print(f"wrote {len(ex)} examples, {len(cond)} prompts -> {RESULTS}")


if __name__ == "__main__":
    main()

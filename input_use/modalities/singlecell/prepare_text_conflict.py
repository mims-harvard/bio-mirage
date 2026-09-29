#!/usr/bin/env python
"""Builds the C2S-Scale evidence conflicts: the gene sentence of a type A cell with a Cell Ontology
description of another type B.

Pairs exclude ancestor/descendant types and need at least two facts of B that A does not hold and
three shared facts (cl_facts.py). `--mode validation` writes prompts that give only the description
of B, once with all atlas labels and once with the choice between A and B. `--mode conflict` keeps
the pairs whose description alone selects B over A in those validation records and, for cells of
type A, writes the gene sentence alone, with the description of A, of B, or of the shared facts, and
the description of B with the fixed average cell sentence, plus two further variants of the
conflict. Supports Figure 3d.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

from input_use.core import conditions as C
from input_use.core.records import Example, write_jsonl, provenance, read_jsonl
from input_use.metrics.cell_ontology import cl_id_for, resolve_free_text_to_cl
from input_use.metrics.mcq import build_options, parse_choice
from input_use.modalities.singlecell.cl_facts import CLFacts

IC_STRATA = [(0.0, 1.5, "distant"), (1.5, 3.0, "nearby"), (3.0, 4.5, "close"), (4.5, 99.0, "sibling")]


def build_pair_table(cl: CLFacts, ids: dict, max_facts: int):
    """Every admissible ordered pair with its ontology statistics and rendered descriptions."""
    import itertools
    rows = []
    for A, B in itertools.permutations(ids, 2):
        ia, ib = ids[A], ids[B]
        if cl.is_ancestor_descendant(ia, ib):
            continue
        d = cl.contrastive(ib, ia)
        shared = cl.shared_facts(ia, ib)
        # Count facts that will be rendered.
        d_render = [f for f in d if not cl.leaks_identity(f, ib, [A])]
        if len(d_render) < 2 or len(shared) < 3:
            continue
        lca = cl.lca(ia, ib)
        ic = cl.ic(lca) if lca else 0.0
        stratum = next(n for lo, hi, n in IC_STRATA if lo <= ic < hi)
        rows.append({
            "A": A, "B": B, "cl_A": ia, "cl_B": ib,
            "ic_lca": round(ic, 4), "lca": cl.names.get(lca, "?"), "stratum": stratum,
            "n_contrastive": len(d_render), "n_contradictory": cl.n_contradictory(ib, ia),
            "n_shared": len(shared),
            # descriptions are stripped of both labels: B's so the answer cannot be read off, and
            # A's so t_neutral/t_A cannot leak the true type either
            "t_B": cl.describe(d, ib, max_facts, extra_strip=[A]),
            "t_A": cl.describe(cl.contrastive(ia, ib), ia, max_facts, extra_strip=[B]),
            "t_neutral": cl.describe(shared, ib, max_facts, extra_strip=[A, B]),
        })
    return rows


def sample_pairs(rows, n_pairs, seed):
    """Even coverage across ic strata, and within a stratum prefer pairs with contradictory facts --
    they are the ones where the description genuinely denies A rather than merely omitting it.
    """
    rng = random.Random(seed)
    by = defaultdict(list)
    for r in rows:
        by[r["stratum"]].append(r)
    per = max(1, n_pairs // len(IC_STRATA))
    out = []
    for _, _, name in IC_STRATA:
        pool = by.get(name, [])
        rng.shuffle(pool)
        pool.sort(key=lambda r: -r["n_contradictory"])
        out += pool[:per]
    rng.shuffle(out)
    return out[:n_pairs]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["validation", "conflict"], required=True)
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--obo", default="input_use/data/cl-full.obo")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n_candidate_pairs", type=int, default=400,
                    help="validation mode: pairs to test (oversampled, since some fail the gates)")
    ap.add_argument("--n_pairs", type=int, default=200, help="conflict mode: pairs to keep")
    ap.add_argument("--n_cells_per_pair", type=int, default=15)
    ap.add_argument("--n_genes", type=int, default=1000)
    ap.add_argument("--max_facts", type=int, default=6)
    ap.add_argument("--validation_records", default=None)
    ap.add_argument("--organism", default="Homo sapiens")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import anndata as ad
    from cell2sentence.utils import generate_vocabulary, generate_sentences
    from input_use.modalities.singlecell.prepare import build_no_modality_sentence

    os.makedirs(a.outdir, exist_ok=True)
    cl = CLFacts(a.obo)
    adata = ad.read_h5ad(a.h5ad)
    cts = adata.obs["cell_type"].astype(str).tolist()
    vocab_labels = sorted({c for c in cts if c.strip().lower() != "unknown"})
    ids = {L: (cl_id_for(L) or resolve_free_text_to_cl(L)) for L in vocab_labels}
    ids = {L: i for L, i in ids.items() if i}
    print(f"[tc] {len(ids)}/{len(vocab_labels)} labels resolve to CL")

    rows = build_pair_table(cl, ids, a.max_facts)
    print(f"[tc] {len(rows)} admissible ordered pairs; by stratum: "
          f"{dict(Counter(r['stratum'] for r in rows))}")

    # ---------------------------------------------------------------- validation mode
    if a.mode == "validation":
        cand = sample_pairs(rows, a.n_candidate_pairs, a.seed)
        json.dump(cand, open(os.path.join(a.outdir, "candidate_pairs.json"), "w"), indent=2)
        examples = []
        for k, r in enumerate(cand):
            pid = f"{r['cl_B']}__vs__{r['cl_A']}"
            full_opts = build_options(r["B"], list(ids), 0, a.seed, k)
            two = [r["A"], r["B"]]
            random.Random(a.seed * 31 + k).shuffle(two)
            examples.append(Example(
                example_id=pid, modality="singlecell", condition=C.TC_VAL_IDENTIFY,
                payload={"prompt": ("Below is a description of a human cell type.\n"
                                    f"Description: {r['t_B']}\n"
                                    "Choose the cell type it describes from the following list:\n"
                                    + "\n".join(f"- {o}" for o in full_opts) + "\n"
                                    "The cell type being described, copied verbatim from the list "
                                    "above, is:"),
                         "options": full_opts, "num_genes": 0, "organism": a.organism},
                intervention=C.TC_VAL_IDENTIFY, ground_truth={"cell_type": r["B"]},
                provenance=provenance(seed=a.seed, pair=pid, A=r["A"], B=r["B"],
                                      ic_lca=r["ic_lca"], stratum=r["stratum"])))
            examples.append(Example(
                example_id=pid, modality="singlecell", condition=C.TC_VAL_DISCRIMINATE,
                payload={"prompt": ("Below is a description of a human cell type.\n"
                                    f"Description: {r['t_B']}\n"
                                    "Which of these two cell types does it describe?\n"
                                    + "\n".join(f"- {o}" for o in two) + "\n"
                                    "The cell type being described, copied verbatim from the list "
                                    "above, is:"),
                         "options": two, "num_genes": 0, "organism": a.organism},
                intervention=C.TC_VAL_DISCRIMINATE, ground_truth={"cell_type": r["B"]},
                provenance=provenance(seed=a.seed, pair=pid, A=r["A"], B=r["B"],
                                      ic_lca=r["ic_lca"], stratum=r["stratum"])))
        write_jsonl(examples, os.path.join(a.outdir, "validation", "examples.jsonl"))
        json.dump({"mode": "validation", "n_candidate_pairs": len(cand),
                   "n_examples": len(examples), "dataset": Path(a.h5ad).name,
                   "by_stratum": dict(Counter(r["stratum"] for r in cand)), "seed": a.seed},
                  open(os.path.join(a.outdir, "validation", "meta.json"), "w"), indent=2)
        print(f"[tc] validation: {len(cand)} pairs x 2 gates = {len(examples)} questions "
              f"-> {a.outdir}/validation")
        return

    # ---------------------------------------------------------------- conflict mode
    if not a.validation_records:
        raise SystemExit("--mode conflict requires --validation_records")
    passed = defaultdict(dict)
    for rec in read_jsonl(a.validation_records):
        p = parse_choice(rec["output"]["raw"], rec["input"]["options"])
        passed[rec["example_id"]][rec["condition"]] = (
            p["cell_type"] == rec["ground_truth"]["cell_type"])
    # vocabulary: gate 2 passes 83% while gate 1 passes only 19%, and the gap is by construction --
    # D(B,A) is built to separate B from A, not from all 30 other labels, so a description can be
    # perfectly conflict-bearing (its job) while failing to single B out of the full list (not its
    # job).
    ok = {pid for pid, g in passed.items() if g.get(C.TC_VAL_DISCRIMINATE)}
    g1 = sum(1 for g in passed.values() if g.get(C.TC_VAL_IDENTIFY))
    g2 = sum(1 for g in passed.values() if g.get(C.TC_VAL_DISCRIMINATE))
    print(f"[tc] validation: {len(passed)} pairs tested | gate1 identify {g1} "
          f"({g1/max(1,len(passed)):.0%}) | gate2 discriminate {g2} "
          f"({g2/max(1,len(passed)):.0%}) | BOTH {len(ok)} ({len(ok)/max(1,len(passed)):.0%})")
    if len(ok) < 20:
        raise SystemExit(f"only {len(ok)} pairs passed both gates -- too few; the ontology text is "
                         f"not discriminative enough for this vocabulary")

    by_pid = {f"{r['cl_B']}__vs__{r['cl_A']}": r for r in rows}
    survivors = [by_pid[p] for p in ok if p in by_pid]
    chosen = sample_pairs(survivors, a.n_pairs, a.seed)
    print(f"[tc] keeping {len(chosen)} pairs; by stratum: "
          f"{dict(Counter(r['stratum'] for r in chosen))}")

    vocabulary = generate_vocabulary(adata)
    sents = generate_sentences(adata, vocabulary)
    idx_by_type = defaultdict(list)
    for i, ct in enumerate(cts):
        idx_by_type[ct].append(i)
    rng = random.Random(a.seed)
    all_idx = sorted({i for r in chosen for i in idx_by_type[r["A"]]})
    topN = {i: sents[i].split(" ")[: a.n_genes] for i in all_idx}
    generic = build_no_modality_sentence(adata, all_idx, topN, a.n_genes, "mean_expression")
    opts_vocab = list(ids)
    # per-A reductio target: the vocabulary type whose lca with A is least informative
    reductio = {A: min((b for b in ids if b != A),
                       key=lambda b: cl.ic(cl.lca(ids[A], ids[b]) or "CL:0000000"))
                for A in {r["A"] for r in chosen}}

    def q(options, sentence, n_genes, ctx=""):
        return (f"The following is a list of {n_genes} gene names ordered by descending expression "
                f"level in a {a.organism} cell. Your task is to give the cell type which this cell "
                f"belongs to based on its gene expression.\n"
                + (f"Context: {ctx}\n" if ctx else "")
                + "Choose your answer from the following list of cell types:\n"
                + "\n".join(f"- {o}" for o in options) + "\n"
                + f"Cell sentence: {sentence}.\n"
                + "The cell type corresponding to these genes, copied verbatim from the list above, is:")

    examples, n_cells = [], 0
    for k, r in enumerate(chosen):
        pool = list(idx_by_type[r["A"]])
        rng.shuffle(pool)
        red_B = reductio[r["A"]]
        t_red = cl.describe(cl.contrastive(ids[red_B], ids[r["A"]]), ids[red_B],
                            a.max_facts, extra_strip=[r["A"], red_B])
        for i in pool[: a.n_cells_per_pair]:
            n_cells += 1
            sent = " ".join(topN[i])
            ng, ngg = len(topN[i]), len(generic.split(" "))
            opts = build_options(r["A"], opts_vocab, 0, a.seed, i)
            gt = {"cell_type": r["A"]}
            prov = provenance(seed=a.seed, pair=f"{r['cl_B']}__vs__{r['cl_A']}",
                              A=r["A"], B=r["B"], ic_lca=r["ic_lca"], stratum=r["stratum"],
                              n_contrastive=r["n_contrastive"],
                              n_contradictory=r["n_contradictory"], reductio_B=red_B)
            for cond, sentence, n, ctx in (
                    (C.TC_XA, sent, ng, ""),
                    (C.TC_XA_TA, sent, ng, r["t_A"]),
                    (C.TC_XA_TB, sent, ng, r["t_B"]),
                    (C.TC_XA_TNEUTRAL, sent, ng, r["t_neutral"]),
                    (C.TC_TB_ONLY, generic, ngg, r["t_B"]),
                    (C.TC_XA_TB_REDUCTIO, sent, ng, t_red)):
                examples.append(Example(
                    example_id=f"cell{i}__{r['cl_B']}", modality="singlecell", condition=cond,
                    payload={"prompt": q(opts, sentence, n, ctx), "options": opts,
                             "num_genes": n, "organism": a.organism,
                             "target_B": r["B"], "context": ctx},
                    intervention=cond, ground_truth=gt, provenance=prov))
            two = [r["A"], r["B"]]
            random.Random(a.seed * 977 + i).shuffle(two)
            examples.append(Example(
                example_id=f"cell{i}__{r['cl_B']}", modality="singlecell",
                condition=C.TC_XA_TB_2WAY,
                payload={"prompt": q(two, sent, ng, r["t_B"]), "options": two,
                         "num_genes": ng, "organism": a.organism,
                         "target_B": r["B"], "context": r["t_B"]},
                intervention=C.TC_XA_TB_2WAY, ground_truth=gt, provenance=prov))

    write_jsonl(examples, os.path.join(a.outdir, "examples.jsonl"))
    json.dump(chosen, open(os.path.join(a.outdir, "pairs.json"), "w"), indent=2)
    json.dump({"mode": "conflict", "dataset": Path(a.h5ad).name, "n_pairs": len(chosen),
               "n_cells": n_cells, "n_examples": len(examples),
               "conditions": sorted({e.condition for e in examples}),
               "n_genes": a.n_genes, "max_facts": a.max_facts, "n_choices": len(opts_vocab),
               "by_stratum": dict(Counter(r["stratum"] for r in chosen)),
               "validation": {"tested": len(passed), "gate1": g1, "gate2": g2, "both": len(ok)},
               "seed": a.seed},
              open(os.path.join(a.outdir, "meta.json"), "w"), indent=2)
    print(f"[tc] {len(chosen)} pairs x {a.n_cells_per_pair} cells = {n_cells} cells "
          f"x {len(C.TEXT_CONFLICT_CONDS)} conditions = {len(examples)} questions -> {a.outdir}")


if __name__ == "__main__":
    main()

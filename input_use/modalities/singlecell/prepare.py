#!/usr/bin/env python
"""Builds C2S-Scale cell type annotation inputs (examples.jsonl, meta.json, candidates.json) from an
AnnData file.

Samples cells balanced across annotated cell types, writes each cell's top --n_genes genes as a gene
sentence, and builds the conditions of interventions.py (gene order shuffling, gene set replacement,
DEG removal). --multiple_choice adds one shuffled option list per cell, shared by all of its
conditions. --text_reliance_tests adds the metadata text conditions of text_conditions.py.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from input_use.core import conditions as C
from input_use.core.records import Example, write_jsonl, provenance
from input_use.metrics.mcq import build_options
from input_use.modalities.singlecell.importance import compute_markers, salient_and_control_genes
from input_use.modalities.singlecell.interventions import make_conditions
from input_use.modalities.singlecell.text_conditions import make_text_conditions

TEXT_METADATA_COLS = ["tissue", "sex", "development_stage", "donor_id"]


def load_source(source, h5ad):
    import scanpy as sc
    import anndata as ad
    if source == "pbmc3k":
        proc = sc.datasets.pbmc3k_processed()
        adata = proc.raw.to_adata()                                # log-normalized counts, all genes
        adata.obs["cell_type"] = proc.obs["louvain"].astype(str).values
        return adata, "scanpy:pbmc3k_processed"
    adata = ad.read_h5ad(h5ad)
    if "cell_type" not in adata.obs and "louvain" in adata.obs:
        adata.obs["cell_type"] = adata.obs["louvain"].astype(str).values
    return adata, Path(h5ad).name


def _row(adata, i):
    x = adata.X[i]
    return np.asarray(x.todense()).ravel() if hasattr(x, "todense") else np.asarray(x).ravel()


def select_balanced_cells(cell_types: list, n_cells: int, seed: int) -> list:
    """Balanced sample of up to n_cells cell indices across `cell_types`, seeded by `seed`. Factored
    out of `build()` so any runner that builds its own per-condition input rather than going through
    examples.jsonl (e.g.
    """
    rng = np.random.RandomState(seed)
    idx_by_type = defaultdict(list)
    for i, ct in enumerate(cell_types):
        if ct.strip().lower() == "unknown":
            continue
        idx_by_type[ct].append(i)
    per = max(1, n_cells // max(1, len(idx_by_type)))
    chosen = []
    for ct, idxs in idx_by_type.items():
        idxs = list(idxs); rng.shuffle(idxs); chosen += idxs[:per]
    rng.shuffle(chosen)
    return chosen[:n_cells]


def build_no_modality_sentence(adata, chosen, topN, n_genes, mode="mean_expression"):
    """The fixed, cell-information-free sentence used for `no_modality` (the M_0 prior baseline)."""
    if mode == "topk_frequency":
        freq = Counter(g for i in chosen for g in topN[i])
        return " ".join(g for g, _ in freq.most_common(n_genes))
    if mode != "mean_expression":
        raise ValueError(f"unknown no_modality_ref {mode!r}")
    X = adata.X
    mean = np.asarray(X.mean(axis=0)).ravel() if hasattr(X, "mean") else np.asarray(X).mean(axis=0)
    names = np.array([str(n).upper() for n in adata.var_names])
    order = np.argsort(-mean)[:n_genes]
    return " ".join(names[order].tolist())


def build(adata, dataset_name, n_genes, k_salient, n_cells, organism, seed, battery="both",
          text_reliance_tests=False, multiple_choice=False, n_choices=0,
          deg_dropout_mode="fraction", no_modality_ref="mean_expression",
          min_log2fc=1.0, max_padj=0.05):
    from cell2sentence.utils import generate_vocabulary, generate_sentences
    # stress_test_1 (scramble_rank/scramble_geneset) needs no marker/de info at all -- only
    # stress_test_2's dropout/jitter conditions are built from salient/control marker genes.
    need_markers = battery in ("stress_test_2", "both")
    need_text = text_reliance_tests
    # n_top=None -> scanpy returns all genes and the log2FC/fdr thresholds decide the DEG set,
    # rather than a top-100 rank cutoff deciding it before the thresholds are ever applied.
    markers = compute_markers(adata, "cell_type", n_top=None, min_log2fc=min_log2fc,
                              max_padj=max_padj) if need_markers else {}
    if need_text:
        missing = [c for c in TEXT_METADATA_COLS if c not in adata.obs.columns]
        if missing:
            raise ValueError(f"--text_reliance_tests needs obs columns {TEXT_METADATA_COLS}; missing "
                             f"{missing} in {dataset_name} (e.g. cross_tissue_immune.h5ad has them; "
                             f"pbmc3k does not)")
        text_meta = {col: adata.obs[col].astype(str).tolist() for col in TEXT_METADATA_COLS}
    vocab = generate_vocabulary(adata)
    sents = generate_sentences(adata, vocab)                        # uppercased ranked gene names
    var_up = [n.upper() for n in adata.var_names]
    name2idx = {n: j for j, n in enumerate(var_up)}
    cell_types = adata.obs["cell_type"].astype(str).tolist()

    chosen = select_balanced_cells(cell_types, n_cells, seed)

    topN = {i: sents[i].split(" ")[:n_genes] for i in chosen}
    generic = build_no_modality_sentence(adata, chosen, topN, n_genes, no_modality_ref)

    # Multiple-choice option vocabulary.
    mc_vocab = sorted({cell_types[i] for i in chosen}) if multiple_choice else []
    if multiple_choice:
        print(f"[sc-prepare] multiple choice: {len(mc_vocab)} label vocabulary, "
              f"n_choices={n_choices or len(mc_vocab)} per question (shuffled per cell)")

    examples = []
    for i in chosen:
        genes = topN[i]
        sentence = " ".join(genes)
        remainder = sents[i].split(" ")[n_genes:]  # this cell's own real genes ranked beyond the visible window
                                                     # (generate_sentences already excludes zero-count genes, so
                                                     # this is never filler) -- substitute pool for
                                                     # scramble_geneset
        salient, control = [], []
        if need_markers:
            expr = _row(adata, i)
            expr_by_gene = {g: float(expr[name2idx[g]]) for g in genes if g in name2idx}
            # fraction mode needs the full DEG set for "% of DEGs" to mean what it says; the older
            # absolute-k series truncates to k_salient. See salient_and_control_genes.
            sc = salient_and_control_genes(genes, cell_types[i], markers, expr_by_gene,
                                           k=(None if deg_dropout_mode == "fraction" else k_salient))
            salient, control = sc["salient"], sc["control"]
            if len(salient) < 3:  # only a meaningful filter when stress_test_2 needs real markers
                continue
        cell_rng = random.Random(seed * 1_000_003 + i)  # deterministic per-cell rank-jitter draws cond ->
        # (cell_sentence, text_fields); text_fields is {} for every
        # non-text condition, so payload/runners stay backward
        # compatible.
        conds = {cond: (s, {}) for cond, s in make_conditions(
            sentence, salient, control, generic, rng=cell_rng,
            remainder_genes=remainder, battery=battery,
            deg_dropout_mode=deg_dropout_mode).items()}
        if need_text:
            text_conds = make_text_conditions(
                sentence, generic, tissue=text_meta["tissue"][i], sex=text_meta["sex"][i],
                development_stage=text_meta["development_stage"][i], donor_id=text_meta["donor_id"][i],
                cell_type=cell_types[i])
            for name, d in text_conds.items():
                conds[name] = (d["cell_sentence"], d["text_fields"])
        gt = {"cell_type": cell_types[i]}
        # One option list per cell, reused across all of that cell's conditions -- so a
        # wt-vs-perturbed accuracy difference is never confounded by a reshuffle. See
        # metrics/mcq.py:build_options.
        options = build_options(cell_types[i], mc_vocab, n_choices, seed, i) if multiple_choice else None
        prov = provenance(
            importance="rank_genes_groups_wilcoxon" if need_markers else "none (no marker computation needed)",
            seed=seed, dataset=dataset_name)
        for cond, (s, text_fields) in conds.items():
            payload = {"cell_sentence": s, "organism": organism,
                       "num_genes": len(s.split(" ")) if s else 0,
                       "salient": salient, "control": control,
                       "text_fields": text_fields}
            if options is not None:
                payload["options"] = options
            examples.append(Example(
                example_id=f"cell{i}", modality="singlecell", condition=cond,
                payload=payload, intervention=cond,
                ground_truth=gt, provenance=prov))
    return examples


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="pbmc3k", help="pbmc3k (scanpy built-in) or 'h5ad'")
    ap.add_argument("--h5ad", default=None)
    ap.add_argument("--n_cells", type=int, default=120)
    ap.add_argument("--n_genes", type=int, default=200)
    ap.add_argument("--k_salient", type=int, default=25)
    ap.add_argument("--organism", default="Homo sapiens")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--battery", choices=["stress_test_1", "stress_test_2", "both", "none"],
                    default="both",
                    help="stress_test_1 = scramble_rank/scramble_geneset only (skips marker/DE "
                         "computation entirely); stress_test_2 = dropout/jitter grounding-gap "
                         "conditions only (needs markers); both = everything (default); none = no "
                         "stress-test conditions at all (e.g. a lightweight wt-only calibration run). "
                         "wt/no_modality are always produced regardless. Combine with "
                         "--text_reliance_tests to ALSO get the metadata-text-ablation suite on the "
                         "SAME sampled cells.")
    ap.add_argument("--text_reliance_tests", action="store_true",
                    help="ALSO produce the metadata-text-ablation suite (text_full/text_drop_donor/"
                         "text_drop_donor_stage/text_drop_donor_stage_sex/text_full_no_modality/"
                         "text_positive_control) on the same sampled cells, alongside whichever "
                         "--battery was requested. Needs obs columns tissue/sex/development_stage/"
                         "donor_id, e.g. --h5ad data/cross_tissue_immune.h5ad.")
    ap.add_argument("--multiple_choice", action="store_true",
                    help="emit payload['options'] -- a per-cell SHUFFLED list of candidate cell-type "
                         "names -- turning the task into forced choice over named options (the model "
                         "is asked for a name, never a letter; see metrics/mcq.py). The same option "
                         "list is reused across every condition of a cell.")
    ap.add_argument("--n_choices", type=int, default=0,
                    help="options per question (0 = the dataset's whole cell-type vocabulary, the "
                         "default). Smaller values keep the ground truth and sample the rest as "
                         "distractors; note this raises chance accuracy to 1/n_choices.")
    ap.add_argument("--deg_dropout_mode", choices=["fraction", "absolute"], default="fraction",
                    help="fraction (default): drop ceil(f*|S+|) DEGs for f in conditions."
                         "DEG_DROPOUT_FRACTIONS, so the dose means the same thing across cells and "
                         "atlases; absolute: the older fixed-k series (k in DEG_DROPOUT_KS).")
    ap.add_argument("--no_modality_ref", choices=["mean_expression", "topk_frequency"],
                    default="mean_expression",
                    help="how the fixed information-free sentence is built (see "
                         "build_no_modality_sentence). mean_expression = genes ranked by mean "
                         "expression over ALL cells in the atlas (the 'average cell').")
    ap.add_argument("--min_log2fc", type=float, default=1.0,
                    help="DEG threshold: minimum one-vs-rest log2 fold change (default 1.0 = 2x).")
    ap.add_argument("--max_padj", type=float, default=0.05,
                    help="DEG threshold: maximum BH/FDR-corrected p-value (default 0.05).")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    adata, dataset_name = load_source(a.source, a.h5ad)
    print(f"[sc-prepare] {dataset_name}: {adata.shape}, types={dict(Counter(adata.obs['cell_type']))}, "
          f"battery={a.battery}, text_reliance_tests={a.text_reliance_tests}, "
          f"multiple_choice={a.multiple_choice}, n_genes(K)={a.n_genes}")
    examples = build(adata, dataset_name, a.n_genes, a.k_salient, a.n_cells, a.organism, a.seed,
                     battery=a.battery, text_reliance_tests=a.text_reliance_tests,
                     multiple_choice=a.multiple_choice, n_choices=a.n_choices,
                     deg_dropout_mode=a.deg_dropout_mode, no_modality_ref=a.no_modality_ref,
                     min_log2fc=a.min_log2fc, max_padj=a.max_padj)
    n_cells = len({e.example_id for e in examples})
    write_jsonl(examples, os.path.join(a.outdir, "examples.jsonl"))
    # Global candidate-label vocabulary (every distinct ground-truth cell type across all examples),
    # written once here so sharded candidate-scoring workers (--shard_index/--num_shards) can load a
    # consistent candidate set via --candidates_file instead of deriving it from just their own
    # shard of cells -- a shard may not see every cell type, which would otherwise give shards
    # mismatched candidate vocabularies and break candidate_scores comparability after merging.
    candidates = sorted({e.ground_truth["cell_type"] for e in examples})
    json.dump(candidates, open(os.path.join(a.outdir, "candidates.json"), "w"), indent=2)
    ct = Counter(e.ground_truth["cell_type"] for e in examples if e.condition == "wt")
    need_markers = a.battery in ("stress_test_2", "both")
    json.dump({"modality": "singlecell", "dataset": dataset_name, "n_cells": n_cells,
               "n_examples": len(examples), "conditions": sorted({e.condition for e in examples}),
               "battery": a.battery, "text_reliance_tests": a.text_reliance_tests,
               "multiple_choice": a.multiple_choice,
               "n_choices": (a.n_choices or len(candidates)) if a.multiple_choice else None,
               "n_genes": a.n_genes, "k_salient": a.k_salient,
               "deg_dropout_mode": a.deg_dropout_mode, "no_modality_ref": a.no_modality_ref,
               "deg_min_log2fc": a.min_log2fc, "deg_max_padj": a.max_padj,
               "organism": a.organism, "seed": a.seed, "cell_type_counts": dict(ct),
               "importance_source": ("scanpy rank_genes_groups (Wilcoxon) over annotated cell types"
                                    if need_markers else "none (no marker computation needed)")},
              open(os.path.join(a.outdir, "meta.json"), "w"), indent=2)
    print(f"[sc-prepare] {n_cells} cells -> {len(examples)} examples -> {a.outdir}")


if __name__ == "__main__":
    main()

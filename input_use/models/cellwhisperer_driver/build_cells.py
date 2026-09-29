"""Builds the cell table and the list of cell pairs per atlas for the CellWhisperer evaluation.
First step of four (build_cells, embed_cells, score_llm, analyze). Runs on CPU in CellWhisperer's
pixi environment.

For each atlas it reads the `wt` queries of the C2S-Scale question set
(single_cell/c2s_scale/deg_removal/<atlas>/examples.jsonl under $INPUT_USE_RESULTS_DIR) and keeps
the cells whose cell sentence has 1,000 genes, so CellWhisperer is evaluated on the same 4,846 cells
as C2S-Scale. It takes the raw counts of those cells from $INPUT_USE_DATA_DIR/prepped/<atlas>.h5ad
(query `cellN` is row N), checks that the label agrees and that the cell sentence lists the cell's
expressed genes in descending order of count, and derives the gene text lists of the top k = 50,
100, 200, 500 and 1,000 genes. It then forms 200 pairs of cells with different labels (seed 0, same
tissue and donor preferred, each cell in at most one pair) for the evidence conflicts of Fig. 3e.

Writes cells.parquet, pairs.parquet, their .jsonl copies for the scoring step,
selected_counts.h5ad (raw counts of the kept cells, read by embed_cells.py) and cells_qc.json under
single_cell/cellwhisperer/<atlas>/ under $INPUT_USE_RESULTS_DIR.

    python input_use/models/cellwhisperer_driver/build_cells.py [--atlases immune1 ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    ATLASES, C2S_RESULTS, KS, N_PAIRS_TARGET, PAIR_SEED, PREPPED, atlas_dir,
    canonical_label, dump_json, read_jsonl, sha256_text, write_jsonl)


def load_c2s_wt(atlas: str):
    path = C2S_RESULTS / atlas / "examples.jsonl"
    rows = read_jsonl(path, cond=lambda r: r["condition"] == "wt")
    assert rows, f"no wt rows in {path}"
    return rows


def make_pairs(cells: pd.DataFrame, seed: int, target: int):
    """Greedy pairing in a seeded order. Tiers: same tissue and donor, then same tissue, then same
    donor, then any. Within a tier the candidate closest in log total counts plus log detected genes
    is taken."""
    rng = np.random.RandomState(seed)
    n = len(cells)
    order = rng.permutation(n)
    canon = cells["canonical_label"].to_numpy()
    tissue = cells["tissue"].fillna("NA").to_numpy()
    donor = cells["donor_id"].fillna("NA").to_numpy()
    has_donor = cells["donor_id"].notna().all()
    ltot = np.log1p(cells["raw_count_sum"].to_numpy(dtype=float))
    lng = np.log(cells["n_detected_genes"].to_numpy(dtype=float))
    used = np.zeros(n, dtype=bool)
    pairs = []
    for i in order:
        if used[i]:
            continue
        cand = np.where((~used) & (canon != canon[i]) & (np.arange(n) != i))[0]
        if len(cand) == 0:
            continue
        st = tissue[cand] == tissue[i]
        sd = (donor[cand] == donor[i]) if has_donor else np.zeros(len(cand), dtype=bool)
        tier = np.where(st & sd, 0, np.where(st, 1, np.where(sd, 2, 3)))
        dist = np.abs(ltot[cand] - ltot[i]) + np.abs(lng[cand] - lng[i])
        j = cand[np.lexsort((dist, tier))[0]]
        pairs.append({"cell_A": cells["cell_id"].iat[i], "cell_B": cells["cell_id"].iat[j],
                      "label_A": cells["ground_truth_label"].iat[i],
                      "label_B": cells["ground_truth_label"].iat[j],
                      "canonical_A": canon[i], "canonical_B": canon[j],
                      "same_tissue": bool(tissue[i] == tissue[j]),
                      "same_donor": (bool(donor[i] == donor[j]) if has_donor else None),
                      "match_tier": int(tier[np.where(cand == j)[0][0]]),
                      "log_count_diff": float(abs(ltot[i] - ltot[j])),
                      "log_ngenes_diff": float(abs(lng[i] - lng[j]))})
        used[i] = used[j] = True
        if len(pairs) >= target:
            break
    return pairs


def swap_partners(cells: pd.DataFrame, seed: int):
    """One random cell with a different label per cell (same tissue when possible), seed-fixed."""
    rng = np.random.RandomState(seed)
    canon = cells["canonical_label"].to_numpy()
    tissue = cells["tissue"].fillna("NA").to_numpy()
    out = []
    for i in range(len(cells)):
        cand = np.where((canon != canon[i]) & (tissue == tissue[i]))[0]
        if len(cand) == 0:
            cand = np.where(canon != canon[i])[0]
        out.append(cells["cell_id"].iat[int(rng.choice(cand))])
    return out


def build_atlas(atlas: str, n_pairs: int):
    import anndata

    qc = {"atlas": atlas}
    wt = load_c2s_wt(atlas)
    qc["n_wt_rows"] = len(wt)
    wt = [r for r in wt if r["payload"]["num_genes"] >= 1000]
    qc["n_condition_complete_cells"] = len(wt)

    ad = anndata.read_h5ad(PREPPED / f"{atlas}.h5ad", backed="r")
    obs = ad.obs
    var_upper = np.array([str(v).upper() for v in ad.var_names])
    var_set = set(var_upper)
    idx = np.array([int(r["example_id"][4:]) for r in wt])
    assert (idx < ad.n_obs).all()
    sub = ad[np.sort(idx)].to_memory()          # ordered by row index
    del ad
    counts = sub.layers["counts"]
    counts = sp.csr_matrix(counts) if sp.issparse(counts) else sp.csr_matrix(np.asarray(counts))
    rowpos = {int(r): p for p, r in enumerate(np.sort(idx))}
    assert np.abs(counts.data - np.round(counts.data)).max() < 1e-6, "counts are not integers"

    recs, bad_label, bad_order, bad_subset, dup_genes = [], 0, 0, 0, 0
    for r in wt:
        eid = r["example_id"]
        ri = int(eid[4:])
        p = rowpos[ri]
        sent = r["payload"]["cell_sentence"]
        genes = sent.split(" ")
        genes = [g for g in genes if g]
        assert len(genes) == 1000, (eid, len(genes))
        if len(set(genes)) != len(genes):
            dup_genes += 1
        gt = r["ground_truth"]["cell_type"]
        obs_label = str(obs["cell_type"].iloc[ri])
        if obs_label != gt:
            bad_label += 1
        row = counts.getrow(p)
        vals = dict(zip(var_upper[row.indices], row.data))
        gvals = [vals.get(g, None) for g in genes]
        if any(v is None or v <= 0 for v in gvals):
            bad_subset += 1
        else:
            if any(gvals[k] < gvals[k + 1] for k in range(len(gvals) - 1)):
                bad_order += 1
        prefixes = {k: genes[:k] for k in KS}
        assert prefixes[50] == prefixes[100][:50] and prefixes[100] == prefixes[200][:100]
        assert prefixes[200] == prefixes[500][:200] and prefixes[500] == prefixes[1000][:500]
        assert prefixes[1000] == genes
        meta = {c: (str(obs[c].iloc[ri]) if c in obs.columns else None)
                for c in ["sex", "assay", "disease", "development_stage", "self_reported_ethnicity"]}
        recs.append({
            "atlas": atlas, "cell_id": eid, "row_index": ri, "barcode": str(obs.index[ri]),
            "ground_truth_label": gt, "canonical_label": canonical_label(atlas, gt),
            "donor_id": (str(obs["donor_id"].iloc[ri]) if "donor_id" in obs.columns else None),
            "tissue": (str(obs["tissue"].iloc[ri]) if "tissue" in obs.columns else None),
            "donor_metadata": json.dumps(meta),
            "raw_count_sum": float(row.sum()), "n_detected_genes": int(row.nnz),
            "zcell_1000_hash": sha256_text(sent),
            **{f"G_{k}": prefixes[k] for k in KS},
        })
    cells = pd.DataFrame(recs).sort_values("row_index").reset_index(drop=True)
    qc.update({"label_mismatch": bad_label,
               "sentence_not_subset_of_nonzero_counts": bad_subset,
               "sentence_not_descending_in_counts": bad_order,
               "sentences_with_duplicate_genes": dup_genes,
               "n_cells": len(cells),
               "n_canonical_labels": int(cells["canonical_label"].nunique()),
               "canonical_label_counts": cells["canonical_label"].value_counts().to_dict(),
               "raw_label_counts": cells["ground_truth_label"].value_counts().to_dict(),
               "has_donor": bool(cells["donor_id"].notna().all()),
               "has_tissue": bool(cells["tissue"].notna().all()),
               "n_tissues": int(cells["tissue"].nunique(dropna=True))})
    assert bad_label == 0 and bad_subset == 0 and bad_order == 0, qc

    pairs = make_pairs(cells, PAIR_SEED, n_pairs)
    pairs = pd.DataFrame(pairs)
    pairs.insert(0, "pair_id", [f"{atlas}_p{i:04d}" for i in range(len(pairs))])
    pairs.insert(1, "atlas", atlas)
    ids = list(pairs["cell_A"]) + list(pairs["cell_B"])
    assert len(ids) == len(set(ids)), "a cell appears in more than one pair"
    assert (pairs["canonical_A"] != pairs["canonical_B"]).all()
    cells["swap_source_cell_id"] = swap_partners(cells, PAIR_SEED)
    pid = {c: p for p, c in zip(pairs["pair_id"], pairs["cell_A"])}
    pid.update({c: p for p, c in zip(pairs["pair_id"], pairs["cell_B"])})
    cells["pair_id"] = cells["cell_id"].map(pid)
    qc.update({"n_pairs": len(pairs),
               "pairs_same_tissue": int(pairs["same_tissue"].sum()),
               "pairs_same_donor": (int(pairs["same_donor"].sum()) if qc["has_donor"] else None),
               "pair_tier_counts": pairs["match_tier"].value_counts().sort_index().to_dict(),
               "pair_label_combos": Counter(
                   tuple(sorted(t)) for t in zip(pairs["canonical_A"], pairs["canonical_B"])
               ).most_common(10)})

    d = atlas_dir(atlas)
    cells.to_parquet(d / "cells.parquet", index=False)
    pairs.to_parquet(d / "pairs.parquet", index=False)
    # JSONL copies for the scoring step (the LLaVA environment has no pandas or pyarrow)
    write_jsonl(d / "cells.jsonl", cells.to_dict(orient="records"))
    write_jsonl(d / "pairs.jsonl", pairs.to_dict(orient="records"))
    sel = sub[[rowpos[int(e[4:])] for e in cells["cell_id"]]].copy()
    sel.X = sp.csr_matrix(sel.layers["counts"]).astype(np.float32)
    del sel.layers["counts"]
    sel.obs = sel.obs[[c for c in ["cell_type", "donor_id", "tissue"] if c in sel.obs.columns]].copy()
    sel.obs["cell_id"] = cells["cell_id"].to_numpy()
    sel.obs.index = cells["cell_id"].to_numpy()
    for c in list(sel.var.columns):
        if c not in ("ensembl_id", "gene_symbols", "feature_name"):
            del sel.var[c]
    sel.obsm.clear(); sel.uns.clear(); sel.varm.clear(); sel.obsp.clear()
    sel.write_h5ad(d / "selected_counts.h5ad")
    qc["selected_counts_h5ad"] = str(d / "selected_counts.h5ad")
    qc["var_has_ensembl_id"] = "ensembl_id" in sel.var.columns
    dump_json(d / "cells_qc.json", qc)
    print(json.dumps({k: v for k, v in qc.items() if k not in ("canonical_label_counts", "raw_label_counts",
                                                                "pair_label_combos")}, indent=1))
    return qc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlases", nargs="+", default=ATLASES)
    ap.add_argument("--n_pairs", type=int, default=N_PAIRS_TARGET)
    a = ap.parse_args()
    for atlas in a.atlases:
        build_atlas(atlas, a.n_pairs)


if __name__ == "__main__":
    main()

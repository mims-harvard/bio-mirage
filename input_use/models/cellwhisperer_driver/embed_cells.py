"""Computes the CellWhisperer transcriptome representation Z_CW of every cell, intact and with
counts permuted across genes. Second step of four (build_cells, embed_cells, score_llm, analyze).
Runs on one GPU in CellWhisperer's pixi environment, which provides the `cellwhisperer` package and
its Geneformer model.

Conditions, all starting from the raw counts in selected_counts.h5ad:
  intact           the cell's own counts
  shuffle_s{0,1,2} the cell's non-zero counts moved to a uniformly random set of genes (seeds 0-2),
                   which is equivalent to permuting the full count vector across genes. Total count,
                   number of detected genes and the multiset of counts are unchanged.
  rankonly_s0      the intact cell's Geneformer tokens in a random order (seed 0)
Each condition passes through the released Geneformer processor (counts normalised per cell, divided
by the gene medians, ranked, truncated at 2,048 tokens) and the transcriptome tower of the released
CLIP checkpoint cellwhisperer_clip_v1.ckpt. The 2,048-d projection is L2-normalised, as in the data
the released chat model was trained on. The shuffle conditions give the permuted panel of Fig. 2f.

Reads and writes single_cell/cellwhisperer/<atlas>/ under $INPUT_USE_RESULTS_DIR:
selected_counts.h5ad in, embeddings.npz and embed_qc.json out.

    python input_use/models/cellwhisperer_driver/embed_cells.py [--atlases immune1 ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch  # (imported before anndata/scanpy for the pixi environment's GLIBCXX)
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    ATLASES, CLIP_CKPT, SHUFFLE_SEEDS, atlas_dir, dump_json, sha256_array)


def load_model():
    """Released CLIP checkpoint -> (Lightning model, Geneformer transcriptome processor, device)."""
    from cellwhisperer.utils.model_io import load_cellwhisperer_model
    assert CLIP_CKPT.exists(), f"CellWhisperer checkpoint missing: {CLIP_CKPT}"
    pl_model, _tokenizer, processor = load_cellwhisperer_model(str(CLIP_CKPT))
    return pl_model, processor, pl_model.model.device


@torch.no_grad()
def embed_token_arrays(model, token_arrays, device, batch_size: int = 64) -> np.ndarray:
    """Embeds variable-length Geneformer token arrays; returns L2-normalised [N, 2048] projections."""
    from cellwhisperer.jointemb.geneformer_model import PAD_TOKEN_ID
    out = []
    for i in range(0, len(token_arrays), batch_size):
        batch = token_arrays[i: i + batch_size]
        lens = [max(1, len(t)) for t in batch]
        maxlen = max(lens)
        mat = np.full((len(batch), maxlen), PAD_TOKEN_ID, dtype=np.int64)
        for r, t in enumerate(batch):
            mat[r, : len(t)] = t
        et = torch.as_tensor(mat, dtype=torch.long, device=device)
        el = torch.as_tensor(lens, dtype=torch.long, device=device)
        _, z = model.get_transcriptome_features(expression_tokens=et, expression_token_lengths=el)
        z = z / z.norm(dim=-1, keepdim=True)
        out.append(z.float().cpu().numpy())
    return np.concatenate(out, 0)


def shuffle_counts(X: sp.csr_matrix, seed: int) -> sp.csr_matrix:
    rng = np.random.RandomState(seed)
    X = sp.csr_matrix(X)
    n, m = X.shape
    indptr, indices, data = [0], [], []
    for i in range(n):
        s, e = X.indptr[i], X.indptr[i + 1]
        vals = X.data[s:e]
        new_cols = rng.choice(m, size=len(vals), replace=False)
        order = np.argsort(new_cols)
        indices.extend(new_cols[order]); data.extend(vals[order]); indptr.append(len(indices))
    return sp.csr_matrix((np.asarray(data, dtype=X.dtype), np.asarray(indices), np.asarray(indptr)),
                         shape=X.shape)


def shuffle_integrity(X: sp.csr_matrix, Y: sp.csr_matrix) -> dict:
    X, Y = sp.csr_matrix(X), sp.csr_matrix(Y)
    same_sum = np.allclose(X.sum(1).A1, Y.sum(1).A1)
    same_nnz = np.array_equal(np.diff(X.indptr), np.diff(Y.indptr))
    same_multiset = all(np.array_equal(np.sort(X.data[X.indptr[i]:X.indptr[i + 1]]),
                                       np.sort(Y.data[Y.indptr[i]:Y.indptr[i + 1]]))
                        for i in range(X.shape[0]))
    # fraction of a cell's detected genes that still carry the same non-zero value after the shuffle
    keep = []
    for i in range(X.shape[0]):
        a = dict(zip(X.indices[X.indptr[i]:X.indptr[i + 1]], X.data[X.indptr[i]:X.indptr[i + 1]]))
        b = dict(zip(Y.indices[Y.indptr[i]:Y.indptr[i + 1]], Y.data[Y.indptr[i]:Y.indptr[i + 1]]))
        keep.append(sum(1 for g, v in a.items() if b.get(g) == v) / max(1, len(a)))
    return {"total_count_unchanged": bool(same_sum), "nnz_unchanged": bool(same_nnz),
            "count_multiset_unchanged": bool(same_multiset),
            "frac_genes_keeping_value_mean": float(np.mean(keep)),
            "frac_genes_keeping_value_max": float(np.max(keep))}


def tokenize(processor, ad):
    ad = ad.copy()
    out = processor(ad, return_tensors="pt", padding=True)
    toks, lens = out["expression_tokens"], out["expression_token_lengths"]
    return [toks[i, : int(lens[i])].cpu().numpy().astype(np.int64) for i in range(toks.shape[0])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlases", nargs="+", default=ATLASES)
    ap.add_argument("--batch_size", type=int, default=32)
    a = ap.parse_args()

    import anndata
    pl_model, processor, device = load_model()
    model = pl_model.model
    print(f"[embed] model on {device}; ckpt={CLIP_CKPT}", flush=True)

    for atlas in a.atlases:
        t0 = time.time()
        d = atlas_dir(atlas)
        ad = anndata.read_h5ad(d / "selected_counts.h5ad")
        X = sp.csr_matrix(ad.X)
        assert np.abs(X.data - np.round(X.data)).max() < 1e-6
        cell_ids = ad.obs["cell_id"].to_numpy().astype(str)
        qc = {"atlas": atlas, "n_cells": int(ad.n_obs), "n_vars": int(ad.n_vars),
              "var_key": ("ensembl_id" if "ensembl_id" in ad.var.columns else "symbol_index")}

        conds, tokens, embs = {}, {}, {}
        conds["intact"] = X
        for s in SHUFFLE_SEEDS:
            conds[f"shuffle_s{s}"] = shuffle_counts(X, s)
            qc[f"shuffle_s{s}_integrity"] = shuffle_integrity(X, conds[f"shuffle_s{s}"])
            assert all(qc[f"shuffle_s{s}_integrity"][k] for k in
                       ["total_count_unchanged", "nnz_unchanged", "count_multiset_unchanged"])
        for name, M in conds.items():
            adc = anndata.AnnData(X=M.astype(np.float32), obs=ad.obs.copy(), var=ad.var.copy())
            tokens[name] = tokenize(processor, adc)
            print(f"[embed] {atlas} {name}: tokenised {len(tokens[name])} cells in {time.time()-t0:.0f}s", flush=True)
        # tokenising the unchanged input twice must give identical tokens
        tok2 = tokenize(processor, anndata.AnnData(X=X.astype(np.float32), obs=ad.obs.copy(), var=ad.var.copy()))
        qc["tokenization_deterministic"] = all(np.array_equal(x, y) for x, y in zip(tokens["intact"], tok2))
        assert qc["tokenization_deterministic"]
        rng = np.random.RandomState(0)
        tokens["rankonly_s0"] = [rng.permutation(t) for t in tokens["intact"]]
        qc["rankonly_token_set_preserved"] = all(np.array_equal(np.sort(x), np.sort(y))
                                                 for x, y in zip(tokens["intact"], tokens["rankonly_s0"]))
        qc["rankonly_order_changed_frac"] = float(np.mean([not np.array_equal(x, y)
                                                            for x, y in zip(tokens["intact"], tokens["rankonly_s0"])]))
        lens = np.array([len(t) for t in tokens["intact"]])
        qc["intact_token_len"] = {"min": int(lens.min()), "median": float(np.median(lens)), "max": int(lens.max()),
                                  "frac_at_2048_cap": float(np.mean(lens >= 2048))}
        for name in tokens:
            embs[name] = embed_token_arrays(model, tokens[name], device, batch_size=a.batch_size)
            assert embs[name].shape == (ad.n_obs, 2048), embs[name].shape
            assert np.isfinite(embs[name]).all()
            print(f"[embed] {atlas} {name}: embedded in {time.time()-t0:.0f}s", flush=True)
        rep = embed_token_arrays(model, tokens["intact"], device, batch_size=a.batch_size)
        qc["embedding_repeat_max_abs_diff"] = float(np.abs(rep - embs["intact"]).max())
        assert qc["embedding_repeat_max_abs_diff"] < 1e-4, qc["embedding_repeat_max_abs_diff"]
        cos = {n: (embs["intact"] * embs[n]).sum(1) for n in embs if n != "intact"}
        qc["cosine_to_intact"] = {n: {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()),
                                      "frac_changed": float(np.mean(v < 1 - 1e-5))} for n, v in cos.items()}
        for n in cos:
            assert qc["cosine_to_intact"][n]["frac_changed"] == 1.0, (n, qc["cosine_to_intact"][n])
        qc["embedding_hash"] = {n: sha256_array(v) for n, v in embs.items()}
        qc["seconds"] = time.time() - t0
        np.savez(d / "embeddings.npz", cell_ids=cell_ids,
                 **{f"emb_{n}": v.astype(np.float32) for n, v in embs.items()},
                 **{f"toklen_{n}": np.array([len(t) for t in tokens[n]]) for n in tokens})
        dump_json(d / "embed_qc.json", qc)
        print(json.dumps({k: v for k, v in qc.items() if k != "embedding_hash"}, indent=1), flush=True)


if __name__ == "__main__":
    main()

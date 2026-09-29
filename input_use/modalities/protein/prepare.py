#!/usr/bin/env python
r"""Builds the protein inputs (examples.jsonl, meta.json) for one of three profiles.

`proteome` (used in the paper) takes every reviewed protein of one organism with curated GOA labels
and 120 to 1,024 residues and builds the BioReason-Pro shuffle conditions: ESM3 alone, ESM3 and
GO-GPT, or all sequence inputs recomputed from the shuffled sequence, plus the text conditions that
keep only GO-GPT, only InterPro, or neither. Supports the BioReason-Pro results of Figure 2c.
`bioreason_pro` builds substitution conditions at UniProt functional residues and control residues
for a smaller protein set, and `protrek` builds sequence conditions and a retrieval pool for
ProTrek, which the paper does not evaluate.

    python -m input_use.modalities.protein.prepare --profile proteome --organisms 9606 \
        --n_proteins 0 --outdir <run>
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from input_use.core import conditions as C
from input_use.core.records import Example, write_jsonl, provenance
from input_use.modalities.protein.sources import uniprot as up
from input_use.modalities.protein.importance import build_salient_and_controls
from input_use.modalities.protein.interventions import make_conditions
from input_use.modalities.protein import interventions as I

ORGANISM_MAP = {9606: "Homo sapiens (Human)", 10090: "Mus musculus (Mouse)",
                559292: "Saccharomyces cerevisiae (strain ATCC 204508 / S288c) (Baker's yeast)",
                83333: "Escherichia coli (strain K12)"}


def build(organism_ids, n_proteins, min_salient, len_min, len_max, require_experimental,
          cache_dir, seed, candidates_per_org, created_after=None):
    examples, kept, pool = [], [], {}
    for oid in organism_ids:
        organism = ORGANISM_MAP.get(oid)          # None when oid==0 -> any organism (fetch per protein)
        if oid and not organism:
            print(f"  [skip] organism_id {oid}: no BioReason-Pro mapping"); continue
        xq = f"length:[{len_min} TO {len_max}]"
        if created_after:   # temporal holdout: UniProt entries first created after the cutoff
            xq += f" AND date_created:[{created_after} TO *]"
        cands = up.find_proteins_with_functional_sites(
            oid, max_records=candidates_per_org, extra_query=xq)
        print(f"  organism {oid}: {len(cands)} candidates")
        for acc in cands:
            if sum(1 for k in kept if k[1] == oid) >= n_proteins:
                break
            try:
                e = up.fetch_entry(acc, cache_dir=cache_dir)
            except Exception as ex:
                print(f"    [skip] {acc}: {ex}"); continue
            if not (len_min <= e["sequence"]["length"] <= len_max):
                continue
            org_name = organism or up.get_organism(e)   # fetched name in any-organism holdout mode
            sc = build_salient_and_controls(e, require_experimental=require_experimental, seed=seed)
            if len(sc.matched_salient_positions()) < min_salient:
                continue
            seq_conds = make_conditions(sc, seed=seed)              # wt/mut_salient/mut_control/scramble
            wt, scr = seq_conds["wt"].sequence, seq_conds["scramble"].sequence
            name = e.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", "")
            gt = {"go_terms": up.go_annotations(e), "protein_name": name,
                  "n_salient_experimental": sc.summary()["n_salient_experimental"]}
            prov = provenance(uniprot_release=sc.uniprot_release, seed=seed,
                              importance="uniprot:functional_sites", organism_id=oid)
            ms_seq, mc_seq = seq_conds["mut_salient"].sequence, seq_conds["mut_control"].sequence
            n_ms, n_mc = len(seq_conds["mut_salient"].mutations), len(seq_conds["mut_control"].mutations)
            pct_ms = round(100.0 * n_ms / max(len(wt), 1), 2)   # % of residues changed
            pct_mc = round(100.0 * n_mc / max(len(wt), 1), 2)
            # Each condition is (embedding_sequence, symbolic_sequence, channel-flags,
            # intervention).
            spec = {
                C.WT:           (wt, wt, {"interpro": "on", "gogpt": "on"}, {"operator": "none"}),
                C.MUT_SALIENT:  (ms_seq, ms_seq, {"interpro": "on", "gogpt": "on"},
                                 {"operator": "alanine_scan", "targets": sc.matched_salient_positions(),
                                  "importance_source": "uniprot:functional_site",
                                  "n_edits": n_ms, "pct_residues_changed": pct_ms}),
                C.MUT_CONTROL:  (mc_seq, mc_seq, {"interpro": "on", "gogpt": "on"},
                                 {"operator": "alanine_scan", "targets": sc.control.positions,
                                  "importance_source": "matched_control",
                                  "n_edits": n_mc, "pct_residues_changed": pct_mc}),
                C.SHUFFLE_ESM3_AND_GOGPT: (scr, scr, {"interpro": "on", "gogpt": "on"},
                                 {"operator": "shuffle_esm3_and_gogpt"}),
                C.SHUFFLE_ESM3_ONLY: (scr, wt, {"interpro": "on", "gogpt": "on"},
                                 {"operator": "shuffle_esm3_only",
                                  "note": "ESM3 sees the shuffled sequence; GO-GPT/InterPro frozen on WT"}),
                C.NO_SYMBOLIC:  (wt, wt, {"interpro": "none", "gogpt": "none"}, {"operator": "drop_symbolic"}),
                C.NO_NAME:      (wt, wt, {"interpro": "no_name", "gogpt": "on"}, {"operator": "strip_name_entry"}),
                C.NO_MODALITY:  (scr, scr, {"interpro": "none", "gogpt": "none"}, {"operator": "no_modality_proxy"}),
            }
            for cond, (sequence, sym_seq, channels, interv) in spec.items():
                examples.append(Example(
                    example_id=acc, modality="protein", condition=cond,
                    payload={"sequence": sequence, "symbolic_sequence": sym_seq,
                             "organism": org_name, "channels": channels},
                    intervention=interv, ground_truth=gt, provenance=prov))
            kept.append((acc, oid))
            pool[acc] = {"wt": wt, "organism": org_name, "gt": gt, "prov": prov}
            print(f"    [keep] {acc} matched_salient={len(sc.matched_salient_positions())} "
                  f"GT_GO={len(gt['go_terms'])} | mut_salient {pct_ms}% / mut_control {pct_mc}% "
                  f"of {len(wt)} aa changed")

    # swap: ESM3 embedding of protein A + GO-GPT/InterPro text of a different protein B. Pairing is
    # a seeded derangement (no protein paired with itself).
    acc_list = [a for a, _ in kept]
    if len(acc_list) >= 2:
        idx = list(range(len(acc_list)))
        rng = random.Random(seed + 777)
        while True:
            rng.shuffle(idx)
            if all(i != j for i, j in enumerate(idx)):
                break
        for i, acc in enumerate(acc_list):
            b = acc_list[idx[i]]
            A, B = pool[acc], pool[b]
            gt_swap = {"go_terms": A["gt"]["go_terms"],          # A = ESM3-embedding source
                       "go_terms_text": B["gt"]["go_terms"],     # B = GO-GPT/InterPro text source
                       "protein_name": A["gt"].get("protein_name", ""), "text_source_id": b}
            examples.append(Example(
                example_id=acc, modality="protein", condition=C.SWAP,
                payload={"sequence": A["wt"], "symbolic_sequence": B["wt"],
                         "organism": A["organism"], "channels": {"interpro": "on", "gogpt": "on"}},
                intervention={"operator": "swap_embedding_text", "embedding_id": acc, "text_id": b},
                ground_truth=gt_swap, provenance=A["prov"]))
        print(f"  [swap] built {len(acc_list)} A-embed+B-text pairs (seeded derangement)")
    return examples, kept


# ---------------------------------------------------------------------------------------------
# Proteome profile: Experiment 1 (ESM3-corruption invariance) over an entire proteome.
PROTEOME_LEN_MAX_DEFAULT = 1024      # GO-GPT truncates the protein encoder input at 1024 aa


def proteome_conditions(wt: str, scr_by_seed):
    """The 8-arm ladder + per-seed shuffle repeats, as {condition: (esm3_sequence, gogpt_sequence,
    channels, intervention)}.
    """
    scr = scr_by_seed[0]
    spec = {
        C.WT:                (wt,  wt,  {"interpro": "wt",    "gogpt": "on"},
                              {"operator": "none"}),
        C.SHUFFLE_ESM3_ONLY: (scr, wt,  {"interpro": "wt",    "gogpt": "on"},
                              {"operator": "shuffle_esm3_only", "shuffle_seed": 0,
                               "note": "ESM3 sees the shuffle; InterPro + GO-GPT frozen on WT"}),
        C.SHUFFLE_ESM3_GOGPT: (scr, scr, {"interpro": "wt",   "gogpt": "on"},
                              {"operator": "shuffle_esm3_gogpt", "shuffle_seed": 0,
                               "note": "ESM3 + GO-GPT re-derived on the shuffle; InterPro frozen on WT"}),
        C.SHUFFLE_ESM3_AND_GOGPT: (scr, scr, {"interpro": "empty", "gogpt": "on"},
                              {"operator": "shuffle_esm3_and_gogpt", "shuffle_seed": 0,
                               "note": "every sequence-derived channel on the shuffle; InterPro is "
                                       "empty because a shuffle matches no InterPro entry"}),
        C.INTERPRO_ONLY:     (scr, wt,  {"interpro": "wt",    "gogpt": "none"},
                              {"operator": "interpro_only", "shuffle_seed": 0}),
        C.GOGPT_ONLY:        (scr, wt,  {"interpro": "none",  "gogpt": "on"},
                              {"operator": "gogpt_only", "shuffle_seed": 0}),
        # Intact-ESM3 twins of the two single-channel rungs.
        C.INTERPRO_ONLY_WT_ESM3: (wt, wt, {"interpro": "wt",   "gogpt": "none"},
                              {"operator": "interpro_only_wt_esm3"}),
        C.GOGPT_ONLY_WT_ESM3: (wt,  wt,  {"interpro": "none",  "gogpt": "on"},
                              {"operator": "gogpt_only_wt_esm3"}),
        C.NO_SYMBOLIC:       (wt,  wt,  {"interpro": "none",  "gogpt": "none"},
                              {"operator": "drop_symbolic"}),
        C.NO_MODALITY:       (scr, scr, {"interpro": "none",  "gogpt": "none"},
                              {"operator": "no_modality_proxy", "shuffle_seed": 0}),
    }
    # Extra permutations, applied only to the two arms whose interpretation could plausibly depend
    # on which permutation was drawn.
    for seed, s in scr_by_seed.items():
        if seed == 0:
            continue
        spec[C.shuffle_seed_cond(C.SHUFFLE_ESM3_ONLY, seed)] = (
            s, wt, {"interpro": "wt", "gogpt": "on"},
            {"operator": "shuffle_esm3_only", "shuffle_seed": seed})
        spec[C.shuffle_seed_cond(C.SHUFFLE_ESM3_GOGPT, seed)] = (
            s, s, {"interpro": "wt", "gogpt": "on"},
            {"operator": "shuffle_esm3_gogpt", "shuffle_seed": seed})
    return spec


def build_proteome(organism_id, len_min, len_max, seed, shuffle_seeds, n_proteins,
                   multi_seed_subset, gaf_path, min_date, interpro_cache, interpro_workers,
                   interpro_type_mode):
    """Build Experiment 1 examples for a whole proteome. Returns (examples, kept, meta)."""
    from input_use.modalities.protein.sources import goa, interpro as ipr

    ann = goa.load_annotations(gaf_path, taxon=f"taxon:{organism_id}")
    print(f"[proteome] {goa.summarise(ann)}")
    if min_date:
        print(f"[proteome] {goa.summarise(ann, min_date=min_date)}")
    gt_all = ann.ground_truth()

    rows = up.stream_proteome(organism_id=organism_id, reviewed=True)
    print(f"[proteome] UniProt: {len(rows)} reviewed entries for organism {organism_id}")

    # Keep proteins that have both a curated GO label (else CAFA cannot score them) and a length
    # both encoders can consume.
    kept, drop_nogo, drop_len = [], 0, 0
    for r in rows:
        acc = r.get("Entry", "")
        seq = (r.get("Sequence") or "").strip()
        if not acc or not seq:
            continue
        if acc not in gt_all:
            drop_nogo += 1
            continue
        if not (len_min <= len(seq) <= len_max):
            drop_len += 1
            continue
        kept.append({"accession": acc, "sequence": seq, "organism": r.get("Organism", ""),
                     "protein_name": r.get("Protein names", ""), "length": len(seq)})
    kept.sort(key=lambda d: d["accession"])
    print(f"[proteome] dropped {drop_nogo} without curated GO, {drop_len} outside "
          f"[{len_min},{len_max}] aa -> {len(kept)} eligible")
    if n_proteins:
        kept = kept[:n_proteins]
        print(f"[proteome] truncated to --n_proteins {n_proteins}")

    # One InterPro lookup per protein, reused by every arm that keeps the block frozen on WT.
    ip_cache = ipr.fetch_many([k["accession"] for k in kept], cache_path=interpro_cache,
                              workers=interpro_workers)

    # Only a subset carries the extra shuffle permutations (see proteome_conditions).
    rng = random.Random(seed + 991)
    multi = set()
    if shuffle_seeds > 1 and multi_seed_subset:
        idx = list(range(len(kept)))
        rng.shuffle(idx)
        multi = {kept[i]["accession"] for i in idx[:multi_seed_subset]}
    elif shuffle_seeds > 1:
        multi = {k["accession"] for k in kept}

    examples, n_ip_hit = [], 0
    for k in kept:
        acc, wt = k["accession"], k["sequence"]
        n_seeds = shuffle_seeds if acc in multi else 1
        scr_by_seed = {s: I.scramble(wt, seed=seed + s).sequence for s in range(n_seeds)}
        ip_entries = ip_cache.get(acc, [])
        ip_text = ipr.format_entries(ip_entries, type_mode=interpro_type_mode)
        n_ip_hit += 1 if ip_entries else 0
        pct_changed = round(100.0 * sum(a != b for a, b in zip(wt, scr_by_seed[0])) / len(wt), 2)
        gt = {"go_terms": sorted(gt_all[acc]), "protein_name": k["protein_name"],
              "go_terms_recent": sorted(ann.ground_truth(accessions=[acc],
                                                         min_date=min_date).get(acc, ()))
              if min_date else []}
        prov = provenance(seed=seed, importance="none (shuffle battery)", organism_id=organism_id,
                          ground_truth="GOA curated evidence", gaf_version=ann.gaf_version,
                          gaf_generated=ann.generated, temporal_cutoff=min_date,
                          interpro_source="interpro-api-by-accession",
                          interpro_type_mode=interpro_type_mode)
        for cond, (esm_seq, go_seq, channels, interv) in proteome_conditions(wt, scr_by_seed).items():
            interv = dict(interv)
            if "shuffle_seed" in interv:
                interv["pct_residues_changed"] = pct_changed
            examples.append(Example(
                example_id=acc, modality="protein", condition=cond,
                payload={"sequence": esm_seq, "symbolic_sequence": go_seq,
                         "organism": k["organism"], "channels": channels,
                         "accession": acc, "interpro_wt": ip_text,
                         "n_interpro_entries": len(ip_entries), "length": k["length"]},
                intervention=interv, ground_truth=gt, provenance=prov))

    meta = {"n_interpro_hit": n_ip_hit, "n_multi_seed": len(multi),
            "gaf_version": ann.gaf_version, "gaf_generated": ann.generated,
            "goa_summary": goa.summarise(ann),
            "goa_summary_recent": goa.summarise(ann, min_date=min_date) if min_date else None}
    print(f"[proteome] {len(kept)} proteins x {len(examples) // max(len(kept), 1)} avg arms "
          f"= {len(examples)} examples | {n_ip_hit}/{len(kept)} have >=1 InterPro entry "
          f"| {len(multi)} proteins carry {shuffle_seeds} shuffle seeds")
    return examples, [(k["accession"], organism_id) for k in kept], meta


def _pool_entry(entry, organism):
    """A retrieval-pool row: accession + sequence + curated function text + GO terms. Returns None if
    the entry has no function comment (no usable retrieval description).
    """
    desc = up.function_text(entry)
    if not desc:
        return None
    name = entry.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", "")
    return {"accession": entry.get("primaryAccession"), "organism": organism,
            "sequence": up.get_sequence(entry), "description": desc,
            "protein_name": name, "go_terms": up.go_annotations(entry)}


DOSES = (0.34, 0.67, 1.0)          # fraction of the matched-salient set mutated (dose-response)
DOSE_TAG = {0.34: "p33", 0.67: "p67", 1.0: "p100"}
MASK_FRAC = 0.15                   # terminal/middle mask block = 15% of the sequence


def protrek_conditions(entry, sc, seed):
    """All ProTrek input-perturbation conditions for one protein. Each is (condition_name, sequence,
    text_override_or_None, intervention_dict).
    """
    seq = sc.sequence
    L = len(seq)
    name = entry.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", "")
    func = up.function_text(entry)
    out = [("wt", seq, None, {"operator": "none"})]

    # (3) dose-response: alanine-scan a growing fraction of functional residues vs matched controls
    msal, ctrl = sc.matched_salient_positions(), list(sc.control.positions)
    n = min(len(msal), len(ctrl))
    msal, ctrl = sorted(msal)[:n], sorted(ctrl)[:n]
    for fr in DOSES:
        tag = DOSE_TAG[fr]
        for who, pos in (("salient", I.frac_positions(msal, fr)), ("control", I.frac_positions(ctrl, fr))):
            c = I.alanine_substitute(seq, pos)
            out.append((f"mut_{who}_{tag}", c.sequence, None,
                        {"operator": "alanine_scan", "targets": pos, "n_edits": len(c.mutations),
                         "dose_frac": fr, "pct_residues_changed": round(100.0 * len(c.mutations) / L, 2),
                         "importance_source": "uniprot:functional_site" if who == "salient" else "matched_control"}))

    # (3b) higher-dose function-centered perturbation: a +/-k window around each functional residue
    # vs the same windows around matched controls.
    for k, tag in ((3, "w3"), (6, "w6")):
        for who, centers in (("salient", msal), ("control", ctrl)):
            pos = I.window_positions(centers, k, L)
            c = I.alanine_substitute(seq, pos)
            out.append((f"mutwin_{who}_{tag}", c.sequence, None,
                        {"operator": "alanine_window", "half_window": k, "centers": centers,
                         "n_edits": len(c.mutations), "pct_residues_changed": round(100.0 * len(c.mutations) / L, 2),
                         "importance_source": "uniprot:functional_site" if who == "salient" else "matched_control"}))
    # window-delete variant at k=3 (removes the functional neighborhood entirely)
    for who, centers in (("salient", msal), ("control", ctrl)):
        pos = I.window_positions(centers, 3, L)
        out.append((f"delwin_{who}_w3", I.delete_positions(seq, pos), None,
                    {"operator": "delete_window", "half_window": 3, "centers": centers,
                     "n_deleted": len(pos), "pct_residues_changed": round(100.0 * len(pos) / L, 2),
                     "importance_source": "uniprot:functional_site" if who == "salient" else "matched_control"}))

    # order-destruction reference: full composition-preserving shuffle
    out.append(("scramble", I.scramble(seq, seed=seed).sequence, None, {"operator": "scramble"}))

    # (1,2) domain necessity (delete) / sufficiency (keep) + matched-size random-region deletion
    span = up.domain_span(entry)
    if span:
        s0, e0, dtype = span
        out.append(("keep_domain", I.keep_span(seq, s0, e0), None,
                    {"operator": "keep_domain", "domain_type": dtype, "span": [s0, e0], "kept": e0 - s0 + 1}))
        out.append(("del_domain", I.delete_span(seq, s0, e0), None,
                    {"operator": "delete_domain", "domain_type": dtype, "span": [s0, e0], "deleted": e0 - s0 + 1}))
        exclude = set(range(s0, e0 + 1)) | up.annotated_positions(entry, up.REGION_TYPES)
        exclude |= {r["position"] for r in up.functional_residues(entry)}
        exclude |= set(range(1, 6)) | set(range(L - 4, L + 1))     # avoid termini
        rs = I.random_nondomain_span(L, e0 - s0 + 1, exclude, seed=seed)
        if rs:
            r0, r1 = rs
            out.append(("del_random", I.delete_span(seq, r0, r1), None,
                        {"operator": "delete_random", "span": [r0, r1], "deleted": r1 - r0 + 1,
                         "importance_source": "matched_control"}))

    # (5) directionality / positional probes
    out.append(("reverse", I.reverse_seq(seq), None, {"operator": "reverse"}))
    K = max(1, round(MASK_FRAC * L))
    for where, tag in (("N", "nterm"), ("C", "cterm"), ("M", "middle")):
        out.append((f"mask_{tag}", I.mask_terminal(seq, K, where=where), None,
                    {"operator": "mask_terminal", "where": where, "k": K}))

    # (4) text-side ablations: sequence stays WT, override the description fed to the text encoder
    if name:
        out.append(("text_nameonly", seq, name, {"operator": "text_name_only"}))
        out.append(("text_namefunc", seq, f"{name}. {func}", {"operator": "text_name_plus_function"}))
    out.append(("text_generic", seq, "This is a protein.", {"operator": "text_generic"}))
    return out


def build_protrek(organism_ids, n_proteins, min_salient, len_min, len_max, require_experimental,
                  cache_dir, seed, candidates_per_org, pool_extra=0, created_after=None):
    """ProTrek (retrieval) profile: per protein emit only the 4 pure-sequence conditions the sequence
    encoder consumes (wt/mut_salient/mut_control/scramble), with the curated function text as the
    retrieval positive.
    """
    examples, kept, pool, seen_acc = [], [], [], set()
    for oid in organism_ids:
        organism = ORGANISM_MAP.get(oid)
        if oid and not organism:
            print(f"  [skip] organism_id {oid}: no mapping"); continue
        xq = f"length:[{len_min} TO {len_max}]"
        if created_after:
            xq += f" AND date_created:[{created_after} TO *]"
        cands = up.find_proteins_with_functional_sites(
            oid, max_records=max(candidates_per_org, n_proteins + pool_extra + 200), extra_query=xq)
        print(f"  organism {oid}: {len(cands)} candidates")
        for acc in cands:
            if sum(1 for k in kept if k[1] == oid) >= n_proteins:
                break
            try:
                e = up.fetch_entry(acc, cache_dir=cache_dir)
            except Exception as ex:
                print(f"    [skip] {acc}: {ex}"); continue
            if not (len_min <= e["sequence"]["length"] <= len_max):
                continue
            org_name = organism or up.get_organism(e)
            desc = up.function_text(e)
            if not desc:                       # retrieval positive must exist
                continue
            sc = build_salient_and_controls(e, require_experimental=require_experimental, seed=seed)
            if len(sc.matched_salient_positions()) < min_salient:
                continue
            conds = protrek_conditions(e, sc, seed)      # all 5 perturbation families
            name = e.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", "")
            gt = {"go_terms": up.go_annotations(e), "protein_name": name, "description": desc,
                  "n_salient_experimental": sc.summary()["n_salient_experimental"]}
            prov = provenance(uniprot_release=sc.uniprot_release, seed=seed,
                              importance="uniprot:functional_sites", organism_id=oid)
            for cond_name, cseq, ctext, interv in conds:
                payload = {"sequence": cseq, "organism": org_name}
                if ctext is not None:
                    payload["text"] = ctext                # text-side condition overrides the description
                examples.append(Example(
                    example_id=acc, modality="protein", condition=cond_name,
                    payload=payload, intervention=interv, ground_truth=gt, provenance=prov))
            kept.append((acc, oid))
            seen_acc.add(acc)
            pool.append(_pool_entry(e, org_name))         # every query protein is in the pool
            print(f"    [keep] {acc} conds={len(conds)} matched_salient={len(sc.matched_salient_positions())} "
                  f"domain={'Y' if up.domain_span(e) else 'N'} GT_GO={len(gt['go_terms'])}")

        # background distractors: extra reviewed proteins (with function text) beyond the kept set
        added = 0
        for acc in cands:
            if added >= pool_extra:
                break
            if acc in seen_acc:
                continue
            try:
                e = up.fetch_entry(acc, cache_dir=cache_dir)
            except Exception:
                continue
            row = _pool_entry(e, organism or up.get_organism(e))
            if row is None:
                continue
            seen_acc.add(acc); pool.append(row); added += 1
        print(f"  [pool] {len([p for p in pool if p])} proteins ({added} background distractors added)")
    pool = [p for p in pool if p]
    return examples, kept, pool


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=["bioreason_pro", "protrek", "proteome"],
                    default="bioreason_pro",
                    help="bioreason_pro: 9 channel/sequence conditions for the generative model (default). "
                         "protrek: 4 pure-sequence conditions + a seq->text retrieval pool for the contrastive model. "
                         "proteome: Experiment 1's 8-arm shuffle ladder over a whole proteome (no "
                         "functional-site requirement, GOA ground truth, InterPro by accession).")
    ap.add_argument("--organisms", type=int, nargs="+", default=[9606])
    ap.add_argument("--n_proteins", type=int, default=20,
                    help="(proteome) 0 = every eligible protein")
    ap.add_argument("--min_salient", type=int, default=4)
    ap.add_argument("--len_min", type=int, default=120)
    ap.add_argument("--len_max", type=int, default=None,
                    help=f"default 450 (bioreason_pro/protrek) or "
                         f"{PROTEOME_LEN_MAX_DEFAULT} (proteome: GO-GPT's encoder cap)")
    # --- proteome profile only ---
    ap.add_argument("--shuffle_seeds", type=int, default=1,
                    help="(proteome) independent permutations per protein for the two shuffle arms")
    ap.add_argument("--multi_seed_subset", type=int, default=3000,
                    help="(proteome) how many proteins get the extra seeds (0 = all of them)")
    ap.add_argument("--gaf", default=None, help="(proteome) GOA GAF path (default: download)")
    ap.add_argument("--min_date", default="20230101",
                    help="(proteome) temporal-holdout cutoff YYYYMMDD for the recent-GO ground "
                         "truth recorded alongside the full one ('' to skip)")
    ap.add_argument("--interpro_cache", default=None,
                    help="(proteome) consolidated InterPro-by-accession cache JSON")
    ap.add_argument("--interpro_workers", type=int, default=8)
    ap.add_argument("--interpro_type_mode", choices=["unknown", "real"], default="unknown",
                    help="(proteome) 'unknown' reproduces the pilot's InterPro block exactly "
                         "(default); 'real' substitutes the true InterPro entry type")
    ap.add_argument("--require_experimental", action="store_true")
    ap.add_argument("--candidates_per_org", type=int, default=200)
    ap.add_argument("--pool_extra", type=int, default=0,
                    help="(protrek) number of extra background proteins to add to the retrieval pool")
    ap.add_argument("--created_after", default=None, help="temporal holdout: only UniProt entries created after YYYY-MM-DD (contamination control)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache_dir", default=str(up.DEFAULT_CACHE))
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    if a.len_max is None:
        a.len_max = PROTEOME_LEN_MAX_DEFAULT if a.profile == "proteome" else 450
    print(f"Preparing protein examples ({a.profile}) -> {a.outdir}")
    pool = None
    extra_meta = {}
    if a.profile == "proteome":
        if len(a.organisms) != 1:
            raise SystemExit("--profile proteome takes exactly one --organisms id (GOA is per-taxon)")
        examples, kept, extra_meta = build_proteome(
            organism_id=a.organisms[0], len_min=a.len_min, len_max=a.len_max,
            seed=a.seed, shuffle_seeds=a.shuffle_seeds, n_proteins=a.n_proteins,
            multi_seed_subset=a.multi_seed_subset, gaf_path=a.gaf,
            min_date=(a.min_date or None), interpro_cache=a.interpro_cache,
            interpro_workers=a.interpro_workers, interpro_type_mode=a.interpro_type_mode)
    elif a.profile == "protrek":
        examples, kept, pool = build_protrek(
            a.organisms, a.n_proteins, a.min_salient, a.len_min, a.len_max, a.require_experimental,
            a.cache_dir, a.seed, a.candidates_per_org, pool_extra=a.pool_extra, created_after=a.created_after)
    else:
        examples, kept = build(a.organisms, a.n_proteins, a.min_salient, a.len_min, a.len_max,
                               a.require_experimental, a.cache_dir, a.seed, a.candidates_per_org,
                               created_after=a.created_after)
    write_jsonl(examples, os.path.join(a.outdir, "examples.jsonl"))
    if pool is not None:
        write_jsonl(pool, os.path.join(a.outdir, "pool.jsonl"))
    # The proteome profile streams sequences rather than caching per-accession entries, so there is
    rel = None
    if a.profile != "proteome" and kept:
        rel = up.fetch_entry(kept[0][0], cache_dir=a.cache_dir).get("_input_use_meta", {}).get("uniprot_release")
    meta = {"modality": "protein", "profile": a.profile, "n_proteins": len(kept),
            "n_examples": len(examples), "n_pool": (len(pool) if pool is not None else None),
            "conditions": sorted({e.condition for e in examples}), "organisms": a.organisms,
            "uniprot_release": rel, "seed": a.seed, "require_experimental": a.require_experimental,
            "created_after": a.created_after, "len_min": a.len_min, "len_max": a.len_max,
            "importance_source": ("none - shuffle battery, no salient/control residues"
                                  if a.profile == "proteome"
                                  else "UniProt curated functional-site features")}
    if a.profile == "proteome":
        meta.update({"ground_truth": "GOA curated evidence codes (CAFA no-knowledge set)",
                     "temporal_cutoff": a.min_date or None,
                     "shuffle_seeds": a.shuffle_seeds,
                     "interpro_type_mode": a.interpro_type_mode, **extra_meta})
    json.dump(meta, open(os.path.join(a.outdir, "meta.json"), "w"), indent=2)
    print(f"DONE: {len(kept)} proteins x {len(examples)//max(len(kept),1)} conditions = {len(examples)} examples"
          + (f" | pool={len(pool)}" if pool is not None else ""))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Builds the BioReason-Pro temporal holdout: proteins added to UniProt after the BioReason-Pro
preprint.

Fetches reviewed UniProt entries of 120 to 1,024 residues created on or after --created_after
(default 2026-03-20) and keeps those with at least one GO cross-reference with non-electronic
evidence, which become the labels. Looks up each protein's InterPro entries and builds the same
shuffle and text conditions as the proteome profile of prepare.py. Writes examples.jsonl and
meta.json to --outdir. Supports the temporal holdout result of RQ1 (appendix "Temporal holdout
(BioReason-Pro)").
"""
from __future__ import annotations

import argparse
import json
import os
import random
import urllib.parse
import urllib.request

from input_use.core.records import Example, write_jsonl, provenance
from input_use.modalities.protein import interventions as I
from input_use.modalities.protein.prepare import proteome_conditions
from input_use.modalities.protein.sources import interpro as ipr

# UniProt marks electronic annotations with these eco codes; everything else is curated.
IEA_CODES = {"ECO:0007669", "ECO:0000501"}
STREAM = "https://rest.uniprot.org/uniprotkb/stream?"


def fetch_candidates(created_after, len_min, len_max, timeout=900):
    """Reviewed entries created on/after the cutoff, as full JSON (needed for GO evidence codes)."""
    q = (f"reviewed:true AND length:[{len_min} TO {len_max}] "
         f"AND date_created:[{created_after} TO *]")
    url = STREAM + urllib.parse.urlencode(
        {"query": q, "format": "json",
         "fields": "accession,protein_name,organism_name,organism_id,sequence,go_id,length"})
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())["results"]


def curated_go(entry):
    """GO ids on this entry with at least one non-iea evidence."""
    out = set()
    for x in entry.get("uniProtKBCrossReferences", []) or []:
        if x.get("database") != "GO":
            continue
        evs = x.get("evidences") or []
        if any(ev.get("evidenceCode") not in IEA_CODES for ev in evs):
            out.add(x.get("id"))
    return {g for g in out if g}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--created_after", default="2026-03-20",
                    help="the BioReason-Pro preprint posting date: no entry created after it can "
                         "be in the training data of a model described in it")
    ap.add_argument("--len_min", type=int, default=120)
    ap.add_argument("--len_max", type=int, default=1024)
    ap.add_argument("--min_go", type=int, default=1, help="minimum curated GO terms to keep a protein")
    ap.add_argument("--shuffle_seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--interpro_cache", default=None)
    ap.add_argument("--interpro_workers", type=int, default=8)
    ap.add_argument("--interpro_type_mode", default="unknown", choices=["unknown", "real"])
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    ents = fetch_candidates(a.created_after, a.len_min, a.len_max)
    print(f"[holdout] UniProt: {len(ents)} reviewed entries created >= {a.created_after}, "
          f"{a.len_min}-{a.len_max} aa", flush=True)

    kept, drop_nogo, drop_seq = [], 0, 0
    for e in ents:
        acc = e.get("primaryAccession")
        seq = ((e.get("sequence") or {}).get("value") or "").strip()
        if not acc or not seq:
            drop_seq += 1
            continue
        go = curated_go(e)
        if len(go) < a.min_go:
            drop_nogo += 1
            continue
        org = (e.get("organism") or {})
        name = ((e.get("proteinDescription") or {}).get("recommendedName") or {}) \
            .get("fullName", {}).get("value", "")
        kept.append({"accession": acc, "sequence": seq, "go": sorted(go),
                     "organism": org.get("scientificName", ""), "organism_id": org.get("taxonId", 0),
                     "protein_name": name, "length": len(seq)})
    kept.sort(key=lambda d: d["accession"])
    print(f"[holdout] dropped {drop_nogo} without a non-IEA GO term, {drop_seq} without a sequence "
          f"-> {len(kept)} eligible", flush=True)
    if not kept:
        raise SystemExit("no eligible proteins")

    ip_cache = ipr.fetch_many([k["accession"] for k in kept], cache_path=a.interpro_cache,
                              workers=a.interpro_workers)

    examples, n_ip_hit = [], 0
    for k in kept:
        acc, wt = k["accession"], k["sequence"]
        scr_by_seed = {s: I.scramble(wt, seed=a.seed + s).sequence for s in range(a.shuffle_seeds)}
        ip_entries = ip_cache.get(acc, [])
        ip_text = ipr.format_entries(ip_entries, type_mode=a.interpro_type_mode)
        n_ip_hit += 1 if ip_entries else 0
        pct = round(100.0 * sum(x != y for x, y in zip(wt, scr_by_seed[0])) / len(wt), 2)
        gt = {"go_terms": k["go"], "protein_name": k["protein_name"]}
        prov = provenance(seed=a.seed, importance="none (shuffle battery)",
                          organism_id=k["organism_id"],
                          ground_truth="UniProt GO cross-references, non-IEA evidence only",
                          temporal_cutoff=a.created_after,
                          interpro_source="interpro-api-by-accession",
                          interpro_type_mode=a.interpro_type_mode)
        for cond, (esm_seq, go_seq, channels, interv) in proteome_conditions(wt, scr_by_seed).items():
            interv = dict(interv)
            if "shuffle_seed" in interv:
                interv["pct_residues_changed"] = pct
            examples.append(Example(
                example_id=acc, modality="protein", condition=cond,
                payload={"sequence": esm_seq, "symbolic_sequence": go_seq,
                         "organism": k["organism"], "channels": channels,
                         "accession": acc, "interpro_wt": ip_text,
                         "n_interpro_entries": len(ip_entries), "length": k["length"]},
                intervention=interv, ground_truth=gt, provenance=prov))

    write_jsonl(examples, os.path.join(a.outdir, "examples.jsonl"))
    n_terms = sorted(len(k["go"]) for k in kept)
    orgs = {}
    for k in kept:
        orgs[k["organism"]] = orgs.get(k["organism"], 0) + 1
    meta = {
        "modality": "protein", "profile": "temporal_holdout",
        "n_proteins": len(kept), "n_examples": len(examples),
        "created_after": a.created_after, "len_min": a.len_min, "len_max": a.len_max,
        "min_go": a.min_go, "shuffle_seeds": a.shuffle_seeds, "seed": a.seed,
        "ground_truth": "UniProt GO cross-references, non-IEA evidence only",
        "selection": ("reviewed + length window + date_created >= cutoff + >=1 non-IEA GO term; "
                      "NO functional-site requirement (that belongs to the salient-residue profile "
                      "and is irrelevant here)"),
        "n_candidates_fetched": len(ents),
        "n_dropped_no_curated_go": drop_nogo,
        "n_interpro_hit": n_ip_hit,
        "curated_go_terms_per_protein": {
            "median": n_terms[len(n_terms) // 2], "mean": round(sum(n_terms) / len(n_terms), 2),
            "min": n_terms[0], "max": n_terms[-1]},
        "organisms": sorted(orgs.items(), key=lambda kv: -kv[1]),
        "conditions": sorted({e.condition for e in examples}),
    }
    with open(os.path.join(a.outdir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[holdout] {len(kept)} proteins -> {len(examples)} examples across "
          f"{len(meta['conditions'])} conditions | {n_ip_hit}/{len(kept)} have InterPro | "
          f"median {meta['curated_go_terms_per_protein']['median']} curated GO terms")
    print(f"[holdout] wrote {a.outdir}/examples.jsonl + meta.json")


if __name__ == "__main__":
    main()

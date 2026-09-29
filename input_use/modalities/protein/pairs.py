#!/usr/bin/env python
"""Builds the protein pairs and inputs for the BioReason-Pro evidence conflicts.

A pair is two reviewed proteins of the same organism that share an InterPro family and differ in one
functional property scored by a readout GO term. The paper uses enzyme vs. pseudoenzyme
(`pseudoenzyme`), organelle targeting (`organelle_targeted`), DNA binding (`dna_binding`) and GO NOT
pairs from the human GOA file (`go_not`). The script also builds other categories that the paper
does not report. For each pair it writes the unmodified inputs of both proteins and the conflicts
that combine the ESM3 representation of one protein with the GO-GPT text, the InterPro text, or
both, from the other. Writes examples.jsonl, pairs.jsonl and meta.json to --outdir. Supports Figure
3b and the appendix figure "Evidence conflicts for BioReason-Pro (RL) on GO NOT pairs".
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from input_use.core import conditions as C
from input_use.core import config as cfg
from input_use.core.records import Example, write_jsonl, provenance
from input_use.modalities.protein.sources import interpro as ipr
from input_use.modalities.protein.sources import uniprot as up

MEROPS_BASE = "https://ftp.ebi.ac.uk/pub/databases/merops/current_release/database_files"
NPH_RE = re.compile(r"^[A-Z]\d+\.9\d\d$")        # MEROPS non-peptidase homologue identifier
PEP_RE = re.compile(r"^[A-Z]\d+\.\d\d\d$")

# ---------------------------------------------------------------------------------------------
# Category A - cofactor specificity
# --------------------------------------------------------------------------------------------- Each
# class maps the ChEBI identifiers UniProt uses in its cofactor comment onto the GO Molecular
# Function term that BioReason-Pro would have to emit to commit to that cofactor.
COFACTOR_CLASSES = {
    "NAD":  {"chebi": {"CHEBI:57540", "CHEBI:57945"}, "go": "GO:0051287", "label": "NAD binding"},
    "NADP": {"chebi": {"CHEBI:58349", "CHEBI:57783"}, "go": "GO:0050661", "label": "NADP binding"},
    "FAD":  {"chebi": {"CHEBI:57692"},                "go": "GO:0071949", "label": "FAD binding"},
    "FMN":  {"chebi": {"CHEBI:58210"},                "go": "GO:0010181", "label": "FMN binding"},
    "Zn":   {"chebi": {"CHEBI:29105"},                "go": "GO:0008270", "label": "zinc ion binding"},
    "Mg":   {"chebi": {"CHEBI:18420"},                "go": "GO:0000287", "label": "magnesium ion binding"},
    "Mn":   {"chebi": {"CHEBI:29035"},                "go": "GO:0030145", "label": "manganese ion binding"},
    "Fe2":  {"chebi": {"CHEBI:29033"},                "go": "GO:0005506", "label": "iron ion binding"},
    "Ca":   {"chebi": {"CHEBI:29108"},                "go": "GO:0005509", "label": "calcium ion binding"},
    "heme": {"chebi": {"CHEBI:30413"},                "go": "GO:0020037", "label": "heme binding"},
}
# Only contrast cofactors that are genuine specificity alternatives for one binding site.
CONTRAST_GROUPS = [{"NAD", "NADP"}, {"FAD", "FMN"}, {"Zn", "Mg", "Mn", "Fe2", "Ca"}]

CHEBI_RE = re.compile(r"ChEBI:(CHEBI:\d+)")
# Default candidate taxa: human, mouse, rat, yeast, E. coli, fly, worm, Arabidopsis.
MODEL_ORGANISMS = [9606, 10090, 10116, 559292, 83333, 7227, 6239, 3702]
CATALYTIC = "GO:0003824"      # catalytic activity
PEPTIDASE = "GO:0008233"      # peptidase activity


def _cofactor_classes(comment: str) -> set:
    """Cofactor classes named in a UniProt cofactor comment, via its ChEBI cross-references."""
    found = set(CHEBI_RE.findall(comment or ""))
    return {name for name, d in COFACTOR_CLASSES.items() if d["chebi"] & found}


def _interpro_ids(field: str) -> set:
    return {x for x in (field or "").split(";") if x.startswith("IPR")}


INTERPRO_ENTRY_LIST = "https://ftp.ebi.ac.uk/pub/databases/interpro/current_release/entry.list"


def interpro_types(data_dir: Optional[Path] = None) -> Dict[str, str]:
    """{ipr accession: entry type} from InterPro's 2.9 MB `entry.list`."""
    dest = Path(data_dir or cfg.DATA_DIR) / "interpro_entry.list"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"[pairs] downloading {INTERPRO_ENTRY_LIST}")
        tmp = str(dest) + ".tmp"
        urllib.request.urlretrieve(INTERPRO_ENTRY_LIST, tmp)
        os.replace(tmp, dest)
    types = {}
    with open(dest) as fh:
        next(fh, None)
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) >= 2:
                types[f[0]] = f[1]
    return types


# **Family only.** Homologous_superfamily is a fold grouping, not an evolutionary family: including
# it pairs rabphilin-3A with a histone demethylase (both have a Zn-finger fold) and prolyl
# endopeptidase with epoxide hydrolase (both alpha/beta hydrolases).
FAMILY_TYPES = {"Family"}

INTERPRO_TREE = "https://ftp.ebi.ac.uk/pub/databases/interpro/current_release/ParentChildTreeFile.txt"


def interpro_family_closure(data_dir=None, types=None):
    """{ipr: its Family-type ancestors, including itself} from InterPro's 660 KB hierarchy file."""
    types = types or interpro_types(data_dir)
    dest = Path(data_dir or cfg.DATA_DIR) / "interpro_ParentChildTreeFile.txt"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"[pairs] downloading {INTERPRO_TREE}")
        tmp = str(dest) + ".tmp"
        urllib.request.urlretrieve(INTERPRO_TREE, tmp)
        os.replace(tmp, dest)
    parent, stack = {}, []
    with open(dest) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if "::" not in line:
                continue
            depth = (len(line) - len(line.lstrip("-"))) // 2
            acc = line.lstrip("-").split("::")[0]
            del stack[depth:]
            if stack:
                parent[acc] = stack[-1]
            stack.append(acc)
    closure = {}

    def anc(a, seen=None):
        if a in closure:
            return closure[a]
        out, cur, guard = set(), a, 0
        while cur and guard < 50:
            if types.get(cur) in FAMILY_TYPES:
                out.add(cur)
            cur = parent.get(cur)
            guard += 1
        closure[a] = out
        return out

    return anc


def _go_ids(field: str) -> set:
    """GO accessions out of UniProt's `go_id` TSV field ('GO:0005524; GO:0004672; ...')."""
    return {x.strip() for x in (field or "").split(";") if x.strip().startswith("GO:")}


def _go_dag():
    from input_use.metrics import go_dag
    return go_dag.load(cfg.GO_OBO)


def _has(go_set, roots, dag) -> bool:
    """True iff the curated GO set contains `root` or anything below it."""
    return any(dag.any_under(go_set, r) for r in roots)


def _fetch_rows(query: str, fields: str) -> List[Dict]:
    return up.stream_proteome(organism_id=0, reviewed=True, fields=fields, extra_query=query)


def _fetch_batched(clauses: List[str], fields: str, extra: str = "", batch: int = 60) -> List[Dict]:
    """or together `clauses` in batches and concatenate the results."""
    rows, seen = [], set()
    for i in range(0, len(clauses), batch):
        chunk = " OR ".join(clauses[i:i + batch])
        q = f"({chunk})" + (f" AND ({extra})" if extra else "")
        for r in _fetch_rows(q, fields):
            if r.get("Entry") and r["Entry"] not in seen:
                seen.add(r["Entry"])
                rows.append(r)
    return rows


def _org_clause(organism_ids) -> str:
    """UniProt clause for a set of taxa. Pairs are always matched within one organism; several taxa are
    queried only to enlarge the pool, because a same-family/different-cofactor contrast is genuinely
    rare inside a single proteome (human yields ~10 such pairs).
    """
    ids = [i for i in (organism_ids or []) if i]
    if not ids:
        return ""
    return "(" + " OR ".join(f"organism_id:{i}" for i in ids) + ")"


def build_cofactor_pairs(organism_ids, len_min: int, len_max: int, max_pairs: int,
                         seed: int = 0, data_dir: Optional[Path] = None) -> List[Dict]:
    """Matched pairs differing in cofactor specificity."""
    ip_type = interpro_types(data_dir)
    fam_closure = interpro_family_closure(data_dir, ip_type)
    oc = _org_clause(organism_ids)
    q = "cc_cofactor:*" + (f" AND {oc}" if oc else "")
    rows = _fetch_rows(q, "accession,protein_name,organism_name,organism_id,length,sequence,"
                          "cc_cofactor,xref_interpro,ec,go_id")
    print(f"[pairs/cofactor] {len(rows)} reviewed entries with a COFACTOR comment")

    cands = []
    for r in rows:
        seq = (r.get("Sequence") or "").strip()
        if not seq or not (len_min <= len(seq) <= len_max):
            continue
        cls = _cofactor_classes(r.get("Cofactor", ""))
        if len(cls) != 1:
            continue          # 0 = unrecognised cofactor; >1 = ambiguous, no single contrast
        fams = set()
        for ip in _interpro_ids(r.get("InterPro", "")):
            fams |= fam_closure(ip)
        if not fams:
            continue
        cands.append({"accession": r["Entry"], "sequence": seq, "organism": r.get("Organism", ""),
                      "organism_id": r.get("Organism (ID)", ""), "protein_name": r.get("Protein names", ""),
                      "cls": next(iter(cls)), "interpro": fams, "ec": r.get("EC number", ""),
                      "go": _go_ids(r.get("Gene Ontology IDs", ""))})
    print(f"[pairs/cofactor] {len(cands)} candidates with one cofactor class and a Family-level "
          f"InterPro entry")

    # Group by (organism, shared InterPro family).
    groups = defaultdict(list)
    for c in cands:
        for ip in c["interpro"]:
            groups[(c["organism_id"], ip)].append(c)

    dag = _go_dag()
    rng = random.Random(seed)
    pairs, used, n_go_reject = [], set(), 0
    for (_org, ip), members in sorted(groups.items()):
        by_cls = defaultdict(list)
        for m in members:
            by_cls[m["cls"]].append(m)
        if len(by_cls) < 2:
            continue
        for group in CONTRAST_GROUPS:
            present = sorted(group & set(by_cls))
            for i in range(len(present)):
                for j in range(i + 1, len(present)):
                    a = rng.choice(by_cls[present[i]])
                    b = rng.choice(by_cls[present[j]])
                    key = tuple(sorted((a["accession"], b["accession"])))
                    if key in used or a["accession"] == b["accession"]:
                        continue
                    ga = COFACTOR_CLASSES[a["cls"]]["go"]
                    gb = COFACTOR_CLASSES[b["cls"]]["go"]
                    # The contrast must be real in the curated annotation, not just in which
                    # cofactor a curator happened to mention: A must carry its own term and not the
                    # other, and vice versa.
                    if not (_has(a["go"], [ga], dag) and not _has(a["go"], [gb], dag)
                            and _has(b["go"], [gb], dag) and not _has(b["go"], [ga], dag)):
                        n_go_reject += 1
                        continue
                    used.add(key)
                    pairs.append({
                        "category": "cofactor", "shared_interpro": ip,
                        "shared_interpro_type": ip_type.get(ip, "?"),
                        "a": a, "b": b,
                        "property": f"{a['cls']} vs {b['cls']} cofactor specificity",
                        "spec_a": {"require": [ga], "forbid": [gb],
                                   "label": COFACTOR_CLASSES[a["cls"]]["label"]},
                        "spec_b": {"require": [gb], "forbid": [ga],
                                   "label": COFACTOR_CLASSES[b["cls"]]["label"]},
                    })
    rng.shuffle(pairs)
    print(f"[pairs/cofactor] built {len(pairs)} matched pairs "
          f"({n_go_reject} rejected: curated GO does not support the contrast)")
    return pairs[:max_pairs] if max_pairs else pairs


# ---------------------------------------------------------------------------------------------
# Category B - enzyme vs pseudoenzyme
# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
# Category D - GO `not` pairs: the only construction where both sides are experimental
# ---------------------------------------------------------------------------------------------
# Every other pair design in this file has an asymmetry: the positive member has a curated
# annotation, and the negative member merely *lacks* one.
EXP_EVIDENCE = {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP"}


def _gaf_experimental(gaf_path, taxon="taxon:9606"):
    """(positives, negatives, term_freq) from a gaf, experimental evidence only."""
    pos, neg, freq = defaultdict(set), defaultdict(set), Counter()
    for ln in open(gaf_path):
        if ln.startswith("!"):
            continue
        f = ln.rstrip("\n").split("\t")
        if len(f) < 14 or f[11] != "protein" or f[6] not in EXP_EVIDENCE:
            continue
        if taxon and not f[12].startswith(taxon):
            continue
        acc, term = f[1], f[4]
        if "NOT" in f[3].split("|"):
            neg[term].add(acc)
        else:
            pos[term].add(acc)
            freq[term] += 1
    return pos, neg, freq


def build_not_pairs(organism_id: int, gaf_path, len_min: int, len_max: int, max_pairs: int,
                    min_ic: float = 0.0, seed: int = 0,
                    data_dir: Optional[Path] = None) -> List[Dict]:
    """Same-family pairs contrasting an experimental positive against an experimental GO `NOT`."""
    import math

    ip_type = interpro_types(data_dir)
    fam_closure = interpro_family_closure(data_dir, ip_type)
    dag = _go_dag()
    pos, neg, freq = _gaf_experimental(gaf_path, taxon=f"taxon:{organism_id}")
    total = sum(freq.values()) or 1
    ic = {t: -math.log(freq[t] / total) for t in freq}
    print(f"[pairs/not] {sum(len(v) for v in neg.values())} experimental NOT annotations "
          f"over {len(neg)} terms")

    rows = up.stream_proteome(organism_id=organism_id, reviewed=True,
                              fields="accession,protein_name,organism_name,organism_id,length,"
                                     "sequence,xref_interpro,go_id")
    info = {}
    for r in rows:
        s = (r.get("Sequence") or "").strip()
        if not s or not (len_min <= len(s) <= len_max):
            continue
        fam = set()
        for x in _interpro_ids(r.get("InterPro", "")):
            fam |= fam_closure(x)
        if not fam:
            continue
        info[r["Entry"]] = {"accession": r["Entry"], "sequence": s,
                            "organism": r.get("Organism", ""),
                            "organism_id": r.get("Organism (ID)", ""),
                            "protein_name": r.get("Protein names", ""),
                            "interpro": fam, "go": _go_ids(r.get("Gene Ontology IDs", ""))}
    print(f"[pairs/not] {len(info)} reviewed proteins pass length/family filters")

    # Best (most specific) contrast per protein pair, so one pair contributes one question and the
    # statistics stay clustered by pair rather than by term.
    best = {}
    n_curated_reject = 0
    for t, negs in neg.items():
        if t not in pos or ic.get(t, 0.0) < min_ic:
            continue
        for b in negs:
            if b in pos.get(t, ()):          # contradictory annotation - skip
                continue
            B = info.get(b)
            if not B:
                continue
            # The gaf `not` is experimental-evidence-only, but the protein's full curated GO can
            # still carry the same term under another evidence code, or carry a descendant of it.
            if dag.any_under(B["go"], t):
                n_curated_reject += 1
                continue
            for a in pos[t]:
                if a == b or a in neg.get(t, ()):
                    continue
                A = info.get(a)
                if not A or not (A["interpro"] & B["interpro"]):
                    continue
                if A["organism_id"] != B["organism_id"]:
                    continue
                # Symmetric to the negative-side check above.
                if not dag.any_under(A["go"], t):
                    n_curated_reject += 1
                    continue
                key = (a, b)
                if key not in best or ic.get(t, 0) > ic.get(best[key][0], 0):
                    best[key] = (t, sorted(A["interpro"] & B["interpro"])[0])

    rng = random.Random(seed)
    pairs = []
    for (a, b), (t, fam) in best.items():
        A, B = info[a], info[b]
        pairs.append({
            "category": "go_not", "shared_interpro": fam,
            "shared_interpro_type": ip_type.get(fam, "?"),
            "a": A, "b": B,
            "property": f"{t} ({dag.name.get(t, '?')}) - experimental positive vs experimental NOT",
            "go_term": t, "go_name": dag.name.get(t, ""), "go_aspect": dag.aspect(t),
            "go_ic": round(ic.get(t, 0.0), 3),
            "spec_a": {"require": [t], "forbid": [], "label": dag.name.get(t, t)},
            "spec_b": {"require": [], "forbid": [t], "label": f"NOT {dag.name.get(t, t)}"},
        })
    rng.shuffle(pairs)
    n_terms = len({p["go_term"] for p in pairs})
    print(f"[pairs/not] built {len(pairs)} pairs over {n_terms} distinct GO terms "
          f"(mean IC {sum(p['go_ic'] for p in pairs)/max(len(pairs),1):.2f}); "
          f"{n_curated_reject} candidate negatives rejected: curated GO carries the term the "
          f"experimental NOT denies")
    return pairs[:max_pairs] if max_pairs else pairs


# ---------------------------------------------------------------------------------------------
# Category C - feature-contrast pairs (generalises the enzyme/pseudoenzyme design)
# ---------------------------------------------------------------------------------------------
# Enzyme-vs-pseudoenzyme is one instance of a general shape: two members of the same InterPro Family
# where one carries a curated sequence feature and the other does not, and where the difference
# shows up as a GO term the model emits anyway.
FEATURE_CONTRASTS = {
    "secreted": {
        "query": "ft_signal:*", "go": ["GO:0005576"], "label": "extracellular region",
        "determinant": "N-terminal signal peptide",
    },
    "membrane": {
        "query": "ft_transmem:*", "go": ["GO:0016020"], "label": "membrane",
        "determinant": "hydrophobic transmembrane helices",
    },
    "organelle_targeted": {
        "query": "ft_transit:*", "go": ["GO:0005739", "GO:0009507"], "label": "mitochondrion/plastid",
        "determinant": "N-terminal transit peptide",
    },
    "dna_binding": {
        "query": "ft_dna_bind:*", "go": ["GO:0003677"], "label": "DNA binding",
        "determinant": "DNA-binding domain residues",
    },
}


def build_feature_pairs(contrast: str, organism_ids, len_min: int, len_max: int, max_pairs: int,
                        seed: int = 0, data_dir: Optional[Path] = None) -> List[Dict]:
    """Pairs from one InterPro Family where one member has a curated feature and the other lacks it."""
    spec = FEATURE_CONTRASTS[contrast]
    ip_type = interpro_types(data_dir)
    fam_closure = interpro_family_closure(data_dir, ip_type)
    dag = _go_dag()
    oc = _org_clause(organism_ids)
    flds = ("accession,protein_name,organism_name,organism_id,length,sequence,xref_interpro,go_id")

    def grab(q):
        return _fetch_rows(q + (f" AND {oc}" if oc else ""), flds)

    pos_rows = grab(spec["query"])
    print(f"[pairs/{contrast}] {len(pos_rows)} entries with {spec['query']}")
    fams_needed = set()
    def mk(r):
        s=(r.get("Sequence") or "").strip()
        if not s or not (len_min <= len(s) <= len_max): return None
        fam=set()
        for x in _interpro_ids(r.get("InterPro","")): fam |= fam_closure(x)
        if not fam: return None
        return {"accession": r["Entry"], "sequence": s, "organism": r.get("Organism",""),
                "organism_id": r.get("Organism (ID)",""), "protein_name": r.get("Protein names",""),
                "interpro": fam, "go": _go_ids(r.get("Gene Ontology IDs",""))}
    pos=[m for m in map(mk,pos_rows) if m and _has(m["go"], spec["go"], dag)]
    for m in pos: fams_needed |= m["interpro"]
    print(f"[pairs/{contrast}] {len(pos)} carry the feature AND the readout GO term")
    if not pos: return []

    # negatives: same families, must lack the readout term (and we drop any that carry the feature)
    neg_rows = _fetch_batched([f"xref:interpro-{f}" for f in sorted(fams_needed)], flds,
                              extra=(oc or "reviewed:true"))
    neg=[m for m in map(mk,neg_rows) if m and not _has(m["go"], spec["go"], dag)]
    print(f"[pairs/{contrast}] {len(neg)} same-family candidates lacking it")

    by_key=defaultdict(list)
    for m in neg:
        for f in m["interpro"]: by_key[(m["organism_id"], f)].append(m)
    rng=random.Random(seed); pairs=[]; used=set(); n_ambiguous=0
    for a in pos:
        opts=[(f,m) for f in a["interpro"] for m in by_key.get((a["organism_id"],f),())
              if m["accession"]!=a["accession"]]
        if not opts: continue
        f,b = rng.choice(opts)
        key=tuple(sorted((a["accession"],b["accession"])))
        if key in used: continue
        # The readout is the one term the positive carries, not the contrast's whole term list.
        carried = [t for t in spec["go"] if dag.any_under(a["go"], t)]
        if len(carried) != 1:
            # Dual-targeted (both organelles): a "which compartment" readout has no single answer,
            # and requiring both makes those pairs a harder question than the rest of the category.
            n_ambiguous += 1
            continue
        t = carried[0]
        label = dag.name.get(t, spec["label"])
        used.add(key)
        pairs.append({"category": contrast, "shared_interpro": f,
                      "shared_interpro_type": ip_type.get(f,"?"), "a": a, "b": b,
                      "property": f"{label} present vs absent ({spec['determinant']})",
                      "spec_a": {"require": [t], "forbid": [], "label": label},
                      "spec_b": {"require": [], "forbid": [t], "label": f"no {label}"}})
    rng.shuffle(pairs)
    print(f"[pairs/{contrast}] built {len(pairs)} matched pairs "
          f"({n_ambiguous} dropped: positive carries more than one of {spec['go']}, so the readout "
          f"term is ambiguous)")
    return pairs[:max_pairs] if max_pairs else pairs


def _download(name: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / name
    if out.exists():
        return out
    url = f"{MEROPS_BASE}/{name}"
    print(f"[pairs/merops] downloading {url}")
    tmp = str(out) + ".tmp"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, out)
    return out


def merops_families(data_dir: Optional[Path] = None):
    """(accession -> set(MEROPS family)) for peptidases and for non-peptidase homologues."""
    path = _download("MID_2_UniProt.txt", Path(data_dir or cfg.DATA_DIR) / "merops")
    pep, nph = defaultdict(set), defaultdict(set)
    with open(path, newline="") as fh:
        rd = csv.reader(fh)
        next(rd, None)
        for row in rd:
            if len(row) < 3:
                continue
            mid, _name, acc = row[0], row[1], row[2]
            if not acc or acc == "-":
                continue
            fam = mid.split(".")[0]
            if NPH_RE.match(mid):
                nph[acc].add(fam)
            elif PEP_RE.match(mid):
                pep[acc].add(fam)
    print(f"[pairs/merops] {len(pep)} peptidase accessions, {len(nph)} non-peptidase-homologue accessions")
    return pep, nph


def build_pseudoenzyme_pairs(organism_ids, len_min: int, len_max: int, max_pairs: int,
                             seed: int = 0, data_dir: Optional[Path] = None) -> List[Dict]:
    """Matched pairs differing in catalytic activity (active enzyme vs pseudoenzyme)."""
    ip_type = interpro_types(data_dir)
    fam_closure = interpro_family_closure(data_dir, ip_type)
    org_clause = _org_clause(organism_ids)
    org_q = f" AND {org_clause}" if org_clause else ""
    flds = ("accession,protein_name,organism_name,organism_id,length,sequence,"
            "xref_interpro,ec,cc_caution,cc_catalytic_activity,go_id")

    inactive_rows = _fetch_rows(f"protein_name:inactive{org_q}", flds)
    print(f"[pairs/pseudo] {len(inactive_rows)} reviewed entries named 'Inactive ...'")

    pep_fam, nph_fam = merops_families(data_dir)
    # Reviewed accessions that MEROPS calls non-peptidase homologues, fetched for their sequences.
    nph_accs = sorted(set(nph_fam) - set(pep_fam))      # not also listed as an active peptidase
    merops_rows = []
    if nph_accs:
        merops_rows = _fetch_batched([f"accession:{a}" for a in nph_accs], flds, extra=org_clause)
        print(f"[pairs/pseudo] {len(merops_rows)} of {len(nph_accs)} MEROPS non-peptidase "
              f"homologues are reviewed")

    def mk(r, source):
        seq = (r.get("Sequence") or "").strip()
        if not seq or not (len_min <= len(seq) <= len_max):
            return None
        ip = set()
        for x in _interpro_ids(r.get("InterPro", "")):
            ip |= fam_closure(x)
        if not ip:
            return None
        return {"accession": r["Entry"], "sequence": seq, "organism": r.get("Organism", ""),
                "organism_id": r.get("Organism (ID)", ""), "protein_name": r.get("Protein names", ""),
                "interpro": ip, "ec": r.get("EC number", ""), "go": _go_ids(r.get("Gene Ontology IDs", "")),
                "caution": r.get("Caution", ""), "source": source,
                "merops_families": sorted(nph_fam.get(r["Entry"], ()))}

    pseudo = [m for m in (mk(r, "uniprot_inactive") for r in inactive_rows) if m]
    pseudo += [m for m in (mk(r, "merops_nph") for r in merops_rows) if m]
    # A protein reached by both routes is one protein, not two.
    seen, uniq = set(), []
    for p in pseudo:
        if p["accession"] not in seen:
            seen.add(p["accession"])
            uniq.append(p)
    pseudo = uniq
    print(f"[pairs/pseudo] {len(pseudo)} pseudoenzyme candidates after length/InterPro filters")
    if not pseudo:
        return []

    # Active partners: same organism, share an InterPro entry, have a catalytic-activity annotation,
    # and are not themselves named "inactive".
    need_ipr = sorted({ip for p in pseudo for ip in p["interpro"]})
    print(f"[pairs/pseudo] querying active partners across {len(need_ipr)} InterPro families")
    # Note: `xref_interpro` is a return field, not a queryable one - searching it is a 400. The
    # search
    active_rows = _fetch_batched([f"xref:interpro-{ip}" for ip in need_ipr], flds,
                                 extra="cc_catalytic_activity:*" + (f" AND {org_clause}" if org_clause else ""))
    actives = []
    for r in active_rows:
        if re.search(r"\binactive\b", r.get("Protein names", ""), re.I):
            continue
        m = mk(r, "active")
        if m:
            actives.append(m)
    print(f"[pairs/pseudo] {len(actives)} active-enzyme candidates")

    by_key = defaultdict(list)
    for m in actives:
        for ip in m["interpro"]:
            by_key[(m["organism_id"], ip)].append(m)

    dag = _go_dag()
    rng = random.Random(seed)
    pairs, used, n_go_reject = [], set(), 0
    for p in pseudo:
        # Prefer the family-specific readout when the pair came from MEROPS: "does it predict
        # peptidase activity" is a much sharper question than "does it predict any catalysis".
        require = [PEPTIDASE] if p["source"] == "merops_nph" else [CATALYTIC]
        # The pseudoenzyme must not itself be annotated with the activity we are asking the model to
        # withhold.
        if _has(p["go"], require, dag):
            n_go_reject += 1
            continue
        options = []
        for ip in p["interpro"]:
            options += [(ip, m) for m in by_key.get((p["organism_id"], ip), ())
                        if m["accession"] != p["accession"] and _has(m["go"], require, dag)]
        if not options:
            continue
        ip, a = rng.choice(options)
        key = tuple(sorted((a["accession"], p["accession"])))
        if key in used:
            continue
        used.add(key)
        pairs.append({
            "category": "pseudoenzyme", "shared_interpro": ip,
            "shared_interpro_type": ip_type.get(ip, "?"),
            "a": a, "b": p,
            "property": "catalytically active vs pseudoenzyme",
            "spec_a": {"require": require, "forbid": [], "label": "catalytic activity"},
            "spec_b": {"require": [], "forbid": require, "label": "no catalytic activity"},
            "pseudo_source": p["source"], "pseudo_caution": p.get("caution", "")[:400],
        })
    rng.shuffle(pairs)
    print(f"[pairs/pseudo] built {len(pairs)} matched pairs "
          f"({n_go_reject} rejected: pseudoenzyme is itself annotated with the target activity)")
    return pairs[:max_pairs] if max_pairs else pairs


# --------------------------------------------------------------------------------------------- Emit
# examples
# ---------------------------------------------------------------------------------------------
def pairs_to_examples(pairs: List[Dict], interpro_cache=None, interpro_workers: int = 8,
                      interpro_type_mode: str = "unknown", seed: int = 0) -> List[Example]:
    """Four conditions per pair: two aligned baselines and the two reciprocal conflicts."""
    accs = sorted({m["accession"] for p in pairs for m in (p["a"], p["b"])})
    ip_cache = ipr.fetch_many(accs, cache_path=interpro_cache, workers=interpro_workers)
    ip_text = {a: ipr.format_entries(ip_cache.get(a, []), type_mode=interpro_type_mode) for a in accs}

    examples = []
    for p in pairs:
        A, B = p["a"], p["b"]
        # The category belongs in the id.
        pair_id = f"{p['category']}:{A['accession']}__{B['accession']}"
        gt = {"pair_id": pair_id, "category": p["category"], "property": p["property"],
              "accession_a": A["accession"], "accession_b": B["accession"],
              "protein_name_a": A["protein_name"], "protein_name_b": B["protein_name"],
              "spec_a": p["spec_a"], "spec_b": p["spec_b"],
              "shared_interpro": p["shared_interpro"]}
        prov = provenance(seed=seed, importance=f"matched pair: {p['category']}",
                          source=("UniProt cc_cofactor + InterPro family" if p["category"] == "cofactor"
                                  else "UniProt 'Inactive ...' / MEROPS non-peptidase homologue"),
                          pseudo_source=p.get("pseudo_source"),
                          interpro_source="interpro-api-by-accession")
        # (ESM3 source, GO-GPT source, InterPro source) per condition.
        layout = {
            C.PAIR_ALIGNED_A:          (A, A, A),
            C.PAIR_ALIGNED_B:          (B, B, B),
            C.PAIR_CONFLICT_A:         (A, B, B),   # both text channels from the partner
            C.PAIR_CONFLICT_B:         (B, A, A),
            C.PAIR_CONFLICT_INTERPRO_A: (A, A, B),  # every sequence-derived signal agrees; only
            C.PAIR_CONFLICT_INTERPRO_B: (B, B, A),  # the InterPro annotation text disagrees
            C.PAIR_CONFLICT_GOGPT_A:   (A, B, A),   # mirror: only GO-GPT disagrees
            C.PAIR_CONFLICT_GOGPT_B:   (B, A, B),
        }
        for cond, (emb, go_src, ip_src) in layout.items():
            conflicting = {m["accession"] for m in (go_src, ip_src)} - {emb["accession"]}
            examples.append(Example(
                example_id=pair_id, modality="protein", condition=cond,
                payload={"sequence": emb["sequence"], "symbolic_sequence": go_src["sequence"],
                         "organism": emb["organism"], "accession": emb["accession"],
                         "channels": {"interpro": "wt", "gogpt": "on"},
                         "interpro_wt": ip_text.get(ip_src["accession"], ""),
                         "embedding_id": emb["accession"], "gogpt_id": go_src["accession"],
                         "interpro_id": ip_src["accession"],
                         "context_id": (sorted(conflicting)[0] if conflicting
                                        else emb["accession"]),
                         "length": len(emb["sequence"])},
                intervention={"operator": "matched_pair_conflict", "category": p["category"],
                              "embedding_id": emb["accession"], "gogpt_id": go_src["accession"],
                              "interpro_id": ip_src["accession"],
                              "is_conflict": bool(conflicting)},
                ground_truth=gt, provenance=prov))
    return examples


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--categories", nargs="+", default=["pseudoenzyme"],
                    choices=["cofactor", "pseudoenzyme", "go_not", "merops"] + list(FEATURE_CONTRASTS))
    ap.add_argument("--organisms", type=int, nargs="+", default=MODEL_ORGANISMS,
                    help="taxon ids to draw candidates from (default: the 8 model organisms). "
                         "Pairs are always matched WITHIN one organism; more taxa only enlarge the "
                         "pool. Pass 0 for every reviewed organism (a much larger query).")
    ap.add_argument("--len_min", type=int, default=120)
    ap.add_argument("--len_max", type=int, default=1024)
    ap.add_argument("--max_pairs", type=int, default=400, help="per category; 0 = unlimited")
    ap.add_argument("--gaf", default=None, help="(go_not) GAF path; default DATA_DIR/goa_human.gaf")
    ap.add_argument("--min_ic", type=float, default=0.0,
                    help="(go_not) drop GO terms below this information content (generic terms)")
    ap.add_argument("--max_per_term", type=int, default=0,
                    help="cap pairs sharing one readout GO term (0 = no cap). Without it `go_not` is "
                         "37%% GPCR/bitter-taste terms and a bootstrap over pairs treats one receptor "
                         "family as hundreds of independent observations.")
    ap.add_argument("--max_per_family", type=int, default=0,
                    help="cap pairs sharing one InterPro family (0 = no cap). Without it "
                         "`dna_binding` is 68%% nuclear hormone receptors.")
    ap.add_argument("--max_per_category", type=int, default=0,
                    help="cap pairs per category after the diversity caps (0 = no cap)")
    ap.add_argument("--min_pairs", type=int, default=100,
                    help="warn if a category yields fewer than this many pairs (it will not "
                         "support a reportable conflict rate)")
    ap.add_argument("--interpro_cache", default=None)
    ap.add_argument("--interpro_workers", type=int, default=8)
    ap.add_argument("--interpro_type_mode", choices=["unknown", "real"], default="unknown")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    pairs = []
    if "cofactor" in a.categories:
        pairs += build_cofactor_pairs(a.organisms, a.len_min, a.len_max, a.max_pairs, a.seed,
                                      data_dir=cfg.DATA_DIR)
    if "pseudoenzyme" in a.categories:
        pairs += build_pseudoenzyme_pairs(a.organisms, a.len_min, a.len_max, a.max_pairs, a.seed,
                                          data_dir=cfg.DATA_DIR)
    if "go_not" in a.categories:
        gaf = a.gaf or str(Path(cfg.DATA_DIR) / "goa_human.gaf")
        pairs += build_not_pairs(a.organisms[0] if a.organisms else 9606, gaf, a.len_min,
                                 a.len_max, a.max_pairs, min_ic=a.min_ic, seed=a.seed,
                                 data_dir=cfg.DATA_DIR)
    if "merops" in a.categories:
        gaf = a.gaf or str(Path(cfg.DATA_DIR) / "goa_human.gaf")
        pairs += build_merops_pairs(a.organisms[0] if a.organisms else 9606, gaf, a.len_min,
                                    a.len_max, a.max_pairs, seed=a.seed, data_dir=cfg.DATA_DIR)
    for c in a.categories:
        if c in FEATURE_CONTRASTS:
            pairs += build_feature_pairs(c, a.organisms, a.len_min, a.len_max, a.max_pairs,
                                         a.seed, data_dir=cfg.DATA_DIR)
    if not pairs:
        raise SystemExit("no matched pairs built - widen --len_max/--organisms, or check network")

    # Diversity caps. Applied after building so the counts reported below are of what was available,
    # not of what a cap let through.
    if a.max_per_term or a.max_per_family or a.max_per_category:
        rng_cap = random.Random(a.seed + 4242)
        shuffled = list(pairs)
        rng_cap.shuffle(shuffled)
        # The term cap only means anything where the readout term varies.
        terms_per_cat = defaultdict(set)
        for p in pairs:
            t = ((p.get("spec_a") or {}).get("require")
                 or (p.get("spec_b") or {}).get("forbid") or ["?"])[0]
            terms_per_cat[p["category"]].add(t)
        term_capped = {c for c, ts in terms_per_cat.items() if len(ts) > 5}
        print(f"[pairs] term cap applies to: {sorted(term_capped) or 'none'} "
              f"(categories with >5 distinct readout terms)")
        per_term, per_fam, per_cat = Counter(), Counter(), Counter()
        kept, dropped = [], Counter()
        for p in shuffled:
            t = ((p.get("spec_a") or {}).get("require")
                 or (p.get("spec_b") or {}).get("forbid") or ["?"])[0]
            fam, cat = p.get("shared_interpro", "?"), p["category"]
            if a.max_per_term and cat in term_capped and per_term[(cat, t)] >= a.max_per_term:
                dropped[f"{cat}:term"] += 1; continue
            if a.max_per_family and per_fam[(cat, fam)] >= a.max_per_family:
                dropped[f"{cat}:family"] += 1; continue
            if a.max_per_category and per_cat[cat] >= a.max_per_category:
                dropped[f"{cat}:category"] += 1; continue
            per_term[(cat, t)] += 1; per_fam[(cat, fam)] += 1; per_cat[cat] += 1
            kept.append(p)
        before = Counter(p["category"] for p in pairs)
        pairs = kept
        after = Counter(p["category"] for p in pairs)
        print(f"[pairs] diversity caps (term={a.max_per_term}, family={a.max_per_family}, "
              f"category={a.max_per_category}):")
        for cat in sorted(before):
            print(f"[pairs]   {cat:<22}{before[cat]:>5} -> {after.get(cat, 0):<5} "
                  f"(dropped: " + ", ".join(f"{k.split(':')[1]} {v}" for k, v in sorted(dropped.items())
                                            if k.startswith(cat + ":")) + ")")

    # Flag an underpowered category rather than letting it into a figure.
    counts = Counter(p["category"] for p in pairs)
    for cat in a.categories:
        n = counts.get(cat, 0)
        if n < a.min_pairs:
            print(f"[pairs] WARNING: category '{cat}' yielded only {n} pairs (< --min_pairs "
                  f"{a.min_pairs}). Its conflict rate will not be estimable - report it as "
                  f"'insufficient matched pairs' or drop it, rather than quoting a rate.")

    examples = pairs_to_examples(pairs, a.interpro_cache, a.interpro_workers,
                                 a.interpro_type_mode, a.seed)
    write_jsonl(examples, os.path.join(a.outdir, "examples.jsonl"))

    def _jsonable(p):
        """`interpro` is carried as a set for the grouping logic; JSON needs a list."""
        out = dict(p)
        for side in ("a", "b"):
            m = dict(out[side])
            for k, v in list(m.items()):
                if isinstance(v, set):
                    m[k] = sorted(v)
            m.pop("sequence", None)
            out[side] = m
        return out

    write_jsonl([_jsonable(p) for p in pairs], os.path.join(a.outdir, "pairs.jsonl"))
    by_cat = defaultdict(int)
    for p in pairs:
        by_cat[p["category"]] += 1
    json.dump({"modality": "protein", "profile": "pairs", "experiment": "matched_pair_conflict",
               "n_pairs": len(pairs), "n_examples": len(examples), "pairs_by_category": dict(by_cat),
               "conditions": C.PAIR_CONDS, "organisms": a.organisms, "seed": a.seed,
               "len_min": a.len_min, "len_max": a.len_max,
               "interpro_type_mode": a.interpro_type_mode},
              open(os.path.join(a.outdir, "meta.json"), "w"), indent=2)
    print(f"DONE: {len(pairs)} pairs ({dict(by_cat)}) x 4 conditions = {len(examples)} examples")


# ---------------------------------------------------------------------------------------------
# Category E - MEROPS activity_status: a deterministic active/inactive label
# --------------------------------------------------------------------------------------------- The
# earlier enzyme/pseudoenzyme route required reading UniProt Note: prose and inferring which GO term
# it denied.
def _merops_tables(data_dir):
    d = Path(data_dir) / "merops"
    status_p = _download("activity_status.txt", d)
    go_p = _download("GO_annotation.txt", d)
    mid2acc_p = _download("MID_2_UniProt.txt", d)

    direct = defaultdict(set)                       # mid -> {'active','inactive'} from direct rows
    for ln in open(status_p, errors="ignore"):
        f = ln.rstrip("\r\n").split("\t")
        if len(f) < 4:
            continue
        mid = f[0].strip().strip("'")
        st = f[2].strip().strip("'").strip()
        qual = f[3].strip().strip("'")
        if qual and qual.upper() != "NULL":
            continue                                # "by similarity" - not a direct call
        if st in ("active", "inactive"):
            direct[mid].add(st)
    status = {m: next(iter(v)) for m, v in direct.items() if len(v) == 1}

    mid_go = defaultdict(set)
    for ln in open(go_p, errors="ignore"):
        f = [x.strip().strip('"') for x in ln.rstrip("\r\n").split("\t")]
        if len(f) >= 3 and f[1] and f[2].startswith("GO:"):
            mid_go[f[1]].add(f[2])

    mid_acc = defaultdict(set)
    with open(mid2acc_p, newline="") as fh:
        rd = csv.reader(fh); next(rd, None)
        for r in rd:
            if len(r) >= 3 and r[2] not in ("-", ""):
                mid_acc[r[0]].add(r[2])
    return status, mid_go, mid_acc


def build_merops_pairs(organism_id, gaf_path, len_min, len_max, max_pairs, seed=0, data_dir=None):
    status, mid_go, mid_acc = _merops_tables(data_dir or cfg.DATA_DIR)
    dag = _go_dag()
    pos_exp, _neg, _freq = _gaf_experimental(gaf_path, taxon=f"taxon:{organism_id}")
    print(f"[pairs/merops] {sum(1 for v in status.values() if v=='inactive')} MEROPS ids labelled "
          f"inactive (direct), {sum(1 for v in status.values() if v=='active')} active")

    fam = lambda m: m.split(".")[0]
    act_acc, inact_acc, act_go, inact_go = defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set)
    for mid, st in status.items():
        f = fam(mid)
        (act_acc if st == "active" else inact_acc)[f] |= mid_acc.get(mid, set())
        (act_go if st == "active" else inact_go)[f] |= mid_go.get(mid, set())

    rows = up.stream_proteome(organism_id=organism_id, reviewed=True,
                              fields="accession,protein_name,organism_name,organism_id,length,"
                                     "sequence,xref_interpro,go_id")
    info = {}
    for r in rows:
        s = (r.get("Sequence") or "").strip()
        if s and len_min <= len(s) <= len_max:
            info[r["Entry"]] = {"accession": r["Entry"], "sequence": s,
                                "organism": r.get("Organism", ""),
                                "organism_id": r.get("Organism (ID)", ""),
                                "protein_name": r.get("Protein names", ""),
                                "interpro": _interpro_ids(r.get("InterPro", "")),
                                "go": _go_ids(r.get("Gene Ontology IDs", ""))}

    rng = random.Random(seed); pairs = []; used = set()
    for f in sorted(set(act_acc) & set(inact_acc)):
        # readout candidates: GO terms MEROPS gives this family's active members and not its
        # inactive ones
        cands = [t for t in (act_go[f] - inact_go[f]) if dag.norm(t)]
        if not cands:
            continue
        A = [info[a] for a in act_acc[f] if a in info]
        B = [info[b] for b in inact_acc[f] if b in info]
        for b in B:
            for t in cands:
                if _has(b["go"], [t], dag):
                    continue                        # inactive member must not already claim it positive member:
                # same organism, carries t with experimental evidence in GOA
                opts = [a for a in A if a["organism_id"] == b["organism_id"]
                        and a["accession"] in pos_exp.get(t, set())]
                if not opts:
                    continue
                a = rng.choice(opts)
                key = (a["accession"], b["accession"])
                if key in used:
                    continue
                used.add(key)
                pairs.append({"category": "merops_activity", "shared_interpro": f,
                              "shared_interpro_type": "MEROPS family", "a": a, "b": b,
                              "go_term": t, "go_name": dag.name.get(t, ""), "go_aspect": dag.aspect(t),
                              "go_ic": 0.0,
                              "property": f"MEROPS active vs inactive - {dag.name.get(t,t)}",
                              "spec_a": {"require": [t], "forbid": [], "label": dag.name.get(t, t)},
                              "spec_b": {"require": [], "forbid": [t], "label": f"NOT {dag.name.get(t,t)}"}})
                break
    rng.shuffle(pairs)
    print(f"[pairs/merops] built {len(pairs)} pairs over "
          f"{len({p['go_term'] for p in pairs})} GO terms, {len({p['shared_interpro'] for p in pairs})} families")
    return pairs[:max_pairs] if max_pairs else pairs


if __name__ == "__main__":
    main()

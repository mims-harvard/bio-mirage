"""UniProt access for the protein experiments: fetches and caches entries through the UniProt REST
API, streams reviewed proteomes, and extracts sequences, functional residue features, domain spans,
GO cross-references and function text.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "input_use/1.0 (research)"})

REST = "https://rest.uniprot.org/uniprotkb"
DEFAULT_CACHE = Path(__file__).resolve().parent.parent / "cache" / "uniprot"

# Per-residue functional-site feature types that define the salient set (catalytic/binding residues
# the paper targets). These are residue-level and functionally interpretable.
FUNCTIONAL_SITE_TYPES = (
    "Active site",     # catalytic residues
    "Binding site",    # ligand / substrate / cofactor binding
    "Metal binding",   # metal-coordinating residues
    "DNA binding",     # (residue-level DNA-contacting; note: also exists as a Region)
    "Site",            # other functionally important single residues (e.g. cleavage, redox)
)
# Secondary residue-level annotations (used for matching/exclusion, optionally as importance).
OTHER_RESIDUE_TYPES = ("Modified residue", "Disulfide bond", "Cross-link", "Glycosylation")
# Region-level structural context (domain membership etc.) - used by importance/protein.py.
REGION_TYPES = ("Domain", "Region", "Motif", "Repeat", "Zinc finger")

# Experimental evidence (curated from experiments) vs. inferred (by similarity / automatic).
EXPERIMENTAL_ECO = {
    "ECO:0000269",  # experimental evidence used in manual assertion
    "ECO:0000314",  # direct assay
    "ECO:0000315",  # mutant phenotype
    "ECO:0007744",  # combinatorial experimental + computational (e.g. ptm proteomics)
}


# --------------------------------------------------------------------------- Fetch + cache
# ---------------------------------------------------------------------------
def fetch_entry(accession: str, cache_dir: os.PathLike = DEFAULT_CACHE,
                force: bool = False, max_retries: int = 4) -> Dict:
    """Fetch a UniProtKB entry as JSON, caching to disk. Records the UniProt release."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{accession}.json"
    if path.exists() and not force:
        return json.loads(path.read_text())

    url = f"{REST}/{accession}.json"
    last = None
    for attempt in range(max_retries):
        try:
            r = _SESSION.get(url, timeout=30)
            if r.status_code == 200:
                entry = r.json()
                entry["_input_use_meta"] = {
                    "uniprot_release": r.headers.get("X-UniProt-Release"),
                    "uniprot_release_date": r.headers.get("X-UniProt-Release-Date"),
                    "fetched_url": url,
                }
                path.write_text(json.dumps(entry))
                return entry
            last = f"HTTP {r.status_code}"
        except requests.RequestException as e:  # network hiccup -> backoff
            last = str(e)
        time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to fetch {accession}: {last}")


def get_sequence(entry: Dict) -> str:
    return entry["sequence"]["value"]


def get_organism(entry: Dict) -> str:
    return entry.get("organism", {}).get("scientificName", "")


def uniprot_release(entry: Dict) -> Optional[str]:
    return entry.get("_input_use_meta", {}).get("uniprot_release")


def verify_sequence(entry: Dict, sequence: str) -> bool:
    """True iff the given sequence exactly matches the UniProt canonical sequence (so that 1-based
    feature positions are valid for it). Perturbation experiments must use the canonical sequence;
    otherwise residue indices are meaningless.
    """
    return get_sequence(entry) == sequence.strip().upper()


# --------------------------------------------------------------------------- Feature extraction
# (the importance signal)
# ---------------------------------------------------------------------------
def _evidence_codes(feature: Dict) -> List[str]:
    return sorted({e.get("evidenceCode", "") for e in feature.get("evidences", []) if e.get("evidenceCode")})


def functional_residues(entry: Dict,
                        types: Iterable[str] = FUNCTIONAL_SITE_TYPES,
                        require_experimental: bool = False) -> List[Dict]:
    """Return per-residue functional-site annotations on the canonical sequence."""
    seq = get_sequence(entry)
    types = set(types)
    out: List[Dict] = []
    for f in entry.get("features", []):
        if f.get("type") not in types:
            continue
        loc = f.get("location", {})
        start = (loc.get("start") or {}).get("value")
        end = (loc.get("end") or {}).get("value")
        if start is None or end is None:
            continue
        ev = _evidence_codes(f)
        is_exp = any(c in EXPERIMENTAL_ECO for c in ev)
        if require_experimental and not is_exp:
            continue
        for pos in range(int(start), int(end) + 1):
            if pos < 1 or pos > len(seq):
                continue
            out.append({
                "position": pos,
                "aa": seq[pos - 1],
                "feature_type": f["type"],
                "description": f.get("description", ""),
                "ligand": (f.get("ligand") or {}).get("name", ""),
                "evidence": ev,
                "experimental": is_exp,
            })
    return out


def function_text(entry: Dict) -> str:
    """Plain-text UniProt function comment(s) - the curated free-text description of what the protein
    does. Used as the retrieval ground-truth ("own description") for ProTrek's text encoder.
    Concatenates every text block of every `commentType == "FUNCTION"` comment, in order.
    """
    out: List[str] = []
    for c in entry.get("comments", []):
        if c.get("commentType") != "FUNCTION":
            continue
        for t in c.get("texts", []):
            v = (t.get("value") or "").strip()
            if v:
                out.append(v)
    return " ".join(out)


def go_annotations(entry: Dict) -> List[str]:
    """Curated GO terms for the entry (UniProt GO cross-references) - used as ground truth."""
    gos = []
    for x in entry.get("uniProtKBCrossReferences", []):
        if x.get("database") == "GO":
            gid = x.get("id")
            if gid and gid.startswith("GO:"):
                gos.append(gid)
    return sorted(set(gos))


def domain_span(entry: Dict, types: Iterable[str] = ("Domain",) + REGION_TYPES):
    """Largest annotated structural region (1-based, inclusive) as (start, end, feature_type), or None
    if the entry has no such feature. Prefers 'Domain' features; falls back to other region types
    (Region/Motif/Repeat/Zinc finger). The largest region is taken as the (typically catalytic)
    functional domain.
    """
    seq_len = entry["sequence"]["length"]
    best = None
    # iterate types in priority order; within the first type that yields any feature, take the
    # largest
    seen_type = None
    for f in entry.get("features", []):
        ft = f.get("type")
        if ft not in set(types):
            continue
        loc = f.get("location", {})
        s = (loc.get("start") or {}).get("value")
        e = (loc.get("end") or {}).get("value")
        if s is None or e is None:
            continue
        s, e = int(s), int(e)
        if not (1 <= s <= e <= seq_len):
            continue
        span = (s, e, ft)
        if best is None or (e - s) > (best[1] - best[0]):
            best = span
    return best


def annotated_positions(entry: Dict, types: Iterable[str]) -> set:
    """Set of 1-based positions covered by any feature of the given types (for control exclusion)."""
    seq_len = entry["sequence"]["length"]
    covered = set()
    types = set(types)
    for f in entry.get("features", []):
        if f.get("type") not in types:
            continue
        loc = f.get("location", {})
        s = (loc.get("start") or {}).get("value")
        e = (loc.get("end") or {}).get("value")
        if s is None or e is None:
            continue
        covered.update(p for p in range(int(s), int(e) + 1) if 1 <= p <= seq_len)
    return covered


# --------------------------------------------------------------------------- Build the experiment
# set: proteins that do carry curated functional-site annotations
# ---------------------------------------------------------------------------
def search_uniprot(query: str, fields: str = "accession", size: int = 100,
                   max_records: int = 500) -> List[Dict]:
    """Query the UniProtKB search API (paginated). Returns the list of result rows."""
    url = f"{REST}/search"
    params = {"query": query, "fields": fields, "format": "json", "size": min(size, 500)}
    rows: List[Dict] = []
    while True:
        r = _SESSION.get(url, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        rows.extend(data.get("results", []))
        if len(rows) >= max_records:
            return rows[:max_records]
        # follow cursor pagination via the Link header
        nxt = r.links.get("next", {}).get("url")
        if not nxt:
            return rows
        url, params = nxt, None


def stream_proteome(organism_id: int = 9606, reviewed: bool = True,
                    fields: str = "accession,id,protein_name,organism_name,length,sequence",
                    extra_query: str = "", timeout: int = 600) -> List[Dict]:
    """Whole proteome in one request, as a list of dicts keyed by the requested fields."""
    parts = []
    if organism_id:
        parts.append(f"organism_id:{organism_id}")
    if reviewed:
        parts.append("reviewed:true")
    if extra_query:
        parts.append(f"({extra_query})")
    params = {"query": " AND ".join(parts), "fields": fields, "format": "tsv"}
    r = _SESSION.get(f"{REST}/stream", params=params, timeout=timeout)
    r.raise_for_status()
    lines = r.text.rstrip("\n").split("\n")
    if len(lines) < 2:
        return []
    header = lines[0].split("\t")
    return [dict(zip(header, ln.split("\t"))) for ln in lines[1:]]


def find_proteins_with_functional_sites(organism_id: int,
                                        reviewed: bool = True,
                                        max_records: int = 200,
                                        extra_query: str = "") -> List[str]:
    """Accessions of (reviewed) proteins that have curated **active site or binding site** annotations
    in the given organism - the candidate population for salient residues.
    """
    parts = [f"organism_id:{organism_id}"] if organism_id else []   # organism_id=0/None -> any organism
    if reviewed:
        parts.append("reviewed:true")
    parts.append("(ft_act_site:* OR ft_binding:*)")  # has an active or binding site
    if extra_query:
        parts.append(f"({extra_query})")
    query = " AND ".join(parts)
    rows = search_uniprot(query, fields="accession", size=500, max_records=max_records)
    return [row["primaryAccession"] for row in rows]

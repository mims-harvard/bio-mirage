"""Reads GO Annotation (GOA) files into per-protein GO term sets, the ground truth for
BioReason-Pro.

Keeps protein annotations of one taxon with the curated evidence codes EXP, IDA, IPI, IMP, IGI, IEP,
TAS and IC and drops NOT annotations. High-throughput codes are optional. Run as a script to
download the human GAF and print its annotation counts per GO aspect.
"""
from __future__ import annotations

import gzip
import os
import shutil
import tempfile
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Set

from input_use.core import config as cfg

GAF_URL = "https://ftp.ebi.ac.uk/pub/databases/GO/goa/HUMAN/goa_human.gaf.gz"

# gaf 2.2 column indices (0-based) for the fields we read.
COL_DB, COL_ACC, COL_SYMBOL, COL_QUALIFIER, COL_GO = 0, 1, 2, 3, 4
COL_EVIDENCE, COL_ASPECT, COL_NAME, COL_TYPE, COL_TAXON, COL_DATE = 6, 8, 9, 11, 12, 13

ASPECT_CODE = {"F": "MF", "P": "BP", "C": "CC"}

CAFA_EVIDENCE = frozenset({"EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "TAS", "IC"})
# High-throughput counterparts, added by GO in 2017. Off by default: including them changes the
# label set relative to the CAFA convention BioReason-Pro's benchmark follows, so it is opt-in.
HTP_EVIDENCE = frozenset({"HTP", "HDA", "HMP", "HGI", "HEP"})


def download_gaf(url: str = GAF_URL, dest: Optional[os.PathLike] = None,
                 force: bool = False) -> Path:
    """Download and decompress the gaf, caching under `cfg.DATA_DIR`. Returns the .gaf path."""
    dest = Path(dest) if dest else Path(cfg.DATA_DIR) / "goa_human.gaf"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return dest
    print(f"[goa] downloading {url}")
    with tempfile.NamedTemporaryFile(suffix=".gaf.gz", delete=False) as tmp:
        tmp_gz = tmp.name
    try:
        urllib.request.urlretrieve(url, tmp_gz)
        tmp_out = str(dest) + ".tmp"
        with gzip.open(tmp_gz, "rb") as fin, open(tmp_out, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        os.replace(tmp_out, dest)          # atomic: never leave a half-written gaf behind
    finally:
        os.unlink(tmp_gz)
    print(f"[goa] wrote {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


@dataclass
class Annotations:
    """Curated GO annotations keyed by UniProt accession."""
    terms: Dict[str, Dict[str, str]] = field(default_factory=lambda: defaultdict(dict))
    aspect_of: Dict[str, str] = field(default_factory=dict)       # go_id -> mf/BP/cc
    gaf_version: str = ""
    generated: str = ""
    n_lines: int = 0
    n_kept: int = 0

    def accessions(self) -> Set[str]:
        return set(self.terms)

    def ground_truth(self, accessions: Optional[Iterable[str]] = None,
                     min_date: Optional[str] = None,
                     aspects: Optional[Iterable[str]] = None) -> Dict[str, Set[str]]:
        """{accession: set(GO terms)}, ready to hand to `metrics/cafa.py` (which propagates)."""
        keep_aspects = set(aspects) if aspects else None
        accs = set(accessions) if accessions is not None else set(self.terms)
        out: Dict[str, Set[str]] = {}
        for acc in accs:
            got = set()
            for go, date in self.terms.get(acc, {}).items():
                if min_date and date < min_date:
                    continue
                if keep_aspects and self.aspect_of.get(go) not in keep_aspects:
                    continue
                got.add(go)
            if got:
                out[acc] = got
        return out


def load_annotations(gaf_path: Optional[os.PathLike] = None,
                     evidence: Iterable[str] = CAFA_EVIDENCE,
                     include_htp: bool = False,
                     taxon: str = "taxon:9606",
                     object_type: str = "protein") -> Annotations:
    """Parse a gaf into `Annotations`, keeping only curated, non-negated protein annotations."""
    gaf_path = Path(gaf_path) if gaf_path else download_gaf()
    codes = set(evidence) | (set(HTP_EVIDENCE) if include_htp else set())
    ann = Annotations()

    opener = gzip.open if str(gaf_path).endswith(".gz") else open
    with opener(gaf_path, "rt") as fh:
        for line in fh:
            if line.startswith("!"):
                if line.startswith("!gaf-version:"):
                    ann.gaf_version = line.split(":", 1)[1].strip()
                elif line.startswith("!date-generated:"):
                    ann.generated = line.split(":", 1)[1].strip()
                continue
            ann.n_lines += 1
            f = line.rstrip("\n").split("\t")
            if len(f) <= COL_DATE:
                continue
            if "NOT" in f[COL_QUALIFIER].split("|"):
                continue
            if f[COL_EVIDENCE] not in codes:
                continue
            if object_type and f[COL_TYPE] != object_type:
                continue
            if taxon and not f[COL_TAXON].startswith(taxon):
                continue
            aspect = ASPECT_CODE.get(f[COL_ASPECT])
            if not aspect:
                continue
            acc, go, date = f[COL_ACC], f[COL_GO], f[COL_DATE]
            prev = ann.terms[acc].get(go)
            if prev is None or date < prev:      # keep the earliest date (see dataclass docstring)
                ann.terms[acc][go] = date
            ann.aspect_of[go] = aspect
            ann.n_kept += 1
    ann.terms = dict(ann.terms)
    return ann


def summarise(ann: Annotations, min_date: Optional[str] = None) -> str:
    """One-line provenance string for meta.json / summary.md."""
    gt = ann.ground_truth(min_date=min_date)
    n_terms = sum(len(v) for v in gt.values())
    per = n_terms / len(gt) if gt else 0.0
    window = f" first annotated >= {min_date}" if min_date else ""
    return (f"GOA {ann.gaf_version} (generated {ann.generated}), curated evidence only"
            f"{window}: {len(gt)} proteins, {n_terms} annotations ({per:.1f}/protein)")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Download + summarise the human GOA ground truth.")
    ap.add_argument("--gaf", default=None, help="existing GAF path (default: download to DATA_DIR)")
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    ap.add_argument("--min_date", default=None, help="temporal holdout cutoff, YYYYMMDD")
    ap.add_argument("--include_htp", action="store_true", help="also accept high-throughput codes")
    a = ap.parse_args()

    path = download_gaf(force=a.force) if not a.gaf else Path(a.gaf)
    ann = load_annotations(path, include_htp=a.include_htp)
    print(f"[goa] parsed {ann.n_lines} rows, kept {ann.n_kept}")
    print("[goa] " + summarise(ann))
    if a.min_date:
        print("[goa] " + summarise(ann, min_date=a.min_date))
    for asp in ("MF", "BP", "CC"):
        gt = ann.ground_truth(aspects=(asp,), min_date=a.min_date)
        print(f"[goa]   {asp}: {len(gt)} proteins, {sum(len(v) for v in gt.values())} annotations")


if __name__ == "__main__":
    main()

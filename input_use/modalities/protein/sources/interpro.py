"""Fetches InterPro entries for UniProt accessions and formats them for BioReason-Pro prompts.
Run as a script to fetch and cache the InterPro entries of a list of accessions.

Results are cached in one JSON file keyed by accession, so an interrupted fetch resumes.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests

from input_use.core import config as cfg

API = "https://www.ebi.ac.uk/interpro/api/entry/InterPro/protein/uniprot"
PAGE_SIZE = 200          # more InterPro entries than any real protein has; avoids paging entirely

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "input_use/1.0 (research)", "Accept": "application/json"})


def _default_cache() -> Path:
    return Path(cfg.DATA_DIR) / "interpro_by_accession.json"


def fetch_one(accession: str, max_retries: int = 4, timeout: int = 30) -> List[Dict]:
    """Precomputed InterPro matches for one accession."""
    url = f"{API}/{accession}/"
    last = None
    for attempt in range(max_retries):
        try:
            r = _SESSION.get(url, params={"page_size": PAGE_SIZE}, timeout=timeout)
            if r.status_code in (204, 404):
                return []                       # no InterPro entries for this protein
            if r.status_code == 200:
                return _parse(r.json(), accession)
            if r.status_code in (429, 500, 502, 503, 504):
                last = f"HTTP {r.status_code}"   # transient -> back off and retry
            else:
                raise RuntimeError(f"InterPro {accession}: HTTP {r.status_code}")
        except requests.RequestException as e:
            last = str(e)
        time.sleep(2 ** attempt)
    raise RuntimeError(f"InterPro fetch failed for {accession}: {last}")


def _parse(payload: Dict, accession: str) -> List[Dict]:
    out: List[Dict] = []
    for res in payload.get("results", []):
        md = res.get("metadata") or {}
        ipr = md.get("accession")
        if not ipr:
            continue
        starts, ends = [], []
        for prot in res.get("proteins", []):
            for loc in prot.get("entry_protein_locations") or []:
                for frag in loc.get("fragments") or []:
                    if frag.get("start") is not None and frag.get("end") is not None:
                        starts.append(int(frag["start"]))
                        ends.append(int(frag["end"]))
        if not starts:
            continue
        out.append({
            "interpro_id": ipr.upper(),
            "entry_name": md.get("name") or "",
            "type": md.get("type") or "unknown",
            "start": min(starts),
            "end": max(ends),
        })
    out.sort(key=lambda d: d["interpro_id"])     # pandas groupby ordering in the reference impl
    return out


def format_entries(entries: Iterable[Dict], type_mode: str = "unknown") -> str:
    """Render to BioReason-Pro's prompt format: `- IPR000494: Receptor L-domain (unknown) [57-480]`."""
    lines = []
    for e in entries:
        t = e.get("type", "unknown") if type_mode == "real" else "unknown"
        lines.append(f"- {e['interpro_id']}: {e['entry_name']} ({t}) [{e['start']}-{e['end']}]")
    return "\n".join(lines)


def _atomic_write(path: Path, obj) -> None:
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh)
    os.replace(tmp, path)        # never leave a half-written cache if we're preempted


def fetch_many(accessions: Iterable[str], cache_path: Optional[os.PathLike] = None,
               workers: int = 8, flush_every: int = 500,
               resume: bool = True) -> Dict[str, List[Dict]]:
    """Fetch (and cache) InterPro matches for many accessions."""
    cache_path = Path(cache_path) if cache_path else _default_cache()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache: Dict[str, List[Dict]] = {}
    if resume and cache_path.exists():
        cache = json.loads(cache_path.read_text())
        print(f"[interpro] resumed {len(cache)} cached accessions from {cache_path}")

    todo = [a for a in dict.fromkeys(accessions) if a not in cache]
    if not todo:
        return cache
    print(f"[interpro] fetching {len(todo)} accessions with {workers} workers ...")

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for acc, entries in zip(todo, pool.map(fetch_one, todo)):
            cache[acc] = entries
            done += 1
            if done % flush_every == 0:
                _atomic_write(cache_path, cache)
                print(f"[interpro]   {done}/{len(todo)}")
    _atomic_write(cache_path, cache)
    n_hit = sum(1 for a in todo if cache.get(a))
    print(f"[interpro] done: {done} fetched, {n_hit} with >=1 InterPro entry "
          f"({100.0 * n_hit / max(done, 1):.1f}%) -> {cache_path}")
    return cache


def assert_shuffle_empty(sequences: Iterable[str], sample: int = 200, email: str = "") -> Dict:
    """Verify the assumption that shuffled sequences carry no InterPro annotation."""
    import random
    import sys

    sys.path.insert(0, str(cfg.require(cfg.BRP_REPO, "BioReason-Pro repo")))
    from interpro_api import run_interproscan_online   # noqa: E402  (repo-local module)

    seqs = list(sequences)
    random.Random(0).shuffle(seqs)
    seqs = seqs[:sample]
    hits = []
    for i, s in enumerate(seqs, 1):
        df = run_interproscan_online(s, email or "anonymous@example.com")
        if df is not None and not df.empty:
            hits.append({"index": i, "n_rows": int(len(df))})
        if i % 25 == 0:
            print(f"[interpro-check] {i}/{len(seqs)} scanned, {len(hits)} with hits")
    return {"n": len(seqs), "n_hit": len(hits), "hits": hits}


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--accessions", nargs="+", help="accessions to fetch")
    ap.add_argument("--accession_file", help="file with one accession per line")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--show", action="store_true", help="print the formatted block for each")
    a = ap.parse_args()

    accs = list(a.accessions or [])
    if a.accession_file:
        accs += [ln.strip() for ln in open(a.accession_file) if ln.strip()]
    cache = fetch_many(accs, cache_path=a.cache, workers=a.workers)
    if a.show:
        for acc in accs:
            print(f"--- {acc} ---")
            print(format_entries(cache.get(acc, [])) or "(no InterPro entries)")


if __name__ == "__main__":
    main()

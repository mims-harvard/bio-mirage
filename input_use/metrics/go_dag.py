"""Minimal parser for go-basic.obo: term names, aspects, and ancestors and descendants over `is_a` and
`part_of`.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Dict, Iterable, Optional, Set

# The three ontology roots. CAFA excludes them from both predictions and ground truth (every protein
# trivially has all three), so ancestor sets here exclude them too.
ROOTS = {
    "GO:0003674": "MF",   # molecular_function
    "GO:0008150": "BP",   # biological_process
    "GO:0005575": "CC",   # cellular_component
}
NAMESPACE_CODE = {"molecular_function": "MF", "biological_process": "BP", "cellular_component": "CC"}


class GODag:
    """Parsed go-basic.obo. Construct once per process - parsing ~47k terms takes ~1 s."""

    def __init__(self, obo_path):
        self.path = str(obo_path)
        self.name: Dict[str, str] = {}
        self.aspect_of: Dict[str, str] = {}
        self.parents: Dict[str, Set[str]] = {}
        self.alt_to_main: Dict[str, str] = {}
        self._children: Optional[Dict[str, Set[str]]] = None   # built lazily; only Exp-2 needs it
        self._anc_cache: Dict[str, frozenset] = {}
        self._parse()

    def _parse(self) -> None:
        cur: Optional[dict] = None
        in_term = False

        def flush(t):
            if not t or t.get("obsolete") or "id" not in t:
                return
            tid = t["id"]
            self.name[tid] = t.get("name", "")
            self.aspect_of[tid] = NAMESPACE_CODE.get(t.get("namespace", ""), "")
            self.parents[tid] = t.get("parents", set())
            for a in t.get("alt_ids", ()):
                self.alt_to_main[a] = tid

        with open(self.path) as fh:
            for raw in fh:
                line = raw.rstrip("\n")
                if line.startswith("["):
                    flush(cur)
                    in_term = line.startswith("[Term]")
                    cur = {"parents": set(), "alt_ids": set()} if in_term else None
                    continue
                if not in_term or cur is None or not line:
                    continue
                key, _, val = line.partition(": ")
                if key == "id":
                    cur["id"] = val.strip()
                elif key == "name":
                    cur["name"] = val.strip()
                elif key == "namespace":
                    cur["namespace"] = val.strip()
                elif key == "alt_id":
                    cur["alt_ids"].add(val.strip())
                elif key == "is_obsolete":
                    cur["obsolete"] = val.strip() == "true"
                elif key == "is_a":
                    # "GO:0005515 ! protein binding"  ->  GO:0005515
                    cur["parents"].add(val.split("!")[0].strip())
                elif key == "relationship":
                    rel, _, target = val.strip().partition(" ")
                    if rel == "part_of":            # cafaeval propagates is_a + part_of only
                        cur["parents"].add(target.split("!")[0].strip())
        flush(cur)
        # Drop edges pointing at obsolete/unknown terms so traversal can't wander off the graph.
        known = set(self.parents)
        for tid, ps in self.parents.items():
            self.parents[tid] = {p for p in ps if p in known}

    # ------------------------------------------------------------------
    def norm(self, term: str) -> Optional[str]:
        """Canonical id for `term` (resolving alt_ids), or None if it is unknown/obsolete."""
        if term in self.parents:
            return term
        return self.alt_to_main.get(term)

    def aspect(self, term: str) -> str:
        """'mf' | 'BP' | 'cc', or '' when the term is not in this ontology release."""
        t = self.norm(term)
        return self.aspect_of.get(t, "") if t else ""

    def _ancestors_cached(self, term: str) -> frozenset:
        """Memoised per instance - `propagate` is called once per (protein, condition), so the same
        handful of terms is walked tens of thousands of times.
        """
        hit = self._anc_cache.get(term)
        if hit is not None:
            return hit
        seen, queue = set(), deque([term])
        while queue:
            t = queue.popleft()
            for p in self.parents.get(t, ()):
                if p not in seen:
                    seen.add(p)
                    queue.append(p)
        out = frozenset(seen - set(ROOTS))
        self._anc_cache[term] = out
        return out

    def ancestors(self, term: str, include_self: bool = True) -> Set[str]:
        """All `is_a`/`part_of` ancestors, roots excluded. Unknown terms give an empty set."""
        t = self.norm(term)
        if not t:
            return set()
        out = set(self._ancestors_cached(t))
        if include_self and t not in ROOTS:
            out.add(t)
        return out

    def propagate(self, terms: Iterable[str]) -> Set[str]:
        """Union of every term's ancestor closure - the same closure cafaeval applies internally."""
        out: Set[str] = set()
        for t in terms:
            out |= self.ancestors(t, include_self=True)
        return out

    # ------------------------------------------------------------------
    @property
    def children(self) -> Dict[str, Set[str]]:
        if self._children is None:
            kids: Dict[str, Set[str]] = {}
            for tid, ps in self.parents.items():
                for p in ps:
                    kids.setdefault(p, set()).add(tid)
            self._children = kids
        return self._children

    def descendants(self, term: str, include_self: bool = True) -> Set[str]:
        """Every term below `term`, e.g. to test whether any catalytic activity term
        (GO:0003824 or below) is in a model's output.
        """
        t = self.norm(term)
        if not t:
            return set()
        kids = self.children
        seen, queue = set(), deque([t])
        while queue:
            cur = queue.popleft()
            for c in kids.get(cur, ()):
                if c not in seen:
                    seen.add(c)
                    queue.append(c)
        if include_self:
            seen.add(t)
        return seen

    def any_under(self, terms: Iterable[str], root: str) -> bool:
        """True iff any of `terms` is `root` or one of its descendants. Implemented by walking each
        term's ancestors (cheap, cached) rather than materialising the descendant set.
        """
        r = self.norm(root)
        if not r:
            return False
        return any(r in self.ancestors(t, include_self=True) for t in terms)


_CACHE: Dict[str, GODag] = {}


def load(obo_path) -> GODag:
    """Process-level cache - the OBO is 31 MB and several call sites want the same parse."""
    key = str(Path(obo_path).resolve())
    if key not in _CACHE:
        _CACHE[key] = GODag(key)
    return _CACHE[key]

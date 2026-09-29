#!/usr/bin/env python
"""Builds cell type descriptions from Cell Ontology relationship axioms for the C2S-Scale evidence
conflicts.

Facts are surface markers present, low or absent, capabilities, anatomical location and
developmental origin. A cell type's facts include the axioms on all its is_a ancestors. The
description of B for a pair (A, B) lists the facts of B that A does not hold, with facts that
contradict A first, and removes the names and synonyms of both types. Used by
prepare_text_conflict.py.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Set, Tuple

# The six relations we render. Chosen because each is populated in the release and maps to a claim a
# biologist would recognise; every other relation is ignored rather than guessed at.
POSITIVE = "RO:0002104"      # has plasma membrane part          -> surface marker present
LOW = "RO:0015016"           # has low plasma membrane amount    -> marker dim
NEGATIVE = "CL:4030046"      # lacks plasma membrane part        -> marker absent
CAPABLE = "RO:0002215"       # capable of                        -> GO biological process
PART_OF = "BFO:0000050"      # part of                           -> uberon structure
DEVELOPS = "RO:0002202"      # develops from                     -> CL precursor

TEMPLATES = {
    POSITIVE: "expresses {label} on its surface",
    LOW: "expresses low levels of {label} on its surface",
    NEGATIVE: "does not express {label} on its surface",
    CAPABLE: "is capable of {label}",
    PART_OF: "is found in the {label}",
    DEVELOPS: "develops from {label}",
}
# Markers first: surface phenotype is what discriminates immune cell types, and it is what a reader
# would weigh most. Ordering is fixed so descriptions are deterministic.
RELATION_PRIORITY = {POSITIVE: 0, NEGATIVE: 1, LOW: 2, CAPABLE: 3, DEVELOPS: 4, PART_OF: 5}

# A marker claim and its denial are mutually exclusive; these pairs are what "contradictory" means.
_CONTRADICTS = {POSITIVE: {NEGATIVE, LOW}, NEGATIVE: {POSITIVE, LOW}, LOW: {POSITIVE, NEGATIVE}}


@dataclass(frozen=True)
class Fact:
    relation: str
    obj: str
    label: str

    def render(self) -> str:
        return TEMPLATES[self.relation].format(label=self.label)


class CLFacts:
    """Parsed full-CL: is_a graph, relationship axioms, names and synonyms."""

    def __init__(self, obo_path: str):
        self.parents: Dict[str, Set[str]] = defaultdict(set)
        self.rels: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
        self.names: Dict[str, str] = {}
        self.synonyms: Dict[str, List[str]] = defaultdict(list)
        self._parse(obo_path)
        self._n_terms = max(1, sum(1 for t in self.names if t.startswith("CL:")))

    def _parse(self, path: str) -> None:
        cur: Optional[str] = None
        obsolete = False
        syn_re = re.compile(r'^synonym: "(.*?)"')
        for line in open(path):
            line = line.rstrip("\n")
            if line == "[Term]":
                cur, obsolete = None, False
            elif line.startswith("id: "):
                cur = line[4:].strip()
            elif not cur:
                continue
            elif line.startswith("is_obsolete: true"):
                obsolete = True
                self.parents.pop(cur, None)
                self.rels.pop(cur, None)
                self.names.pop(cur, None)
                cur = None
            elif obsolete:
                continue
            elif line.startswith("name: "):
                self.names[cur] = line[6:].strip()
            elif line.startswith("synonym: "):
                m = syn_re.match(line)
                if m:
                    self.synonyms[cur].append(m.group(1))
            elif line.startswith("is_a: "):
                self.parents[cur].add(line[6:].split()[0].strip())
            elif line.startswith("relationship: "):
                parts = line[len("relationship: "):].split()
                if len(parts) >= 2 and parts[0] in TEMPLATES:
                    self.rels[cur].add((parts[0], parts[1]))

    # ---- graph ----
    @lru_cache(maxsize=None)
    def ancestors(self, term: str) -> frozenset:
        """`term` plus every is_a ancestor. CL is a dag (multiple inheritance is common), so this is a
        reachable set, not a path -- a single lineage walk would miss half of a Treg's properties.
        """
        seen, stack = {term}, [term]
        while stack:
            for p in self.parents.get(stack.pop(), ()):
                if p not in seen:
                    seen.add(p)
                    stack.append(p)
        return frozenset(seen)

    @lru_cache(maxsize=None)
    def n_descendants(self, term: str) -> int:
        return sum(1 for t in self.names if t.startswith("CL:") and term in self.ancestors(t))

    def ic(self, term: str) -> float:
        """Resnik information content: -log(p(term)), p estimated structurally as the fraction of CL
        terms subsumed by it. A general term (`cell`) scores ~0; a leaf scores high.
        """
        return -math.log(max(self.n_descendants(term), 1) / self._n_terms)

    def lca(self, a: str, b: str) -> Optional[str]:
        """Most informative common ancestor. Used instead of hop counts because CL branches at very
        uneven rates -- two hops in the T-cell subtree covers far less biology than two hops near
        the root, so hop distance is not comparable across the ontology.
        """
        common = self.ancestors(a) & self.ancestors(b)
        return max(common, key=self.ic) if common else None

    def is_ancestor_descendant(self, a: str, b: str) -> bool:
        return a in self.ancestors(b) or b in self.ancestors(a)

    # ---- facts ----
    @lru_cache(maxsize=None)
    def facts(self, term: str) -> frozenset:
        """All axioms on `term` and its is_a ancestors (see module docstring on why closure matters).
        Objects with no label in the release are dropped -- an unlabelled id cannot be templated
        into a sentence, and inventing one would break the 'every word comes from the ontology'
        guarantee.
        """
        out = set()
        for anc in self.ancestors(term):
            for rel, obj in self.rels.get(anc, ()):
                lab = self.names.get(obj)
                if lab:
                    out.add(Fact(rel, obj, lab))
        return frozenset(out)

    def contrastive(self, target: str, other: str) -> List[Fact]:
        """D(target, other) = facts of `target` not held by `other`, contradictory ones first."""
        f_other = self.facts(other)
        other_by_obj = defaultdict(set)
        for f in f_other:
            other_by_obj[f.obj].add(f.relation)
        shared = {(f.relation, f.obj) for f in f_other}

        def contradicts(f: Fact) -> bool:
            return bool(_CONTRADICTS.get(f.relation, set()) & other_by_obj.get(f.obj, set()))

        d = [f for f in self.facts(target) if (f.relation, f.obj) not in shared]
        return sorted(d, key=lambda f: (not contradicts(f), RELATION_PRIORITY[f.relation], f.obj))

    def shared_facts(self, a: str, b: str) -> List[Fact]:
        """Facts true of both -- the neutral description, matched to the conflicting one in length,
        register and vocabulary while carrying nothing that discriminates a from b.
        """
        fb = {(f.relation, f.obj) for f in self.facts(b)}
        return sorted((f for f in self.facts(a) if (f.relation, f.obj) in fb),
                      key=lambda f: (RELATION_PRIORITY[f.relation], f.obj))

    def n_contradictory(self, target: str, other: str) -> int:
        f_other = defaultdict(set)
        for f in self.facts(other):
            f_other[f.obj].add(f.relation)
        return sum(1 for f in self.facts(target)
                   if _CONTRADICTS.get(f.relation, set()) & f_other.get(f.obj, set()))

    # ---- text ----
    def strip_identity(self, text: str, term: str, extra: Iterable[str] = ()) -> str:
        """Remove the term's name and every synonym so the description cannot be decoded from a label
        rather than from its content. Longest-first, so 'naive B cell' is removed before 'B cell'
        could match inside it and leave a fragment behind.
        """
        names = set(self.synonyms.get(term, [])) | {self.names.get(term, "")} | set(extra)
        out = text
        for n in sorted((n for n in names if n and len(n) > 2), key=len, reverse=True):
            out = re.sub(re.escape(n), "this cell type", out, flags=re.IGNORECASE)
        return out

    def _identity_strings(self, term: str, extra: Iterable[str] = ()) -> List[str]:
        names = set(self.synonyms.get(term, [])) | {self.names.get(term, "")} | set(extra)
        return sorted((n for n in names if n and len(n) > 2), key=len, reverse=True)

    def leaks_identity(self, fact: Fact, target: str, extra: Iterable[str] = ()) -> bool:
        """Does this fact's own label contain the target's name or a synonym?"""
        lab = fact.label.lower()
        return any(n.lower() in lab for n in self._identity_strings(target, extra))

    def describe(self, facts: List[Fact], target: str, max_facts: int = 6,
                 extra_strip: Iterable[str] = ()) -> str:
        """Render up to `max_facts` facts as one sentence."""
        picked = [f for f in facts if not self.leaks_identity(f, target, extra_strip)][:max_facts]
        if not picked:
            return ""
        clauses = [f.render() for f in picked]
        body = clauses[0] if len(clauses) == 1 else \
            ", ".join(clauses[:-1]) + (", and " if len(clauses) > 2 else " and ") + clauses[-1]
        # belt and braces: the sentence frame itself should still never contain the label
        return self.strip_identity(f"This cell {body}.", target, extra_strip)

#!/usr/bin/env python
"""Extracts Cell Ontology (or other OBO) terms from generated text by exact string matching.

    from input_use.metrics.ontology_ner import OntologyNER
    ner = OntologyNER("input_use/data/cl-full.obo")
    ner.extract("This population consists of naive B cells and CD8-positive T cells in the spleen.")

Only term names and EXACT synonyms are indexed. Strings that map to more than one term are dropped,
and a single-word phrase must equal a term's primary name, so a one-word synonym such as "atlas"
does not match. Matching is longest first on word boundaries, so a nested term is not counted twice.
Each mention records its matched span and character offsets, and its information content (negative
log of the fraction of ontology terms it subsumes) allows general terms such as "cell" to be filtered
with `min_ic`. input_use.core.analyze_rationale_removal uses it to find cell type names in C2S-Scale
rationales.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set

# Only these prefixes are treated as reportable entities. CL = cell type, UBERON = anatomy/tissue.
KINDS = {"CL": "cell_type", "UBERON": "tissue"}
_SYN_EXACT = re.compile(r'^synonym: "(.*?)" EXACT')
_TOKEN = re.compile(r"[A-Za-z0-9+\-]+")


@dataclass(frozen=True)
class Mention:
    span: str          # the literal text matched, as it appeared
    term: str          # ontology id
    name: str          # the ontology's canonical label for that id
    kind: str          # cell_type | tissue
    ic: float          # information content; higher = more specific a claim
    start: int         # character offsets into the generation, for audit
    end: int


class OntologyNER:
    def __init__(self, obo_path: str, min_ic: float = 2.0):
        self.min_ic = min_ic
        self.index: Dict[str, str] = {}        # normalised phrase -> term id
        self.names: Dict[str, str] = {}
        self.name_phrases: Set[str] = set()    # phrases that are a term's primary label
        self._parents: Dict[str, Set[str]] = defaultdict(set)
        self._load(obo_path)
        self._n_terms = {k: max(1, sum(1 for t in self.names if t.startswith(k))) for k in KINDS}
        self._ic_cache: Dict[str, float] = {}

    # ---------------- ontology ----------------
    def _load(self, path: str) -> None:
        collisions: Set[str] = set()
        cur: Optional[str] = None
        obsolete = False
        phrases: Dict[str, str] = {}
        for line in open(path):
            line = line.rstrip("\n")
            if line == "[Term]":
                cur, obsolete = None, False
            elif line.startswith("id: "):
                cur = line[4:].strip()
                if cur.split(":")[0] not in KINDS:
                    cur = None
            elif not cur:
                continue
            elif line.startswith("is_obsolete: true"):
                obsolete = True
                self.names.pop(cur, None)
                cur = None
            elif obsolete:
                continue
            elif line.startswith("name: "):
                nm = line[6:].strip()
                self.names[cur] = nm
                self._add(phrases, collisions, nm, cur)
                self.name_phrases.add(self.normalise(nm))
            elif line.startswith("synonym: "):
                m = _SYN_EXACT.match(line)
                if m:
                    self._add(phrases, collisions, m.group(1), cur)
            elif line.startswith("is_a: "):
                self._parents[cur].add(line[6:].split()[0].strip())
        # ambiguous phrases are dropped entirely rather than arbitrarily resolved
        self.index = {p: t for p, t in phrases.items() if p not in collisions}

    def _add(self, phrases, collisions, phrase: str, term: str) -> None:
        p = self.normalise(phrase)
        if not p or len(p) < 3:
            return
        if p in phrases and phrases[p] != term:
            collisions.add(p)
        else:
            phrases[p] = term

    @staticmethod
    def normalise(text: str) -> str:
        """Lowercase, collapse whitespace and hyphens/underscores to single spaces, drop punctuation
        other than '+' (CD4+ style markers). Applied identically to the index and to the text, so the
        two can only match on a canonical form."""
        t = text.strip().lower()
        t = re.sub(r"[\s_/\-]+", " ", t)
        t = re.sub(r"[^\w\s+]", "", t)
        return re.sub(r"\s+", " ", t).strip()

    def ancestors(self, term: str) -> Set[str]:
        seen, stack = {term}, [term]
        while stack:
            for p in self._parents.get(stack.pop(), ()):
                if p not in seen:
                    seen.add(p)
                    stack.append(p)
        return seen

    def ic(self, term: str) -> float:
        """-log(fraction of same-prefix terms subsumed by `term`). Computed once per term."""
        if term not in self._ic_cache:
            pref = term.split(":")[0]
            n = sum(1 for t in self.names if t.startswith(pref) and term in self.ancestors(t))
            self._ic_cache[term] = -math.log(max(n, 1) / self._n_terms[pref])
        return self._ic_cache[term]

    # ---------------- extraction ----------------
    def extract(self, text: str, min_ic: Optional[float] = None) -> List[Mention]:
        """Longest-match-first scan over a word-boundary token grid.

        Plural handling is one explicit, deterministic rule -- if an n-gram fails to match, retry with
        a trailing 's' stripped from its final token ("naive B cells" -> "naive b cell"). Stated here
        because it is the only morphological liberty taken; there is no stemmer and no lemmatiser,
        both of which would introduce behaviour that is hard to audit.
        """
        thr = self.min_ic if min_ic is None else min_ic
        toks = [(m.group(0), m.start(), m.end()) for m in _TOKEN.finditer(text)]
        norm = [self.normalise(t) for t, _, _ in toks]
        n = len(toks)
        consumed = [False] * n
        out: List[Mention] = []
        MAXLEN = 9                                   # longest CL label is well under 9 tokens
        for size in range(min(MAXLEN, n), 0, -1):    # longest first
            for i in range(0, n - size + 1):
                if any(consumed[i:i + size]):
                    continue
                phrase = " ".join(norm[i:i + size]).strip()
                if not phrase:
                    continue
                term = self.index.get(phrase)
                singular = None
                if term is None and phrase.endswith("s"):
                    singular = phrase[:-1]
                    term = self.index.get(singular)
                if term is None:
                    continue
                # single word that is only an alias, never a primary label -> word-sense collision
                if size == 1 and not ({phrase, singular} & self.name_phrases):
                    continue
                v = self.ic(term)
                if v < thr:
                    continue                          # too general to count as a specific claim
                for j in range(i, i + size):
                    consumed[j] = True
                out.append(Mention(span=text[toks[i][1]:toks[i + size - 1][2]], term=term,
                                   name=self.names.get(term, "?"),
                                   kind=KINDS[term.split(":")[0]], ic=round(v, 4),
                                   start=toks[i][1], end=toks[i + size - 1][2]))
        return sorted(out, key=lambda m: m.start)

    @staticmethod
    def to_dicts(ms: List[Mention]) -> List[dict]:
        return [asdict(m) for m in ms]

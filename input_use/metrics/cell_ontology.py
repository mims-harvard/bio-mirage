"""Cell Ontology lookups for cell type matching: ids of the 35 cross-tissue immune atlas labels,
resolution of free-text names and exact synonyms, `is_a` ancestor matching and hop distance.

Reads the ontology file set by `input_use.core.config.CL_OBO`. Used by input_use.metrics.singlecell.
"""
from __future__ import annotations

import re
from collections import deque
from functools import lru_cache

from input_use.core import config as cfg

# cell_type (lowercased) -> CL id, hardcoded from cross_tissue_immune.h5ad's
# cell_type/cell_type_ontology_term_id columns (confirmed 1:1, 35 distinct pairs; the dataset's
# literal "unknown" row carries no CL id and is intentionally absent here).
IMMUNE_ATLAS_CL_IDS = {
    "mast cell": "CL:0000097",
    "macrophage": "CL:0000235",
    "cd4-positive helper t cell": "CL:0000492",
    "lymphocyte": "CL:0000542",
    "megakaryocyte": "CL:0000556",
    "alveolar macrophage": "CL:0000583",
    "erythroid lineage cell": "CL:0000764",
    "plasmacytoid dendritic cell": "CL:0000784",
    "plasma cell": "CL:0000786",
    "memory b cell": "CL:0000787",
    "naive b cell": "CL:0000788",
    "alpha-beta t cell": "CL:0000789",
    "gamma-delta t cell": "CL:0000798",
    "regulatory t cell": "CL:0000815",
    "precursor b cell": "CL:0000817",
    "pro-b cell": "CL:0000826",
    "germinal center b cell": "CL:0000844",
    "classical monocyte": "CL:0000860",
    "non-classical monocyte": "CL:0000875",
    "naive thymus-derived cd4-positive, alpha-beta t cell": "CL:0000895",
    "naive thymus-derived cd8-positive, alpha-beta t cell": "CL:0000900",
    "effector memory cd4-positive, alpha-beta t cell": "CL:0000905",
    "cd8-positive, alpha-beta memory t cell": "CL:0000909",
    "cd16-negative, cd56-bright natural killer cell, human": "CL:0000938",
    "cd16-positive, cd56-dim natural killer cell, human": "CL:0000939",
    "mucosal-associated invariant t cell": "CL:0000940",
    "plasmablast": "CL:0000980",
    "conventional dendritic cell": "CL:0000990",
    "dendritic cell, human": "CL:0001056",
    "effector memory cd8-positive, alpha-beta t cell, terminally differentiated": "CL:0001062",
    "group 3 innate lymphoid cell": "CL:0001071",
    "cd8-positive, alpha-beta memory t cell, cd45ro-positive": "CL:0001203",
    "t follicular helper cell": "CL:0002038",
    "progenitor cell": "CL:0011026",
}


def _parse_obo(path) -> dict:
    """[Term] stanzas -> {CL id: {parent CL ids via is_a}}. Ignores relationship:/part_of and every
    other stanza field -- only the is_a taxonomic hierarchy is needed for lineage distance.
    """
    parents: dict = {}
    term_id = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line == "[Term]":
                term_id = None
            elif line.startswith("id: "):
                term_id = line[4:].strip()
                parents.setdefault(term_id, set())
            elif line.startswith("is_a: ") and term_id:
                parent = line[6:].split()[0].strip()   # drop trailing {...}/! comment
                parents[term_id].add(parent)
    return parents


@lru_cache(maxsize=1)
def _parents() -> dict:
    return _parse_obo(str(cfg.CL_OBO))


_SYNONYM_RE = re.compile(r'^synonym: "(.*)" (EXACT|BROAD|NARROW|RELATED)')


def _parse_name_index(path) -> dict:
    """[Term] stanzas -> {normalized name/exact-synonym string: CL id}, for free-text resolution
    (coarse_match's free-text fallback in metrics/singlecell.py, for answers given as open-ended
    text rather than picked from the fixed 35-label vocabulary).
    """
    index: dict = {}
    ambiguous: set = set()
    term_id, obsolete = None, False
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "[Term]":
                term_id, obsolete = None, False
            elif line.startswith("id: "):
                term_id = line[4:].strip()
            elif line.startswith("is_obsolete: true"):
                obsolete = True
            elif term_id and not obsolete:
                key = None
                if line.startswith("name: "):
                    key = line[6:].strip().lower()
                else:
                    m = _SYNONYM_RE.match(line)
                    if m and m.group(2) == "EXACT":
                        key = m.group(1).strip().lower()
                if key is not None:
                    if key in index and index[key] != term_id:
                        ambiguous.add(key)
                    else:
                        index[key] = term_id
    for key in ambiguous:
        index.pop(key, None)
    return index


@lru_cache(maxsize=1)
def _name_index() -> dict:
    return _parse_name_index(str(cfg.CL_OBO))


_ARTICLES_RE = re.compile(r"^(a|an|the)\s+")
_TRAILING_PUNCT_RE = re.compile(r"[.\s]+$")
_PARENTHETICAL_RE = re.compile(r"\s*\([^)]*\)")
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_free_text(text: str) -> str:
    t = (text or "").strip().lower()
    t = _PARENTHETICAL_RE.sub("", t)   # e.g. "regulatory T cell (Treg)" -> "regulatory T cell"
    t = _WHITESPACE_RE.sub(" ", t)
    t = _TRAILING_PUNCT_RE.sub("", t)
    t = _ARTICLES_RE.sub("", t)
    return t.strip()


def resolve_free_text_to_cl(text: str):
    """Cell Ontology id whose normalized name or exact synonym equals the normalized free-text
    answer, or None.
    """
    return _name_index().get(_normalize_free_text(text))


def cl_coarse_match(pred_cl: str, gt_cl: str) -> bool:
    """True iff pred_cl == gt_cl, or pred_cl is a true ancestor of gt_cl along is_a edges (any number
    of hops up, unbounded) -- i.e. the prediction names a broader category that the ground truth
    falls under.
    """
    if pred_cl == gt_cl:
        return True
    return pred_cl in _ancestor_distances(gt_cl, _parents())


def _ancestor_distances(term_id: str, parents: dict) -> dict:
    """bfs up the is_a graph from term_id; returns {ancestor CL id: hop distance}, including term_id
    itself at distance 0.
    """
    dist = {term_id: 0}
    queue = deque([term_id])
    while queue:
        node = queue.popleft()
        for parent in parents.get(node, ()):
            if parent not in dist:
                dist[parent] = dist[node] + 1
                queue.append(parent)
    return dist


def cl_distance(cl_a: str, cl_b: str):
    """Hop-count between two CL ids via their nearest common ancestor along is_a edges. None if they
    share no common ancestor at all (shouldn't happen -- CL is one connected dag under CL:0000000).
    """
    if cl_a == cl_b:
        return 0
    parents = _parents()
    dist_a = _ancestor_distances(cl_a, parents)
    dist_b = _ancestor_distances(cl_b, parents)
    shared = set(dist_a) & set(dist_b)
    if not shared:
        return None
    return min(dist_a[anc] + dist_b[anc] for anc in shared)


def cl_id_for(cell_type: str):
    """Look up a (possibly free-text) cell-type string against the 35 known immune-atlas labels. None
    if it's not a recognized label (e.g. a different dataset, or a stale free-generation phrasing
    that doesn't exactly match one of the 35) -- callers should fall back to the string heuristic.
    """
    return IMMUNE_ATLAS_CL_IDS.get((cell_type or "").strip().lower())

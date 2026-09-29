"""Selects functional residues and control residues in a protein from its UniProt features.

The salient set is the residues annotated as active, binding, metal binding, DNA binding or other
functional sites. Each salient residue is paired with a control residue of the same amino acid that
carries no functional, structural or region annotation and lies away from the termini.
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from input_use.modalities.protein.sources import uniprot as up


@dataclass
class ResidueSet:
    positions: List[int]                 # 1-based
    residues: List[Dict]                 # full annotation dicts (pos, aa, feature_type, evidence, ...)
    role: str                            # "salient" | "control"


@dataclass
class SalientControl:
    accession: str
    sequence: str
    uniprot_release: Optional[str]
    salient: ResidueSet
    control: ResidueSet
    unmatched_salient: List[Dict] = field(default_factory=list)  # S+ residues with no AA-matched control
    params: Dict = field(default_factory=dict)

    def matched_salient_positions(self) -> List[int]:
        """Salient positions that received an AA-matched control - the set used for the fair,
        equal-count S+ vs S- comparison. (Unmatched salient residues, e.g.
        """
        return [c["matched_to"] for c in self.control.residues]

    def summary(self) -> Dict:
        exp = sum(1 for r in self.salient.residues if r["experimental"])
        return {
            "accession": self.accession,
            "length": len(self.sequence),
            "n_salient": len(self.salient.positions),
            "n_salient_experimental": exp,
            "n_control": len(self.control.positions),
            "n_unmatched": len(self.unmatched_salient),
            "salient_feature_types": sorted({r["feature_type"] for r in self.salient.residues}),
            "uniprot_release": self.uniprot_release,
        }


# Positions that a control residue must not overlap (anything functionally/structurally annotated).
DEFAULT_EXCLUDE = (
    up.FUNCTIONAL_SITE_TYPES + up.OTHER_RESIDUE_TYPES + up.REGION_TYPES
    + ("Helix", "Beta strand", "Turn")  # known secondary structure (where available)
)


def build_salient_and_controls(
    entry: Dict,
    salient_types=up.FUNCTIONAL_SITE_TYPES,
    require_experimental: bool = False,
    exclude_types=DEFAULT_EXCLUDE,
    n_controls_per_salient: int = 1,
    avoid_termini: int = 5,
    conservation: Optional[Dict[int, float]] = None,
    conservation_tol: float = 0.15,
    seed: int = 0,
) -> SalientControl:
    """Build S+ (functional residues) and AA-matched S- (non-annotated controls) for one entry."""
    rng = random.Random(seed)
    seq = up.get_sequence(entry)
    n = len(seq)
    acc = entry["primaryAccession"]

    # --- S+ : functional residues (deduped by position; keep the highest-evidence annotation) ---
    fr = up.functional_residues(entry, types=salient_types, require_experimental=require_experimental)
    by_pos: Dict[int, Dict] = {}
    for r in fr:
        cur = by_pos.get(r["position"])
        if cur is None or (r["experimental"] and not cur["experimental"]):
            by_pos[r["position"]] = r
    salient_residues = [by_pos[p] for p in sorted(by_pos)]
    salient_positions = [r["position"] for r in salient_residues]

    # --- candidate control positions: not annotated by anything, not near termini, not salient ---
    excluded = up.annotated_positions(entry, exclude_types) | set(salient_positions)
    candidates_by_aa: Dict[str, List[int]] = defaultdict(list)
    for pos in range(avoid_termini + 1, n - avoid_termini + 1):
        if pos in excluded:
            continue
        candidates_by_aa[seq[pos - 1]].append(pos)
    for aa in candidates_by_aa:
        rng.shuffle(candidates_by_aa[aa])

    # --- match each salient residue to control(s) of the same AA (+ optional conservation) ---
    control_residues: List[Dict] = []
    unmatched: List[Dict] = []
    used = set()
    for sr in salient_residues:
        aa = sr["aa"]
        want = n_controls_per_salient
        pool = candidates_by_aa.get(aa, [])
        picked = 0
        for q in list(pool):
            if q in used:
                continue
            if conservation is not None:
                cs = conservation.get(sr["position"])
                cq = conservation.get(q)
                if cs is not None and cq is not None and abs(cs - cq) > conservation_tol:
                    continue
            used.add(q)
            control_residues.append({
                "position": q, "aa": seq[q - 1], "feature_type": "control",
                "matched_to": sr["position"], "matched_feature": sr["feature_type"],
                "description": "", "ligand": "", "evidence": [], "experimental": False,
            })
            picked += 1
            if picked >= want:
                break
        if picked < want:
            unmatched.append(sr)

    return SalientControl(
        accession=acc,
        sequence=seq,
        uniprot_release=up.uniprot_release(entry),
        salient=ResidueSet(salient_positions, salient_residues, "salient"),
        control=ResidueSet([r["position"] for r in control_residues], control_residues, "control"),
        unmatched_salient=unmatched,
        params={
            "salient_types": list(salient_types), "require_experimental": require_experimental,
            "n_controls_per_salient": n_controls_per_salient, "avoid_termini": avoid_termini,
            "conservation_matched": conservation is not None, "seed": seed,
        },
    )

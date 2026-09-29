"""Locations of every model output the code reads or writes under the results directory.

The results directory is `INPUT_USE_RESULTS_DIR`, by default `results/` at the repository root. It
is laid out as <modality>/<model>/<experiment>:

    dna/bioreason/perturbations            biological input perturbations, released SFT and RL
    dna/bioreason/text_conditions          text conditions with intact and shuffled Evo2
    dna/bioreason/genome_dependent         query strata and the Evo2 probe on 165 queries
    dna/bioreason/evidence_conflicts       Evo2 of one query with the text of another
    dna/bioreason/linear_probes            Evo2 probes before and after the projection
    dna/bioreason/auxiliary_supervision    training exports and generations for RQ4
    dna/chatnt                             ChatNT on its splice site and promoter tasks
    protein/bioreason_pro/...              perturbations and evidence conflicts, SFT and RL
    protein/prot2text_v2                   Prot2Text-V2 on function description generation
    protein/esm3_cache                     ESM3 representations, one file per sequence
    single_cell/c2s_scale/...              DEG removal, evidence conflicts and rationales
    single_cell/cellwhisperer              CellWhisperer on the C2S-Scale cells

Every constant is a string path. The module imports only the standard library, so the ChatNT,
Prot2Text-V2 and CellWhisperer drivers load it by file path from their own environments.
"""
from __future__ import annotations

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.environ.get("INPUT_USE_RESULTS_DIR") or os.path.join(REPO_ROOT, "results")


def _r(*parts: str) -> str:
    return os.path.join(RESULTS, *parts)


# ---- DNA: BioReason -------------------------------------------------------------------------
BIOREASON = _r("dna", "bioreason")
# examples.jsonl, records_{rl,sft}.jsonl, metrics_{rl,sft}.json, per_query_accuracy_{rl,sft}.json,
# pairs.json (donor pairs and query strata), lookup.json, evo2_pooled.npz
DNA_PERTURBATIONS = os.path.join(BIOREASON, "perturbations")
# records_{rl,sft}.jsonl (the perturbation run and the text conditions scored together),
# records_shuffled_evo2_arms_{rl,sft}.jsonl, metrics_{rl,sft}.json, per_query_accuracy_{rl,sft}.json,
# validity_{rl,sft}.json
DNA_TEXT_CONDITIONS = os.path.join(BIOREASON, "text_conditions")
DNA_GENOME_DEPENDENT = os.path.join(BIOREASON, "genome_dependent")
DNA_GENOME_DEPENDENT_QUERIES = os.path.join(DNA_GENOME_DEPENDENT, "genome_dependent_queries.json")
DNA_MINORITY_LABEL_QUERIES = os.path.join(DNA_GENOME_DEPENDENT, "minority_label_queries.json")
DNA_GENOME_DEPENDENT_PROBE = os.path.join(DNA_GENOME_DEPENDENT, "evo2_probe.json")
DNA_EVIDENCE_CONFLICTS = os.path.join(BIOREASON, "evidence_conflicts")
DNA_LINEAR_PROBES = os.path.join(BIOREASON, "linear_probes")
# query id -> split of the pathway network split (train, id_test, ood_test)
DNA_NETWORK_SPLIT = os.path.join(BIOREASON, "network_split.json")

# Auxiliary sequence supervision (RQ4). Each export holds train.csv, val.csv, test.csv, meta.json.
AUXILIARY = os.path.join(BIOREASON, "auxiliary_supervision")
AUX_WINDOWS_257BP = os.path.join(AUXILIARY, "windows_257bp")
AUX_TARGET_257BP = os.path.join(AUXILIARY, "windows_257bp_ref_var_target")
AUX_TARGET_2048BP = os.path.join(AUXILIARY, "windows_2048bp_ref_var_target")
AUX_NO_TARGET_2048BP = os.path.join(AUXILIARY, "windows_2048bp")
# Generations on the 145 held-out queries: <condition>_seed<seed>.json
AUX_GENERATIONS = os.path.join(AUXILIARY, "generations")

# ---- DNA: ChatNT ----------------------------------------------------------------------------
CHATNT = _r("dna", "chatnt")

# ---- Protein: BioReason-Pro -----------------------------------------------------------------
BIOREASON_PRO = _r("protein", "bioreason_pro")
PROTEIN_PERTURBATIONS = {ck: os.path.join(BIOREASON_PRO, f"perturbations_{ck}") for ck in ("rl", "sft")}
PROTEIN_EVIDENCE_CONFLICTS = {ck: os.path.join(BIOREASON_PRO, f"evidence_conflicts_{ck}")
                              for ck in ("rl", "sft")}
PROTEIN_EVIDENCE_CONFLICTS_SHUFFLED_ESM3 = {
    ck: os.path.join(BIOREASON_PRO, f"evidence_conflicts_shuffled_esm3_{ck}") for ck in ("rl", "sft")}
PROTEIN_LINEAR_PROBES = os.path.join(BIOREASON_PRO, "linear_probes")
ESM3_CACHE = _r("protein", "esm3_cache")

# ---- Protein: Prot2Text-V2 ------------------------------------------------------------------
PROT2TEXT = _r("protein", "prot2text_v2")

# ---- Single cell: C2S-Scale -----------------------------------------------------------------
C2S_SCALE = _r("single_cell", "c2s_scale")
C2S_DEG_REMOVAL = os.path.join(C2S_SCALE, "deg_removal")          # <atlas>/c2s_scale_{2b,27b}/
C2S_DEG_REMOVAL_SUMMARY = os.path.join(C2S_SCALE, "deg_removal_summary.json")
C2S_RANDOM_EXPRESSED = os.path.join(C2S_SCALE, "random_expressed")
C2S_EVIDENCE_CONFLICTS = os.path.join(C2S_SCALE, "evidence_conflicts")
C2S_RATIONALES = os.path.join(C2S_SCALE, "rationales")            # <atlas>/records_rationale_*.jsonl
C2S_RATIONALE_REMOVAL = os.path.join(C2S_SCALE, "rationale_removal")
C2S_RATIONALE_ORDER = os.path.join(C2S_SCALE, "rationale_order")

# ---- Single cell: CellWhisperer -------------------------------------------------------------
CELLWHISPERER = _r("single_cell", "cellwhisperer")

# ---- Filters shared by the Prot2Text-V2 and CellWhisperer evidence conflict panels -----------
EVIDENCE_CONFLICT_FILTERS = _r("evidence_conflict_filters.csv")


def dna_records(run_dir: str, ck: str) -> str:
    """BioReason records of checkpoint `ck` ('rl' or 'sft') in `run_dir`."""
    return os.path.join(run_dir, f"records_{ck.lower()}.jsonl")


def dna_shuffled_evo2_arm_records(ck: str) -> str:
    """Records of the three text conditions with shuffled Evo2 alone, before they were scored
    together with the perturbation run into dna_records(DNA_TEXT_CONDITIONS, ck)."""
    return os.path.join(DNA_TEXT_CONDITIONS, f"records_shuffled_evo2_arms_{ck.lower()}.jsonl")


def dna_per_query(run_dir: str, ck: str) -> str:
    """Per-query accuracy with bootstrap intervals of checkpoint `ck` in `run_dir`."""
    return os.path.join(run_dir, f"per_query_accuracy_{ck.lower()}.json")


def deg_removal(atlas: str) -> str:
    """C2S-Scale question set and records of one atlas under DEG removal."""
    return os.path.join(C2S_DEG_REMOVAL, atlas)


def rationales(atlas: str) -> str:
    """C2S-Scale rationale prompts and generations of one atlas."""
    return os.path.join(C2S_RATIONALES, atlas)

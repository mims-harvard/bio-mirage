"""Data and results paths, each read from an environment variable with a default relative to the
repository.

`INPUT_USE_RESULTS_DIR` holds the released model outputs, `INPUT_USE_DATA_DIR` the ontologies and
reference annotations, `INPUT_USE_GO_OBO`, `INPUT_USE_CL_OBO` and `INPUT_USE_CL_FULL_OBO` the
ontology files, and `INPUT_USE_MODELS_ROOT` the checkouts of the evaluated models.
"""
from __future__ import annotations

import os
from pathlib import Path


def _path(env: str, default) -> Path:
    return Path(os.environ[env]) if os.environ.get(env) else Path(default)


HOME = Path(os.environ.get("INPUT_USE_HOME") or Path(__file__).resolve().parents[1])

# Reference ontologies. setup/fetch_ontologies.sh downloads both.
GO_OBO = _path("INPUT_USE_GO_OBO", HOME / "data" / "go-basic.obo")
CL_OBO = _path("INPUT_USE_CL_OBO", HOME / "data" / "cl-basic.obo")
CL_FULL_OBO = _path("INPUT_USE_CL_FULL_OBO", HOME / "data" / "cl-full.obo")

# Downloaded reference data: the GOA annotation file, the InterPro cache, the ribosomal gene list.
DATA_DIR = _path("INPUT_USE_DATA_DIR", HOME / "data")

# Question sets, model outputs and scores. One directory per run.
RESULTS_DIR = _path("INPUT_USE_RESULTS_DIR", HOME.parent / "results")

# Checkouts of the evaluated models, each with its own environment.
MODELS_ROOT = _path("INPUT_USE_MODELS_ROOT", HOME.parent.parent)

# BioReason-Pro and BioReason (DNA) checkouts, imported by their runners.
BRP_REPO = _path("INPUT_USE_BRP_REPO", MODELS_ROOT / "BioReason-Pro")
DNA_REPO = _path("INPUT_USE_DNA_REPO", MODELS_ROOT / "BioReason")

# Information accretion weights for the weighted F_max, computed from the GOA snapshot used for
# scoring by `python -m input_use.metrics.make_ia`.
GO_IA = _path("INPUT_USE_GO_IA", DATA_DIR / "IA_goa_human.txt")


def require(path, what: str = "path") -> Path:
    """Return `path` if it exists, else exit with a message naming the missing path."""
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"[input_use.config] {what} not found: {p}\n"
                         f"  Set the matching INPUT_USE_* environment variable.")
    return p

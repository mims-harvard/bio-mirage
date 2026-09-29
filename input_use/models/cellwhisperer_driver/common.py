"""Shared paths, constants and helpers for the CellWhisperer evaluation on the five C2S-Scale atlases.

Every CellWhisperer step imports this module, so it uses only the standard library and numpy: it
runs both in CellWhisperer's pixi environment (build_cells.py, embed_cells.py, analyze.py) and in
the LLaVA environment of the CellWhisperer repository (score_llm.py), which has no pandas.

Paths come from environment variables and input_use/core/paths.py. `INPUT_USE_RESULTS_DIR` holds the
C2S-Scale question sets `single_cell/c2s_scale/deg_removal/<atlas>/examples.jsonl`, and every output
is written under its `single_cell/cellwhisperer/` subdirectory. `INPUT_USE_DATA_DIR` holds the
atlases as `prepped/<atlas>.h5ad` with raw counts in `layers['counts']`. The CellWhisperer checkout,
with its released CLIP checkpoint, chat checkpoints and Geneformer resources, is
`INPUT_USE_CELLWHISPERER_REPO`, or `cellwhisperer` under `INPUT_USE_MODELS_ROOT`. The base Mistral
model id is read from checkpoints.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(os.environ.get("INPUT_USE_DATA_DIR") or REPO_ROOT / "input_use" / "data")
MODELS_ROOT = Path(os.environ.get("INPUT_USE_MODELS_ROOT") or REPO_ROOT.parent)
CW_REPO = Path(os.environ.get("INPUT_USE_CELLWHISPERER_REPO") or MODELS_ROOT / "cellwhisperer")

PREPPED = DATA_DIR / "prepped"                               # <atlas>.h5ad, the C2S-Scale inputs
CHAT_MODELS = {"default": CW_REPO / "results" / "chat_models" / "default",
               "celltype": CW_REPO / "results" / "chat_models" / "celltype"}
CLIP_CKPT = CW_REPO / "results" / "models" / "jointemb" / "cellwhisperer_clip_v1.ckpt"

sys.path.insert(0, str(REPO_ROOT / "input_use" / "core"))
from checkpoints import ckpt  # noqa: E402
import paths as RD  # noqa: E402

OUT = Path(RD.CELLWHISPERER)
C2S_RESULTS = Path(RD.C2S_DEG_REMOVAL)                       # <atlas>/examples.jsonl


def mistral_base() -> str:
    """Hugging Face id (or local directory) of the base Mistral instruct model of the chat model."""
    return os.environ.get("INPUT_USE_CELLWHISPERER_TEXT_BASE") or ckpt("cellwhisperer_text_base")


ATLASES = ["immune1", "immune2", "immune3", "pancreas", "lung"]
KS = [50, 100, 200, 500, 1000]
SHUFFLE_SEEDS = [0, 1, 2]
N_PAIRS_TARGET = 200
PAIR_SEED = 0
BOOT_N = 2000
BOOT_SEED = 0

# CellWhisperer's own evaluation conversation (config.yaml of the CellWhisperer repository, llava_eval)
QUESTION = "Which cell type is this cell?"
RESPONSE_PREFIX = "This cell is a "
PRE_PROMPT_USER = ("Respond to my request regarding a sample of cells characterized by its "
                   "top-expressed genes being {}")
PRE_PROMPT_ASSISTANT = "Sure. What's your request?"
CONV_MODE = "mistral_instruct"

# Answer string per atlas label. immune1/2/3 labels are Cell Ontology names and are used verbatim.
# pancreas and lung labels are study shorthand. Their CL id is the first id in
# input_use/core/analyze_rationale_removal.py::MANUAL_TYPE_TERMS. The string is the Tabula Sapiens
# `cell_ontology_class` spelling of that term when the chat model's training data contain it (the
# released chat model was fine-tuned on "pancreatic alpha cell", not the CL label "pancreatic A cell"),
# otherwise the CL name. Labels that share a CL term share one answer (pancreas stellate x2, lung
# basal x4).
LABEL_MAP = {
    "pancreas": {
        "acinar": "pancreatic acinar cell",          # CL:0002064 (Tabula Sapiens spelling)
        "alpha": "pancreatic alpha cell",            # CL:0000171 (Tabula Sapiens; CL label "pancreatic A cell")
        "beta": "pancreatic beta cell",              # CL:0000169 (Tabula Sapiens; CL label "type B pancreatic cell")
        "delta": "pancreatic delta cell",            # CL:0000173 (Tabula Sapiens; CL label "pancreatic D cell")
        "gamma": "Pancreatic PP cell",               # CL:0002275 (Tabula Sapiens, capital P as in the dataset)
        "epsilon": "pancreatic epsilon cell",        # CL:0005019 (not in Tabula Sapiens; CL name)
        "ductal": "pancreatic ductal cell",          # CL:0002079 (Tabula Sapiens spelling)
        "endothelial": "endothelial cell",           # CL:0000115 (Tabula Sapiens spelling)
        "macrophage": "macrophage",                  # CL:0000235 (Tabula Sapiens spelling)
        "mast": "mast cell",                         # CL:0000097 (Tabula Sapiens spelling)
        "schwann": "schwann cell",                   # CL:0002573 (Tabula Sapiens; CL label "Schwann cell")
        "activated_stellate": "pancreatic stellate cell",   # CL:0002410 (Tabula Sapiens spelling)
        "quiescent_stellate": "pancreatic stellate cell",   # CL:0002410
        "t_cell": "T cell",                          # CL:0000084 (7 cells, never sampled)
    },
    "lung": {
        "Basal": "basal cell",                       # CL:0000646 (Tabula Sapiens spelling)
        "Differentiating.Basal": "basal cell",       # CL:0000646
        "Transitioning.Basal": "basal cell",         # CL:0000646
        "Proliferating.Basal": "basal cell",         # CL:0000646 (first id; CL:4033082 also accepted)
        "Ciliated": "ciliated cell",                 # CL:0000064 (Tabula Sapiens spelling)
        "Secretory": "secretory cell",               # CL:0000151 (Tabula Sapiens spelling)
        "Suprabasal": "suprabasal cell",             # CL:7770004 (not in Tabula Sapiens; CL name)
    },
}


def canonical_label(atlas: str, label: str) -> str:
    m = LABEL_MAP.get(atlas)
    if m is None:
        return label
    if label not in m:
        raise KeyError(f"{atlas}: no canonical name for label {label!r}")
    return m[label]


def answer_text(canonical: str) -> str:
    return RESPONSE_PREFIX + canonical


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_array(a) -> str:
    import numpy as np
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float32))
    return hashlib.sha256(a.tobytes()).hexdigest()


def read_jsonl(path, cond=None):
    out = []
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if cond is None or cond(r):
                out.append(r)
    return out


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def dump_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)
    os.replace(tmp, path)


def _json_default(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def atlas_dir(atlas: str) -> Path:
    d = OUT / atlas
    d.mkdir(parents=True, exist_ok=True)
    return d

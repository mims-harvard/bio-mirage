"""Writes run_manifest.json for the CellWhisperer evaluation: the CellWhisperer repository commit,
sizes and SHA-256 hashes of the CLIP checkpoint, Geneformer files and chat checkpoints, the package
versions of the given Python environments, the prompt and preprocessing settings, and the manifests
that score_llm.py wrote. Optional, run after analyze.py in either environment.

Writes single_cell/cellwhisperer/run_manifest.json under $INPUT_USE_RESULTS_DIR.

    python input_use/models/cellwhisperer_driver/manifest.py --pythons <pixi python> <LLaVA python>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import (  # noqa: E402
    BOOT_N, BOOT_SEED, CHAT_MODELS, CLIP_CKPT, CONV_MODE, CW_REPO, KS, N_PAIRS_TARGET,
    OUT, PAIR_SEED, PRE_PROMPT_ASSISTANT, PRE_PROMPT_USER, PREPPED, QUESTION, RESPONSE_PREFIX, SHUFFLE_SEEDS,
    ATLASES, LABEL_MAP, dump_json, mistral_base)


def sh(cmd, cwd=None):
    try:
        return subprocess.check_output(cmd, shell=True, cwd=cwd, stderr=subprocess.STDOUT, timeout=600).decode().strip()
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"


def sha256_file(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def freeze(python: Path) -> str:
    """pip freeze, or for a venv without pip the dist-info names and versions in its site-packages."""
    out = sh(f"{python} -m pip freeze 2>/dev/null | head -400")
    if out.strip():
        return out
    sp = sorted(Path(python).parents[1].glob("lib/python*/site-packages/*.dist-info"))   # venv dir, not the base interpreter
    return "\n".join(p.name[:-len(".dist-info")].replace("-", "==", 1) for p in sp)


def dir_manifest(d: Path, patterns=(".json", ".safetensors", ".bin", ".model", ".ckpt")):
    out = {}
    for p in sorted(Path(d).iterdir()):
        if p.is_file() and p.suffix in patterns:
            out[p.name] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pythons", nargs="*", default=[sys.executable],
                    help="interpreters of the pixi and LLaVA environments")
    a = ap.parse_args()
    base = mistral_base()
    base_entry = ({"path": base, "files": dir_manifest(Path(base))} if Path(base).is_dir() else {"id": base})
    man = {
        "cellwhisperer_repo": {"path": str(CW_REPO), "commit": sh("git rev-parse HEAD", CW_REPO),
                               "status": sh("git status --short", CW_REPO),
                               "submodules": sh("git submodule status --recursive", CW_REPO)},
        "driver_code": {"files": {p.name: sha256_file(p) for p in sorted(HERE.glob("*.py"))}},
        "artifacts": {
            "cellwhisperer_clip_v1.ckpt": {"path": str(CLIP_CKPT), "bytes": CLIP_CKPT.stat().st_size, "sha256": sha256_file(CLIP_CKPT)},
            "geneformer-12L-30M": dir_manifest(CW_REPO / "resources" / "geneformer-12L-30M"),
            "ensembl_gene_symbol_map.csv": {"sha256": sha256_file(CW_REPO / "resources" / "ensembl_gene_symbol_map.csv")},
            **{f"chat_model_{k}": {"path": str(v), "files": dir_manifest(v)} for k, v in CHAT_MODELS.items()},
            "mistral_base": base_entry,
            "prepped_atlases": {a: {"path": str(PREPPED / f"{a}.h5ad"), "bytes": (PREPPED / f"{a}.h5ad").stat().st_size} for a in ATLASES},
        },
        "environments": {py: {"freeze": freeze(Path(py))} for py in a.pythons},
        "configuration": {
            "atlases": ATLASES, "ks": KS, "shuffle_seeds": SHUFFLE_SEEDS, "n_pairs_target": N_PAIRS_TARGET,
            "pair_seed": PAIR_SEED, "bootstrap": {"n": BOOT_N, "seed": BOOT_SEED},
            "question": QUESTION, "response_prefix": RESPONSE_PREFIX, "pre_prompt_user": PRE_PROMPT_USER,
            "pre_prompt_assistant": PRE_PROMPT_ASSISTANT, "conv_mode": CONV_MODE, "label_map": LABEL_MAP,
            "transcriptome_preprocessing": "GeneformerTranscriptomeProcessor (src/cellwhisperer/jointemb/geneformer_model.py): "
                                           "raw counts / n_counts * 1e4 / gene median, rank desc, truncate 2048 tokens; "
                                           "embedding = get_transcriptome_features()[1], L2-normalised (2048-d)",
            "chat_inference": "LlavaLlamaForCausalLM (modules/LLaVA), bf16, mlp2x_8t_gelu projector (8 tokens), "
                              "conv template mistral_instruct, labels via preprocess_llama_2, last assistant block scored; "
                              "config.tokenizer_model_max_length lifted to None so no prompt is truncated (spliced lengths recorded)",
            "text_gate_inference": "Mistral-7B-Instruct-v0.2 (base), same template without <image>",
        },
        "scoring_manifests": {},
    }
    for p in sorted(OUT.glob("*/*_manifest.json")):
        man["scoring_manifests"][f"{p.parent.name}/{p.name}"] = json.load(open(p))
    dump_json(OUT / "run_manifest.json", man)
    print(f"wrote {OUT / 'run_manifest.json'}")


if __name__ == "__main__":
    main()

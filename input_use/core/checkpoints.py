"""Hugging Face ids of every model and dataset the code loads, read from a JSON file.

The file maps a short key to a repository id. `checkpoints.json` at the repository root holds the ids
the paper cites. Point `INPUT_USE_CHECKPOINTS` at another file to load different ids.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

# No package imports: the ChatNT, Prot2Text-V2 and CellWhisperer drivers load this file by path from their own environments.
PATH = Path(os.environ.get("INPUT_USE_CHECKPOINTS") or Path(__file__).resolve().parents[2] / "checkpoints.json")


@lru_cache(maxsize=None)
def _table() -> dict:
    if not PATH.exists():
        raise FileNotFoundError(f"{PATH} not found")
    return json.loads(PATH.read_text())


def ckpt(key: str) -> str:
    """Repository id stored under `key`."""
    value = _table().get(key)
    if not value:
        raise KeyError(f"no id for '{key}' in {PATH}")
    return value


def name(key: str) -> str:
    """Last path component of the id under `key`, the name a local mirror uses."""
    return ckpt(key).rstrip("/").split("/")[-1]


def hub_dir(key: str) -> str:
    """Directory name the Hugging Face cache gives the id under `key`."""
    return "models--" + ckpt(key).replace("/", "--")

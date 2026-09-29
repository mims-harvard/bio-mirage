"""Schema and JSONL helpers for the records every model runner writes and every scorer reads.

`Example` is a model-independent perturbed input written by a question set builder. `Record` adds
the exact input the model received and its raw and parsed output. The released model outputs are
JSONL files of `Record` rows, one per query and condition.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List

SCHEMA_VERSION = "1.0"


@dataclass
class Example:
    """Model-agnostic perturbed input (output of `prepare`)."""
    example_id: str                       # biological entity (UniProt acc, cell id, ...)
    modality: str                         # "protein" | "singlecell" | "dna"
    condition: str                        # see core.conditions
    payload: Dict[str, Any] = field(default_factory=dict)       # perturbed modality content + channel flags
    intervention: Dict[str, Any] = field(default_factory=dict)  # operator, targets, importance source, evidence
    ground_truth: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION


@dataclass
class Record:
    """Full record after model inference (output of `run`)."""
    record_id: str
    experiment: str
    model: str
    example_id: str
    modality: str
    condition: str
    intervention: Dict[str, Any] = field(default_factory=dict)
    input: Dict[str, Any] = field(default_factory=dict)    # exact input the model received
    output: Dict[str, Any] = field(default_factory=dict)   # {"raw": ..., "parsed": {...}}
    ground_truth: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION


def record_from_example(ex: Example, experiment: str, model: str,
                        input: Dict[str, Any], output: Dict[str, Any]) -> Record:
    return Record(
        record_id=f"{experiment}::{model}::{ex.example_id}::{ex.condition}",
        experiment=experiment, model=model, example_id=ex.example_id, modality=ex.modality,
        condition=ex.condition, intervention=ex.intervention, input=input, output=output,
        ground_truth=ex.ground_truth, provenance=ex.provenance,
    )


# ----- JSONL I/O -----
def write_jsonl(items: Iterable[Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(asdict(it) if hasattr(it, "__dataclass_fields__") else it) + "\n")


def append_jsonl(item: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(asdict(item) if hasattr(item, "__dataclass_fields__") else item) + "\n")


def read_jsonl(path: str | Path) -> List[dict]:
    """Tolerates a truncated final line and nothing else."""
    lines = [l for l in open(path) if l.strip()]
    out = []
    for i, line in enumerate(lines):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
            print(f"[records] dropping truncated final line of {path} (interrupted write)")
    return out


def read_examples(path: str | Path) -> List[Example]:
    out = []
    for d in read_jsonl(path):
        d.pop("schema_version", None)
        out.append(Example(**d))
    return out


def provenance(**kw) -> Dict[str, Any]:
    kw.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S"))
    return kw


def shard_by_id(items: list, id_fn, shard_index: int, num_shards: int) -> list:
    """Keep only the items belonging to this shard, grouping by id_fn(item) (e.g. example_id) rather
    than by list position -- so every condition for a given cell stays in the same shard.
    """
    if num_shards <= 1:
        return items
    ids = sorted({id_fn(it) for it in items})
    keep = set(ids[shard_index::num_shards])
    return [it for it in items if id_fn(it) in keep]

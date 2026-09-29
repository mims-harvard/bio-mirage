#!/usr/bin/env python
"""Runs BioReason-Pro (SFT or RL) on an examples.jsonl question set and writes records.jsonl, inside
the BioReason-Pro environment.

Each example states which sequence ESM3 encodes and how the InterPro and GO-GPT text is formed: from
that sequence, from another sequence, fixed to the unmodified protein, empty, dropped, or with the
full-length InterPro name entry removed. The runner computes the text with the BioReason-Pro
pipeline, generates with greedy decoding, and stores the exact input with the parsed GO terms,
reasoning trace and functional summary. `--num_shards` splits the examples by protein.

    python -m input_use.models.bioreason_pro.runner --examples <run>/examples.jsonl --records_out <run>/records.jsonl --model_type rl
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from input_use.core import config as cfg
from input_use.core.records import read_examples, record_from_example, append_jsonl, shard_by_id
from input_use.metrics.protein import extract_go, extract_summary, extract_think

BRP = cfg.require(cfg.BRP_REPO, "BioReason-Pro repo (INPUT_USE_BRP_REPO)")
sys.path.insert(0, str(BRP)); sys.path.insert(0, str(BRP / "gogpt" / "src"))
import predict as P  # BioReason-Pro inference pipeline (validated)

NAME_FRAC = 0.8  # InterPro entries spanning >=80% of the protein are treated as the name/family
# entry


def strip_name_entry(interpro_text: str, seq_len: int) -> str:
    keep = []
    for line in (interpro_text or "").splitlines():
        m = re.search(r"\[(\d+)-(\d+)\]", line)
        if m and (int(m.group(2)) - int(m.group(1)) + 1) >= NAME_FRAC * seq_len:
            continue  # drop the full-length (protein-naming) entry
        keep.append(line)
    return "\n".join(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", required=True)
    ap.add_argument("--records_out", required=True)
    ap.add_argument("--experiment", default="protein_brp")
    ap.add_argument("--model_type", choices=["sft", "rl"], default="rl")
    ap.add_argument("--max_new_tokens", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=32,
                    help="vLLM generation batch. The pilot used 4, which underfills an 80 GB H100 "
                         "(vLLM reserves ~61 GB of KV cache); raise until throughput plateaus.")
    ap.add_argument("--shard_index", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1,
                    help="split by PROTEIN (never by condition) so every arm of a protein is scored "
                         "in one place and no shard is missing its own wt reference")
    ap.add_argument("--gpu_memory_utilization", type=float, default=None,
                    help="Fraction of the card vLLM may reserve. Upstream defaults to 0.9, which "
                         "leaves ~8 GB on an 80 GB H100 - enough for the RL checkpoint but NOT for "
                         "sft, which loads a second ~3 GB ESM3 (protein_vllm.py assigns "
                         "self.protein_model twice). At 0.9 the sft proteome run OOM-ed on 11.3% of "
                         "batches and the pairs run on 96.9%; those rows are written EMPTY, not "
                         "dropped, so the loss is silent until you count them. Use ~0.80 for sft.")
    a = ap.parse_args()
    model_name = f"bioreason_pro_{a.model_type}"
    outdir = Path(a.records_out).parent
    interm = outdir / "intermediate"; interm.mkdir(parents=True, exist_ok=True)  # Sharing either would race: run_bioreason_stage appends
    # incrementally, and concurrent appends from N jobs interleave
    # rows. `core.score` globs records.jsonl* so the shards are
    # read back as one run.
    suffix = f".{a.shard_index}" if a.num_shards > 1 else ""
    tsv = str(interm / f"brp_responses{suffix}.tsv")
    a.records_out = str(a.records_out) + suffix

    examples = read_examples(a.examples)
    if a.num_shards > 1:
        examples = shard_by_id(examples, lambda e: e.example_id, a.shard_index, a.num_shards)
        print(f"[brp-runner] shard {a.shard_index}/{a.num_shards}: {len(examples)} examples")
    accs = sorted({e.example_id for e in examples})
    print(f"[brp-runner] {len(examples)} examples / {len(accs)} proteins")

    # The symbolic channel is derived from the example's `symbolic_sequence` (default = the
    # embedding `sequence`).
    SCAN_FLAGS = {"on", "no_name"}

    def needs_scan(ex):
        return ex.payload["channels"]["interpro"] in SCAN_FLAGS

    def needs_symbolic(ex):
        ch = ex.payload["channels"]
        return needs_scan(ex) or ch["gogpt"] != "none"

    def sym_seq(ex):
        return ex.payload.get("symbolic_sequence", ex.payload["sequence"])

    def sym_id(seq):
        return "sym" + hashlib.sha1(seq.encode()).hexdigest()[:16]

    def _rows(predicate):
        """Unique (by sequence content) rows for whichever stage needs them."""
        seen, rows = set(), []
        for ex in examples:
            if not predicate(ex):
                continue
            ss = sym_seq(ex); sid = sym_id(ss)
            if sid not in seen:
                seen.add(sid)
                rows.append({"protein_id": sid, "organism": ex.payload["organism"],
                             "sequence": ss, "sequence_length": str(len(ss))})
        return rows

    # The two stages get their own row sets.
    scan_rows = _rows(needs_scan)
    gogpt_rows = _rows(lambda ex: ex.payload["channels"]["gogpt"] != "none")
    print(f"[brp-runner] symbolic channel: {len(scan_rows)} sequences need InterProScan, "
          f"{len(gogpt_rows)} need GO-GPT")
    # per-shard checkpoint paths.
    interpro_by_sid = (P.run_interpro_stage(scan_rows, str(interm / f"interpro{suffix}.json"), resume=True)
                       if scan_rows else {})
    gogpt_by_sid = (P.run_gogpt_stage(gogpt_rows, str(interm / f"gogpt{suffix}.json"), resume=True)
                    if gogpt_rows else {})

    def resolve_ip(ex):
        flag = ex.payload["channels"]["interpro"]
        if flag == "none" or flag == "empty":
            # "empty" is not the same as "none": the block is present in the prompt template but has
            # no entries, because a shuffled sequence genuinely matches no InterPro entry.
            return ""
        if flag == "wt":
            return ex.payload.get("interpro_wt", "")     # precomputed per accession, frozen on WT
        ss = sym_seq(ex); ip = interpro_by_sid.get(sym_id(ss), "")
        # name-strip uses the length of the symbolic sequence (the protein the text describes)
        return strip_name_entry(ip, len(ss)) if flag == "no_name" else ip

    def resolve_go(ex):
        if ex.payload["channels"]["gogpt"] == "none":
            return ""
        return gogpt_by_sid.get(sym_id(sym_seq(ex)), "")

    proteins, interpro_results, gogpt_results, by_rid = [], {}, {}, {}
    for ex in examples:
        rid = f"{ex.example_id}__{ex.condition}"
        ip, go = resolve_ip(ex), resolve_go(ex)
        proteins.append({"protein_id": rid, "organism": ex.payload["organism"],
                         "sequence": ex.payload["sequence"], "sequence_length": str(len(ex.payload["sequence"]))})
        interpro_results[rid] = ip; gogpt_results[rid] = go
        by_rid[rid] = (ex, ip, go)

    # run_bioreason_stage reads the memory fraction from the module-level GEN_DEFAULTS rather than
    # from `args`, so overriding it means mutating that dict before the call - the same route
    # dms_prompt.py uses for the chat template.
    if a.gpu_memory_utilization is not None:
        P.GEN_DEFAULTS["gpu_memory_utilization"] = a.gpu_memory_utilization
        print(f"[brp-runner] vLLM gpu_memory_utilization -> {a.gpu_memory_utilization}", flush=True)

    args = SimpleNamespace(output=tsv, model_type=a.model_type, resume=True,
                           batch_size=a.batch_size,
                           max_new_tokens=a.max_new_tokens, temperature=0.0, top_p=0.95,
                           repetition_penalty=1.0, go_embeddings_path=None)
    t0 = time.time()
    print(f"[brp-runner] running BioReason-Pro on {len(proteins)} (protein,condition) rows "
          f"at batch_size={a.batch_size} ...")
    P.run_bioreason_stage(proteins, interpro_results, gogpt_results, args)
    dt = time.time() - t0
    print(f"[brp-runner] generation stage: {dt / 60:.1f} min "
          f"({dt / max(len(proteins), 1):.2f} s/row) - compare across --batch_size to tune")

    # merge model output TSV -> records.jsonl (exact input + raw/parsed output)
    csv.field_size_limit(10 ** 9)   # generated_response can exceed the 128 KB default field limit
    resp = {r["protein_id"]: r.get("generated_response", "") for r in csv.DictReader(open(tsv), delimiter="\t")}
    Path(a.records_out).unlink(missing_ok=True)
    n, n_empty = 0, 0
    for rid, (ex, ip, go) in by_rid.items():
        raw = resp.get(rid, "")
        n_empty += 0 if raw else 1
        rec = record_from_example(
            ex, a.experiment, model_name,
            input={"organism": ex.payload["organism"], "sequence": ex.payload["sequence"],
                   "symbolic_sequence": sym_seq(ex), "interpro": ip, "gogpt": go,
                   "channels": ex.payload["channels"]},
            output={"raw": raw, "parsed": {"go_terms": sorted(extract_go(raw)),
                                            "summary": extract_summary(raw), "think": extract_think(raw)}},
        )
        append_jsonl(rec, a.records_out); n += 1
    print(f"[brp-runner] DONE: wrote {n} records -> {a.records_out}")
    if n_empty:
        # run_bioreason_stage skips a batch on cuda OOM, which otherwise shows up only as a quietly
        # missing row. Surface it here so a shard that lost work is not scored as if it were
        # complete.
        print(f"[brp-runner] WARNING: {n_empty}/{n} rows have an EMPTY response (OOM-skipped batch "
              f"or truncated generation). Re-run with --resume to fill them, or lower --batch_size.")


if __name__ == "__main__":
    main()

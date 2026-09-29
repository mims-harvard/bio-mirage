#!/usr/bin/env python
"""Scores a C2S-Scale multiple-choice cell type annotation run: records.jsonl to metrics_mcq.json and
summary_mcq.md.

Per condition it reports the share of answers that name an option, accuracy (exact and Cell Ontology
ancestor match) with a bootstrap interval, chance accuracy from the number of options, and answer
position diagnostics.

    python -m input_use.core.score_mcq --rundir <run>[/<model>]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from input_use.core.records import read_jsonl
from input_use.core.stats import bootstrap_ci
from input_use.metrics.mcq import chance_accuracy, position_diagnostics
from input_use.metrics.singlecell import coarse_match, fine_match


def _mean(v):
    return round(sum(v) / len(v), 4) if v else None


def _safe_coarse(pred: str, gt: str):
    """coarse_match, but a missing Cell Ontology file degrades to "not applicable" instead of taking
    the whole run's scoring down with it.
    """
    try:
        return coarse_match(pred, gt)
    except (FileNotFoundError, OSError):
        return None


def score_rundir(rundir: Path) -> dict:
    records = read_jsonl(rundir / "records.jsonl")
    if not records:
        raise SystemExit(f"no records in {rundir}/records.jsonl")

    by_cond = defaultdict(list)
    for r in records:
        opts = (r.get("input") or {}).get("options") or []
        parsed = (r.get("output") or {}).get("parsed") or {}
        by_cond[r["condition"]].append({
            "options": opts,
            "pred": parsed.get("cell_type"),
            "in_options": bool(parsed.get("in_options")),
            "match_kind": parsed.get("match_kind", "none"),
            "gt": (r.get("ground_truth") or {}).get("cell_type"),
            "raw": (r.get("output") or {}).get("raw", ""),
        })

    out = {}
    for cond, rows in sorted(by_cond.items()):
        correct = [1.0 if fine_match(x["pred"] or "", x["gt"] or "") else 0.0 for x in rows]
        compliant = [1.0 if x["in_options"] else 0.0 for x in rows]
        comp_rows = [x for x in rows if x["in_options"]]
        coarse = [c for c in (_safe_coarse(x["pred"] or "", x["gt"] or "") for x in rows)
                  if c is not None]
        # Strict regime: only exact/normalized resolutions count as an answer at all -- see the
        # module docstring on why the lenient containment rule needs its influence made visible.
        STRICT = ("exact", "normalized")
        strict = [1.0 if (x["match_kind"] in STRICT and fine_match(x["pred"] or "", x["gt"] or ""))
                  else 0.0 for x in rows]
        lo, hi = bootstrap_ci(correct) if correct else (None, None)
        out[cond] = {
            "n": len(rows),
            "compliance": _mean(compliant),
            "compliance_strict": _mean([1.0 if x["match_kind"] in STRICT else 0.0 for x in rows]),
            "accuracy": _mean(correct),
            "accuracy_strict": _mean(strict),
            "accuracy_ci95": [round(lo, 4), round(hi, 4)] if lo is not None else None,
            "accuracy_of_compliant": _mean(
                [1.0 if fine_match(x["pred"] or "", x["gt"] or "") else 0.0 for x in comp_rows]),
            # None (not 0.0) when no label in this dataset is in the CL id table -- see module
            # docstring
            "accuracy_coarse": _mean([1.0 if c else 0.0 for c in coarse]) if coarse else None,
            "chance": chance_accuracy(rows),
            "match_kinds": dict(Counter(x["match_kind"] for x in rows).most_common()),
            "diagnostics": position_diagnostics(rows),
            "non_compliant_examples": [x["raw"][:120] for x in rows if not x["in_options"]][:5],
        }

    if "wt" in out:
        for cond, m in out.items():
            if cond != "wt" and m["accuracy"] is not None and out["wt"]["accuracy"] is not None:
                m["delta_vs_wt"] = round(m["accuracy"] - out["wt"]["accuracy"], 4)

    meta = {}
    for p in (rundir / "meta.json", rundir.parent / "meta.json"):
        if p.exists():
            meta = json.loads(p.read_text())
            break
    return {"modality": "singlecell", "answer_mode": "multiple_choice",
            "model": records[0].get("model"), "experiment": records[0].get("experiment"),
            "n_records": len(records), "n_cells": len({r["example_id"] for r in records}),
            "dataset": meta.get("dataset"), "n_genes_K": meta.get("n_genes"),
            "n_choices": meta.get("n_choices"), "conditions": out}


def write_summary(rundir: Path, m: dict) -> None:
    L = [f"# Multiple-choice cell-type annotation - {m.get('model')}", "",
         f"- experiment: `{m.get('experiment')}`",
         f"- dataset: `{m.get('dataset')}`",
         f"- K (genes per cell sentence): **{m.get('n_genes_K')}**",
         f"- options per question: **{m.get('n_choices')}** (shuffled per cell)",
         f"- {m['n_cells']} cells, {m['n_records']} queries", "",
         "| condition | n | compliance | accuracy | 95% CI | chance | acc (strict) | "
         "acc (compliant) | acc (coarse/CL) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for cond, c in m["conditions"].items():
        ci = f"{c['accuracy_ci95'][0]:.3f}-{c['accuracy_ci95'][1]:.3f}" if c["accuracy_ci95"] else "-"
        L.append(f"| `{cond}` | {c['n']} | {c['compliance']:.3f} | **{c['accuracy']:.3f}** | {ci} | "
                 f"{c['chance']:.3f} | {c['accuracy_strict']:.3f} | "
                 f"{c['accuracy_of_compliant'] if c['accuracy_of_compliant'] is not None else '-'} | "
                 f"{c['accuracy_coarse'] if c['accuracy_coarse'] is not None else 'n/a'} |")
    L += ["", "## Option-order and degeneracy diagnostics", "",
          "A model answering from the transcriptome should show a flat chosen-position histogram "
          "(options are re-shuffled per cell) and spread its answers across the label vocabulary.", "",
          "| condition | first-opt | last-opt | distinct labels | label entropy | top choices |",
          "|---|---|---|---|---|---|"]
    for cond, c in m["conditions"].items():
        d = c["diagnostics"]
        top = ", ".join(f"{lbl} ({n})" for lbl, n in d["most_common_choices"][:3])
        L.append(f"| `{cond}` | {d['first_option_rate']:.3f} | {d['last_option_rate']:.3f} | "
                 f"{d['n_distinct_labels_chosen']}/{d['n_options']} | "
                 f"{d['chosen_label_entropy_norm']:.3f} | {top} |")
    d = m["conditions"].get("wt", {}).get("diagnostics", {})
    if d:
        L += ["", f"Shuffle sanity check: the ground truth sits in first position "
                  f"{d['gt_position_first_rate']:.3f} of the time "
                  f"(expected {d['gt_position_first_rate_expected']:.3f} = 1/{d['n_options']}).",
              "", f"Chosen-position histogram (deciles, `wt`): {d['chosen_position_hist']}"]
    nc = m["conditions"].get("wt", {}).get("non_compliant_examples") or []
    if nc:
        L += ["", "## Sample non-compliant generations (`wt`)", ""] + [f"- `{s}`" for s in nc]
    (rundir / "summary_mcq.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rundir", required=True)
    a = ap.parse_args()
    rundir = Path(a.rundir)
    m = score_rundir(rundir)
    (rundir / "metrics_mcq.json").write_text(json.dumps(m, indent=2))
    write_summary(rundir, m)
    print(json.dumps({c: {k: v for k, v in d.items()
                          if k in ("n", "accuracy", "compliance", "chance", "accuracy_coarse")}
                      for c, d in m["conditions"].items()}, indent=2))
    print(f"[score-mcq] -> {rundir}/metrics_mcq.json, {rundir}/summary_mcq.md")


if __name__ == "__main__":
    main()

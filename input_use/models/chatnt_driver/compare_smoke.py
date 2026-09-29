#!/usr/bin/env python
"""Checks that the PyTorch ChatNT used for the evaluation agrees with the official jax implementation.

Compares smoke_jax.json with smoke_torch.json on the notebook example and on the queries both ran:
generated text, likelihood label, first generated token and the difference between the Yes and No
log-likelihoods. Writes smoke_compare.json and exits 1 if any label disagrees.

    python input_use/models/chatnt_driver/compare_smoke.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chatnt_common import RESULTS  # noqa: E402


def main():
    J = json.load(open(os.path.join(RESULTS, "smoke_jax.json")))
    T = json.load(open(os.path.join(RESULTS, "smoke_torch.json")))
    out = {"jax_backend": J["backend"], "torch_gpu": T["gpu"], "rows": []}
    jn, tn = J["notebook_example"], T["notebook_example"]
    out["notebook_example"] = {
        "jax_generated": jn["generated"], "torch_generated": tn["raw_generation"],
        "same_text": jn["generated"].strip() == tn["raw_generation"].strip(),
        "jax_label": jn["parsed_label"], "torch_label": tn["parsed_label"],
        "jax_first_token": jn["first_token_argmax"], "torch_first_token": tn["first_token_argmax"],
        "jax_margin": jn["margin"], "torch_margin": tn["margin"],
        "abs_margin_diff": abs(jn["margin"] - tn["margin"]),
        "jax_matches_notebook_output": jn["matches_notebook_output"],
    }
    trows = {r["example_id"]: r for t in T["integrity"].values() for r in t["rows"]}
    disagree = 0
    for r in J["examples"]:
        t = trows.get(r["example_id"])
        if t is None:
            continue
        tm = t["L_yes"] - t["L_no"]
        row = {"example_id": r["example_id"], "gold": r["gold_A"], "jax_generated": r["generated"],
               "torch_generated": t["raw_generation"], "jax_label_lik": "Yes" if r["margin"] > 0 else "No",
               "torch_label_lik": t["pred_lik"], "jax_margin": r["margin"], "torch_margin": tm,
               "abs_margin_diff": abs(r["margin"] - tm), "jax_first_token": r["first_token_argmax"],
               "torch_first_token": t["first_token_argmax"]}
        disagree += int(row["jax_label_lik"] != row["torch_label_lik"])
        out["rows"].append(row)
    out["n_compared"] = len(out["rows"])
    out["n_label_disagreements"] = disagree
    out["max_abs_margin_diff"] = max([r["abs_margin_diff"] for r in out["rows"]] + [out["notebook_example"]["abs_margin_diff"]])
    out["notebook_label_agrees"] = out["notebook_example"]["jax_label"] == out["notebook_example"]["torch_label"]
    with open(os.path.join(RESULTS, "smoke_compare.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2))
    for r in out["rows"]:
        print(f"{r['example_id']:32s} gold {r['gold']:3s} jax {r['jax_margin']:+7.3f} torch {r['torch_margin']:+7.3f} "
              f"| {r['jax_generated']!r} | {r['torch_generated']!r}")
    sys.exit(0 if disagree == 0 and out["notebook_label_agrees"] else 1)


if __name__ == "__main__":
    main()

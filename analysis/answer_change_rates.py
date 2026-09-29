#!/usr/bin/env python
"""Counts, per query, how often the answer changes between a reference condition and a perturbed or
conflicting condition, for the BioReason, BioReason-Pro and C2S-Scale 27B conditions of Figures 2
and 3.

A label answer counts as changed when the mapped label differs. A GO term answer counts as changed when
the emitted GO term set differs, with the mean Jaccard of the two sets also reported. Each rate carries
a 95% Wilson interval. Rows cover the text conditions with intact and shuffled Evo2 (BioReason), the
ESM3 shuffles and evidence conflicts (BioReason-Pro), gene order shuffling and resampling from all
expressed genes (C2S-Scale), and the DNA and cell sentence evidence conflicts. The `wt_repeat` row
gives the rate at which re-decoding the identical BioReason prompt changes the label, quoted in the
RQ1 appendix section "Biological Input Perturbations". Where the scorer already stored a rate, the
script asserts that it reproduces it.

Reads INPUT_USE_RESULTS_DIR and writes outputs/analysis/answer_change_rates.{csv,json}.
"""
from __future__ import annotations

import csv
import glob
import hashlib
import json
import math
import os
OUT_DIR = os.path.join(os.environ.get("INPUT_USE_OUTPUT_DIR", "outputs"), "analysis")
os.makedirs(OUT_DIR, exist_ok=True)
REPO_DIR = os.environ.get("INPUT_USE_HOME", ".")
import sys
from collections import Counter, defaultdict

sys.path.insert(0, REPO_DIR)
from input_use.metrics.dna import label_of, normalize_label_text, canonical_gold  # noqa: E402
from input_use.core import paths as RD  # noqa: E402


ROWS = []


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - s) / d, (c + s) / d)


def read_jsonl(path):
    with open(path) as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def rec(panel, model, checkpoint, group, arm, reference, n, changed, in_figure, source, **extra):
    lo, hi = wilson(changed, n)
    row = {"panel": panel, "model": model, "checkpoint": checkpoint, "group": group, "arm": arm,
           "reference": reference, "n": n, "n_changed": changed,
           "pct_changed": round(100 * changed / n, 2) if n else None,
           "pct_same": round(100 * (n - changed) / n, 2) if n else None,
           "ci_low_pct": round(100 * lo, 2) if n else None, "ci_high_pct": round(100 * hi, 2) if n else None,
           "in_figure": in_figure, "source": source}
    row.update(extra)
    ROWS.append(row)
    tag = "*" if in_figure else " "
    print(f"[{panel}]{tag} {model} {checkpoint or ''} | {group} | {arm} vs {reference}: "
          f"n={n} changed={changed} ({row['pct_changed']}%) same={row['pct_same']}%"
          + (" " + " ".join(f"{k}={v}" for k, v in extra.items() if not isinstance(v, str)) if extra else ""))
    return row


def label_transition(ref_pred, ref_ok, arm_pred, arm_ok):
    """Classify one row of a label-valued task."""
    if ref_pred == arm_pred:
        return "same_correct" if ref_ok else "same_wrong"
    if ref_ok and not arm_ok:
        return "right_to_wrong"
    if arm_ok and not ref_ok:
        return "wrong_to_right"
    return "wrong_to_wrong"


def compare_labels(panel, model, ck, group, arm, ref, pairs, in_figure, source, **extra):
    """pairs: iterable of (ref_pred, ref_ok, arm_pred, arm_ok)."""
    c = Counter(label_transition(*p) for p in pairs)
    n = sum(c.values())
    changed = c["right_to_wrong"] + c["wrong_to_right"] + c["wrong_to_wrong"]
    return rec(panel, model, ck, group, arm, ref, n, changed, in_figure, source,
               same_correct=c["same_correct"], same_wrong=c["same_wrong"],
               right_to_wrong=c["right_to_wrong"], wrong_to_right=c["wrong_to_right"],
               wrong_to_wrong=c["wrong_to_wrong"], **extra)


# ------------------------------------------------------------------------------------------------
# a.
DNA_ARMS_VS_WT = ["wt_repeat", "scramble", "no_gene", "scramble_no_gene", "no_pathway",
                  "scramble_no_pathway", "no_textkey", "scramble_no_textkey", "no_dna"]
DNA_RUNGS = [("no_gene", "scramble_no_gene"),   # (wt, scramble) is already the scramble-vs-wt row above
             ("no_pathway", "scramble_no_pathway"), ("no_textkey", "scramble_no_textkey")]


def load_dna(path, conds):
    by = defaultdict(dict)
    golds = set()
    for r in read_jsonl(path):
        golds.add((r.get("ground_truth") or {}).get("answer", ""))
        if r["condition"] in conds:
            by[r["condition"]][r["example_id"]] = r
    labels = sorted(golds - {""})
    out = defaultdict(dict)
    for cond, d in by.items():
        for eid, r in d.items():
            pred = label_of(r["output"].get("raw", ""), labels)
            ok = pred is not None and normalize_label_text(pred) == canonical_gold(r["ground_truth"].get("answer", ""))
            out[cond][eid] = (pred, ok)
    return out, labels


def panel_a():
    for ck in ("RL", "SFT"):
        f = RD.dna_records(RD.DNA_TEXT_CONDITIONS, ck)
        conds = set(DNA_ARMS_VS_WT) | {"wt"}
        by, _ = load_dna(f, conds)
        assert all(len(by[c]) == 1449 for c in conds), {c: len(by[c]) for c in conds}
        stored = json.load(open(f"{RD.DNA_TEXT_CONDITIONS}/metrics_{ck.lower()}.json"))["experiment1_2_deltas"]
        for arm in DNA_ARMS_VS_WT:
            row = compare_labels("a", "BioReason", ck, "all 1,449 rows", arm, "wt",
                                 [(*by["wt"][e], *by[arm][e]) for e in by["wt"]],
                                 in_figure=(ck == "RL" and arm not in ("wt_repeat", "no_dna")), source=f)
            # the scorer already stores this rate; assert we reproduce it
            assert abs(row["pct_changed"] / 100 - stored[arm]["answer_changed_rate"]) < 1e-3, (arm, row["pct_changed"], stored[arm])
        for intact, shuffled in DNA_RUNGS:
            compare_labels("a", "BioReason", ck, "all 1,449 rows", shuffled, intact,
                           [(*by[intact][e], *by[shuffled][e]) for e in by[intact]],
                           in_figure=(ck == "RL"), source=f)


# ------------------------------------------------------------------------------------------------
# b.
PROT_CONDS = ["wt", "shuffle_esm3_only", "shuffle_esm3_only_s1", "shuffle_esm3_only_s2",
              "gogpt_only_wt_esm3", "gogpt_only", "interpro_only_wt_esm3", "interpro_only",
              "no_symbolic", "no_modality"]
PROT_RUNGS = [("GO-GPT & InterPro", "wt", "shuffle_esm3_only"),
              ("GO-GPT & InterPro", "wt", "shuffle_esm3_only_s1"),
              ("GO-GPT & InterPro", "wt", "shuffle_esm3_only_s2"),
              ("GO-GPT", "gogpt_only_wt_esm3", "gogpt_only"),
              ("InterPro", "interpro_only_wt_esm3", "interpro_only"),
              ("None", "no_symbolic", "no_modality")]
PROT_VS_WT = [("GO-GPT", "gogpt_only_wt_esm3"), ("InterPro", "interpro_only_wt_esm3"), ("None", "no_symbolic"),
              ("GO-GPT", "gogpt_only"), ("InterPro", "interpro_only"), ("None", "no_modality")]


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def compare_sets(panel, model, ck, group, arm, ref, pairs, in_figure, source, **extra):
    """pairs: iterable of ((set, raw_sha), (set, raw_sha))."""
    n = changed = raw_same = empty_ref = empty_arm = 0
    js = []
    for (sr, hr), (sa, ha) in pairs:
        n += 1
        changed += sr != sa
        raw_same += hr == ha
        empty_ref += not sr
        empty_arm += not sa
        js.append(jaccard(sr, sa))
    js.sort()
    return rec(panel, model, ck, group, arm, ref, n, changed, in_figure, source,
               mean_jaccard=round(sum(js) / n, 4) if n else None,
               median_jaccard=round(js[n // 2], 4) if n else None,
               pct_jaccard_ge_0p5=round(100 * sum(j >= 0.5 for j in js) / n, 2) if n else None,
               pct_raw_generation_identical=round(100 * raw_same / n, 2) if n else None,
               n_raw_generation_identical=raw_same,
               n_empty_go_set_reference=empty_ref, n_empty_go_set_arm=empty_arm, **extra)


def load_protein(dirpath, conds):
    by = defaultdict(dict)
    for f in sorted(glob.glob(f"{dirpath}/records.jsonl.*")):
        for r in read_jsonl(f):
            if r["condition"] in conds:
                raw = r["output"].get("raw", "") or ""
                by[r["condition"]][r["example_id"]] = (frozenset(r["output"]["parsed"]["go_terms"]),
                                                       hashlib.sha1(raw.encode()).hexdigest())
    return by


def panel_b():
    for ck in ("RL", "SFT"):
        src = RD.PROTEIN_PERTURBATIONS[ck.lower()]
        by = load_protein(src, set(PROT_CONDS))
        print(f"[b] {ck} rows per condition: " + " ".join(f"{c}={len(by[c])}" for c in PROT_CONDS))
        assert len(by["wt"]) == 14102, len(by["wt"])
        # the proteome scorer already stores P(GO set == GO set at wt) as `retention_rate` and
        # mean(1 - Jaccard) as `delta_vs_wt`; every arm-vs-wt row here must reproduce both
        stored = json.load(open(f"{src}/metrics.json"))
        ret, dj = stored["retention_rate"], stored["delta_vs_wt"]

        def check(arm, row):
            if arm in ret:
                assert row["n"] == ret[arm]["n"], (arm, row["n"], ret[arm]["n"])
                assert abs(row["pct_same"] / 100 - ret[arm]["rate"]) < 1e-3, (arm, row["pct_same"], ret[arm])
            v = dj.get(arm)
            if isinstance(v, dict):
                v = v.get("mean", v.get("delta", v.get("rate")))
            if isinstance(v, (int, float)):
                assert abs((1 - row["mean_jaccard"]) - v) < 1e-3, (arm, row["mean_jaccard"], dj[arm])
                row["stored_delta_vs_wt_1_minus_jaccard"] = v
        for group, arm in PROT_VS_WT:
            common = sorted(set(by["wt"]) & set(by[arm]))
            row = compare_sets("b", "BioReason-Pro", ck, group, arm, "wt", [(by["wt"][e], by[arm][e]) for e in common],
                               in_figure=False, source=src)
            check(arm, row)
        for group, intact, shuffled in PROT_RUNGS:
            common = sorted(set(by[intact]) & set(by[shuffled]))
            row = compare_sets("b", "BioReason-Pro", ck, group, shuffled, intact,
                               [(by[intact][e], by[shuffled][e]) for e in common],
                               in_figure=(ck == "RL"), source=src)
            if intact == "wt":
                check(shuffled, row)


# ------------------------------------------------------------------------------------------------
# c.
ATLASES = ["immune1", "immune2", "immune3", "pancreas", "lung"]


def load_c2s(path, conds):
    by = defaultdict(dict)
    for r in read_jsonl(path):
        if r["condition"] in conds:
            by[r["example_id"]][r["condition"]] = (r["output"]["parsed"]["cell_type"], r["ground_truth"]["cell_type"])
    return by


def panel_c():
    from input_use.core.analyze_deg_removal import ARMS, DOSES
    core = ["wt", "no_modality", "scramble_rank", "scramble_geneset"]
    arms = [f"{pre}_{d}" for name, pre in ARMS.items() for d in DOSES if not (name == "bottom" and d == "p100")]
    exp1 = json.load(open(RD.C2S_DEG_REMOVAL_SUMMARY))
    pooled = defaultdict(list)
    for ds in ATLASES:
        f_orig = f"{RD.deg_removal(ds)}/c2s_scale_27b/records.jsonl"
        f_new = f"{RD.C2S_RANDOM_EXPRESSED}/{ds}/c2s_scale_27b/records.jsonl"
        orig = load_c2s(f_orig, set(core + arms))
        new = load_c2s(f_new, {"random_expressed"})
        cells = sorted(c for c, v in orig.items() if all(k in v for k in core + arms))
        assert len(cells) == exp1[f"{ds}|27b"]["n_cells"], (ds, len(cells))
        assert all("random_expressed" in new[c] for c in cells), ds

        def pairs(arm, src):
            out = []
            for c in cells:
                wp, gt = orig[c]["wt"]
                ap, _ = src[c][arm]
                out.append((wp, wp == gt, ap, ap == gt))
            return out

        for arm, src, f in (("scramble_rank", orig, f_orig), ("random_expressed", new, f_new)):
            p = pairs(arm, src)
            pooled[arm].extend(p)
            row = compare_labels("c", "C2S-Scale 27B", "", ds, arm, "wt", p, in_figure=True, source=f)
            # check: the wt accuracy implied by our rows must equal deg_removal_summary.json
            wt_acc = (row["same_correct"] + row["right_to_wrong"]) / row["n"]
            assert abs(wt_acc - exp1[f"{ds}|27b"]["wt"]) < 5e-5, (ds, wt_acc, exp1[f"{ds}|27b"]["wt"])
    for arm, p in pooled.items():
        compare_labels("c", "C2S-Scale 27B", "", "pooled 5 atlases", arm, "wt", p, in_figure=False,
                       source="single_cell/c2s_scale/deg_removal + random_expressed")


# ------------------------------------------------------------------------------------------------
# d.
SWAP_ARMS = ["swap_both_donor", "swap_both_gene_donor", "swap_both_pathway_donor", "swap_variant_donor"]


def panel_d():
    pairs = json.load(open(f"{RD.DNA_PERTURBATIONS}/pairs.json"))
    donors = {eid: d["donor_answer"] for eid, d in pairs["donors"].items()}
    for ck in ("RL", "SFT"):
        f = RD.dna_records(RD.DNA_PERTURBATIONS, ck)
        by, labels = load_dna(f, {"wt", "swap_variant_donor"})
        arms = {"swap_variant_donor": (f, by["swap_variant_donor"])}
        f2 = RD.dna_records(RD.DNA_EVIDENCE_CONFLICTS, ck)
        if os.path.exists(f2):
            by2, labels2 = load_dna(f2, set(SWAP_ARMS))
            assert labels2 == labels
            for arm in SWAP_ARMS:
                if by2.get(arm):
                    arms[arm] = (f2, by2[arm])
        wt = by["wt"]
        for arm in SWAP_ARMS:
            if arm not in arms:
                continue
            src, sw = arms[arm]
            for rowset, eids in (("all rows", sorted(sw)),
                                 ("wt-correct rows (panel conditioning)", sorted(e for e in sw if wt[e][1]))):
                pr = [(*wt[e], *sw[e]) for e in eids]
                to_A = sum(1 for e in eids
                           if sw[e][0] != wt[e][0] and sw[e][0] is not None
                           and normalize_label_text(sw[e][0]) == canonical_gold(donors[e]))
                compare_labels("d", "BioReason", ck, rowset, arm, "wt", pr,
                               in_figure=(ck == "RL" and arm != "swap_variant_donor" and rowset.startswith("wt-correct")),
                               source=src, changed_to_A_label=to_A)


# ------------------------------------------------------------------------------------------------
# e.
def panel_e():
    from input_use.core import conditions as C
    from input_use.core.config import GO_OBO
    from input_use.metrics import go_dag
    src = f"{RD.PROTEIN_EVIDENCE_CONFLICTS['rl']}/records.jsonl"
    by = defaultdict(dict)
    for r in read_jsonl(src):
        pid = r["ground_truth"]["pair_id"]
        raw = r["output"].get("raw", "") or ""
        by[pid][r["condition"]] = (frozenset(r["output"]["parsed"]["go_terms"]), hashlib.sha1(raw.encode()).hexdigest(),
                                   r["ground_truth"])
    panel_cats = {"pseudoenzyme", "organelle_targeted", "dna_binding"}
    gate = None
    if os.path.exists(GO_OBO):
        dag = go_dag.load(GO_OBO)

        def satisfies(pred, spec):
            if not all(dag.any_under(pred, r) for r in spec.get("require", [])):
                return False
            return not any(dag.any_under(pred, r) for r in spec.get("forbid", []))

        gate = {}
        for pid, conds in by.items():
            gt = conds["aligned_a"][2]
            gate[pid] = satisfies(conds["aligned_a"][0], gt["spec_a"]) and satisfies(conds["aligned_b"][0], gt["spec_b"])
    else:
        print(f"[e] WARNING: GO OBO not found at {GO_OBO}; gate-passing row set skipped")

    rowsets = [("all pairs, 4 categories", lambda pid: True),
               ("3 panel categories", lambda pid: by[pid]["aligned_a"][2]["category"] in panel_cats)]
    if gate is not None:
        rowsets.append(("3 panel categories, gate-passing (panel rows)",
                        lambda pid: by[pid]["aligned_a"][2]["category"] in panel_cats and gate[pid]))
    for fam, fam_conds in C.PAIR_CONFLICT_FAMILIES.items():
        for rs_name, keep in rowsets:
            pairs, to_other = [], 0
            for pid, conds in by.items():
                if not keep(pid):
                    continue
                for cond in fam_conds:
                    own, other = C.PAIR_CONFLICT_LAYOUT[cond]
                    ref, arm, oth = conds[f"aligned_{own}"], conds[cond], conds[f"aligned_{other}"]
                    pairs.append(((ref[0], ref[1]), (arm[0], arm[1])))
                    to_other += arm[0] == oth[0]
            compare_sets("e", "BioReason-Pro", "RL", rs_name, fam, "aligned_<own> (all channels from the ESM3 protein)",
                         pairs, in_figure=rs_name.endswith("(panel rows)"), source=src,
                         pct_identical_to_text_protein_aligned=round(100 * to_other / len(pairs), 2) if pairs else None,
                         n_identical_to_text_protein_aligned=to_other,
                         arms=" + ".join(fam_conds))


# ------------------------------------------------------------------------------------------------
# f.
def panel_f():
    src = f"{RD.C2S_EVIDENCE_CONFLICTS}/c2s_scale_27b/records.jsonl"
    by = defaultdict(dict)
    B = {}
    for r in read_jsonl(src):
        by[r["example_id"]][r["condition"]] = (r["output"]["parsed"]["cell_type"], r["ground_truth"]["cell_type"])
        if r["condition"] == "tc_xA":
            B[r["example_id"]] = r["provenance"]["B"]
    arms = ["tc_xA_tB", "tc_xA_tB_2way", "tc_xA_tA", "tc_xA_tneutral"]
    cells = sorted(c for c, v in by.items() if all(k in v for k in ["tc_xA"] + arms))
    cond = [c for c in cells if by[c]["tc_xA"][0] == by[c]["tc_xA"][1]]
    for arm in arms:
        for rs_name, eids in (("all cells", cells), ("tc_xA-correct cells (panel conditioning)", cond)):
            pr, to_B = [], 0
            for c in eids:
                rp, gt = by[c]["tc_xA"]
                ap, _ = by[c][arm]
                pr.append((rp, rp == gt, ap, ap == gt))
                to_B += (ap != rp) and ap == B[c]
            compare_labels("f", "C2S-Scale 27B", "", rs_name, arm, "tc_xA (cell sentence only)", pr,
                           in_figure=(arm == "tc_xA_tB" and rs_name.startswith("tc_xA-correct")), source=src,
                           changed_to_B_label=to_B)


def main():
    panel_a()
    panel_c()
    panel_d()
    panel_f()
    panel_e()
    panel_b()
    keys = []
    for r in ROWS:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(f"{OUT_DIR}/answer_change_rates.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(ROWS)
    json.dump(ROWS, open(f"{OUT_DIR}/answer_change_rates.json", "w"), indent=1)
    print(f"wrote {len(ROWS)} rows -> {OUT_DIR}/answer_change_rates.{{csv,json}}")


if __name__ == "__main__":
    main()

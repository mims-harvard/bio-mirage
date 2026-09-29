"""GO term prediction scoring with the CAFA evaluator (cafaeval), using BioReason-Pro's evaluation
settings.

`cafa_fmax` returns F_max, and weighted F_max when an information accretion file is given, per
condition. `cafa_f1_bootstrap` recomputes CAFA F1 from per-protein counts for a percentile bootstrap
over proteins, and `verify_against_cafaeval` checks that this recomputation matches cafaeval.
"""
from __future__ import annotations

import math
import os
import shutil
import tempfile

from cafaeval.evaluation import cafa_eval

# cafaeval reports namespaces by their full OBO name; map to the mf/BP/cc codes used in summaries.
ASPECT_CODE = {"biological_process": "BP", "molecular_function": "MF", "cellular_component": "CC"}

# cafa_eval params pinned to BioReason-Pro's evals/cafa_evals.py + run_cafa_eval.sh settings.
NORM = "cafa"
PROP = "max"
TH_STEP = 0.99  # score=1.0 predictions -> a single operating point (no real threshold sweep)


def _safe(name: str) -> str:
    """Filename-safe token (cafaeval derives the prediction's name from the file name)."""
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in name)


def _write_pred(path: str, rows) -> None:
    """rows: iterable of (target_id, term_set). Proteins with no predicted terms are omitted (a missing
    prediction is a recall miss in CAFA, exactly as in BioReason-Pro's pipeline).
    """
    with open(path, "w") as fh:
        for tid, terms in rows:
            for t in sorted(terms):
                fh.write(f"{tid}\t{t}\t1.0\n")


def _eval_dir(pred_dir: str, gt_file: str, obo_path: str, ia, n_cpu: int):
    """Run cafa_eval once over a directory of prediction files; return {filename: {aspect_code: (f,
    f_w_or_None)}} from the single best operating point.
    """
    _df, best = cafa_eval(obo_path, pred_dir, gt_file, ia, norm=NORM, prop=PROP,
                          th_step=TH_STEP, n_cpu=n_cpu)
    out = {}
    if not best or "f" not in best:
        return out
    fb = best["f"].reset_index()             # one row per (filename, ns) at the F-best threshold
    has_w = "f_w" in fb.columns              # present only when an IA file was supplied
    for _, row in fb.iterrows():
        fname = str(row["filename"])
        code = ASPECT_CODE.get(row["ns"], row["ns"])
        fw = float(row["f_w"]) if has_w else None
        out.setdefault(fname, {})[code] = (float(row["f"]), fw)
    return out


def _overall(by_aspect):
    """Mean F_max over the aspects present for this prediction file - identical to BioReason-Pro's
    `overall_mean_f1` / `overall_mean_weighted_f1` (mean over `df['ns'].unique()`).
    """
    f_vals = [f for f, _ in by_aspect.values()]
    w_vals = [w for _, w in by_aspect.values() if w is not None]
    f = sum(f_vals) / len(f_vals) if f_vals else 0.0
    fw = sum(w_vals) / len(w_vals) if w_vals else None
    return f, fw


def cafa_fmax(preds_by_cond, gt_by_protein, obo_path, ia_path=None, n_cpu: int = 0, workdir=None):
    """Compute CAFA F_max per condition with the official cafaeval engine."""
    ia = ia_path if (ia_path and os.path.exists(ia_path)) else None
    tmp = workdir or tempfile.mkdtemp(prefix="input_use_cafa_")
    try:
        result = {}
        for gi, (cover, conds) in enumerate(coverage_groups(preds_by_cond).items()):
            # One cafa_eval call per coverage group, each against ground truth restricted to the
            # proteins those conditions were asked about.
            sub_gt = {p: t for p, t in gt_by_protein.items() if p in cover}
            pred_dir = os.path.join(tmp, f"pred{gi}")
            os.makedirs(pred_dir, exist_ok=True)
            gt_file = os.path.join(tmp, f"gt{gi}.tsv")
            with open(gt_file, "w") as fh:
                for pid, terms in sub_gt.items():
                    for t in sorted(terms):
                        fh.write(f"{pid}\t{t}\n")
            for cond in conds:
                _write_pred(os.path.join(pred_dir, f"{_safe(cond)}.tsv"),
                            preds_by_cond[cond].items())

            per_file = _eval_dir(pred_dir, gt_file, obo_path, ia, n_cpu)
            for cond in conds:
                by_aspect = per_file.get(f"{_safe(cond)}.tsv", {})
                f, fw = _overall(by_aspect)
                result[cond] = {
                    "f": f, "f_w": fw,
                    "by_aspect": {a: v[0] for a, v in by_aspect.items()},
                    "by_aspect_w": ({a: v[1] for a, v in by_aspect.items()} if ia else None),
                    "n_aspects": len(by_aspect), "n_proteins": len(cover),
                }
        return result, bool(ia)
    finally:
        if workdir is None:
            shutil.rmtree(tmp, ignore_errors=True)


def coverage_groups(preds_by_cond):
    """{frozenset(protein ids): [conditions with exactly that coverage]}."""
    groups = {}
    for cond, by_prot in preds_by_cond.items():
        groups.setdefault(frozenset(by_prot), []).append(cond)
    return groups


# ---------------------------------------------------------------------------------------------
# Protein-level bootstrap CIs
# ---------------------------------------------------------------------------------------------
# `cafa_fmax` above returns a point estimate, matching BioReason-Pro's reporting.

def _cafa_f1_parts(pred_by_prot, gt_by_prot, dag, aspects=("MF", "BP", "CC"), ia=None):
    """Per-protein, per-aspect (tp, n_pred, n_gt) after ancestor propagation."""
    root_of = {v: k for k, v in _go_dag_roots().items()}
    parts = {a: {} for a in aspects}
    for prot, gt in gt_by_prot.items():
        gt_p = dag.propagate(gt)
        pred_p = dag.propagate(pred_by_prot.get(prot, ()))
        for a in aspects:
            g = {t for t in gt_p if dag.aspect_of.get(t) == a}
            if not g:
                continue                     # no ground truth in this aspect -> protein not scored here
            g = g | {root_of[a]}
            p = {t for t in pred_p if dag.aspect_of.get(t) == a}
            if p:
                p = p | {root_of[a]}
            if ia is None:
                parts[a][prot] = (len(p & g), len(p), len(g))
            else:
                # fsum: set iteration order is not stable across processes and float addition is not
                # associative, so a plain sum makes the weighted parts run-dependent in the last
                # bits. Exact summation removes that.
                w = lambda s: math.fsum(ia.get(t, 0.0) for t in s)
                parts[a][prot] = (w(p & g), w(p), w(g))
    return parts


def _go_dag_roots():
    from input_use.metrics.go_dag import ROOTS
    return ROOTS


def load_ia(path):
    """Read an Information-Accretion file into {term: weight}."""
    if not path or not os.path.exists(path):
        return {}
    ia = {}
    with open(path) as fh:
        for line in fh:
            bits = line.split()
            if len(bits) < 2:
                continue
            try:
                ia[bits[0]] = float(bits[1])
            except ValueError:
                continue
    return ia


def _f1_from_parts(parts, aspects, index_by_aspect=None):
    """CAFA-normalised F1, averaged over the aspects that have any scored protein."""
    fs = []
    for a in aspects:
        rows = parts[a]
        idx = index_by_aspect[a] if index_by_aspect else list(rows)
        if not idx:
            continue
        p_sum = p_n = 0.0
        r_sum = 0.0
        for prot in idx:
            tp, npred, ngt = rows[prot]
            if npred:
                p_sum += tp / npred
                p_n += 1
            # Under IA weighting `ngt` is a sum of weights, not a count, and it is 0 for a protein
            # whose only annotation in this aspect carries zero information (the root).
            if ngt:
                r_sum += tp / ngt
        prec = p_sum / p_n if p_n else 0.0
        rec = r_sum / len(idx)
        fs.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return sum(fs) / len(fs) if fs else 0.0


def cafa_f1_bootstrap(preds_by_cond, gt_by_protein, obo_path, n_boot: int = 2000,
                      alpha: float = 0.05, seed: int = 0,
                      aspects=("MF", "BP", "CC"), paired_to: str = "wt", ia=None):
    """CAFA F1 per condition with a percentile bootstrap over proteins."""
    import random

    from input_use.metrics import go_dag

    dag = go_dag.load(obo_path)
    # Restrict to one coverage group at a time: an arm run on a subset must be resampled - and
    # paired against wt - over that subset, not over every protein in the run.
    groups = coverage_groups(preds_by_cond)
    if len(groups) > 1:
        out = {}
        for cover, conds in groups.items():
            sub_preds = {c: preds_by_cond[c] for c in conds}
            ref = paired_to if paired_to in sub_preds else None
            if ref is None and paired_to in preds_by_cond:
                # give the group its own reference by restricting wt to this coverage
                sub_preds[paired_to] = {p: v for p, v in preds_by_cond[paired_to].items() if p in cover}
                ref = paired_to
            sub_gt = {p: t for p, t in gt_by_protein.items() if p in cover}
            res = cafa_f1_bootstrap(sub_preds, sub_gt, obo_path, n_boot=n_boot, alpha=alpha,
                                    seed=seed, aspects=aspects, paired_to=ref or paired_to, ia=ia)
            for c in conds:
                out[c] = res[c]
            out.setdefault("_bootstrap", res.get("_bootstrap"))
        return out

    parts_by_cond = {c: _cafa_f1_parts(p, gt_by_protein, dag, aspects, ia=ia)
                     for c, p in preds_by_cond.items()}
    # A protein is bootstrappable only where it is scored, which is aspect-specific; resample the
    # protein list and project it into each aspect's scored subset.
    per_aspect_members = {a: {p for c in parts_by_cond for p in parts_by_cond[c][a]} for a in aspects}
    proteins = sorted(gt_by_protein)

    point = {c: _f1_from_parts(parts_by_cond[c], aspects) for c in parts_by_cond}
    by_aspect = {c: {a: _f1_from_parts(parts_by_cond[c], (a,)) for a in aspects}
                 for c in parts_by_cond}

    rng = random.Random(seed)
    draws = {c: [] for c in parts_by_cond}
    ddraws = {c: [] for c in parts_by_cond}
    n = len(proteins)
    for _ in range(n_boot):
        sample = [proteins[rng.randrange(n)] for _ in range(n)]
        idx = {a: [p for p in sample if p in per_aspect_members[a]] for a in aspects}
        base = None
        for c, parts in parts_by_cond.items():
            # restrict to proteins this condition scored (a shard may miss a row)
            idx_c = {a: [p for p in idx[a] if p in parts[a]] for a in aspects}
            v = _f1_from_parts(parts, aspects, idx_c)
            draws[c].append(v)
            if c == paired_to:
                base = v
        if base is not None:
            for c in parts_by_cond:
                ddraws[c].append(draws[c][-1] - base)

    def ci(vals):
        if not vals:
            return None
        s = sorted(vals)
        lo = s[int((alpha / 2) * len(s))]
        hi = s[min(len(s) - 1, int((1 - alpha / 2) * len(s)))]
        return [float(lo), float(hi)]

    out = {}
    for c in parts_by_cond:
        out[c] = {"f": float(point[c]), "ci": ci(draws[c]), "by_aspect": by_aspect[c],
                  "n_scored": {a: len(parts_by_cond[c][a]) for a in aspects}}
        if paired_to in parts_by_cond and c != paired_to:
            out[c]["delta_vs_wt"] = float(point[c] - point[paired_to])
            out[c]["delta_ci"] = ci(ddraws[c])
    out["_bootstrap"] = {"n_boot": n_boot, "alpha": alpha, "seed": seed,
                         "n_proteins": n, "paired_to": paired_to}
    return out


def verify_against_cafaeval(preds_by_cond, gt_by_protein, obo_path, tol: float = 1e-4,
                            ref=None, paired_to: str = "wt", ia=None):
    """Check the fast F1 reimplementation reproduces cafaeval's F_max on this data."""
    from input_use.metrics import go_dag

    if ref is None:
        ref, _ = cafa_fmax(preds_by_cond, gt_by_protein, obo_path)
    dag = go_dag.load(obo_path)
    out, worst = {}, 0.0
    for cover, conds in coverage_groups(preds_by_cond).items():
        sub_gt = {p: t for p, t in gt_by_protein.items() if p in cover}
        for c in conds:
            fast = _f1_from_parts(_cafa_f1_parts(preds_by_cond[c], sub_gt, dag, ia=ia),
                                  ("MF", "BP", "CC"))
            got = ref.get(c, {}).get("f_w" if ia is not None else "f", 0.0) or 0.0
            d = abs(fast - got)
            worst = max(worst, d)
            out[c] = {"cafaeval": float(got), "fast": float(fast), "abs_diff": float(d)}
    # delta agreement: the quantity the CIs are placed on
    worst_d = 0.0
    if paired_to in out:
        bc, bf = out[paired_to]["cafaeval"], out[paired_to]["fast"]
        for c, d in out.items():
            if c.startswith("_") or c == paired_to:
                continue
            dd = abs((d["cafaeval"] - bc) - (d["fast"] - bf))
            d["delta_diff"] = float(dd)
            worst_d = max(worst_d, dd)
    out["_max_abs_diff"] = float(worst)
    out["_max_delta_diff"] = float(worst_d)
    out["_ok"] = bool(worst_d <= tol)
    return out

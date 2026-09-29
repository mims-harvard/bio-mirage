"""Multiple-choice cell type annotation: option list construction, answer parsing, answer position
diagnostics and chance accuracy.
"""
from __future__ import annotations

import random
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence


def build_options(ground_truth: str, vocabulary: Sequence[str], n_choices: int,
                  seed: int, cell_index: int) -> List[str]:
    """The shuffled option list for one cell."""
    vocab = list(dict.fromkeys(vocabulary))          # de-dup, order-stable
    if ground_truth not in vocab:
        vocab.append(ground_truth)
    rng = random.Random((seed * 1_000_003) ^ (cell_index * 2_654_435_761))
    if n_choices is None or n_choices <= 0 or n_choices >= len(vocab):
        opts = list(vocab)
    else:
        distractors = [c for c in vocab if c != ground_truth]
        opts = rng.sample(distractors, k=min(n_choices - 1, len(distractors))) + [ground_truth]
    rng.shuffle(opts)
    return opts


# ---------------------------------------------------------------------------------------------
# answer parsing
# ---------------------------------------------------------------------------------------------

def _norm(s: str) -> str:
    """Casefold + strip punctuation/whitespace differences, so 'CD4+ T cells.' and 'cd4+ t cells'
    compare equal. Hyphens/underscores collapse to spaces because label vocabularies mix the two
    (`activated_stellate` vs `activated stellate`).
    """
    s = s.strip().lower()
    s = re.sub(r"[\s_\-]+", " ", s)
    s = re.sub(r"[^\w\s+]", "", s)                   # keep '+' (CD4-positive labels use it)
    return s.strip()


def parse_choice(raw: str, options: Sequence[str]) -> Dict[str, object]:
    """Map a model's raw generation onto one of `options`."""
    text = (raw or "").strip()
    text = text.split("\n")[0].strip()
    text = re.sub(r"<[^>]+>", "", text)              # stray Gemma control tokens
    text = text.strip().strip(".").strip()

    for o in options:                                 # 1. exact
        if text == o:
            return {"cell_type": o, "in_options": True, "match_kind": "exact"}

    nt = _norm(text)
    norm_opts = [(o, _norm(o)) for o in options]
    for o, no in norm_opts:                           # 2. normalized exact
        if nt and nt == no:
            return {"cell_type": o, "in_options": True, "match_kind": "normalized"}

    hits = [o for o, no in norm_opts if no and no in nt]   # 3. unique containment
    if len(hits) == 1:
        return {"cell_type": hits[0], "in_options": True, "match_kind": "contained"}
    if len(hits) > 1:                                 # prefer the longest only if unambiguous
        longest = max(hits, key=lambda o: len(_norm(o)))
        if sum(1 for o in hits if len(_norm(o)) == len(_norm(longest))) == 1:
            return {"cell_type": longest, "in_options": True, "match_kind": "contained_longest"}

    starts = [o for o, no in norm_opts if nt and no.startswith(nt)]   # 4. truncated generation
    if len(starts) == 1:
        return {"cell_type": starts[0], "in_options": True, "match_kind": "prefix"}

    return {"cell_type": text, "in_options": False, "match_kind": "none"}


# ---------------------------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------------------------

def position_diagnostics(rows: List[dict]) -> Dict[str, object]:
    """Is the model reading the options, or exploiting their layout?"""
    import math
    chosen_pos, gt_pos, chosen_labels = [], [], []
    for r in rows:
        opts = r.get("options") or []
        if not opts:
            continue
        gt = r.get("gt")
        if gt in opts:
            gt_pos.append(opts.index(gt) / max(1, len(opts) - 1))
        pred = r.get("pred")
        if pred in opts:
            chosen_pos.append(opts.index(pred) / max(1, len(opts) - 1))
            chosen_labels.append(pred)

    def hist(vals, nb=10):
        h = [0] * nb
        for v in vals:
            h[min(nb - 1, int(v * nb))] += 1
        n = max(1, len(vals))
        return [round(c / n, 4) for c in h]

    n_opts = Counter(len(r.get("options") or []) for r in rows).most_common(1)
    n_opts = n_opts[0][0] if n_opts else 0
    counts = Counter(chosen_labels)
    tot = max(1, sum(counts.values()))
    ent = -sum((c / tot) * math.log(c / tot) for c in counts.values()) if counts else 0.0
    max_ent = math.log(n_opts) if n_opts > 1 else 1.0
    return {
        "n_options": n_opts,
        "n_scored": len(chosen_pos),
        "chosen_position_hist": hist(chosen_pos),
        "first_option_rate": round(sum(1 for r in rows
                                       if (r.get("options") or [None])[0] == r.get("pred"))
                                   / max(1, len(rows)), 4),
        "last_option_rate": round(sum(1 for r in rows
                                      if (r.get("options") or [None])[-1] == r.get("pred"))
                                  / max(1, len(rows)), 4),
        "chosen_label_entropy_norm": round(ent / max_ent, 4) if max_ent else 0.0,
        "n_distinct_labels_chosen": len(counts),
        "most_common_choices": counts.most_common(5),
        "gt_position_first_rate": round(sum(1 for p in gt_pos if p == 0.0) / max(1, len(gt_pos)), 4),
        "gt_position_first_rate_expected": round(1.0 / n_opts, 4) if n_opts else None,
    }


def chance_accuracy(rows: List[dict]) -> Optional[float]:
    """Mean 1/n_options -- the accuracy a uniform random chooser gets. The number every MCQ result has
    to be read against; with a 35-label vocabulary it is 2.9%, so accuracy above chance alone is
    weak evidence and worth printing next to the result rather than leaving to the reader.
    """
    ns = [len(r.get("options") or []) for r in rows if r.get("options")]
    return round(sum(1.0 / n for n in ns) / len(ns), 4) if ns else None

"""Yes/No answer parsing and scoring for ChatNT's own binary DNA tasks, such as splice donors, splice
acceptors and TATA promoters.
"""
from __future__ import annotations

import re

_YES = re.compile(r"\b(yes|present|positive|detected|contains?|true|likely)\b", re.I)
_NO = re.compile(r"\b(no|not|absent|negative|none|lacks?|false|unlikely)\b", re.I)


def extract_yesno(raw: str) -> str:
    """Return 'yes' | 'no' | '' from a (possibly verbose) generation. Prefer the leading token, then
    fall back to the first yes/no-bearing keyword.
    """
    t = (raw or "").strip().lower()
    if not t:
        return ""
    first = re.findall(r"[a-z]+", t[:24])
    if first:
        if first[0] in ("yes", "yeah", "yep"):
            return "yes"
        if first[0] in ("no", "nope"):
            return "no"
    # leading "not"/"absent"/... => no ; "present"/... => yes (check no first: "not present")
    head = t[:60]
    if _NO.search(head):
        return "no"
    if _YES.search(head):
        return "yes"
    return ""


def label_to_yesno(label) -> str:
    """Dataset binary label -> yes/no. 1/'1'/True/'yes' => yes (motif present)."""
    s = str(label).strip().lower()
    return "yes" if s in ("1", "yes", "true", "positive") else "no"


def yesno_correct(pred: str, label) -> bool:
    return pred in ("yes", "no") and pred == label_to_yesno(label)

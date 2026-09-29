"""Parsers for BioReason-Pro generations (GO term ids, the `<think>` reasoning trace, the functional
summary) and GO set Jaccard similarity.
"""
from __future__ import annotations

import re
from typing import Set

GO_RE = re.compile(r"GO:\d{7}")
THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
SUMMARY_RE = re.compile(r"Functional Summary:\s*(.+?)(?:\n[-*]|\n\n|$)", re.S | re.I)


# ---- response parsers (used by the runner) ----
def extract_go(text: str) -> Set[str]:
    return set(GO_RE.findall(text or ""))


def extract_think(text: str) -> str:
    m = THINK_RE.search(text or "")
    return m.group(1).strip() if m else ""


def extract_summary(text: str) -> str:
    m = SUMMARY_RE.search(text or "")
    if m:
        return " ".join(m.group(1).split())
    if "</think>" in (text or ""):
        return " ".join(text.split("</think>")[-1].split())[:1500]
    return ""


# ---- self-consistency metric (used by core/score.py; F_max lives in metrics/cafa.py) ----
def go_jaccard(a, b) -> float:
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if (a | b) else 1.0

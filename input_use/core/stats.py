"""Resampling statistics for the scorers: percentile bootstrap intervals for a mean and a one-sided
paired sign-flip permutation test.

Pure Python with fixed seeds, so it runs unchanged in each model's environment.
"""
from __future__ import annotations

import random
from typing import List, Optional, Tuple


def bootstrap_ci(values: List[float], n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> Tuple[Optional[float], Optional[float]]:
    """Percentile bootstrap CI for the mean of `values` (defaults to 95%). (None, None) if empty."""
    v = [x for x in values if x is not None]
    if not v:
        return (None, None)
    rng = random.Random(seed)
    n = len(v)
    means = sorted(sum(v[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (round(lo, 4), round(hi, 4))


def perm_test_paired_gt(a: List[float], b: List[float], n_perm: int = 10000,
                        seed: int = 0) -> Optional[float]:
    """One-sided paired permutation test, H1: mean(a - b) > 0, by random sign-flips of the paired
    differences. Returns a p-value, or None for <3 pairs. Robust to ties (all-zero diffs -> p=1).
    """
    if len(a) != len(b) or len(a) < 3:
        return None
    d = [x - y for x, y in zip(a, b)]
    if all(abs(x) < 1e-12 for x in d):
        return 1.0
    obs = sum(d) / len(d)
    rng = random.Random(seed)
    ge = 0
    for _ in range(n_perm):
        s = sum(x if rng.random() < 0.5 else -x for x in d) / len(d)
        if s >= obs - 1e-12:
            ge += 1
    return round((ge + 1) / (n_perm + 1), 4)


def mean_ci(values: List[float], seed: int = 0):
    """Convenience: (mean, (lo, hi)) for a list of 0/1 (or real) values."""
    v = [x for x in values if x is not None]
    if not v:
        return (None, (None, None))
    return (round(sum(v) / len(v), 4), bootstrap_ci(v, seed=seed))

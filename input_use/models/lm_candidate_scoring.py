"""Candidate log-likelihood scoring for decoder-only language models loaded with
`AutoModelForCausalLM`, such as C2S-Scale.

`score_candidates` returns the mean per-token log-likelihood of each candidate answer after the
prompt. `debias` subtracts a per-candidate baseline score that does not depend on the input (for
example the mean over cells from `mean_scores`), leaving the part of the score that depends on the
input.
"""
from __future__ import annotations

from typing import Dict, List

import torch


@torch.no_grad()
def score_candidates(model, tokenizer, prompt: str, candidates: List[str], device) -> Dict[str, float]:
    """Mean per-token log-likelihood of each candidate as a continuation of `prompt`, under `model`."""
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
    n_prompt = prompt_ids.shape[0]
    sep = "" if prompt.endswith(" ") else " "
    scores: Dict[str, float] = {}
    for cand in candidates:
        full_ids = tokenizer(prompt + sep + cand, return_tensors="pt").input_ids.to(device)
        cand_len = full_ids.shape[1] - n_prompt
        if cand_len <= 0:
            scores[cand] = float("-inf")
            continue
        logits = model(full_ids).logits[0]                     # [seq_len, vocab]
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        target_ids = full_ids[0, n_prompt:]                     # candidate's own token ids
        pred_log_probs = log_probs[n_prompt - 1: -1]            # next-token predictions aligned to them
        token_lls = pred_log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        scores[cand] = float(token_lls.mean())
    return scores


def argmax_candidate(scores: Dict[str, float]) -> str:
    return max(scores, key=scores.get)


def mean_scores(score_dicts: List[Dict[str, float]]) -> Dict[str, float]:
    """Per-candidate mean over a list of {candidate: score} dicts (all assumed to share the same
    candidate set) -- used to build the empirical wt-mean baseline for models with no `no_modality`
    reference available."""
    candidates = score_dicts[0].keys()
    n = len(score_dicts)
    return {c: sum(d[c] for d in score_dicts) / n for c in candidates}


def debias(scores: Dict[str, float], baseline: Dict[str, float]) -> Dict[str, float]:
    """Subtract each candidate's fixed, input-independent baseline score, leaving only the part that
    depends on this specific input -- see module docstring.
    """
    return {c: scores[c] - baseline.get(c, 0.0) for c in scores}

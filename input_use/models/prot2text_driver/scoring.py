"""BERTScore, ROUGE-L and corpus BLEU per generation, with the model, layer and truncation settings of
the released Prot2Text-V2 benchmark script.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


class Scorers:
    def __init__(self, device="cuda", which=("biobert", "roberta"), batch_size=64):
        from bert_score import BERTScorer
        from transformers import AutoTokenizer
        self.scorers, self.toks = {}, {}
        for name in which:
            model_type, layers = C.BERTSCORE_MODELS[name]
            self.scorers[name] = BERTScorer(model_type=model_type, num_layers=layers, batch_size=batch_size,
                                            nthreads=4, all_layers=False, idf=False, device=device,
                                            rescale_with_baseline=False)
            self.toks[name] = AutoTokenizer.from_pretrained(model_type)

    def truncate(self, name, texts):
        tok = self.toks[name]
        ids = tok(list(texts), padding="max_length", truncation=True, max_length=C.BERTSCORE_TRUNC,
                  return_tensors="pt")["input_ids"]
        return tok.batch_decode(ids, skip_special_tokens=True)

    def bertscore(self, name, cands, refs):
        """P, R, F1 arrays for aligned lists, after the benchmark's truncation."""
        cands = self.truncate(name, cands)
        refs = self.truncate(name, refs)
        P, R, F = self.scorers[name].score(cands, refs, verbose=False)
        return P.numpy(), R.numpy(), F.numpy()


def rouge_l(cands, refs):
    from rouge_score import rouge_scorer
    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    return np.array([sc.score(r, c)["rougeL"].fmeasure for c, r in zip(cands, refs)])


def bleu_corpus(cands, refs, max_order):
    """Corpus-level bleu as benchmark.py computes it (evaluate 'bleu'); None if the metric module
    cannot be loaded offline.
    """
    try:
        import evaluate
        m = evaluate.load("bleu")
        return float(m.compute(predictions=list(cands), references=[[r] for r in refs], max_order=max_order)["bleu"])
    except Exception as e:  # noqa: BLE001
        print(f"bleu unavailable: {e}", file=sys.stderr)
        return None

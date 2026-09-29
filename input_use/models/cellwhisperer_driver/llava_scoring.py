"""Scores candidate answers with the released CellWhisperer chat model (a LLaVA model on
Mistral-7B-Instruct-v0.2) or with the base Mistral model, by teacher forcing. Imported by
score_llm.py and runs in the LLaVA environment of the CellWhisperer repository (its modules/LLaVA
package, transformers 4.37.2).

Prompts and label masks come from the LLaVA code of the CellWhisperer repository: the
`mistral_instruct` conversation template, `preprocess_llama_2` for token ids and masking, and only
the last assistant turn scored, as in CellWhisperer's perplexity evaluation. Z_CW enters as one
`<image>` placeholder that `prepare_inputs_labels_for_multimodal` replaces by the 8 projector tokens.
For each candidate answer the scorer returns the summed log-likelihood of the answer tokens, their
mean (perplexity is exp of minus the mean) and the number of tokens.

Both released chat checkpoints set tokenizer_model_max_length to 2,048, and the LLaVA code silently
truncates longer prompts to it. Prompts with the top-1,000 gene text exceed that length, so the
scorer sets the attribute to None and records each prompt's length and whether it lies within 2,048
tokens.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
QUESTION = "Which cell type is this cell?"
PRE_PROMPT_USER = ("Respond to my request regarding a sample of cells characterized by its "
                   "top-expressed genes being {}")
PRE_PROMPT_ASSISTANT = "Sure. What's your request?"
CONV_MODE = "mistral_instruct"


def _set_default_conversation():
    from llava import conversation as conversation_lib
    conversation_lib.default_conversation = conversation_lib.conv_templates[CONV_MODE]
    return conversation_lib


def make_sources(genes: Optional[List[str]], question: str, answer: str, image: bool):
    """The conversation in CellWhisperer's evaluation JSON layout (question, then '\\n<image>')."""
    conv = []
    if genes:
        conv.append({"from": "human", "value": PRE_PROMPT_USER.format(", ".join(genes))})
        conv.append({"from": "gpt", "value": PRE_PROMPT_ASSISTANT})
    conv.append({"from": "human", "value": f"{question}\n<image>" if image else question})
    conv.append({"from": "gpt", "value": answer})
    return [conv]


def last_block_mask(labels: torch.Tensor) -> torch.Tensor:
    """Boolean mask of the final contiguous run of non-IGNORE labels (1-d)."""
    valid = labels != IGNORE_INDEX
    mask = torch.zeros_like(valid)
    idx = torch.nonzero(valid, as_tuple=False).flatten()
    if len(idx) == 0:
        return mask
    end = int(idx[-1])
    start = end
    while start - 1 >= 0 and valid[start - 1]:
        start -= 1
    mask[start:end + 1] = True
    return mask


class Scorer:
    def __init__(self, model_dir: str, kind: str, lift_cap: bool = True, device: str = "cuda",
                 dtype=torch.bfloat16):
        assert kind in ("chat", "base")
        from transformers import AutoTokenizer
        self.kind = kind
        self.model_dir = str(model_dir)
        self.conversation_lib = _set_default_conversation()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, use_fast=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.unk_token
        if kind == "chat":
            cfg = json.load(open(Path(self.model_dir) / "config.json"))
            arch = cfg["architectures"][0]
            from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
            classes = {"LlavaLlamaForCausalLM": LlavaLlamaForCausalLM}
            self.model = classes[arch].from_pretrained(self.model_dir, torch_dtype=dtype,
                                                       low_cpu_mem_usage=True).to(device)
            self.released_cap = getattr(self.model.config, "tokenizer_model_max_length", None)
            if lift_cap:
                self.model.config.tokenizer_model_max_length = None
            self.num_projector_tokens = int(self.model.config.mm_projector_type.split("_")[1].strip("t"))
            assert self.model.config.mm_hidden_size == 2048
            assert not getattr(self.model.config, "mm_use_im_start_end", False)
        else:
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(self.model_dir, torch_dtype=dtype,
                                                              low_cpu_mem_usage=True).to(device)
            self.released_cap = None
            self.num_projector_tokens = 0
        self.model.eval()
        self.device = device
        self.max_position_embeddings = int(self.model.config.max_position_embeddings)
        self.attn_implementation = getattr(self.model.config, "_attn_implementation", None)
        self.tokenizer_model_max_length = int(min(self.tokenizer.model_max_length, 10**9))

    # ---- prompt -> ids/labels -------------------------------------------------------------
    def encode(self, genes: Optional[List[str]], answer: str, image: bool, question: str = QUESTION) -> Dict:
        from llava.train.train import preprocess_llama_2
        assert (self.kind == "chat") == image, "image prompts need the chat model; text-only the base"
        sources = make_sources(genes, question, answer, image)
        d = preprocess_llama_2(copy.deepcopy(sources), self.tokenizer, has_image=image)
        ids, labels = d["input_ids"][0], d["labels"][0]
        if not image:  # the text branch pads/truncates with tokenizer.model_max_length; assert none
            assert len(ids) < self.tokenizer_model_max_length, len(ids)
            assert (ids != self.tokenizer.pad_token_id).all() or self.tokenizer.pad_token_id == self.tokenizer.unk_token_id
        mask = last_block_mask(labels)
        n_ans = int(mask.sum())
        assert n_ans > 0, "empty answer span (tokenization mismatch?)"
        labels = torch.where(mask, labels, torch.full_like(labels, IGNORE_INDEX))
        n_img = int((ids == IMAGE_TOKEN_INDEX).sum())
        assert n_img == (1 if image else 0)
        spliced = len(ids) + (self.num_projector_tokens - 1 if image else 0)
        return {"input_ids": ids, "labels": labels, "n_answer_tokens": n_ans,
                "prompt_token_count": int(len(ids) - n_ans), "spliced_len": int(spliced),
                "within_released_cap": bool(self.released_cap is None or spliced <= self.released_cap)}

    # ---- batched forward -------------------------------------------------------------------
    @torch.inference_mode()
    def score(self, items: List[Dict], images: Optional[List[np.ndarray]] = None,
              token_budget: int = 24000, max_batch: int = 64) -> List[Dict]:
        """items: encode() outputs; images: one 2048-d vector per item (chat) or None (base).
        Returns per item: logp_sum, logp_mean, n_tokens (answer tokens scored)."""
        n = len(items)
        order = sorted(range(n), key=lambda i: len(items[i]["input_ids"]))
        out: List[Optional[Dict]] = [None] * n
        i = 0
        while i < n:
            L = len(items[order[i]]["input_ids"])
            b = 1
            while (i + b < n and b < max_batch
                   and (b + 1) * len(items[order[i + b]]["input_ids"]) <= token_budget):
                b += 1
            batch_idx = order[i:i + b]
            res = self._forward(batch_idx, items, images)
            for j, r in zip(batch_idx, res):
                out[j] = r
            i += b
        return out  # type: ignore

    def _forward(self, batch_idx, items, images):
        pad = self.tokenizer.pad_token_id
        maxlen = max(len(items[j]["input_ids"]) for j in batch_idx)
        B = len(batch_idx)
        ids = torch.full((B, maxlen), pad, dtype=torch.long)
        labels = torch.full((B, maxlen), IGNORE_INDEX, dtype=torch.long)
        attn = torch.zeros((B, maxlen), dtype=torch.long)
        for r, j in enumerate(batch_idx):
            x = items[j]["input_ids"]
            ids[r, :len(x)] = x
            labels[r, :len(x)] = items[j]["labels"]
            attn[r, :len(x)] = 1
        ids, labels, attn = ids.to(self.device), labels.to(self.device), attn.to(self.device)
        if self.kind == "chat":
            img = torch.from_numpy(np.stack([images[j] for j in batch_idx]).astype(np.float32))
            img = img.to(self.device, dtype=self.model.dtype)
            (_, position_ids, attn2, _, inputs_embeds, labels2) = self.model.prepare_inputs_labels_for_multimodal(
                ids, None, attn, None, labels, img)
            from transformers import LlamaForCausalLM
            outp = LlamaForCausalLM.forward(self.model, inputs_embeds=inputs_embeds, attention_mask=attn2,
                                            position_ids=position_ids, use_cache=False, return_dict=True)
            logits, lab = outp.logits, labels2
        else:
            outp = self.model(input_ids=ids, attention_mask=attn, use_cache=False, return_dict=True)
            logits, lab = outp.logits, labels
        # next-token alignment: logits[:, t] predicts lab[:, t+1]
        tgt = lab[:, 1:]
        valid = tgt != IGNORE_INDEX
        res = []
        for r in range(B):
            pos = torch.nonzero(valid[r], as_tuple=False).flatten()
            lg = logits[r, pos, :].float()
            lp = torch.log_softmax(lg, dim=-1).gather(1, tgt[r, pos].unsqueeze(1)).squeeze(1)
            res.append({"logp_sum": float(lp.sum()), "logp_mean": float(lp.mean()), "n_tokens": int(len(pos))})
            assert len(pos) == items[batch_idx[r]]["n_answer_tokens"], (len(pos), items[batch_idx[r]]["n_answer_tokens"])
        return res

    def generate(self, genes, image_vec: Optional[np.ndarray], question: str = QUESTION, max_new_tokens: int = 48) -> str:
        """Greedy generation, for inspecting individual answers; not used in scoring."""
        from llava.mm_utils import tokenizer_image_token
        conv = self.conversation_lib.conv_templates[CONV_MODE].copy()
        if genes:
            conv.append_message(conv.roles[0], PRE_PROMPT_USER.format(", ".join(genes)))
            conv.append_message(conv.roles[1], PRE_PROMPT_ASSISTANT)
        conv.append_message(conv.roles[0], f"{question}\n<image>" if image_vec is not None else question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        with torch.inference_mode():
            if image_vec is not None:
                ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
                img = torch.from_numpy(np.asarray(image_vec, dtype=np.float32)).to(self.device, dtype=self.model.dtype).unsqueeze(0)
                out = self.model.generate(ids, images=img, do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
                                          use_cache=True, pad_token_id=self.tokenizer.eos_token_id)
                return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()
            ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            out = self.model.generate(ids, do_sample=False, num_beams=1, max_new_tokens=max_new_tokens, use_cache=True,
                                      pad_token_id=self.tokenizer.eos_token_id)
            return self.tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

    def manifest(self) -> Dict:
        import transformers
        return {"model_dir": self.model_dir, "kind": self.kind, "released_tokenizer_model_max_length": self.released_cap,
                "max_position_embeddings": self.max_position_embeddings, "num_projector_tokens": self.num_projector_tokens,
                "attn_implementation": self.attn_implementation, "dtype": str(self.model.dtype),
                "transformers": transformers.__version__, "torch": torch.__version__, "conv_mode": CONV_MODE}

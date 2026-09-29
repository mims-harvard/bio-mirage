"""Shared constants and helpers for the Prot2Text-V2 evaluation on function description generation.

Sets the result, run and Hugging Face cache paths under protein/prot2text_v2 of
$INPUT_USE_RESULTS_DIR, pins the model, encoder, dataset and tokenizer revisions, and reproduces the
released data loader's prompt (protein name, taxon, one placeholder token per residue), tokenization
and generation settings. Also defines a seeded shuffle of the amino acid sequence that keeps its
composition.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "core"))
import paths as RD  # noqa: E402

PROJ = RD.PROT2TEXT
HF_HOME = f"{PROJ}/hf"
HUB = f"{HF_HOME}/hub"
DATA = f"{PROJ}/data"
RUNS = f"{PROJ}/runs"
RESULTS = f"{PROJ}/results"
REPO = f"{PROJ}/repo"

# ---- pinned revisions -----------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "core"))
from checkpoints import ckpt  # noqa: E402

MODEL_ID = ckpt("prot2text")
MODEL_REV = "ec57acd96f2a24912f30bf5596eb4cc435a92682"
ESM_ID = ckpt("prot2text_esm")
ESM_REV = "476b639933c8baad5ad09a60ac1a87f987b656fc"
DATASET_ID = ckpt("prot2text_data")
DATASET_REV = "96122499290b40ab9e0c3094164127961d2a015c"
REPO_URL = ckpt("prot2text_repo")
REPO_COMMIT = "0d75d4642bd3d2aae1cce4c371635407598880a9"
# mirrors carry the same vocabulary; they differ in the chat template.
LLAMA_TOK = {
    "llama31": (ckpt("llama_tokenizer_31"), "4699cc75b550f9c6f3173fb80f4703b62d946aa5"),
    "llama3_old": (ckpt("llama_tokenizer_old"), "d10aef7999a2b5ba950ab3974312feeedbfe0b77"),
}
BERTSCORE_MODELS = {
    # (model_type, num_layers) exactly as scripts/benchmark.py resolves them
    "roberta": (ckpt("bertscore_roberta"), 17),          # lang="en" -> roberta-large, layer 17
    "biobert": (ckpt("bertscore_biobert"), 24),
}
BERTSCORE_TRUNC = 495       # benchmark.py retokenises to max_length=495 and decodes before scoring


def snapshot(repo_id, rev):
    """Local snapshot directory of a pinned hub revision (jobs run with HF_HUB_OFFLINE=1)."""
    p = os.path.join(HUB, "models--" + repo_id.replace("/", "--"), "snapshots", rev)
    if not os.path.isdir(p):
        raise FileNotFoundError(p)
    return p


SYSTEM_MESSAGE = (
    "You are a scientific assistant specialized in protein function "
    "predictions. Given the sequence embeddings and other information "
    "of a protein, describe its function clearly and concisely in "
    "professional language. "
)
PLACEHOLDER = "<|reserved_special_token_1|>"
PLACEHOLDER_ID = 128003
PAD_TOKEN = "<|reserved_special_token_0|>"
PAD_ID = 128002
EOS_ID = 128009
OFFICIAL_MAX_SEQ = 1021        # dataset.py default max_sequence_length (N-terminal truncation)
UNKNOWN = "unknown"            # dataset.py substitutes this for a missing Full Name / taxon

# scripts/generate_instruct.py + the model card: beam 4, greedy, up to 1024 new tokens
GEN_KWARGS = dict(max_new_tokens=1024, eos_token_id=EOS_ID, pad_token_id=PAD_ID,
                  return_dict_in_generate=False, num_beams=4, length_penalty=1.0,
                  temperature=1.0, do_sample=False, top_p=1.0, top_k=50)
GEN_SEED = 0


def user_message(fullname: str, taxon: str, sequence: str) -> str:
    """dataset.py: 'Protein name: ' + fullname + ' ; Taxon: ' + taxon + ' ; Sequence embeddings: ' +
    placeholder * (len(sequence) + 2). The +2 is the esm bos and eos.
    """
    return ("Protein name: " + fullname
            + " ; Taxon: " + taxon
            + " ; Sequence embeddings: "
            + PLACEHOLDER * (len(sequence) + 2))


def conversation(fullname: str, taxon: str, sequence: str):
    return [{"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user_message(fullname, taxon, sequence)}]


def check_vendored_prompt():
    """The four prompt lines and the system message must still be in the repository file."""
    src = open(os.path.join(REPO, "dataset", "dataset.py")).read()
    for needle in ('"Protein name: " + fullname', '+ " ; Taxon: " + taxon',
                   '+ " ; Sequence embeddings: "', "self.placeholder_token * (len(sequence) + 2)",
                   '"You are a scientific assistant specialized in protein function "',
                   '"professional language. "', "placeholder_token: str = '<|reserved_special_token_1|>'",
                   "max_sequence_length: Optional[int] = 1021",
                   'fullname = "unknown" if pd.isna(fullname) else fullname'):
        assert needle in src, needle
    return hashlib.sha256(src.encode()).hexdigest()


# ---- sequences --------------------------------------------------------------------------------
def seq_hash(seq: str) -> str:
    return hashlib.sha256(seq.encode()).hexdigest()[:16]


def shuffle_sequence(seq: str, seed: int, protein_id: str) -> str:
    """Composition-preserving permutation of the amino-acid string. The stream is seeded from the
    (seed, protein) pair so a protein's shuffle does not depend on where it sits in the table.
    """
    rng = random.Random(f"prot2text_shuffle:{seed}:{protein_id}")
    chars = list(seq)
    rng.shuffle(chars)
    return "".join(chars)


def shuffle_ok(orig: str, shuf: str) -> dict:
    return {"len_equal": len(orig) == len(shuf),
            "composition_equal": Counter(orig) == Counter(shuf),
            "hash_differs": seq_hash(orig) != seq_hash(shuf),
            "degenerate": len(set(orig)) <= 1}


def request_id(condition: str, seed: int, protein_id: str) -> str:
    return f"{condition}|{seed}|{protein_id}"


# ---- io -----------------------------------------------------------------------------------------
def read_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def versions():
    import importlib
    out = {}
    for m in ("torch", "transformers", "tokenizers", "bert_score", "evaluate", "rouge_score",
              "nltk", "pandas", "numpy", "accelerate", "safetensors"):
        try:
            out[m] = importlib.import_module(m).__version__
        except Exception:
            out[m] = None
    return out


def load_tokenizers(template="llama31"):
    from transformers import AutoTokenizer
    esm_tok = AutoTokenizer.from_pretrained(snapshot(ESM_ID, ESM_REV))
    rid, rev = LLAMA_TOK[template]
    llama_tok = AutoTokenizer.from_pretrained(snapshot(rid, rev), pad_token=PAD_TOKEN)
    assert llama_tok.convert_tokens_to_ids(PLACEHOLDER) == PLACEHOLDER_ID
    assert llama_tok.convert_tokens_to_ids(PAD_TOKEN) == PAD_ID
    assert llama_tok.convert_tokens_to_ids("<|eot_id|>") == EOS_ID
    return esm_tok, llama_tok


def tokenize_request(esm_tok, llama_tok, fullname, taxon, sequence):
    """Prompt ids (1-D LongTensor) and protein ids (1-D), exactly as dataset.py builds them."""
    prompt_ids = llama_tok.apply_chat_template(conversation(fullname, taxon, sequence),
                                               add_generation_prompt=True, tokenize=True,
                                               padding=False, return_tensors="pt")[0]
    prot_ids = esm_tok([sequence], add_special_tokens=True, return_attention_mask=False,
                       return_tensors="pt")["input_ids"][0]
    n_ph = int((prompt_ids == PLACEHOLDER_ID).sum())
    if n_ph != prot_ids.numel():
        raise ValueError(f"placeholder count {n_ph} != protein token count {prot_ids.numel()} "
                         f"(len {len(sequence)}): {sequence[:30]}...")
    return prompt_ids, prot_ids


VENDORED_PROMPT_SHA = check_vendored_prompt()

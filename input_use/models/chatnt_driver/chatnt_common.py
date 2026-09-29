"""Shared constants and helpers for the ChatNT evaluation on its own binary DNA tasks (splice donors,
splice acceptors, TATA promoters).

Sets the Hugging Face cache under dna/chatnt of $INPUT_USE_RESULTS_DIR before transformers is
imported, pins the model and dataset revisions, and defines the prompt builder, answer parsing, the
sequence shuffles that keep base composition or dinucleotide counts, and `ChatNTTorch`, a batched
wrapper around the released PyTorch model that returns the log-likelihoods of " Yes" and " No" as
the first answer token and an optional greedy generation.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "core"))
import paths as RD  # noqa: E402

ROOT = RD.CHATNT
DATA_RAW = os.path.join(ROOT, "data", "raw")
RESULTS = os.path.join(ROOT, "results")
HF_HOME = os.path.join(ROOT, "hf")
# the cache location is forced here, before transformers / huggingface_hub are imported.
os.environ["HF_HOME"] = HF_HOME
os.environ["HF_HUB_CACHE"] = os.path.join(HF_HOME, "hub")
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_HOME, "datasets")
os.environ["XDG_CACHE_HOME"] = os.path.join(ROOT, "xdg")          # get_chatNT's joblib cache
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "core"))
from checkpoints import ckpt  # noqa: E402

MODEL_ID = ckpt("chatnt")
MODEL_REV = "862ba01ce8206f289700e35aa5a2805304c155ca"
DATA_ID = ckpt("chatnt_data")
DATA_REV = "0b44d80063c454fced739eadda26555a735c1d9d"
REPO_URL = ckpt("chatnt_repo")

# dataset task id -> name used in Datasets_overview.csv / the paper
TASKS = {
    "NT_splice_sites_donors": "Splice donors",
    "NT_splice_sites_acceptors": "Splice acceptors",
    "NT_promoter_tata": "TATA promoters",
}
TASK_ORDER = list(TASKS)

MAX_PER_CLASS = 1000
SELECT_SEED = 0
SHUFFLE_SEEDS = (0, 1, 2)
DINUC_SEED = 0
BOOT_N = 2000
BOOT_SEED = 0
GC_TOL = 0.02

# --- the released inference interface (text_generation.py / inference.ipynb) ---
CONTEXT = ("A chat between a curious user and an artificial intelligence assistant that can "
           "handle bio sequences. The assistant gives helpful, detailed, and polite answers to "
           "the user's questions. USER: ")
ENGLISH_MAX_LENGTH = 512
BIO_MAX_LENGTH = 512
RESAMPLED_LENGTH = 64       # embeddings inserted per <DNA>; the decoder drops that many trailing
# pads
MAX_NEW_TOKENS = 24         # secondary (generation) score; answers in the data are 10-20 tokens
ANSWERS = ("Yes", "No")

CONDITIONS = ["intact", "shuffle", "swap", "shuffle_dinuc"]

# the notebook's own example (unchanged), used as the smoke test
NOTEBOOK_ENGLISH = ("A chat between a curious user and an artificial intelligence assistant that "
                    "can handle bio sequences. The assistant gives helpful, detailed, and polite "
                    "answers to the user's questions. USER: Is there any evidence of an acceptor "
                    "splice site in this sequence <DNA> ? ASSISTANT:")
NOTEBOOK_DNA = ("ATCGGAAAAAGATCCAGAAAGTTATACCAGGCCAATGGGAATCACCTATTACGTGGATAATAGCGATAGTATGTTACCTAT"
                "AAATTTAACTACGTGGATATCAGGCAGTTACGTTACCAGTCAAGGAGCACCCAAAACTGTCCAGCAACAAGTTAATTTACC"
                "CATGAAGATGTACTGCAAGCCTTGCCAACCAGTTAAAGTAGCTACTCATAAGGTAATAAACAGTAATATCGACTTTTTATC"
                "CATTTTGATAATTGATTTATAACAGTCTATAACTGATCGCTCTACATAATCTCTATCAGATTACTATTGACACAAACAGAA"
                "ACCCCGTTAATTTGTATGATATATTTCCCGGTAAGCTTCGATTTTTAATCCTATCGTGACAATTTGGAATGTAACTTATTT"
                "CGTATAGGATAAACTAATTTACACGTTTGAATTCCTAGAATATGGAGAATCTAAAGGTCCTGGCAATGCCATCGGCTTTCA"
                "ATATTATAATGGACCAAAAGTTACTCTATTAGCTTCCAAAACTTCGCGTGAGTACATTAGAACAGAAGAATAACCTTCAAT"
                "ATCGAGAGAGTTACTATCACTAACTATCCTATG")
NOTEBOOK_EXPECTED = "Yes, an acceptor splice site is present in this nucleotide sequence."


def sha(s) -> str:
    if isinstance(s, str):
        s = s.encode()
    return hashlib.sha256(s).hexdigest()


def build_prompt(question: str) -> str:
    """Exactly `TextGenerationPipeline.preprocess` in the released text_generation.py."""
    space = " "
    if question[-1] == " ":
        space = ""
    return CONTEXT + question + space + "ASSISTANT:"


def parse_exchanges(s: str):
    ex = json.loads(s)
    assert len(ex) == 2 and ex[0]["role"] == "USER" and ex[1]["role"] == "ASSISTANT", ex
    return ex[0]["message"], ex[1]["message"]


def label_from(label_field: str) -> str:
    v = json.loads(label_field)
    assert isinstance(v, list) and len(v) == 1 and v[0] in ("yes", "no"), label_field
    return "Yes" if v[0] == "yes" else "No"


def opposite(label: str) -> str:
    return "No" if label == "Yes" else "Yes"


def gc_fraction(seq: str) -> float:
    return sum(1 for c in seq if c in "GCgc") / len(seq)


def composition(seq: str) -> dict:
    return dict(sorted(Counter(seq).items()))


def dinuc_counts(seq: str) -> dict:
    return dict(sorted(Counter(seq[i:i + 2] for i in range(len(seq) - 1)).items()))


def rng_for(seed: int, sample_id: int, salt: int = 0) -> np.random.Generator:
    """One generator per (seed, example): the shuffle of an example does not depend on how many other
    examples were shuffled before it.
    """
    return np.random.default_rng([int(seed), int(sample_id), int(salt)])


def shuffle_mono(seq: str, rng: np.random.Generator) -> str:
    """Uniform permutation of the positions. Preserves length and every symbol count."""
    arr = np.array(list(seq))
    return "".join(arr[rng.permutation(len(arr))])


def shuffle_dinuc(seq: str, rng: np.random.Generator) -> str:
    """Altschul & Erickson (1985) dinucleotide-preserving shuffle (the ushuffle algorithm, k=2).
    Preserves the first symbol, the last symbol, every symbol count and every dinucleotide count.
    """
    s = list(seq)
    n = len(s)
    if n < 3:
        return seq
    letters = sorted(set(s))
    edges = {a: [] for a in letters}
    for i in range(n - 1):
        edges[s[i]].append(s[i + 1])
    last = s[-1]
    while True:                                   # draw last edges until they form a tree into `last`
        last_edge = {}
        for a in letters:
            if a == last or not edges[a]:
                continue
            last_edge[a] = edges[a][int(rng.integers(len(edges[a])))]
        ok = True
        for a in last_edge:
            v, seen = a, set()
            while v != last:
                if v in seen or v not in last_edge:
                    ok = False
                    break
                seen.add(v)
                v = last_edge[v]
            if not ok:
                break
        if ok:
            break
    new_edges = {}
    for a in letters:
        e = list(edges[a])
        if a in last_edge:
            e.remove(last_edge[a])
        e = [e[i] for i in rng.permutation(len(e))]
        if a in last_edge:
            e.append(last_edge[a])
        new_edges[a] = e
    out, ptr, v = [s[0]], {a: 0 for a in letters}, s[0]
    for _ in range(n - 1):
        nxt = new_edges[v][ptr[v]]
        ptr[v] += 1
        out.append(nxt)
        v = nxt
    res = "".join(out)
    assert dinuc_counts(res) == dinuc_counts(seq) and composition(res) == composition(seq)
    return res


_WORD = re.compile(r"^[^A-Za-z]*([A-Za-z]+)")


def parse_generation_label(text: str):
    """Yes / No when the generation is unambiguous, else None. First word wins; otherwise the label is
    taken only when exactly one of the two words occurs anywhere.
    """
    if text is None:
        return None
    t = text.strip()
    m = _WORD.match(t)
    if m:
        w = m.group(1).lower()
        if w == "yes":
            return "Yes"
        if w == "no":
            return "No"
    has_yes = re.search(r"\byes\b", t, re.I) is not None
    has_no = re.search(r"\bno\b", t, re.I) is not None
    if has_yes and not has_no:
        return "Yes"
    if has_no and not has_yes:
        return "No"
    return None


def package_versions():
    import importlib
    out = {}
    for name in ("torch", "transformers", "tokenizers", "sentencepiece", "numpy", "pandas",
                 "pyarrow", "scipy", "jax", "jaxlib", "haiku", "huggingface_hub", "safetensors",
                 "accelerate", "nucleotide_transformer"):
        try:
            m = importlib.import_module(name)
            out[name] = getattr(m, "__version__", "installed")
        except Exception as e:  # noqa: BLE001
            out[name] = f"not importable: {type(e).__name__}"
    return out


def repo_commit():
    p = os.path.join(ROOT, "env", "nucleotide-transformer", ".git", "HEAD")
    try:
        head = open(p).read().strip()
        if head.startswith("ref:"):
            return open(os.path.join(ROOT, "env", "nucleotide-transformer", ".git",
                                     head.split()[1])).read().strip()
        return head
    except OSError:
        return None


# ---------------------------------------------------------------------------------------------- The
# PyTorch (Hugging Face) ChatNT: the released custom code, loaded with trust_remote_code.
class ChatNTTorch:
    """Batched wrapper around the released `TorchMultiOmicsModel`."""

    def __init__(self, device="cuda", dtype="bfloat16"):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.device = device
        self.dtype = getattr(torch, dtype)
        self.english_tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, subfolder="english_tokenizer", revision=MODEL_REV)
        self.bio_tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, subfolder="bio_tokenizer", revision=MODEL_REV)
        self.model, info = AutoModel.from_pretrained(
            MODEL_ID, trust_remote_code=True, revision=MODEL_REV, torch_dtype=self.dtype,
            output_loading_info=True)
        self.loading_info = {k: sorted(map(str, v)) if isinstance(v, (list, set)) else str(v)
                             for k, v in info.items()}
        assert not info.get("missing_keys") and not info.get("unexpected_keys"), self.loading_info
        assert not info.get("mismatched_keys"), self.loading_info
        self.model.to(device).eval()
        self.pad_id = self.english_tokenizer.pad_token_id
        self.eos_id = self.english_tokenizer.eos_token_id
        self.bio_pad_id = self.bio_tokenizer.pad_token_id
        assert self.pad_id == 2 and self.eos_id == 2 and self.bio_pad_id == 1
        self.dna_id = self.english_tokenizer.convert_tokens_to_ids("<DNA>")
        assert self.dna_id == 32000, self.dna_id
        # answer token ids in the data's own formatting ("assistant: Yes, ..."): the tokens that
        # `prompt + " Yes"` adds beyond `prompt`, verified per prompt in `answer_ids`
        probe = build_prompt("Is there any evidence of an acceptor splice site in this sequence <DNA>?")
        self.answer_token_ids = {a: self.answer_ids(probe, " " + a) for a in ANSWERS}
        self.answer_token_ids_nospace = {a: self.answer_ids(probe, a) for a in ANSWERS}
        for a in ANSWERS:
            assert len(self.answer_token_ids[a]) == 1, (a, self.answer_token_ids[a])

    # --- tokenisation, verbatim from text_generation.py ---
    def tokenize_english(self, prompts):
        return self.english_tokenizer(prompts, return_tensors="pt", padding="max_length",
                                      truncation=True, max_length=ENGLISH_MAX_LENGTH).input_ids

    def tokenize_bio(self, seqs):
        return self.bio_tokenizer(seqs, return_tensors="pt", padding="max_length",
                                  max_length=BIO_MAX_LENGTH, truncation=True).input_ids.unsqueeze(1)

    def prompt_len(self, prompt):
        return len(self.english_tokenizer(prompt).input_ids)

    def bio_len(self, seq):
        return len(self.bio_tokenizer(seq).input_ids)

    def answer_ids(self, prompt, answer):
        base = self.english_tokenizer(prompt).input_ids
        full = self.english_tokenizer(prompt + answer).input_ids
        assert full[:len(base)] == base, ("answer tokenisation changes the prompt tokens", prompt[-40:], answer)
        return full[len(base):]

    def bio_hash(self, seq):
        return sha(self.tokenize_bio([seq]).numpy().tobytes())

    @staticmethod
    def _first_pad(tokens, pad_id):
        is_pad = tokens == pad_id
        assert bool(is_pad.any(dim=1).all()), "a prompt fills the whole English window"
        return is_pad.int().argmax(dim=1)

    def run_batch(self, prompts, seqs, max_new_tokens=MAX_NEW_TOKENS, generate=True):
        """Returns one dict per row: L_yes / L_no (log-softmax of the first answer token in the data's
        formatting), the no-space variants, the greedy generation, and token counts.
        """
        torch = self.torch
        B = len(prompts)
        assert len(seqs) == B
        eng = self.tokenize_english(prompts)
        bio = self.tokenize_bio(seqs)
        # no truncation, and room for the 64 inserted embeddings plus the answer
        for i, (p, s) in enumerate(zip(prompts, seqs)):
            pl, bl = self.prompt_len(p), self.bio_len(s)
            assert pl + RESAMPLED_LENGTH + max_new_tokens <= ENGLISH_MAX_LENGTH, (pl, p[-60:])
            assert bl <= BIO_MAX_LENGTH, (bl, len(s))
            assert int((eng[i] != self.pad_id).sum()) == pl, "english tokens truncated"
            assert int((bio[i, 0] != self.bio_pad_id).sum()) == bl, "bio tokens truncated"
        eng = eng.to(self.device)
        bio = bio.to(self.device)
        first_pad = self._first_pad(eng, self.pad_id)
        ar = torch.arange(B, device=self.device)
        with torch.no_grad():
            outs = self.model(multi_omics_tokens_ids=(eng, bio),
                              projection_english_tokens_ids=eng, projected_bio_embeddings=None)
            proj = outs["projected_bio_embeddings"]
            step0 = outs["logits"][ar, first_pad - 1].float()          # (B, V) next-token logits
            logp0 = torch.log_softmax(step0, dim=-1)
            rows = []
            for i in range(B):
                r = {"n_english_tokens": int(first_pad[i]),
                     "n_bio_tokens": int((bio[i, 0] != self.bio_pad_id).sum()),
                     "first_token_argmax": int(step0[i].argmax()),
                     "L_yes": float(logp0[i, self.answer_token_ids["Yes"][0]]),
                     "L_no": float(logp0[i, self.answer_token_ids["No"][0]]),
                     "L_yes_nospace": float(logp0[i, self.answer_token_ids_nospace["Yes"][0]])
                     if len(self.answer_token_ids_nospace["Yes"]) == 1 else None,
                     "L_no_nospace": float(logp0[i, self.answer_token_ids_nospace["No"][0]])
                     if len(self.answer_token_ids_nospace["No"]) == 1 else None}
                rows.append(r)
            if not generate:
                for r in rows:
                    r["raw_generation"] = None
                    r["generation_token_ids"] = None
                return rows
            # greedy loop of the released pipeline, over the batch
            tokens = eng.clone()
            cur = first_pad.clone()
            done = torch.zeros(B, dtype=torch.bool, device=self.device)
            gen = [[] for _ in range(B)]
            cur_logits = step0
            for _ in range(max_new_tokens):
                nxt = cur_logits.argmax(dim=-1)
                for i in range(B):
                    if done[i]:
                        continue
                    t = int(nxt[i])
                    if t == self.eos_id:
                        done[i] = True
                        continue
                    tokens[i, cur[i]] = t
                    gen[i].append(t)
                    cur[i] += 1
                    if int(cur[i]) >= ENGLISH_MAX_LENGTH:
                        done[i] = True
                if bool(done.all()):
                    break
                outs = self.model(multi_omics_tokens_ids=(tokens, bio),
                                  projection_english_tokens_ids=eng, projected_bio_embeddings=proj)
                cur_logits = outs["logits"][ar, cur - 1].float()
            for i in range(B):
                rows[i]["raw_generation"] = self.english_tokenizer.decode(gen[i])
                rows[i]["generation_token_ids"] = gen[i]
        return rows

    def teacher_forced_logprob(self, prompt, seq, answer):
        """General multi-token path: log p(answer tokens | prompt, DNA) with the projected DNA
        embeddings computed from the prompt alone (as at the pipeline's first step). Used only as a
        cross-check of the single-token fast path.
        """
        torch = self.torch
        ids = self.answer_ids(prompt, answer)
        eng = self.tokenize_english([prompt]).to(self.device)
        bio = self.tokenize_bio([seq]).to(self.device)
        fp = int(self._first_pad(eng, self.pad_id)[0])
        with torch.no_grad():
            outs = self.model(multi_omics_tokens_ids=(eng, bio),
                              projection_english_tokens_ids=eng, projected_bio_embeddings=None)
            proj = outs["projected_bio_embeddings"]
            full = eng.clone()
            full[0, fp:fp + len(ids)] = torch.tensor(ids, device=self.device)
            outs = self.model(multi_omics_tokens_ids=(full, bio),
                              projection_english_tokens_ids=eng, projected_bio_embeddings=proj)
            lp = torch.log_softmax(outs["logits"][0].float(), dim=-1)
            total = 0.0
            for k, t in enumerate(ids):
                total += float(lp[fp - 1 + k, t])
        return total, ids

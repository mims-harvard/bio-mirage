"""Builds the BioReason input conditions for one disease prediction query.

Genome conditions edit the reference and variant DNA sequences and keep the prompt text. They
shuffle the variant or both sequences (base composition is kept), replace the variant with the
reference, replace the variant with the variant of a query with a different disease label
(`swap_variant_donor`), drop the DNA, or give a fixed random sequence (`no_modality`). Text
conditions keep both sequences and remove or mask fields of the prompt text: the chromosome, the
pathway network, the gene list, the gene descriptions, the asterisk that marks the variant gene, and
the gene symbols. The appendix table "Text fields present in each text condition" describes the text
conditions drawn in Figures 2 and 3.
"""
from __future__ import annotations

import random
import re
from typing import Dict, Optional

from input_use.core import conditions as C

BASES = "ACGT"

CHROM_PREFIX = "chromosome number"
NETWORK_PREFIX = "network definition of the pathway"
GENELIST_PREFIX = "genes in the pathway"
_STRIP_PREFIXES = (NETWORK_PREFIX, GENELIST_PREFIX, CHROM_PREFIX)

STEM_GENE_RE = re.compile(r"effect of this (\S+) allele")


# --------------------------------------------------------------------------------------------- DNA

def scramble_seq(seq: str, rng: random.Random) -> str:
    """Composition-preserving shuffle (destroys positional/sequence info, keeps base composition)."""
    ch = list(seq)
    rng.shuffle(ch)
    return "".join(ch)


def neutral_sequence(length: int) -> str:
    """A fixed pseudo-random sequence (identical for every example) - a valid but information-free DNA
    input for the no-modality (M_0) regime, so the model must predict from the question text alone.
    """
    r = random.Random(20260609)
    return "".join(r.choice(BASES) for _ in range(max(1, length)))


# -------------------------------------------------------------------------------------------- text

def _lines(question: str):
    return question.splitlines()


def field_line(question: str, prefix: str) -> Optional[str]:
    """The first line starting with `prefix` (case-insensitive), or None."""
    for ln in _lines(question):
        if ln.strip().lower().startswith(prefix):
            return ln
    return None


def drop_field(question: str, prefix: str) -> str:
    """Remove the line starting with `prefix`; everything else, including blank lines, is preserved."""
    keep = [ln for ln in _lines(question) if not ln.strip().lower().startswith(prefix)]
    return "\n".join(keep).strip()


def strip_pathway(question: str) -> str:
    """Remove the pathway-network + gene-list + chromosome lines (keeps the question stem)."""
    keep = [ln for ln in _lines(question)
            if not ln.strip().lower().startswith(_STRIP_PREFIXES)]
    return "\n".join(keep).strip()


def stem_gene(question: str) -> Optional[str]:
    """The gene named in the question stem ('...effect of this tardbp allele...'). Matches all 1,449
    rows of the KEGG dataset.
    """
    m = STEM_GENE_RE.search(question)
    return m.group(1) if m else None


def _kegg_gene_symbols(question: str):
    """Gene symbols from the 'Genes in the pathway:' list (the symbol before each ';')."""
    syms = set()
    for line in _lines(question):
        if line.strip().lower().startswith(GENELIST_PREFIX):
            body = line.split(":", 1)[1] if ":" in line else line
            for part in body.split("|"):
                sym = part.split(";")[0].strip()
                if re.fullmatch(r"[A-Za-z0-9-]{2,}", sym):
                    syms.add(sym)
    return {s for s in syms if s.lower() not in {"q", "cxi"}}


def mask_genes(question: str) -> str:
    """mask every gene symbol in the question (gene list, network, and the stem 'this tardbp allele')
    with 'gene', removing the textual answer key while keeping the DNA + pathway structure.
    """
    q = question
    for s in sorted(_kegg_gene_symbols(question), key=len, reverse=True):
        q = re.sub(r"\b" + re.escape(s) + r"\b", "GENE", q)
    return q


def mask_text_key(question: str) -> str:
    """Remove the whole textual answer key: strip the pathway/gene-list lines (which carry the gene
    descriptions, e.g. 'tar DNA binding protein') and mask the gene symbols in the remaining stem.
    Leaves a gene/pathway-free question + the real DNA - the DNA-only condition.
    """
    syms = _kegg_gene_symbols(question)
    q = strip_pathway(question)
    for s in sorted(syms, key=len, reverse=True):
        q = re.sub(r"\b" + re.escape(s) + r"\b", "GENE", q)
    return q


def strip_descriptions(question: str) -> str:
    """Keep every gene symbol in the gene list, drop the free-text description after each ';'."""
    out = []
    for ln in _lines(question):
        if ln.strip().lower().startswith(GENELIST_PREFIX) and ":" in ln:
            head, body = ln.split(":", 1)
            syms = [p.split(";")[0].strip() for p in body.split("|")]
            out.append(f"{head}: " + " | ".join(s for s in syms if s))
        else:
            out.append(ln)
    return "\n".join(out).strip()


def remove_star_marker(question: str) -> str:
    """Remove the '*' that marks the variant gene inside the network string."""
    out = []
    for ln in _lines(question):
        if ln.strip().lower().startswith(NETWORK_PREFIX):
            out.append(ln.replace("*", ""))
        else:
            out.append(ln)
    return "\n".join(out).strip()


# -------------------------------------------------------------------------------------- assembly

def _payload(dna, q, dna_ch, path_ch, **extra) -> dict:
    return {"dna_sequences": list(dna), "question": q,
            "channels": {"dna": dna_ch, "pathway": path_ch}, **extra}


def make_conditions(reference: str, variant: str, question: str, seed: int = 0,
                    donor_variant: Optional[str] = None,
                    donor_example_id: Optional[str] = None) -> Dict[str, dict]:
    """Return {condition: payload} for one KEGG example."""
    rng = random.Random(seed)
    conds: Dict[str, dict] = {}

    # --- Exp 1: genome arms. `question` is passed through untouched in every one of these.
    conds[C.WT] = _payload([reference, variant], question, "wt", "full")
    conds[C.WT_REPEAT] = _payload([reference, variant], question, "wt", "full")
    conds[C.SCRAMBLE_VARIANT] = _payload([reference, scramble_seq(variant, rng)], question,
                                         "scramble_variant", "full")
    conds[C.SCRAMBLE] = _payload([scramble_seq(reference, rng), scramble_seq(variant, rng)],
                                 question, "scramble", "full")
    conds[C.REVERT_VARIANT_BLOCK] = _payload([reference, reference], question,
                                             "revert_variant_block", "full")
    if donor_variant is not None:
        conds[C.SWAP_VARIANT_DONOR] = _payload([reference, donor_variant], question,
                                               "swap_variant_donor", "full",
                                               donor_example_id=donor_example_id)
    conds[C.NO_DNA] = _payload([], question, "none", "full")
    neutral = neutral_sequence(len(variant))
    conds[C.NO_MODALITY] = _payload([neutral, neutral], question, "neutral", "full")

    # --- Exp 2: text arms. [reference, variant] is passed through untouched in every one of these.
    dna = [reference, variant]
    conds[C.NO_CHROMOSOME] = _payload(dna, drop_field(question, CHROM_PREFIX), "wt", "no_chromosome")
    conds[C.NO_NETWORK] = _payload(dna, drop_field(question, NETWORK_PREFIX), "wt", "no_network")
    conds[C.NO_GENELIST] = _payload(dna, drop_field(question, GENELIST_PREFIX), "wt", "no_genelist")
    conds[C.NO_DESCRIPTIONS] = _payload(dna, strip_descriptions(question), "wt", "no_descriptions")
    conds[C.NO_STAR_MARKER] = _payload(dna, remove_star_marker(question), "wt", "no_star_marker")
    conds[C.NO_GENE] = _payload(dna, mask_genes(question), "wt", "gene_masked")
    conds[C.NO_TEXTKEY] = _payload(dna, mask_text_key(question), "wt", "textkey_removed")
    # Leave-one-in (stem gene only, all three labelled lines gone). Byte-identical to what a
    # separate `gene_symbol_only` arm would produce, so it is kept under its existing name.
    conds[C.NO_PATHWAY] = _payload(dna, strip_pathway(question), "wt", "stripped")
    return conds

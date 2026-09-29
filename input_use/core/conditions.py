"""Names of the input conditions stored in each record's `condition` field, shared by all models.

Covers the unmodified input (`wt`) and removal of the biological input (`no_modality`); the
BioReason-Pro sequence shuffles, text channel removals, deep mutational scanning conditions, residue
permutations and protein pair conditions for evidence conflicts; the C2S-Scale gene order, gene set
and DEG removal conditions and the single cell text conditions; and the BioReason disease prediction
conditions (DNA shuffles, donor variants, removal of the DNA blocks or of prompt text fields).
Helper functions build the per-seed and per-dose names.
"""

# Shared
WT = "wt"                         # full, unperturbed input (reference)
NO_MODALITY = "no_modality"       # remove the biological modality entirely -> M_0 / language prior

# Protein (BioReason-Pro): the modality enters as a dense ESM3 embedding (sequence) plus the
# symbolic InterPro/GO text channel; conditions can target either.
MUT_SALIENT = "mut_salient"       # alanine-scan salient residues S+ (UniProt functional sites);
                                  # symbolic channel re-derived on the mutated sequence
MUT_CONTROL = "mut_control"       # alanine-scan matched control residues S- (same count,
                                  # AA-matched); symbolic channel re-derived on the mutated sequence
                                  # The two sequence-shuffle conditions are named by which channels
                                  # are updated to reflect the shuffle.
SHUFFLE_ESM3_ONLY = "shuffle_esm3_only"  # the shuffled sequence updates only the ESM3 embedding; the
                                  # GO-GPT/InterPro text stays frozen on WT (not updated).
                                  # Isolates reliance on the dense ESM3 embedding: output ~ WT
                                  # => embedding ignored.
SHUFFLE_ESM3_AND_GOGPT = "shuffle_esm3_and_gogpt"  # the shuffled sequence updates both the ESM3 embedding and
                                  # the GO-GPT/InterPro text (both re-derived on the shuffle) -
                                  # full modality destruction; accuracy can survive only if the
                                  # embedding helps.
SWAP = "swap"                     # ESM3 embedding of protein A + GO-GPT/InterPro text of a
                                  # different protein B. If the prediction matches B's GO (not A's),
                                  # the model follows the symbolic channel rather than its dense
                                  # embedding.
NO_SYMBOLIC = "no_symbolic"       # drop InterPro+GO text, keep real sequence embedding
NO_NAME = "no_name"               # strip the protein-name InterPro entry (answer leak), keep rest
                                  # (re-derived on WT, then the full-length name entry removed)

# Isolating GO-GPT requires holding InterPro frozen on WT while only GO-GPT is re-derived, which is
# what SHUFFLE_ESM3_GOGPT does.
SHUFFLE_ESM3_GOGPT = "shuffle_esm3_gogpt"  # shuffled sequence updates the ESM3 embedding and GO-GPT;
                                  # InterPro stays frozen on WT. The clean test of whether
                                  # GO-GPT carries sequence information the model consumes.
INTERPRO_ONLY = "interpro_only"   # embedding shuffled, GO-GPT dropped, InterPro frozen on WT - is
                                  # the InterPro block alone sufficient to recover the answer?
GOGPT_ONLY = "gogpt_only"         # embedding shuffled, InterPro dropped, GO-GPT frozen on WT - is
                                  # the GO-GPT block alone sufficient?

# --- The two arms that complete the {ESM3 intact, ESM3 shuffled} x {text channel kept} cross ------
# INTERPRO_ONLY and GOGPT_ONLY above were both run with the embedding destroyed, so the ladder
# supports a wild-type-vs-shuffled ESM3 comparison only at its two ends: with both text channels
# present (WT vs SHUFFLE_ESM3_ONLY) and with both removed (NO_SYMBOLIC vs NO_MODALITY).
INTERPRO_ONLY_WT_ESM3 = "interpro_only_wt_esm3"  # ESM3 intact, GO-GPT dropped, InterPro frozen on WT
GOGPT_ONLY_WT_ESM3 = "gogpt_only_wt_esm3"        # ESM3 intact, InterPro dropped, GO-GPT on WT

# The proteome-scale Experiment 1 battery, in figure order (ceiling -> floors).
PROTEIN_SHUFFLE_LADDER = [WT, SHUFFLE_ESM3_ONLY, SHUFFLE_ESM3_GOGPT, SHUFFLE_ESM3_AND_GOGPT,
                          INTERPRO_ONLY, GOGPT_ONLY, INTERPRO_ONLY_WT_ESM3, GOGPT_ONLY_WT_ESM3,
                          NO_SYMBOLIC, NO_MODALITY]

# --- Experiment 2: matched protein-context conflict ---------------------------------------------
# `SWAP` above pairs a protein with a random other protein, which shows only that the model follows
# whichever text it is given.
PAIR_ALIGNED_A = "aligned_a"      # (E_A, C_A) - baseline: does the model get A's property right?
PAIR_ALIGNED_B = "aligned_b"      # (E_B, C_B) - baseline for B
PAIR_CONFLICT_A = "conflict_a"    # (E_A, C_B) - A's embedding, B's InterPro/GO-GPT context
PAIR_CONFLICT_B = "conflict_b"    # (E_B, C_A) - the reciprocal, controlling for pair asymmetry

# --- Which text channel drives the override?
PAIR_CONFLICT_INTERPRO_A = "conflict_interpro_a"  # ESM3=A, GO-GPT=A, InterPro=B
PAIR_CONFLICT_INTERPRO_B = "conflict_interpro_b"  # ESM3=B, GO-GPT=B, InterPro=A
PAIR_CONFLICT_GOGPT_A = "conflict_gogpt_a"        # ESM3=A, GO-GPT=B, InterPro=A
PAIR_CONFLICT_GOGPT_B = "conflict_gogpt_b"        # ESM3=B, GO-GPT=A, InterPro=B

PAIR_CONDS = [PAIR_ALIGNED_A, PAIR_ALIGNED_B, PAIR_CONFLICT_A, PAIR_CONFLICT_B,
              PAIR_CONFLICT_INTERPRO_A, PAIR_CONFLICT_INTERPRO_B,
              PAIR_CONFLICT_GOGPT_A, PAIR_CONFLICT_GOGPT_B]
# (embedding+GO-GPT source, conflicting-channel source) per conflict arm, keyed 'a'/'b' for the two
# pair members. The scorer reads which side "won" from this rather than from the condition name.
PAIR_CONFLICT_LAYOUT = {
    PAIR_CONFLICT_A:          ("a", "b"),
    PAIR_CONFLICT_B:          ("b", "a"),
    PAIR_CONFLICT_INTERPRO_A: ("a", "b"),
    PAIR_CONFLICT_INTERPRO_B: ("b", "a"),
    PAIR_CONFLICT_GOGPT_A:    ("a", "b"),
    PAIR_CONFLICT_GOGPT_B:    ("b", "a"),
}
PAIR_CONFLICT_FAMILIES = {
    "both channels swapped":  [PAIR_CONFLICT_A, PAIR_CONFLICT_B],
    "InterPro text only":     [PAIR_CONFLICT_INTERPRO_A, PAIR_CONFLICT_INTERPRO_B],
    "GO-GPT only":            [PAIR_CONFLICT_GOGPT_A, PAIR_CONFLICT_GOGPT_B],
}


DMS_CORRECT = "dms_correct"       # sequence = variant v, symbolic_sequence = variant v. The
                                  # reference arm: every input belongs to the same variant.
DMS_SHUFFLED = "dms_shuffled"     # sequence = variant pi(v) (a seeded derangement within the
                                  # assay), symbolic_sequence = variant v.
DMS_NO_MODALITY = "dms_no_modality"  # sequence = composition-preserving scramble, symbolic channel
                                  # unchanged. The floor: does the ESM3 channel matter at all,
                                  # before asking whether the correct one matters.

DMS_POSITIVE_CONTROL = "dms_positive_control"

PROTEIN_DMS_CONDS = [DMS_CORRECT, DMS_SHUFFLED, DMS_NO_MODALITY, DMS_POSITIVE_CONTROL]


def shuffle_seed_cond(condition: str, seed: int) -> str:
    """Per-seed name for a shuffle arm (`shuffle_esm3_only` -> `shuffle_esm3_only_s1`)."""
    return condition if seed == 0 else f"{condition}_s{seed}"

# Single-cell (Cell2Sentence): the modality is the rank-ordered gene "cell sentence" (text).
TOP_DEG_DROPOUT = "top_deg_dropout"             # graded dropout of marker genes S+ (k-series only)
MATCHED_NONDEG_DROPOUT = "matched_nondeg_dropout"  # graded dropout of matched control genes S- (k-series only)

# Stress test 1, split into two single-variable-at-a-time conditions over the same visible window
# (top-N genes by expression -- N=200 by default): does the model need the right genes, the right
# order, or both?
SCRAMBLE_RANK = "scramble_rank"           # same visible gene set as wt, order permuted within it
SCRAMBLE_GENESET = "scramble_geneset"  # different (random) gene set of the same size, own natural
                                           # rank order preserved for whichever genes are shown

DEG_DROPOUT_KS = [1, 2, 5, 10, 25]

# Dose axis as a fraction of the cell's own DEG set, not an absolute count.
DEG_DROPOUT_FRACTIONS = [0.10, 0.25, 0.50, 0.75, 1.00]


def top_deg_dropout_k(k: int) -> str:
    return f"{TOP_DEG_DROPOUT}_k{k}"


def matched_nondeg_dropout_k(k: int) -> str:
    return f"{MATCHED_NONDEG_DROPOUT}_k{k}"


def _pct(frac: float) -> int:
    return int(round(frac * 100))


def top_deg_dropout_pct(frac: float) -> str:
    return f"{TOP_DEG_DROPOUT}_p{_pct(frac)}"


def matched_nondeg_dropout_pct(frac: float) -> str:
    return f"{MATCHED_NONDEG_DROPOUT}_p{_pct(frac)}"


# Two further dose-matched deletion arms, so the dose-response curve has the three-way contrast the
# draft's Experiment 1 asks for ("Top vs random vs bottom deletion") rather than only
# top-vs-control:  top highest-salience DEGs      -- if importance drives the model, this breaks it
# fastest bottom lowest-salience DEGs       -- same gene class (still DEGs passing both cutoffs),
# just the weakest ones.
TISSUE_NONE = "tissue_none"                 # no tissue stated (reference)
TISSUE_TRUE = "tissue_true"                 # the cell's actual tissue
TISSUE_FALSE = "tissue_false"               # a different tissue where this type is equally common
TISSUE_ONLY = "tissue_only"                 # false tissue + the information-free cell sentence -> yields
                                            # y_text, the metadata-only prediction that the
TISSUE_FABRICATED = "tissue_fabricated"     # an invented batch identifier: zero mutual information with
                                            # cell type by construction.
TISSUE_CONDS = [TISSUE_NONE, TISSUE_TRUE, TISSUE_FALSE, TISSUE_ONLY, TISSUE_FABRICATED]

# --- Experiment 2: ontology-grounded transcriptome-text conflict ----------------------------------
# The cell sentence is identical across all conflict conditions; only the accompanying
# ontology-derived description changes.
TC_VAL_IDENTIFY = "tc_val_identify"        # gate 1: does t_B alone name B, against the full option list?
TC_VAL_DISCRIMINATE = "tc_val_discriminate"  # gate 2: does t_B alone pick B over A in a two-way choice?
                                           # Gate 1 alone admits descriptions equally compatible with A,
                                           # which would not be conflict-bearing.
TC_XA = "tc_xA"                            # cell sentence alone (baseline)
TC_XA_TA = "tc_xA_tA"                      # + description consistent with A (agreement control)
TC_XA_TB = "tc_xA_tB"                      # + description of B  <- the conflict
TC_XA_TNEUTRAL = "tc_xA_tneutral"          # + facts true of both (length/register/vocabulary control)
TC_TB_ONLY = "tc_tB_only"                  # description of B + information-free sentence -> y_text,
TC_XA_TB_REDUCTIO = "tc_xA_tB_reductio"    # B from the most distant lineage available for this A: if the
                                           # model follows even this, no ontology stratification is
                                           # needed
TC_XA_TB_2WAY = "tc_xA_tB_2way"            # conflict with only {A, B} offered.
TEXT_CONFLICT_CONDS = [TC_XA, TC_XA_TA, TC_XA_TB, TC_XA_TNEUTRAL, TC_TB_ONLY, TC_XA_TB_REDUCTIO,
                       TC_XA_TB_2WAY]

BOTTOM_DEG_DROPOUT = "bottom_deg_dropout"
RANDOM_DROPOUT = "random_dropout"


def bottom_deg_dropout_pct(frac: float) -> str:
    return f"{BOTTOM_DEG_DROPOUT}_p{_pct(frac)}"


def random_dropout_pct(frac: float) -> str:
    return f"{RANDOM_DROPOUT}_p{_pct(frac)}"


# Single-cell condition batteries -- lets `prepare`/runners generate/run only a subset (e.g.
STRESS_TEST_1_CONDS = [SCRAMBLE_RANK, SCRAMBLE_GENESET]
STRESS_TEST_2_CONDS = ([top_deg_dropout_k(k) for k in DEG_DROPOUT_KS] +
                      [matched_nondeg_dropout_k(k) for k in DEG_DROPOUT_KS])
STRESS_TEST_2_PCT_CONDS = ([top_deg_dropout_pct(f) for f in DEG_DROPOUT_FRACTIONS] +
                           [matched_nondeg_dropout_pct(f) for f in DEG_DROPOUT_FRACTIONS] +
                           # bottom at 100% deletes the same set as top at 100% (all of S+), so it
                           # is omitted rather than run as a duplicate of an existing condition.
                           [bottom_deg_dropout_pct(f) for f in DEG_DROPOUT_FRACTIONS if f < 1.0] +
                           [random_dropout_pct(f) for f in DEG_DROPOUT_FRACTIONS])

# Single-cell text reliance: a second single-cell axis, orthogonal to the marker-gene dropout above.
TEXT_FULL = "text_full"                                  # tissue+sex+development_stage+donor_id, wt cell sentence
TEXT_DROP_DONOR = "text_drop_donor"                        # tissue+sex+development_stage, wt cell sentence
TEXT_DROP_DONOR_STAGE = "text_drop_donor_stage"            # tissue+sex, wt cell sentence
TEXT_DROP_DONOR_STAGE_SEX = "text_drop_donor_stage_sex"    # tissue only, wt cell sentence
TEXT_FULL_NO_MODALITY = "text_full_no_modality"            # tissue+sex+development_stage+donor_id, but paired with the
                                                            # no_modality cell sentence/ embedding -- the diagnostic: does
                                                            # stated metadata alone substitute for a genuinely informative
                                                            # signal?
TEXT_POSITIVE_CONTROL = "text_positive_control"            # the ground-truth cell type stated directly in the
                                                            # prompt/question (nothing else -- no tissue/sex/stage/donor),
                                                            # wt cell sentence.

TEXT_RELIANCE_CONDS = [TEXT_FULL, TEXT_DROP_DONOR, TEXT_DROP_DONOR_STAGE, TEXT_DROP_DONOR_STAGE_SEX,
                       TEXT_FULL_NO_MODALITY, TEXT_POSITIVE_CONTROL]  # + plain `wt` (already baseline machinery) = the full 7-point
                       # suite

# DNA - KEGG disease-pathway (BioReason): modality = the [reference, variant] pair (dense, via
# Evo2); the question carries a symbolic pathway/gene channel.
SCRAMBLE = "scramble"
NO_PATHWAY = "no_pathway"         # strip the pathway-network + gene-list lines, keep the DNA + stem
NO_GENE = "no_gene"
NO_TEXTKEY = "no_textkey"         # remove the whole textual answer key: strip the pathway/gene list
                                  # and mask gene symbols + their descriptions in the stem ->
                                  # DNA-only input.

# --- KEGG battery (released BioReason DNA checkpoints) ------------------------------------------
# The three DNA experiments all run on KEGG, because that is the only task the released checkpoints
# (dna_sft and dna_rl in checkpoints.json) were trained for.
WT_REPEAT = "wt_repeat"                       # identical input, generated a second time. The decoding noise
                                              # floor: without it an "N% of answers changed" number from any
                                              # other arm is uninterpretable.
SCRAMBLE_VARIANT = "scramble_variant"         # composition-preserving shuffle of the variant block only;
                                              # the reference block is untouched. Sharper than scramble,
                                              # which destroys both and so also removes the locus.
SCRAMBLE_BOTH = SCRAMBLE                      # alias: scramble already shuffles both blocks
REVERT_VARIANT_BLOCK = "revert_variant_block"
SWAP_VARIANT_DONOR = "swap_variant_donor"     # variant block <- a different example's variant block (seeded
                                              # donor map, no self-pairs). The 050dna.tex `S_j` arm: if the
                                              # answer follows the donor, the genome is read.
NO_DNA = "no_dna"                             # both DNA blocks omitted from the prompt entirely. Distinct
                                              # from NO_MODALITY, which substitutes a fixed information-free
                                              # sequence and so keeps the prompt shape.

# Exp 2 field ablations.
NO_CHROMOSOME = "no_chromosome"       # drop the "Chromosome Number:" line
NO_NETWORK = "no_network"             # drop the "Network Definition of the pathway:" line
NO_GENELIST = "no_genelist"           # drop the "Genes in the pathway:" line
NO_DESCRIPTIONS = "no_descriptions"   # keep gene symbols in the gene list, drop the text after each
# ';'
NO_STAR_MARKER = "no_star_marker"     # remove the `*` that marks the variant gene inside the
                                      # network string (present in 1,357 of 1,449 rows) -- the model
                                      # can still see every gene, just not which one is mutated The
                                      # leave-one-in arm (stem gene only, all three labelled lines
                                      # dropped) is NO_PATHWAY above: it is byte-identical to what a
                                      # separate `gene_symbol_only` would produce, and it already
                                      # carries a prior

KEGG_GENOME_CONDS = [WT, WT_REPEAT, SCRAMBLE_VARIANT, SCRAMBLE, REVERT_VARIANT_BLOCK,
                     SWAP_VARIANT_DONOR, NO_DNA, NO_MODALITY]
KEGG_TEXT_CONDS = [NO_CHROMOSOME, NO_NETWORK, NO_GENELIST, NO_DESCRIPTIONS, NO_STAR_MARKER,
                   NO_GENE, NO_TEXTKEY, NO_PATHWAY]
KEGG_CONDS = KEGG_GENOME_CONDS + KEGG_TEXT_CONDS       # 16 arms; `wt` is shared by both experiments

REVERT_VARIANT = "revert_variant"  # revert the variant base to reference (remove the mutation) ->
# expected benign (salient S+)
EDIT_CONTROL = "edit_control"      # change a matched random non-variant base, variant intact ->
# expected unchanged (control S-)

# DNA - sequence tasks (ChatNT native Yes/No tasks: splice sites, promoters, enhancers, ...). The
# only channel is the DNA; conditions perturb the single sequence with edits that should flip a
# "Yes".
DINUC_SHUFFLE = "dinuc_shuffle"          # dinucleotide-preserving shuffle (keeps 1/2-mer composition,
                                         # destroys motif) - survival here => k-mer-composition
                                         # shortcut
REVERSE_COMPLEMENT = "reverse_complement"  # reverse-complement strand flip (directional motifs should
# change)

DESCRIPTIONS = {
    WT: "full unperturbed input (reference)",
    NO_MODALITY: "biological modality removed entirely (M_0 / language-prior baseline)",
    MUT_SALIENT: "alanine-scan of UniProt functional-site residues (salient S+); both channels re-derived on the mutant",
    MUT_CONTROL: "alanine-scan of AA-matched non-functional residues (control S-); both channels re-derived on the mutant",
    SHUFFLE_ESM3_ONLY: "sequence shuffled; ONLY the ESM3 embedding updated, GO-GPT/InterPro text frozen on WT (isolates ESM3-embedding reliance)",
    SHUFFLE_ESM3_AND_GOGPT: "sequence shuffled; BOTH the ESM3 embedding and the GO-GPT/InterPro text updated on the shuffle (full modality destruction). NOTE: a shuffle matches no InterPro entry, so this arm also ablates InterPro - use shuffle_esm3_gogpt to isolate GO-GPT",
    SHUFFLE_ESM3_GOGPT: "sequence shuffled; ESM3 embedding AND GO-GPT updated on the shuffle, InterPro FROZEN on WT (isolates whether GO-GPT carries sequence information the model uses)",
    PAIR_ALIGNED_A: "matched pair, aligned: protein A's ESM3 embedding + protein A's own context (validation gate)",
    PAIR_ALIGNED_B: "matched pair, aligned: protein B's ESM3 embedding + protein B's own context (validation gate)",
    PAIR_CONFLICT_A: "matched pair, CONFLICT: protein A's ESM3 embedding + protein B's InterPro/GO-GPT context",
    PAIR_CONFLICT_B: "matched pair, CONFLICT: protein B's ESM3 embedding + protein A's InterPro/GO-GPT context",
    INTERPRO_ONLY: "ESM3 embedding shuffled, GO-GPT dropped, InterPro frozen on WT (is the InterPro block alone sufficient?)",
    GOGPT_ONLY: "ESM3 embedding shuffled, InterPro dropped, GO-GPT frozen on WT (is the GO-GPT block alone sufficient?)",
    INTERPRO_ONLY_WT_ESM3: "ESM3 embedding INTACT, GO-GPT dropped, InterPro frozen on WT (intact-ESM3 reference for interpro_only)",
    GOGPT_ONLY_WT_ESM3: "ESM3 embedding INTACT, InterPro dropped, GO-GPT on WT (intact-ESM3 reference for gogpt_only)",
    SWAP: "ESM3 embedding of protein A + GO-GPT/InterPro text of a different protein B (does the prediction match A's GO or B's GO?)",
    NO_SYMBOLIC: "InterPro+GO text dropped; only the ESM3 sequence embedding remains",
    NO_NAME: "protein-name InterPro entry stripped (removes the literal answer leak)",
    SCRAMBLE_RANK: "stress test 1a: SAME visible gene set as wt, order permuted within it -- isolates "
                   "sensitivity to rank order alone",
    SCRAMBLE_GENESET: "stress test 1b: visible gene set swapped for a different (random) same-size "
                        "set, natural order preserved for whichever genes are shown -- isolates "
                        "sensitivity to gene identity alone",
    TEXT_FULL: "tissue+sex+development_stage+donor_id in the prompt/question text, wt cell sentence",
    TEXT_DROP_DONOR: "tissue+sex+development_stage in the prompt/question text (donor_id dropped), wt cell sentence",
    TEXT_DROP_DONOR_STAGE: "tissue+sex in the prompt/question text (donor_id+development_stage dropped), wt cell sentence",
    TEXT_DROP_DONOR_STAGE_SEX: "tissue only in the prompt/question text, wt cell sentence",
    TEXT_FULL_NO_MODALITY: "tissue+sex+development_stage+donor_id in the prompt/question text, but paired "
                           "with the no_modality cell sentence/embedding (decisive: does stated metadata "
                           "alone substitute for a genuinely informative signal?)",
    TEXT_POSITIVE_CONTROL: "the ground-truth cell type stated directly in the prompt/question text (no "
                           "other metadata), wt cell sentence -- sanity-check ceiling, not a rung on the "
                           "field-ablation ladder",
    NO_PATHWAY: "pathway/gene names stripped from the question; only the DNA sequences remain",
    REVERT_VARIANT: "variant base reverted to reference (mutation removed) - expected benign (salient S+)",
    EDIT_CONTROL: "a matched random non-variant base changed, variant intact - expected unchanged (control S-)",
    WT_REPEAT: "identical input generated a second time - the decoding noise floor every other arm's "
               "answer-change rate is read against",
    SCRAMBLE_VARIANT: "variant DNA block shuffled (composition-preserving), reference block and question "
                      "text untouched",
    REVERT_VARIANT_BLOCK: "variant DNA block replaced by the reference block, so both DNA inputs are "
                          "identical - full-length real DNA, variant information removed",
    SWAP_VARIANT_DONOR: "variant DNA block replaced by a different example's variant block (seeded donor "
                        "map, no self-pairs); does the answer follow the donor?",
    NO_DNA: "both DNA blocks omitted from the prompt - strict text-only (unlike no_modality, which keeps "
            "the prompt shape by substituting a fixed information-free sequence)",
    NO_CHROMOSOME: "the 'Chromosome Number:' line dropped; DNA and every other text field intact",
    NO_NETWORK: "the 'Network Definition of the pathway:' line dropped; DNA and every other text field intact",
    NO_GENELIST: "the 'Genes in the pathway:' line dropped; DNA and every other text field intact",
    NO_DESCRIPTIONS: "gene SYMBOLS kept in the gene list, the free-text description after each ';' dropped "
                     "- separates 'the symbol is the answer key' from 'the description is'",
    NO_STAR_MARKER: "the '*' marking the variant gene inside the network string removed (it is present in "
                    "1,415 of 1,449 KEGG rows); every gene still visible, but not which one is mutated",
    SCRAMBLE: "both reference+variant sequences shuffled (DNA destroyed, question text frozen)",
    NO_GENE: "every gene symbol masked to 'GENE' in all three places it appears (network string, gene "
             "list, question stem); the free-text gene DESCRIPTIONS deliberately survive",
    NO_TEXTKEY: "the whole textual answer key removed: all three labelled lines stripped AND the gene "
                "symbol masked in the remaining stem, leaving real DNA and no gene name anywhere",
}
DESCRIPTIONS.update({
    top_deg_dropout_k(k): f"graded dropout: top-{k} marker genes (S+) removed outright, mimicking scRNA-seq "
                          f"capture dropout (k={k} of up to {max(DEG_DROPOUT_KS)})"
    for k in DEG_DROPOUT_KS
})
DESCRIPTIONS.update({
    matched_nondeg_dropout_k(k): f"graded dropout: top-{k} matched control genes (S-) removed outright "
                                 f"(k={k} of up to {max(DEG_DROPOUT_KS)})"
    for k in DEG_DROPOUT_KS
})

# --- Experiment 4: residue-level grounding (ESM3 embedding-row permutation) ---------------------
# Experiments 1-3 all perturb the sequence, which necessarily perturbs the symbolic channel too and
# confounds "the model ignores ESM3" with "the model lost its InterPro/GO-GPT context".
PERMUTE_RESIDUES = "permute_residues"       # within-protein permutation of the residue rows
REVERSE_RESIDUES = "reverse_residues"       # order flipped; local adjacency preserved
MEAN_POOL_RESIDUES = "mean_pool_residues"   # every residue row <- this protein's own mean row
NO_ESM3 = "no_esm3"                         # the protein block dropped from the prompt entirely
NO_PPI_TEXT = "no_ppi_text"                 # ppi task only: partner names dropped, ESM3 intact
WT_REPEAT = "wt_repeat"
ROLL_RESIDUES = "roll_residues"


def roll_residues_k(k: int) -> str:
    """Cyclic shift of the residue rows by k (`roll_residues_k10`)."""
    return f"{ROLL_RESIDUES}_k{k}"


RESIDUE_PERM_SEEDS = [0, 1, 2, 3, 4]
RESIDUE_PERM_CONDS = [shuffle_seed_cond(PERMUTE_RESIDUES, s) for s in RESIDUE_PERM_SEEDS]

# The arms generated. `mean_pool_residues` is not optional: it is the dynamic-range denominator.
RESIDUE_CONDS = ([WT, WT_REPEAT] + RESIDUE_PERM_CONDS
                 + [roll_residues_k(10), MEAN_POOL_RESIDUES, NO_ESM3])
RESIDUE_PPI_EXTRA_CONDS = [NO_PPI_TEXT]


DESCRIPTIONS.update({
    PERMUTE_RESIDUES: "ESM3 per-residue embedding rows permuted WITHIN the protein; sequence, "
                      "InterPro, GO-GPT and prompt byte-identical. Destroys residue-to-index "
                      "correspondence while keeping every vector a real ESM3 vector",
    REVERSE_RESIDUES: "residue rows reversed - same multiset of vectors as the permutation, but "
                      "local neighbourhoods preserved (reverse ~ wt while permute << wt would mean "
                      "the model uses local structure, not absolute indexing)",
    MEAN_POOL_RESIDUES: "every residue row replaced by this protein's own mean residue vector: "
                        "residue-level detail destroyed, protein-level signal and token count kept",
    NO_ESM3: "the protein content block omitted entirely (strict no-input arm). The ONLY arm whose "
             "prompt is not byte-identical to wt - it also loses the literal 'Protein: ' prefix",
    NO_PPI_TEXT: "PPI task only: the interaction-partner names dropped from the symbolic channel, "
                 "ESM3 intact - does the model need to know the partner at all?",
    WT_REPEAT: "identical input, generated again in a different batch. At temperature 0 any output "
               "difference is vLLM continuous-batching nondeterminism: the noise floor that an "
               "'outputs changed under permutation' rate must be read against",
    ROLL_RESIDUES: "residue rows cyclically shifted by k",
})
DESCRIPTIONS.update({
    shuffle_seed_cond(PERMUTE_RESIDUES, s): DESCRIPTIONS[PERMUTE_RESIDUES] + f" (replicate seed {s})"
    for s in RESIDUE_PERM_SEEDS if s != 0
})
DESCRIPTIONS[roll_residues_k(10)] = "residue rows cyclically shifted by 10 (claims should move by -10)"


# DNA-specific wording for the reused shared conditions (score/readme pick modality-appropriate
# text).
DNA_DESCRIPTIONS = {
    SCRAMBLE: "both reference+variant sequences shuffled (DNA destroyed, question text frozen)",
    NO_MODALITY: "the DNA sequences replaced with a fixed information-free sequence (M_0: predict from text alone)",
}

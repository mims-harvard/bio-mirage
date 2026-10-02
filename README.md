# When Do Biological Reasoning Models Use Their Biological Inputs?

**Authors**
* Ada Fang
* Nikitha Thoduguli
* Lukas Fesser
* Hanlin Zhang
* Sham Kakade
* Marinka Zitnik

[Preprint](https://arxiv.org/abs/2610.00898) | [Website](https://zitniklab.hms.harvard.edu/biomirage)

Biological reasoning models use post-training to connect a large language model to biological
inputs, which include representations from a biological foundation model and biological text such
as gene and pathway descriptions, functional annotations, and gene lists. Benchmark accuracy of these
models is taken as evidence that the language model reasons over its biological inputs. Predictive
annotations in the text can support accurate answers even when the sequence or transcriptomic
representation contributes little, so benchmark accuracy alone does not identify which input drives
a prediction. We test which biological inputs contribute to the predictions of six biological
reasoning models across DNA, protein, and single cell tasks. We perturb one biological input while
holding the query and the other inputs fixed, construct evidence conflicts that pair the foundation
model representation of one genome, protein, or cell with the text of another, fit linear probes to
the representations the language model receives, and compare reasoning traces with the biological
inputs.

This repository contains the code that builds the question sets, runs each model on its released
checkpoints, scores the outputs, and produces the figures and tables in the paper. Model outputs are
distributed separately, one JSON record per query and condition. Set `INPUT_USE_RESULTS_DIR` to
point at them.

## Layout

    input_use/                     installable package
        core/                      record schema, condition names, bootstrap, scorers
        metrics/                   label matching, IA-weighted F_max, GO and Cell Ontology, MCQ parsing
        modalities/                question sets, perturbations, evidence conflicts, probes
        models/                    one runner per evaluated model
        data/                      HGNC ribosomal protein list; ontologies are downloaded here
    figures/                       one script per paper figure, plus shared panels and style
    analysis/                      numbers quoted in the text and appendix, bootstrap intervals
    experiments/
        auxiliary_supervision/     data exports, generation and scoring for the auxiliary target
    setup/                         ontology download and the DNA runtime environment

## Install

    pip install -e ".[scoring]"
    bash setup/fetch_ontologies.sh          # go-basic.obo, cl-basic.obo, cl-full.obo

The probe numbers were produced with scikit-learn 1.9.0, numpy 2.5.1 and scipy 1.18.0, and the
pin in `pyproject.toml` holds them there: under scikit-learn 1.5.2 the pooled protein within-family
AUROC reads 0.800 instead of 0.822, the difference concentrated in the smallest families.

Paths are environment variables, read in `input_use/core/config.py`:

| Variable | Holds | Default |
| --- | --- | --- |
| `INPUT_USE_RESULTS_DIR` | released model outputs, laid out as below | `results/` in this repository |
| `INPUT_USE_OUTPUT_DIR` | figures and analysis outputs | `outputs/` |
| `INPUT_USE_DATA_DIR` | ontologies, reference annotations, atlases, the KEGG split CSVs (`kegg/`) | `input_use/data/` |
| `INPUT_USE_MODELS_ROOT` | checkouts of the evaluated models | the directory that contains this repository |
| `INPUT_USE_DNA_REPO`, `INPUT_USE_BRP_REPO`, `INPUT_USE_CELLWHISPERER_REPO` | one model checkout each | `BioReason`, `BioReason-Pro`, `cellwhisperer` under `INPUT_USE_MODELS_ROOT` |
| `INPUT_USE_HOME` | the `input_use` package directory | located from `config.py` |
| `INPUT_USE_CHECKPOINTS` | model and dataset ids (see below) | `checkpoints.json` in this repository |

`input_use/core/paths.py` names every location under `INPUT_USE_RESULTS_DIR` that the code reads or
writes:

    dna/bioreason/
        perturbations/                    examples, records_{rl,sft}.jsonl, metrics, pairs.json, evo2_pooled.npz
        text_conditions/                  the text conditions with intact and shuffled Evo2
        genome_dependent/                 the 165 genome-dependent queries and the Evo2 probe on them
        evidence_conflicts/               Evo2 of one query with the text of another
        linear_probes/                    Evo2 probes before and after the BioReason projection
        auxiliary_supervision/            training exports (windows_257bp, windows_257bp_ref_var_target,
                                          windows_2048bp, windows_2048bp_ref_var_target) and generations/
        network_split.json                query id -> split of the pathway network split
    dna/chatnt/                           ChatNT data, predictions and results
    protein/bioreason_pro/
        perturbations_{rl,sft}/           GO term prediction on the proteome with ESM3 intact and shuffled
        evidence_conflicts_{rl,sft}/      ESM3 of one protein with the text of another
        evidence_conflicts_shuffled_esm3_{rl,sft}/
        linear_probes/                    ESM3 probes before and after the BioReason-Pro projection
    protein/prot2text_v2/                 Prot2Text-V2 runs and results
    protein/esm3_cache/                   ESM3 representations, one file per sequence
    single_cell/c2s_scale/
        deg_removal/<atlas>/              question set and records per model, DEG removal conditions
        deg_removal_summary.json          accuracy per atlas, model and condition
        random_expressed/                 genes resampled from all expressed genes
        evidence_conflicts/               cell sentence of one cell with the description of another
        rationales/<atlas>/               rationale prompts and generations
        rationale_removal/                referenced genes under DEG removal
        rationale_order/                  referenced genes with the cell sentence in random order
    single_cell/cellwhisperer/            CellWhisperer cells, embeddings, scores and analysis
    evidence_conflict_filters.csv         filtered Prot2Text-V2 and CellWhisperer conflict rates

Each evaluated model has its own dependencies. The runners in `input_use/models/` are executed
inside that model's environment and import only its released package.

## Models and data

Model and dataset identifiers are not stored in the code. Every Hugging Face id the code loads is
read from `checkpoints.json` at the repository root, which lists the public checkpoints and datasets
the paper evaluates. Set `INPUT_USE_CHECKPOINTS` to another file to load different ids.

| Model | Keys in `checkpoints.json` |
| --- | --- |
| BioReason | `dna_sft`, `dna_rl`, `dna_text_base`, `dna_encoder`; data `kegg_data` |
| ChatNT | `chatnt`, `chatnt_data`, `chatnt_repo` |
| BioReason-Pro | `protein_sft`, `protein_rl`, `protein_encoder` |
| Prot2Text-V2 | `prot2text`, `prot2text_esm`, `prot2text_data`, `prot2text_repo`, `llama_tokenizer_31`, `llama_tokenizer_old`, `bertscore_roberta`, `bertscore_biobert` |
| Cell2Sentence-Scale | `c2s_2b`, `c2s_27b` |
| CellWhisperer | `cellwhisperer_text_base`; the CLIP and chat checkpoints are downloaded with the CellWhisperer repository |

BioReason records are stored as `records_sft.jsonl` and `records_rl.jsonl`. Other evaluation data
are reviewed human proteins from UniProt with InterPro and GO annotations, five single cell atlases
from CELLxGENE and GEO `GSE84133` and `GSE280502`, and the Cell Ontology.

## Results

Run every command from the repository root. Scripts read model outputs from `INPUT_USE_RESULTS_DIR`
and write to `INPUT_USE_OUTPUT_DIR` (default `outputs/`): figures to `outputs/figures/`, everything
else to `outputs/analysis/`. Figures that draw bootstrap intervals or the within-family protein table
read them from `outputs/analysis/`, so run the matching analysis script first; without it the
figure is drawn with no interval.

| Paper element | Command |
| --- | --- |
| Fig. 2, input perturbations | `python figures/perturbations_and_conflicts.py` |
| Fig. 3, evidence conflicts | `python figures/evidence_conflicts.py` |
| Fig. 4a-b, linear probes | `python figures/probes.py` |
| Fig. 4, DEG removal in C2S-Scale | `python figures/rq1_data.py` (writes `fig_rq1_singlecell`) |
| Fig. 5a and 5c, SFT and RL checkpoints | `python figures/sft_vs_rl_panels.py` |
| Fig. 6, reasoning traces and rationales | `python figures/rationale_panels.py`, `figures/trace_base_pairs.py`, then `figures/reasoning_traces.py` |
| Fig. 7, auxiliary sequence supervision | `python figures/auxiliary_supervision_main.py` |
| Appendix, predicted base pairs for the 257 bp auxiliary condition | `python figures/auxiliary_supervision_matrices.py` |
| Appendix, BioReason-Pro evidence conflicts on GO NOT pairs | `python figures/go_not_conflicts.py` |
| Appendix, genes referenced in C2S-Scale rationales | `python figures/rationale_genes.py` |
| Figs. 2f and 3e, CellWhisperer | see [`input_use/models/cellwhisperer_driver/README.md`](input_use/models/cellwhisperer_driver/README.md) |
| Per-query disease prediction accuracy and intervals | `python analysis/dna_per_query_accuracy.py`, `analysis/dna_per_query_text_conditions.py` |
| Answer and GO set change rates | `python analysis/answer_change_rates.py` |
| Within-family protein probe | `python analysis/protein_family_probe.py`, then `analysis/protein_family_table.py` |
| Protein probe on the spliced-in embedding | `python analysis/protein_probe_mean_embedding.py`, `analysis/protein_probe_projected.py` |
| Genome-dependent subset | `python analysis/dna_genome_dependent_rows.py` |
| Evo2 probe | `python analysis/extract_evo2_raw.py`, `analysis/extract_evo2_projected.py` (GPU, BioReason environment), then `analysis/dna_probe_raw.py`, `analysis/dna_probe_projected.py` |
| Bootstrap intervals | `python analysis/bootstrap_evidence_swap.py`, `bootstrap_dna_probe.py`, `bootstrap_c2s.py`, `bootstrap_protein_shuffle.py` |

Each figure script also writes a CSV beside the figure holding every drawn value with its sample size.
Scripts named `*_panels.py` hold the loaders and panels that the figure scripts share. The paper
figures were drawn with matplotlib 3.9.2 and numpy 1.26.4. The layout checks in
`figures/reasoning_traces.py` and `figures/auxiliary_supervision_main.py` measure rendered text and
fail under matplotlib 3.11, and the figures that read parquet files need `pyarrow`.

### Auxiliary supervision

    python experiments/auxiliary_supervision/export_257bp_windows.py     # 257 bp windows around the edit
    python experiments/auxiliary_supervision/export_ref_var_target.py    # adds the ref>var target line
    python experiments/auxiliary_supervision/export_2048bp_windows.py    # 2,048 bp windows around the edit
    source setup/dna_runtime_env.sh                                      # required by every BioReason run
    python train_dna_qwen.py --kegg_data_dir_local <export> --truncate_dna_per_side 0 \
        --max_length_dna {512,2048} --seed {23,24,25}                    # in the BioReason repo
    python experiments/auxiliary_supervision/evaluate_generation.py --ckpt <checkpoint> --data <export> \
        --truncate-per-side 0 --conditions wt,same,shuffle,swap --seed 11
    python experiments/auxiliary_supervision/score_seeds.py

## Contact

For questions, please leave a GitHub issue or contact Ada Fang at <ada_fang@g.harvard.edu>.

## License

The code in this package is licensed under the MIT License.

## Citation

If you use this code in your research, please cite the following [preprint](https://arxiv.org/abs/2610.00898):
```
@misc{fang2026biologicalreasoningmodelsuse,
      title={When Do Biological Reasoning Models Use Their Biological Inputs?}, 
      author={Ada Fang and Nikitha Thoduguli and Lukas Fesser and Hanlin Zhang and Sham M. Kakade and Marinka Zitnik},
      year={2026},
      eprint={2610.00898},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2610.00898}, 
}
```

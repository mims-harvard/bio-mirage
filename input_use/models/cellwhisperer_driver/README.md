# CellWhisperer

The CellWhisperer numbers of Figs. 2f and 3e are computed on the 4,846 C2S-Scale cells by four
scripts in `input_use/models/cellwhisperer_driver/`, run in the environments of the CellWhisperer
repository. Set `INPUT_USE_CELLWHISPERER_REPO` to that checkout (default `cellwhisperer` under
`INPUT_USE_MODELS_ROOT`), with its released CLIP checkpoint, chat checkpoints and Geneformer files
in place. Atlases are read from `INPUT_USE_DATA_DIR/prepped/<atlas>.h5ad` and the C2S-Scale question
sets from `single_cell/c2s_scale/deg_removal/` under `INPUT_USE_RESULTS_DIR`. Outputs go to
`single_cell/cellwhisperer/` under `INPUT_USE_RESULTS_DIR`.

    python input_use/models/cellwhisperer_driver/build_cells.py          # pixi env, CPU: cells and pairs
    python input_use/models/cellwhisperer_driver/embed_cells.py          # pixi env, GPU: intact and shuffled Z_CW
    python input_use/models/cellwhisperer_driver/score_llm.py --atlas <atlas> --task expA --model default
    python input_use/models/cellwhisperer_driver/score_llm.py --atlas <atlas> --task expB --model default
    python input_use/models/cellwhisperer_driver/score_llm.py --atlas <atlas> --task textgate --model base
                                                                         # LLaVA env, GPU, per atlas
    python input_use/models/cellwhisperer_driver/analyze.py              # pixi env, CPU: analysis_default/results.json
    python figures/perturbations_and_conflicts.py                        # draws the panels

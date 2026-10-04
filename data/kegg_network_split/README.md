# KEGG pathway network split

The auxiliary supervision experiment uses a repartition of the 1,449 rows of
[`wanglab/kegg`](https://huggingface.co/datasets/wanglab/kegg) by pathway network. Row content is unchanged.

| file | rows | role |
|---|---|---|
| `train_network_split.csv` | 1,159 | training |
| `id_test_network_split.csv` | 145 | validation |
| `ood_test_network_split.csv` | 145 | held-out (9 pathway networks absent from training) |

Rebuild the three files (md5-checked against the originals):

    python build_network_split.py --out $INPUT_USE_DATA_DIR/kegg

`network_split_to_wanglab_kegg.csv` maps each row to its `wanglab/kegg` split and row index.

## Seen and unseen genomes

`auxiliary_supervision_seen_unseen.csv` lists the 290 validation and held-out queries behind the
unseen-genome table. A genome is seen when both 257-bp windows match a training query's.

| | seen | unseen |
|---|---|---|
| held-out | 135 | 10 |
| validation | 83 | 62 |

Of the 72 unseen, 31 are a new variant at a trained reference window and 41 are fully unseen.

Edited base pair accuracy, three seeds:

| arm | unseen (72) | seen (218) |
|---|---|---|
| auxiliary, 257 bp | 11/11/10, mean 0.148 | 132/161/159, mean 0.691 |
| auxiliary, 2,048 bp | 7/11/5, mean 0.106 | 87/156/153, mean 0.606 |
| no auxiliary, 2,048 bp | 4/4/1, mean 0.042 | 19/14/21, mean 0.083 |
| most frequent training pair (T>C) | 8/72, 0.111 | 22/218, 0.101 |

## DNA inputs

Windows are cut around the first differing base before training, and training uses
`--truncate_dna_per_side 0`.

| arm | window | edit offset | `--max_length_dna` |
|---|---|---|---|
| auxiliary, 257 bp | 257 bp | 128 | 512 |
| auxiliary, 2,048 bp | 2,048 bp | 1,024 | 2,048 |
| no auxiliary, 2,048 bp | 2,048 bp | 1,024 | 2,048 |

The disease prediction evaluation (`input_use/models/bioreason/runner.py`) uses 2,049 bp around the
first differing base (1,024 bp per side, the tokenizer keeps 2,048), `--max_length_text 1280`, no
prompt truncation, and greedy decoding.

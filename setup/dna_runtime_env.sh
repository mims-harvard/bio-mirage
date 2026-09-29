#!/usr/bin/env bash
# Sets the environment variables the BioReason DNA runner needs. Source it before every BioReason run
# from the repo root:
#
#   source setup/dna_runtime_env.sh
#
# BIOREASON_USE_VORTEX_PYTORCH_LINEAR=1 keeps the Evo2 encoder on standard PyTorch linear layers.
# Without it Evo2 runs FP8 input projections that the trained DNA projection never saw, and the
# generations are wrong without any error being raised. TORCHDYNAMO_DISABLE=1 turns off torch.compile.
export BIOREASON_USE_VORTEX_PYTORCH_LINEAR=1
export TORCHDYNAMO_DISABLE=1

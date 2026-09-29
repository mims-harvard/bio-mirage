#!/bin/bash
# Downloads the Gene Ontology (go-basic.obo) and Cell Ontology (cl-basic.obo, cl.obo saved as
# cl-full.obo) files that the GO term and cell type scorers read. Writes to INPUT_USE_DATA_DIR, or to
# input_use/data when it is unset.
#
#   bash setup/fetch_ontologies.sh
set -euo pipefail
OUT="${INPUT_USE_DATA_DIR:-$(dirname "$0")/../input_use/data}"
mkdir -p "$OUT"
curl -L -o "$OUT/go-basic.obo"  http://purl.obolibrary.org/obo/go/go-basic.obo
curl -L -o "$OUT/cl-basic.obo"  http://purl.obolibrary.org/obo/cl/cl-basic.obo
curl -L -o "$OUT/cl-full.obo"   http://purl.obolibrary.org/obo/cl.obo

#!/usr/bin/env bash
# Full data preprocessing for the hetero pipeline — no make, no sudo.
# Produces everything under data/training/ that make_graph.sh and training need:
#   download TCGA -> prepare/process omics -> priors -> data_wrapper -> splits.
#
# The PKT knowledge graph (data/prior_knowledge/PKT: nodes.zip + edges.zip) is
# downloaded from Hugging Face in step [0]; both it and the TCGA omics are fetched
# here so the pipeline is self-contained.
#
# Usage (from the repo root, after setup_env.sh):
#   bash prepare_data.sh                 # full chain
#   SEEDS="42 43 44" bash prepare_data.sh
#   SKIP_DOWNLOAD=1 bash prepare_data.sh # skip ALL downloads (raw + PKT already present)
set -euo pipefail

ENV_NAME="${ENV_NAME:-gnn}"
SEEDS="${SEEDS:-42}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"

export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
run() { conda run -n "$ENV_NAME" python "$@"; }

if [ "$SKIP_DOWNLOAD" != "1" ]; then
    echo "==> [0/5] download PKT knowledge graph (nodes.zip + edges.zip) from Hugging Face"
    run scripts/download_pkt.py

    echo "==> [1/5] download TCGA pan-cancer + prepare (transpose/parquet)"
    run scripts/download_pancan.py --out data/raw/tcga_pancan
    run scripts/preprocessing/omics/prepare_pancan.py \
        --src data/raw/tcga_pancan --out data/raw/tcga_pancan/processed
else
    echo "==> [0-1/5] SKIP_DOWNLOAD=1 — using existing PKT + data/raw/tcga_pancan"
fi

echo "==> [2/5] process omics (filter common samples, normalize)"
run scripts/preprocessing/omics/process_data_pancan.py

echo "==> [3/5] priors (BioGRID / miRDB / TFLink -> GGI / miRNA / TF)"
run scripts/preprocessing/priors/get_raw_data.py
run scripts/preprocessing/priors/refseq2gene.py
run scripts/preprocessing/priors/load_interaction.py

echo "==> [4/5] data_wrapper -> data/training/* (expression/cnv/labels/tf_nodes/variance)"
run scripts/preprocessing/omics/data_wrapper.py

echo "==> [5/5] stratified splits for seeds: $SEEDS"
run -m multiomics_kg_hgnn.pancancer_prediction.preprocessing.make_splits --seeds $SEEDS

echo ""
echo "==> done. data/training/ is populated. Next:"
echo "    bash make_graph.sh && bash train_and_eval.sh"

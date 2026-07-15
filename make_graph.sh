#!/usr/bin/env bash
# Build the heterogeneous backbone graph on the server — no make, no sudo.
# Two steps: (1) unified variance feature selection on the train split,
#            (2) build the HeteroData template from the selected panels.
#
# Usage (from the repo root, after setup_env.sh):
#   bash make_graph.sh
#
# Override any knob via env vars, e.g.:
#   SEED=43 TOP_GENES=900 METAPATH=--metapath bash make_graph.sh
set -euo pipefail

ENV_NAME="${ENV_NAME:-gnn}"
SEED="${SEED:-42}"
TOP_GENES="${TOP_GENES:-700}"
TOP_TF="${TOP_TF:-200}"
TOP_MIRNA="${TOP_MIRNA:-100}"
GO_MIN_SUPPORT="${GO_MIN_SUPPORT:-3}"
METAPATH="${METAPATH:-}"                 # set to "--metapath" to add miRNA-miRNA / TF-TF
GRAPH_FLAGS="${GRAPH_FLAGS:-}"           # extra builder flags, e.g. "--no-go" (optimized model)
SPLIT_DIR="${SPLIT_DIR:-data/training/splits/splits_seed_${SEED}}"
FS_DIR="${FS_DIR:-data/training/feature_selection/splits_seed_${SEED}}"
OUT_DIR="${OUT_DIR:-data/prior_knowledge/hetero}"

# so `python -m multiomics_kg_hgnn...` resolves even without an editable install
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
run() { conda run -n "$ENV_NAME" python "$@"; }

echo "==> feature selection (variance, train-only) | seed=$SEED genes=$TOP_GENES tf=$TOP_TF mirna=$TOP_MIRNA"
run -m multiomics_kg_hgnn.pancancer_prediction.preprocessing.feature_selection \
    --split-dir "$SPLIT_DIR" \
    --top-genes "$TOP_GENES" --top-tf "$TOP_TF" --top-mirna "$TOP_MIRNA" \
    --out-dir "$FS_DIR"

echo "==> build hetero template ${METAPATH:+(with metapaths)}"
run scripts/preprocessing/priors/build_hetero_graph.py \
    --gene-list "$FS_DIR/selected_genes.csv" \
    --tf-list "$FS_DIR/selected_tf.csv" \
    --mirna-list "$FS_DIR/selected_mirna.txt" \
    --go-min-support "$GO_MIN_SUPPORT" $METAPATH $GRAPH_FLAGS \
    --out-dir "$OUT_DIR" --force

echo ""
echo "==> done. Template + node/edge CSVs in: $OUT_DIR"

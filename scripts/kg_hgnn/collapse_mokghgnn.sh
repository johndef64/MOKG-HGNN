#!/usr/bin/env bash
# Feature-collapse study — MOKG-HGNN (heterogeneous, this model).
# For each gene count in the grid, rebuild the template with that many genes
# (per seed, leakage-free) and train the hetero model. Results go under
# results/feature_collapse/mokghgnn_g<N>/<seed>/.
#
# The hypothesis (proposta B): as genes shrink, the multi-scale KG scaffold
# (pathway/GO/disease) keeps performance up while a molecular-only model collapses.
#
# Usage (from the repo root):
#   bash scripts/kg_hgnn/collapse_mokghgnn.sh
#   SEEDS="42 43 44 45 46" GENE_GRID="700 500 300 150 100 50 20" bash scripts/kg_hgnn/collapse_mokghgnn.sh
#   BACKBONE=rgcn CONFIG=configs/config_kg_hgnn_best.yml bash scripts/kg_hgnn/collapse_mokghgnn.sh
set -euo pipefail

ENV_NAME="${ENV_NAME:-gnn}"
CONFIG="${CONFIG:-configs/config_kg_hgnn.yml}"       # "best available" MOKG-HGNN config
BACKBONE="${BACKBONE:-hgt}"
GENE_GRID="${GENE_GRID:-700 500 300 150 100 50 20}"  # feature-collapse grid (down)
SEEDS="${SEEDS:-42 43 44 45 46}"                     # split seeds (default 5)
MODEL_SEED="${MODEL_SEED:-2025}"                     # fixed model init seed
TOP_TF="${TOP_TF:-200}"
TOP_MIRNA="${TOP_MIRNA:-100}"
GO_MIN_SUPPORT="${GO_MIN_SUPPORT:-3}"
METAPATH="${METAPATH:-}"
OUT_ROOT="${OUT_ROOT:-results/feature_collapse}"

export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
run() { conda run --no-capture-output -n "$ENV_NAME" python -u "$@"; }

echo "############################################################"
echo "# MOKG-HGNN feature-collapse | backbone=$BACKBONE"
echo "# genes: $GENE_GRID | seeds: $SEEDS | out: $OUT_ROOT"
echo "############################################################"

for G in $GENE_GRID; do
  for S in $SEEDS; do
    SPLIT_DIR="data/training/splits/splits_seed_${S}"
    FS_DIR="data/training/feature_selection/collapse_g${G}_seed${S}"
    TEMPLATE="data/prior_knowledge/hetero/collapse_g${G}_seed${S}.pt"
    EXP="feature_collapse/mokghgnn_g${G}"
    echo ""
    echo "===== MOKG-HGNN | genes=$G | split seed=$S ====="

    # 1) split (idempotent)
    run -m multiomics_kg_hgnn.pancancer_prediction.preprocessing.make_splits --seeds "$S"

    # 2) feature selection with THIS gene count + template (per seed)
    run -m multiomics_kg_hgnn.pancancer_prediction.preprocessing.feature_selection \
        --split-dir "$SPLIT_DIR" \
        --top-genes "$G" --top-tf "$TOP_TF" --top-mirna "$TOP_MIRNA" \
        --out-dir "$FS_DIR"
    run scripts/preprocessing/priors/build_hetero_graph.py \
        --gene-list "$FS_DIR/selected_genes.csv" \
        --tf-list "$FS_DIR/selected_tf.csv" \
        --mirna-list "$FS_DIR/selected_mirna.txt" \
        --go-min-support "$GO_MIN_SUPPORT" $METAPATH \
        --out-dir data/prior_knowledge/hetero --force
    cp data/prior_knowledge/hetero/hetero_graph_template.pt "$TEMPLATE"

    # 3) per-run config. results_dir stays "results"; experiment_name carries the
    #    collapse sub-path (feature_collapse/mokghgnn_gN) so runs land under OUT_ROOT.
    RUN_CFG="$(mktemp --suffix=.yml)"
    MKCFG="$(mktemp --suffix=.py)"
    cat > "$MKCFG" <<'PY'
import sys, yaml
src, dst, mseed, split_dir, template, exp, backbone = sys.argv[1:8]
cfg = yaml.safe_load(open(src))
cfg["project"]["seed"] = int(mseed)
cfg["project"]["experiment_name"] = exp        # e.g. feature_collapse/mokghgnn_g300
cfg["paths"]["results_dir"] = "results"
cfg["data"]["split_dir"] = split_dir
cfg["data"]["template_path"] = template
cfg["model"]["backbone"] = backbone
yaml.safe_dump(cfg, open(dst, "w"))
PY
    run "$MKCFG" "$CONFIG" "$RUN_CFG" "$MODEL_SEED" "$SPLIT_DIR" "$TEMPLATE" "$EXP" "$BACKBONE"
    rm -f "$MKCFG"

    # 4) train
    run scripts/kg_hgnn/train.py --config "$RUN_CFG"
    rm -f "$RUN_CFG"
  done
done

echo ""
echo "==> MOKG-HGNN feature-collapse done. Results under: $OUT_ROOT/mokghgnn_g*/"

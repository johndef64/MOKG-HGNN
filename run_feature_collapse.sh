#!/usr/bin/env bash
# Feature-collapse experiment (thesis P1): "does the graph save performance?"
# Runs BOTH models across a decreasing gene grid and collects the results, so the
# degradation slope of the heterogeneous model (MOKG-HGNN) can be compared to the
# homogeneous baseline (MOGNN-TF). Results under results/feature_collapse/.
#
# Usage (from the repo root):
#   bash run_feature_collapse.sh                 # both models, default grid + seeds
#   MODELS="mokghgnn" bash run_feature_collapse.sh   # only one model
#   GENE_GRID="700 300 100 50" SEEDS="42 43 44" bash run_feature_collapse.sh
set -euo pipefail

ENV_NAME="${ENV_NAME:-gnn}"
MODELS="${MODELS:-mognntf mokghgnn}"                  # which models to run
GENE_GRID="${GENE_GRID:-700 500 300 150 100 50 20}"  # feature-collapse grid (down)
SEEDS="${SEEDS:-42 43 44 45 46}"                     # 5 seeds by default
MODEL_SEED="${MODEL_SEED:-2025}"
OUT_ROOT="${OUT_ROOT:-results/feature_collapse}"
# configs: "best available" for each model
MOGNNTF_CONFIG="${MOGNNTF_CONFIG:-configs/config_final.yml}"
MOKG_CONFIG="${MOKG_CONFIG:-configs/config_kg_hgnn.yml}"
BACKBONE="${BACKBONE:-hgt}"

export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
mkdir -p "$OUT_ROOT"

echo "############################################################"
echo "# FEATURE-COLLAPSE experiment"
echo "# models: $MODELS | genes: $GENE_GRID | seeds: $SEEDS"
echo "# out: $OUT_ROOT"
echo "############################################################"

for M in $MODELS; do
  case "$M" in
    mognntf)
      echo ""; echo "########## MOGNN-TF (homogeneous baseline) ##########"
      # --resume: the grid is ~35 long runs; re-launching after an interruption
      # must skip what is already in the summary instead of retraining it.
      conda run --no-capture-output -n "$ENV_NAME" python -u scripts/collapse_mognntf.py \
          --config "$MOGNNTF_CONFIG" \
          --gene-grid $GENE_GRID --seeds $SEEDS --model-seed "$MODEL_SEED" \
          --out-root "$OUT_ROOT" --resume
      ;;
    mokghgnn)
      echo ""; echo "########## MOKG-HGNN (heterogeneous, backbone=$BACKBONE) ##########"
      GENE_GRID="$GENE_GRID" SEEDS="$SEEDS" MODEL_SEED="$MODEL_SEED" \
        CONFIG="$MOKG_CONFIG" BACKBONE="$BACKBONE" OUT_ROOT="$OUT_ROOT" \
        bash scripts/kg_hgnn/collapse_mokghgnn.sh
      ;;
    *) echo "unknown model: $M (use mognntf | mokghgnn)" >&2; exit 1 ;;
  esac
done

echo ""
echo "########## AGGREGATE ##########"
conda run --no-capture-output -n "$ENV_NAME" python -u scripts/kg_hgnn/collapse_aggregate.py \
    --results "$OUT_ROOT"

echo ""
echo "==> done. Comparison table + plot in: $OUT_ROOT/"

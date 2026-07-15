#!/usr/bin/env bash
# Train the FINAL models (post tuning + ablation) — no make, no sudo.
# Two variants of the same tuned hetero_sage model, multi-seed for the final
# thesis numbers (mean ± s.d. + paired comparison):
#
#   full       complete multi-scale graph: gene + pathway + GO + disease
#              (rich / interpretable version)          -> best_model_config.yml
#   optimized  same model, graph WITHOUT the GO scale  -> best_model_config_optimized.yml
#              (ablation-driven: removing GO gave the best validation-F1)
#
# NO metapath anywhere: the ablation showed it useless (+0.0002) and it was never
# in the tuning. Only the semantic scales differ between the two variants.
#
# Each variant is run over SEEDS via train_and_eval.sh (split seed varies, model
# seed fixed, feature-selection + template rebuilt per seed, leakage-free). Results
# land in results/<experiment_name>/ with the per-seed mean ± s.d. aggregated.
#
# Usage (from the repo root):
#   bash train_best_models.sh                 # both variants, 10 seeds (default)
#   RUNS=5 bash train_best_models.sh          # fewer seeds
#   VARIANTS="optimized" bash train_best_models.sh   # only one variant
#
# Run it detached on the server so an SSH drop doesn't kill it:
#   nohup bash train_best_models.sh > train_best.log 2>&1 &
#   tail -f train_best.log
set -euo pipefail

ENV_NAME="${ENV_NAME:-gnn}"
BACKBONE="${BACKBONE:-hetero_sage}"       # the winning backbone (tuning)
RUNS="${RUNS:-5}"                         # seeds for the final multi-seed estimate
                                          # (5 is enough here; runs converge ~epoch
                                          # 80 via early stopping, ~1h each, so 5
                                          # seeds x 2 variants ~= 10h. Set RUNS=10
                                          # for tighter error bands if time allows.)
START_SEED="${START_SEED:-42}"
VARIANTS="${VARIANTS:-full optimized}"    # which final models to train

FULL_CONFIG="${FULL_CONFIG:-configs/best_model_config.yml}"
OPT_CONFIG="${OPT_CONFIG:-configs/best_model_config_optimized.yml}"

echo "############################################################"
echo "# FINAL MODELS | backbone=$BACKBONE | runs(seeds)=$RUNS from $START_SEED"
echo "# variants: $VARIANTS | (no metapath — ablation-confirmed useless)"
echo "############################################################"

for V in $VARIANTS; do
  echo ""
  echo "===================== final model: $V ====================="
  case "$V" in
    full)
      # complete graph: all semantic scales, no --no-* flags, no metapath.
      CONFIG="$FULL_CONFIG" BACKBONE="$BACKBONE" START_SEED="$START_SEED" \
        GRAPH_FLAGS="" METAPATH="" \
        bash train_and_eval.sh --runs "$RUNS"
      ;;
    optimized)
      # same model, graph built WITHOUT the GO scale (--no-go), rebuilt per seed.
      CONFIG="$OPT_CONFIG" BACKBONE="$BACKBONE" START_SEED="$START_SEED" \
        GRAPH_FLAGS="--no-go" METAPATH="" \
        bash train_and_eval.sh --runs "$RUNS"
      ;;
    *)
      echo "unknown variant: $V (use 'full' | 'optimized')" >&2; exit 1 ;;
  esac
done

echo ""
echo "==> done. Final per-seed metrics under results/<experiment_name>/ :"
echo "    full      -> results/best_model_full/"
echo "    optimized -> results/best_model_optimized/"
echo "    (each run's metrics.json aggregated to mean ± s.d. by train_and_eval.sh)"

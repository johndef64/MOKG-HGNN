#!/usr/bin/env bash
# Hyperparameter tuning (Optuna) for the heterogeneous model — no make, no sudo.
# Tunes model/training knobs on a FIXED template; objective = validation macro-F1.
# Assumes setup_env.sh + make_graph.sh have run (template must exist).
#
# Usage (from the repo root):
#   bash tune.sh                       # 35 trials, 10h (paper-style budget)
#   N_TRIALS=20 TIMEOUT_HOURS=4 bash tune.sh
#
# After tuning, apply the winning params to configs/config_kg_hgnn.yml and run
# the full multi-seed evaluation with train_and_eval.sh --runs N.
set -euo pipefail

ENV_NAME="${ENV_NAME:-gnn}"
CONFIG="${CONFIG:-configs/config_kg_hgnn.yml}"
N_TRIALS="${N_TRIALS:-35}"
TIMEOUT_HOURS="${TIMEOUT_HOURS:-10}"
TUNE_EPOCHS="${TUNE_EPOCHS:-60}"
TUNE_PATIENCE="${TUNE_PATIENCE:-12}"
STUDY_NAME="${STUDY_NAME:-kg_hgnn_optuna}"
OUT_DIR="${OUT_DIR:-results/optuna}"

export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

TEMPLATE="data/prior_knowledge/hetero/hetero_graph_template.pt"
if [ ! -f "$TEMPLATE" ]; then
    echo "ERROR: template not found ($TEMPLATE). Run make_graph.sh first." >&2
    exit 1
fi

echo "==> Optuna tuning | $N_TRIALS trials | ${TIMEOUT_HOURS}h timeout | epochs $TUNE_EPOCHS"
echo "    objective: validation macro-F1 | template: fixed | study: $STUDY_NAME"
# --no-capture-output + python -u: stream progress LIVE. Without it, `conda run`
# buffers stdout and nothing appears until the whole study ends.
export PYTHONUNBUFFERED=1
conda run --no-capture-output -n "$ENV_NAME" python -u scripts/kg_hgnn/run_optuna.py \
    --config "$CONFIG" \
    --n-trials "$N_TRIALS" --timeout-hours "$TIMEOUT_HOURS" \
    --tune-epochs "$TUNE_EPOCHS" --tune-patience "$TUNE_PATIENCE" \
    --study-name "$STUDY_NAME" --out-dir "$OUT_DIR"

echo ""
echo "==> done. Best params in: $OUT_DIR/$STUDY_NAME/best.json"
echo "    full report:          $OUT_DIR/$STUDY_NAME/optuna_trials_report.csv"
echo "    next: copy best params into $CONFIG, then: bash train_and_eval.sh --runs 5"

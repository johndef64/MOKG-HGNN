#!/usr/bin/env bash
# Train the heterogeneous model and evaluate it on the test split — no make, no sudo.
# Assumes the env is set up (setup_env.sh) and the template is built (make_graph.sh).
#
# Usage (from the repo root):
#   bash train_and_eval.sh
#
# Override via env vars, e.g.:
#   CONFIG=configs/config_kg_hgnn.yml bash train_and_eval.sh
#   bash train_and_eval.sh --no-eval          # train only
set -euo pipefail

ENV_NAME="${ENV_NAME:-gnn}"
CONFIG="${CONFIG:-configs/config_kg_hgnn.yml}"

# so the entrypoints resolve the package without an editable install
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
run() { conda run -n "$ENV_NAME" python "$@"; }

echo "==> training ($CONFIG)"
echo "    the per-epoch log shows time/epoch and ETA; a run dir is written under results/"
run scripts/kg_hgnn/train.py --config "$CONFIG"

if [ "${1:-}" = "--no-eval" ]; then
    echo "==> --no-eval: skipping evaluation"
    exit 0
fi

echo ""
echo "==> evaluating the latest checkpoint for this config"
run scripts/kg_hgnn/evaluate.py --config "$CONFIG"

echo ""
echo "==> done. Metrics / checkpoint / logs are in the run dir printed above (results/...)."

#!/usr/bin/env bash
# Tune all three backbones (hgt, hetero_sage, rgcn) in sequence.
# Simple fixed split of the total budget: 20h / 3 backbones ~= 6.5h each.
# One separate Optuna study per backbone (best config each).
#
# Usage (from the repo root):
#   bash tune_all.sh
set -euo pipefail

# ~6.5h each x 3 = ~19.5h total (under 20h). Each study also stops at N_TRIALS.
HOURS_EACH="${HOURS_EACH:-6.5}"
N_TRIALS="${N_TRIALS:-35}"

for bb in hgt hetero_sage rgcn; do
    echo ""
    echo "===================== tuning backbone=$bb (${HOURS_EACH}h) ====================="
    BACKBONE="$bb" TIMEOUT_HOURS="$HOURS_EACH" N_TRIALS="$N_TRIALS" bash tune.sh \
        || echo "[tune_all] study '$bb' ended with a non-zero status; continuing."
done

echo ""
echo "==> done. Best params: results/optuna/kg_hgnn_optuna_{hgt,hetero_sage,rgcn}/best.json"

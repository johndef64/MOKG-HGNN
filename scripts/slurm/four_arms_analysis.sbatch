#!/bin/bash
# =============================================================================
# Version-3 factorial experiment, STEP 3/3: tables and figures (CPU only).
# compare.py refuses to run if any of the 6 x 35 runs is missing.
#
#   sbatch scripts/slurm/four_arms_analysis.sbatch
# =============================================================================
#SBATCH --job-name=fa-analysis
#SBATCH --output=results/four_arms/slurm_logs/%x-%j.out
#SBATCH --error=results/four_arms/slurm_logs/%x-%j.err
#SBATCH --qos=low
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --requeue

set -euo pipefail
source scripts/slurm/no_shallow_env.sh

python -u scripts/four_arms/compare.py
python -u scripts/four_arms/per_class_four_arms.py
python -u scripts/four_arms/plot_four_arms.py

echo "=== end: $(date '+%F %T')"
echo "    tables: $NS_REPO/results/four_arms/tables/ | figures: $NS_REPO/results/four_arms/figures/"

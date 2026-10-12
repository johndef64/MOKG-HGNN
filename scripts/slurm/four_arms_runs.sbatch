#!/bin/bash
# =============================================================================
# Version-3 factorial experiment, STEP 2/3: the new arms, 35 (level, split) pairs.
#
# Job array: task i -> level LEVELS[i / 5], split SEEDS[i % 5]; each task runs the three
# new arms on its pair: kg_s (MOKG-HGNN + shallow, about 1 h at 700 genes), sh_kg and
# sh_tf (shallow only, minutes). At most 3 tasks at once (%3). Any free GPU.
# Resumable: a run whose JSON exists is skipped (JSONs are written atomically).
#
#   sbatch scripts/slurm/four_arms_runs.sbatch
#   only the 700-gene pairs:  sbatch --array=0-4 scripts/slurm/four_arms_runs.sbatch
# =============================================================================
#SBATCH --job-name=fa-runs
#SBATCH --array=0-34%3
#SBATCH --output=results/four_arms/slurm_logs/%x-%A_%a.out
#SBATCH --error=results/four_arms/slurm_logs/%x-%A_%a.err
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --requeue

set -euo pipefail
source scripts/slurm/no_shallow_env.sh

LEVELS=(700 500 300 150 100 50 20)
SEEDS=(42 43 44 45 46)
LEVEL="${LEVELS[$((SLURM_ARRAY_TASK_ID / 5))]}"
SEED="${SEEDS[$((SLURM_ARRAY_TASK_ID % 5))]}"
echo "    pair:   $LEVEL genes, split $SEED"

python -u scripts/slurm/no_shallow_preflight.py --gpu
python -u scripts/four_arms/run_four_arms.py --levels "$LEVEL" --seeds "$SEED"

echo "=== end: $(date '+%F %T')"

#!/bin/bash
# =============================================================================
# No-shallow matched experiment, STEP 1/3: preflight + sanity check (ticket §4).
#
#   static checks : frozen splits, 35 panels, configs, variant model, existing series
#   reproduction  : the ORIGINAL with-shallow MOGNN-TF re-run twice on level 700,
#                   split 42; must match the existing value within 0.02 macro-F1.
# Writes experiments/no_shallow_matched/results/sanity_check.json. STEP 2 starts only
# if this job ends OK (afterok) and the runner re-checks the JSON (passed=true, same
# GPU model): a failed reproduction stops the experiment, as the ticket requires.
#
# Submit from the repository root (or use no_shallow_submit.sh for the whole chain):
#   cd ~/private/MOKG-HGNN && sbatch scripts/slurm/no_shallow_sanity.sbatch
#
# GPU model: the gate requires the SAME GPU model for sanity and runs. Keep the same
# --gres type in all GPU steps (rtx3090 -> iknos-gpu1 or iknos-gpu3, same model).
# =============================================================================
#SBATCH --job-name=ns-sanity
#SBATCH --output=experiments/no_shallow_matched/slurm_logs/%x-%j.out
#SBATCH --error=experiments/no_shallow_matched/slurm_logs/%x-%j.err
#SBATCH --qos=low
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:30:00
#SBATCH --requeue

set -euo pipefail
source scripts/slurm/no_shallow_env.sh

python -u scripts/slurm/no_shallow_preflight.py --gpu

# --regenerate-panels: rebuild all 35 panels HERE, with this machine's libraries, even
# if a panels/ cache was copied from elsewhere; the runs then reuse these audited panels.
# num_workers keeps the config value (1): on Linux the DataLoader worker is forked
# cheaply, so the run is byte-for-byte the original configuration.
srun python -u experiments/no_shallow_matched/sanity_check.py --repeats 2 --regenerate-panels

echo "=== end: $(date '+%F %T')"

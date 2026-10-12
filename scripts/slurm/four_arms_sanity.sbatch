#!/bin/bash
# =============================================================================
# Version-3 factorial experiment (graph x shallow), STEP 1/3: sanity check.
# Plan: docs/versione_3/piano_v3_quattro_bracci.md (§6).
#
#   static checks : node vocabularies of the 35 (level, split) verified on their template,
#                   configs, new model classes (same initial weights, widths, backward)
#   reproduction  : arm kg (MOKG-HGNN) re-run through the new code at 700 genes, split 42;
#                   must match the existing run within 0.02 macro-F1
# Writes results/four_arms/sanity_check.json; STEP 2 starts only if this job ends OK.
#
#   sbatch scripts/slurm/four_arms_sanity.sbatch     (or four_arms_submit.sh for the chain)
# =============================================================================
#SBATCH --job-name=fa-sanity
#SBATCH --output=results/four_arms/slurm_logs/%x-%j.out
#SBATCH --error=results/four_arms/slurm_logs/%x-%j.err
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --requeue

set -euo pipefail
source scripts/slurm/no_shallow_env.sh

python -u scripts/slurm/no_shallow_preflight.py --gpu
python -u scripts/four_arms/sanity_check.py --repro 1

echo "=== end: $(date '+%F %T')"

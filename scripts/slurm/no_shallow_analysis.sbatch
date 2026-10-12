#!/bin/bash
# =============================================================================
# No-shallow matched experiment, STEP 3/3: tables, figures, report (CPU only).
#
# compare.py refuses to run if any of the 35 runs is missing or inconsistent with
# the audited panels and splits (no --allow-partial here), so a report is produced
# only from the complete series. No GPU requested: the job goes to iknos-head.
#
#   cd ~/private/MOKG-HGNN && sbatch scripts/slurm/no_shallow_analysis.sbatch
# =============================================================================
#SBATCH --job-name=ns-analysis
#SBATCH --output=results/no_shallow/slurm_logs/%x-%j.out
#SBATCH --error=results/no_shallow/slurm_logs/%x-%j.err
#SBATCH --qos=low
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --requeue

set -euo pipefail
source scripts/slurm/no_shallow_env.sh

python -u scripts/no_shallow/compare.py
python -u scripts/no_shallow/plot_comparison.py
python -u scripts/no_shallow/report.py

echo "=== end: $(date '+%F %T')"
echo "    report: $NS_REPO/results/no_shallow/REPORT_no_shallow.md"

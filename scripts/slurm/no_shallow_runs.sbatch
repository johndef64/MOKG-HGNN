#!/bin/bash
# =============================================================================
# No-shallow matched experiment, STEP 2/3: the 35 runs (7 levels x 5 splits).
#
# Job array: one task per gene-panel level, each running the 5 splits of that level
# in sequence. Task i -> level LEVELS[i]. At most 3 tasks at once (%3), as the cluster
# etiquette asks. Every task re-checks the sanity gate (passed=true, same torch).
# Any free GPU (--gres=gpu:1): each run JSON records the GPU model it ran on.
#
# Resumable at run granularity: a run whose JSON exists is skipped, and JSONs are
# written atomically. If a 'low' task is preempted and requeued, only the run that was
# in progress is redone (its incomplete MOGNN-TF folder stays in runs_raw/ and is
# never read: the comparison uses the JSONs only).
#
#   cd ~/private/MOKG-HGNN && sbatch scripts/slurm/no_shallow_runs.sbatch
#   one level only (e.g. 700):  sbatch --array=0 scripts/slurm/no_shallow_runs.sbatch
# =============================================================================
#SBATCH --job-name=ns-runs
#SBATCH --array=0-6%3
#SBATCH --output=experiments/no_shallow_matched/slurm_logs/%x-%A_%a.out
#SBATCH --error=experiments/no_shallow_matched/slurm_logs/%x-%A_%a.err
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --requeue

set -euo pipefail
source scripts/slurm/no_shallow_env.sh

LEVELS=(700 500 300 150 100 50 20)
LEVEL="${LEVELS[$SLURM_ARRAY_TASK_ID]}"
echo "    level:  $LEVEL genes (splits 42-46)"

python -u scripts/slurm/no_shallow_preflight.py --gpu
srun python -u experiments/no_shallow_matched/run_no_shallow.py --levels "$LEVEL"

echo "=== end: $(date '+%F %T')"

#!/bin/bash
# Submit the whole no-shallow chain with dependencies, from the iknos LOGIN node:
#
#   cd <repository root>
#   bash scripts/slurm/no_shallow_submit.sh            # sanity -> 35 runs -> analysis
#   bash scripts/slurm/no_shallow_submit.sh --skip-sanity   # sanity already passed HERE
#
#   sanity   (GPU)  must end OK, otherwise the next steps never start
#   runs     (GPU)  array of 7 tasks, starts afterok:sanity
#   analysis (CPU)  starts afterok on ALL array tasks
#
# If a step fails, its dependants stay in the queue with reason DependencyNeverSatisfied:
# read the .err log, cancel them (scancel), fix, and resubmit (finished runs are kept).
set -euo pipefail

cd "$(dirname "$0")/../.."          # repository root: the #SBATCH paths are relative to it
mkdir -p experiments/no_shallow_matched/slurm_logs

SKIP_SANITY=0
[ "${1:-}" = "--skip-sanity" ] && SKIP_SANITY=1

# fail fast on the login node (environment + inputs), before queueing anything
# same environment choice as no_shallow_env.sh: the venv gnn-ns if it exists, else gnn
if [ -z "${NS_ENV:-}" ]; then
    NS_ENV=gnn
    [ -x "/home/apps/venvs/$USER/gnn-ns/bin/python" ] && NS_ENV="$USER/gnn-ns"
fi
export NS_ENV                       # the jobs inherit it (sbatch exports the environment)
echo "[submit] environment: $NS_ENV"
source /home/apps/iknos-env.sh
iknos_activate "$NS_ENV"
python scripts/slurm/no_shallow_preflight.py

if [ "$SKIP_SANITY" = "1" ]; then
    python - <<'PY'
import json, sys
s = json.load(open("experiments/no_shallow_matched/results/sanity_check.json"))
if not s.get("passed"):
    sys.exit("sanity_check.json does not say passed=true: submit without --skip-sanity")
print("[submit] using existing sanity check:", s["env"]["cuda_device"], "|", s["date"])
PY
    RUNS=$(sbatch --parsable scripts/slurm/no_shallow_runs.sbatch)
else
    SANITY=$(sbatch --parsable scripts/slurm/no_shallow_sanity.sbatch)
    echo "[submit] sanity  : job $SANITY"
    RUNS=$(sbatch --parsable --dependency=afterok:"$SANITY" scripts/slurm/no_shallow_runs.sbatch)
fi
echo "[submit] runs    : array job $RUNS (7 tasks, max 3 at once)"
ANALYSIS=$(sbatch --parsable --dependency=afterok:"$RUNS" scripts/slurm/no_shallow_analysis.sbatch)
echo "[submit] analysis: job $ANALYSIS"
echo
echo "follow:  squeue -u \$USER -o '%.10i %.12j %.8T %.10M %.10L %R'"
echo "logs:    experiments/no_shallow_matched/slurm_logs/"

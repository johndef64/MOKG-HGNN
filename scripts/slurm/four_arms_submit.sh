#!/bin/bash
# Submit the version-3 chain with dependencies, from the iknos LOGIN node:
#
#   cd <repository root>
#   bash scripts/slurm/four_arms_submit.sh                 # sanity -> 35 pairs -> analysis
#   bash scripts/slurm/four_arms_submit.sh --skip-sanity   # sanity already passed HERE
#
# If a step fails, its dependants stay in the queue with reason DependencyNeverSatisfied:
# read the .err log, cancel them (scancel), fix, and resubmit (finished runs are kept).
set -euo pipefail

cd "$(dirname "$0")/../.."          # repository root: the #SBATCH paths are relative to it
mkdir -p results/four_arms/slurm_logs

SKIP_SANITY=0
[ "${1:-}" = "--skip-sanity" ] && SKIP_SANITY=1

if [ -z "${NS_ENV:-}" ]; then
    NS_ENV=gnn
    [ -x "/home/apps/venvs/$USER/gnn-ns/bin/python" ] && NS_ENV="$USER/gnn-ns"
fi
export NS_ENV
echo "[submit] environment: $NS_ENV"
source /home/apps/iknos-env.sh
iknos_activate "$NS_ENV"
python scripts/slurm/no_shallow_preflight.py

if [ "$SKIP_SANITY" = "1" ]; then
    python - <<'PY'
import json, sys
s = json.load(open("results/four_arms/sanity_check.json"))
if not (s.get("passed") and s.get("reproduction_done")):
    sys.exit("sanity_check.json: not passed or reproduction not done; submit without --skip-sanity")
print("[submit] using existing sanity check:", s["env"]["cuda_device"], "|", s["date"])
PY
    RUNS=$(sbatch --parsable scripts/slurm/four_arms_runs.sbatch)
else
    SANITY=$(sbatch --parsable scripts/slurm/four_arms_sanity.sbatch)
    echo "[submit] sanity  : job $SANITY"
    RUNS=$(sbatch --parsable --dependency=afterok:"$SANITY" scripts/slurm/four_arms_runs.sbatch)
fi
echo "[submit] runs    : array job $RUNS (35 tasks, max 3 at once)"
ANALYSIS=$(sbatch --parsable --dependency=afterok:"$RUNS" scripts/slurm/four_arms_analysis.sbatch)
echo "[submit] analysis: job $ANALYSIS"
echo
echo "follow:  squeue -u \$USER -o '%.10i %.12j %.8T %.10M %.10L %R'"
echo "logs:    results/four_arms/slurm_logs/"

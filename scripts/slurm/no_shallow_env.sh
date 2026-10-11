# Sourced by every no-shallow sbatch script (do not execute it directly).
# Activates the Python environment on the iknos cluster and moves to the repository.
#
#   NS_REPO : repository root on the cluster   (default: the folder sbatch was called from,
#             i.e. the repository root, as no_shallow_submit.sh does)
#   NS_ENV  : environment for iknos_activate    (default: the user venv "$USER/gnn-ns" if it
#             exists, otherwise the official "gnn" environment: Python 3.10, torch 2.7.0 +
#             CUDA 12.8, PyTorch Geometric 2.7.0). gnn-ns = venv on gnn that only adds
#             optuna and statsmodels (imported by the MOGNN-TF runner, missing in gnn).

NS_REPO="${NS_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [ -z "${NS_ENV:-}" ]; then
    NS_ENV=gnn
    [ -x "/home/apps/venvs/${USER:-$(id -un)}/gnn-ns/bin/python" ] && NS_ENV="${USER:-$(id -un)}/gnn-ns"
fi

source /home/apps/iknos-env.sh
iknos_activate "$NS_ENV"

cd "$NS_REPO"

# headless plotting, threads matched to the allocation, unbuffered logs
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

echo "=== Job ${SLURM_JOB_ID:-local}${SLURM_ARRAY_TASK_ID:+ (array task $SLURM_ARRAY_TASK_ID)} | ${SLURM_JOB_NAME:-}"
echo "    node:   $(hostname) | start: $(date '+%F %T')"
echo "    GPU:    ${CUDA_VISIBLE_DEVICES:-none} | cores: ${SLURM_CPUS_PER_TASK:-?} | mem: ${SLURM_MEM_PER_NODE:-?} MB"
echo "    env:    $NS_ENV | repo: $NS_REPO | python: $(python --version 2>&1) ($(type python 2>&1 | head -1))"
[ "${SLURM_RESTART_COUNT:-0}" -gt 0 ] && echo "    NB: restart number $SLURM_RESTART_COUNT (job was requeued)"
true

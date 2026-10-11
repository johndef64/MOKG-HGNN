# Sourced by every no-shallow sbatch script (do not execute it directly).
# Activates the Python environment on the iknos cluster and moves to the repository.
#
#   NS_REPO : repository root on the cluster   (default: $HOME/private/MOKG-HGNN)
#   NS_ENV  : environment for iknos_activate    (default: the official "gnn" environment:
#             Python 3.10, torch 2.7.0 + CUDA 12.8, PyTorch Geometric 2.7.0)

NS_REPO="${NS_REPO:-$HOME/private/MOKG-HGNN}"
NS_ENV="${NS_ENV:-gnn}"

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
echo "    env:    $NS_ENV | repo: $NS_REPO | python: $(python --version 2>&1)"
[ "${SLURM_RESTART_COUNT:-0}" -gt 0 ] && echo "    NB: restart number $SLURM_RESTART_COUNT (job was requeued)"
true

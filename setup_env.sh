#!/usr/bin/env bash
# Create the conda env `gnn` and install this repo into it — no make, no sudo.
# conda installs everything under your home dir, so root is never needed.
#
# Usage (from the repo root):
#   bash setup_env.sh
#
# Override the env name:  ENV_NAME=myenv bash setup_env.sh
set -euo pipefail

ENV_NAME="${ENV_NAME:-gnn}"
PY_VER="${PY_VER:-3.10}"
# CUDA build index for torch 2.7.0 (matches environment.yml). Override for a
# different CUDA, or set TORCH_INDEX="" and CPU wheels will be used.
TORCH_INDEX="${TORCH_INDEX:-https://data.pyg.org/whl/torch-2.7.0+cu128.html}"

echo "==> using conda: $(command -v conda)"

# 1) create the env if it does not exist yet
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "==> env '$ENV_NAME' already exists, reusing it"
else
    echo "==> creating env '$ENV_NAME' (python $PY_VER)"
    conda create -y -n "$ENV_NAME" "python=$PY_VER" pip
fi

# 2) run the remaining steps inside the env (no `conda activate` needed)
run() { conda run -n "$ENV_NAME" "$@"; }

# 3) torch + PyG from the CUDA wheel index FIRST (they are not on plain PyPI)
echo "==> installing torch / torch_geometric (${TORCH_INDEX:-CPU wheels})"
if [ -n "$TORCH_INDEX" ]; then
    run python -m pip install --find-links "$TORCH_INDEX" torch==2.7.0 torch_geometric==2.7.0
else
    run python -m pip install torch==2.7.0 torch_geometric==2.7.0
fi

# 4) install the repo (editable) + the rest of the deps from pyproject
echo "==> pip install -e . (repo + remaining deps)"
run python -m pip install -e .

# 5) sanity check
echo "==> verifying the install"
run python -c "import torch, torch_geometric, multiomics_kg_hgnn; \
print('torch', torch.__version__, '| PyG', torch_geometric.__version__, \
'| cuda', torch.cuda.is_available())"

echo ""
echo "==> done. Activate with:  conda activate $ENV_NAME"
echo "    then run training:     python scripts/kg_hgnn/train.py --config configs/config_kg_hgnn.yml"

# Activate the env at the end so that the user can immediately run training without
conda activate $ENV_NAME
"""Training entrypoint for the heterogeneous multi-scale model (multiomics_kg_hgnn).

Thin launcher (same pattern as scripts/train.py for MOGNN-TF): loads a YAML
config and drives the runner in the package. Kept separate from the MOGNN-TF
scripts so the two models never overlap.

Usage:
    conda run -n gnn python scripts/kg_hgnn/train.py [--config configs/config_kg_hgnn.yml]
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from multiomics_gnn.config.loader import load_config  # reuse the simple YAML loader
from multiomics_kg_hgnn.pancancer_prediction.experiments.runner import run_experiment


def main():
    ap = argparse.ArgumentParser(description="Train the hetero multi-scale model.")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "config_kg_hgnn.yml"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_experiment(cfg)


if __name__ == "__main__":
    main()

"""Single-run training entrypoint.

Usage:
    python scripts/train.py [--config configs/config.yml]
"""
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from multiomics_gnn.config.loader import load_config
from multiomics_gnn.pancancer_prediction.experiments.experiment_runner import ExperimentRunner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "config.yml"),
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    runner = ExperimentRunner()
    runner.run_experiment(cfg)


if __name__ == "__main__":
    main()

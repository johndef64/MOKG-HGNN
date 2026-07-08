"""Feature-collapse study — MOGNN-TF (homogeneous baseline).

Runs the BEST MOGNN-TF config (config_final.yml: gcn_tf, variance FS) at each
gene count in the grid, over N split seeds with a fixed model seed. This is the
molecular-only baseline the heterogeneous model is compared against: the
hypothesis is that MOGNN-TF degrades faster as genes shrink.

Results are written under results/feature_collapse/mognntf_g<N>/ by the MOGNN-TF
runner (one folder per run), and a summary CSV is aggregated at the end.

    conda run -n gnn python scripts/collapse_mognntf.py
    conda run -n gnn python scripts/collapse_mognntf.py \
        --gene-grid 700 500 300 150 100 50 20 --seeds 42 43 44 45 46
"""

import argparse
import copy
import csv
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from multiomics_gnn.config.loader import load_config
from multiomics_gnn.pancancer_prediction.experiments.experiment_runner import (
    ExperimentRunner, ExperimentDataLoader)


def _set(d, dotted, value):
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def main():
    ap = argparse.ArgumentParser(description="MOGNN-TF feature-collapse study.")
    ap.add_argument("--config", default="configs/config_final.yml",
                    help="Best MOGNN-TF config (default: the paper final config).")
    ap.add_argument("--gene-grid", type=int, nargs="+",
                    default=[700, 500, 300, 150, 100, 50, 20])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46],
                    help="Split seeds (default 5).")
    ap.add_argument("--model-seed", type=int, default=2025)
    ap.add_argument("--out-root", default="results/feature_collapse")
    ap.add_argument("--smoke", action="store_true", help="3 epochs, to validate the pipeline.")
    args = ap.parse_args()

    base = load_config(args.config)
    os.makedirs(args.out_root, exist_ok=True)

    # load the omics once, reuse across all runs
    print("[collapse-mognntf] loading omics data once...")
    loader = ExperimentDataLoader(config=base)
    expression_data, cnv_data, mirna_data = loader.load_raw_data()
    runner = ExperimentRunner(expression_data, cnv_data, mirna_data)

    rows = []
    for g in args.gene_grid:
        for s in args.seeds:
            cfg = copy.deepcopy(base)
            _set(cfg, "data.num_gene", int(g))
            _set(cfg, "project.split_seed", int(s))
            _set(cfg, "project.seed", int(args.model_seed))
            _set(cfg, "paths.results_dir", args.out_root)
            if args.smoke:
                _set(cfg, "train.num_epochs", 3)
            name = f"mognntf_g{g}"
            print(f"\n[collapse-mognntf] === genes={g} | split seed={s} ===")
            summary = runner.run_experiment(cfg, experiment_name=name)
            rows.append({
                "model": "mognn-tf", "genes": g, "split_seed": s,
                "test_macro_f1": summary.get("test_f1_macro"),
                "test_accuracy": summary.get("test_accuracy"),
                "val_macro_f1": summary.get("val_f1_macro"),
            })
            # write incrementally so partial results survive an interruption
            _write_summary(rows, args.out_root)

    print(f"\n[collapse-mognntf] done. Summary: {args.out_root}/collapse_mognntf_summary.csv")


def _write_summary(rows, out_root):
    path = os.path.join(out_root, "collapse_mognntf_summary.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()

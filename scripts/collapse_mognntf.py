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
    ap.add_argument("--resume", action="store_true",
                    help="Skip (genes, seed) pairs already in the summary CSV.")
    args = ap.parse_args()

    base = load_config(args.config)
    os.makedirs(args.out_root, exist_ok=True)

    # The grid is ~35 long runs: keep what previous invocations already produced
    # instead of truncating the summary on every start.
    rows = _read_summary(args.out_root)
    done = {(int(r["genes"]), int(r["split_seed"])) for r in rows} if args.resume else set()
    if done:
        print(f"[collapse-mognntf] resume: {len(done)} run(s) already done, skipping them")

    # load the omics once, reuse across all runs
    print("[collapse-mognntf] loading omics data once...")
    loader = ExperimentDataLoader(config=base)
    expression_data, cnv_data, mirna_data = loader.load_raw_data()
    runner = ExperimentRunner(expression_data, cnv_data, mirna_data)

    # Fair collapse: as the genes shrink, miRNA and TF shrink by the SAME fraction,
    # so the baseline is starved of features exactly like MOKG-HGNN (whose miRNA/TF
    # nodes fall off the graph when their target genes are dropped). Keeping
    # num_mirna/num_tf fixed would leave MOGNN-TF with ~86% of its features at 20
    # genes vs our ~10%, making any slope comparison meaningless.
    top_g = max(args.gene_grid)
    base_mirna = int(base.get("data", {}).get("num_mirna", 100))
    base_tf = int(base.get("data", {}).get("num_tf", 200))
    print(f"[collapse-mognntf] scaling modalities with genes (at {top_g} genes: "
          f"{base_mirna} miRNA, {base_tf} TF)")

    for g in args.gene_grid:
        frac = g / top_g
        n_mirna = max(1, round(base_mirna * frac))
        n_tf = max(1, round(base_tf * frac))
        for s in args.seeds:
            if (int(g), int(s)) in done:
                print(f"[collapse-mognntf] skip genes={g} seed={s} (already done)")
                continue
            cfg = copy.deepcopy(base)
            _set(cfg, "data.num_gene", int(g))
            _set(cfg, "data.num_mirna", int(n_mirna))
            _set(cfg, "data.num_tf", int(n_tf))
            _set(cfg, "project.split_seed", int(s))
            _set(cfg, "project.seed", int(args.model_seed))
            # Nest the runs like MOKG-HGNN: results_dir carries the gene level
            # (mognntf_gN) and the runner appends "run_<ts>" as the run dir, giving
            # OUT_ROOT/mognntf_gN/run_<ts>/ instead of OUT_ROOT/mognntf_gN_<ts>/.
            _set(cfg, "paths.results_dir", os.path.join(args.out_root, f"mognntf_g{g}"))
            if args.smoke:
                _set(cfg, "train.num_epochs", 3)
            print(f"\n[collapse-mognntf] === genes={g} | miRNA={n_mirna} | TF={n_tf} "
                  f"| split seed={s} ===")
            summary = runner.run_experiment(cfg, experiment_name="run")
            rows.append({
                "model": "mognn-tf", "genes": g, "mirna": n_mirna, "tf": n_tf,
                "split_seed": s,
                "test_macro_f1": summary.get("test_f1_macro"),
                "test_accuracy": summary.get("test_accuracy"),
                "val_macro_f1": summary.get("val_f1_macro"),
            })
            # write incrementally so partial results survive an interruption
            _write_summary(rows, args.out_root)

    print(f"\n[collapse-mognntf] done. Summary: {args.out_root}/collapse_mognntf_summary.csv")


SUMMARY = "collapse_mognntf_summary.csv"


def _read_summary(out_root):
    """Rows from a previous invocation, so --resume can skip finished runs."""
    path = os.path.join(out_root, SUMMARY)
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        return [r for r in csv.DictReader(fh) if r.get("genes") and r.get("split_seed")]


def _write_summary(rows, out_root):
    path = os.path.join(out_root, SUMMARY)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "genes", "mirna", "tf", "split_seed",
                                           "test_macro_f1", "test_accuracy", "val_macro_f1"])
        w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()

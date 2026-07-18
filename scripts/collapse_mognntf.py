"""Feature-collapse study — MOGNN-TF (homogeneous baseline), FEATURE-ALIGNED.

Runs MOGNN-TF at each gene count in the grid over N split seeds, but on the EXACT
same molecular panel as MOKG-HGNN at that level: for each (genes, seed) it
regenerates the unified feature selection (the same one that builds the hetero
graph), derives the miRNA/TF that SURVIVE in the graph (edge to a selected gene),
and forces MOGNN-TF to use precisely those genes/TF/miRNA — bypassing its own
variance FS. Without this the two models would see different features (TFs overlap
only ~56/98, genes differ), making the slope comparison meaningless.

Results are written under results/feature_collapse/mognntf_g<N>/run_<ts>/ (nested
like MOKG-HGNN), and a summary CSV is aggregated at the end.

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
REPO = HERE.parent
SRC = REPO / "src"
for p in (str(SRC), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd

from multiomics_gnn.config.loader import load_config
from multiomics_gnn.pancancer_prediction.experiments.experiment_runner import (
    ExperimentRunner, ExperimentDataLoader)
from multiomics_kg_hgnn.pancancer_prediction.preprocessing import feature_selection as kgfs
from scripts.kg_hgnn.survived_features import survived as survived_features


def _panel(fs_dir, split_dir, genes, tf, mirna):
    """Regenerate the unified FS at this level, then the survived miRNA/TF panel.

    Returns (gene_list, survived_tf, survived_mirna). The FS is deterministic, so
    this reproduces exactly the panel MOKG-HGNN used at this (genes, seed)."""
    kgfs.select_features(split_dir=split_dir, out_dir=fs_dir,
                         top_genes=genes, top_tf=tf, top_mirna=mirna)
    gene_list = pd.read_csv(os.path.join(fs_dir, "selected_genes.csv"))["symbol"].astype(str).tolist()
    surv_mirna, surv_tf = survived_features(fs_dir)
    return gene_list, surv_tf, surv_mirna


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

    # Feature alignment: at each level the FS uses the SAME top-K as the hetero
    # collapse (top_tf=200, top_mirna=100 requested), then survival on the graph
    # decides how many actually remain — exactly as for MOKG-HGNN. MOGNN-TF is then
    # forced onto that identical panel (genes/TF/miRNA by name).
    fs_top_tf = int(base.get("data", {}).get("num_tf", 200))
    fs_top_mirna = int(base.get("data", {}).get("num_mirna", 100))
    fs_root = os.path.join(args.out_root, "_aligned_fs")
    print(f"[collapse-mognntf] feature-aligned to MOKG-HGNN "
          f"(FS request: top_tf={fs_top_tf}, top_mirna={fs_top_mirna}; survival trims both)")

    for g in args.gene_grid:
        for s in args.seeds:
            if (int(g), int(s)) in done:
                print(f"[collapse-mognntf] skip genes={g} seed={s} (already done)")
                continue
            split_dir = f"data/training/splits/splits_seed_{s}"
            fs_dir = os.path.join(fs_root, f"g{g}_seed{s}")
            gene_list, surv_tf, surv_mirna = _panel(fs_dir, split_dir, g, fs_top_tf, fs_top_mirna)

            cfg = copy.deepcopy(base)
            _set(cfg, "data.num_gene", int(g))
            _set(cfg, "data.num_tf", len(surv_tf))
            _set(cfg, "data.num_mirna", len(surv_mirna))
            # explicit panels -> the runner bypasses its own FS and uses THESE
            _set(cfg, "data.gene_list", gene_list)
            _set(cfg, "data.tf_list", surv_tf)
            _set(cfg, "data.mirna_keep", surv_mirna)
            _set(cfg, "project.split_seed", int(s))
            _set(cfg, "project.seed", int(args.model_seed))
            # Nest the runs like MOKG-HGNN: results_dir carries the gene level
            # (mognntf_gN) and the runner appends "run_<ts>" as the run dir, giving
            # OUT_ROOT/mognntf_gN/run_<ts>/ instead of OUT_ROOT/mognntf_gN_<ts>/.
            _set(cfg, "paths.results_dir", os.path.join(args.out_root, f"mognntf_g{g}"))
            if args.smoke:
                _set(cfg, "train.num_epochs", 3)
            print(f"\n[collapse-mognntf] === genes={g} | miRNA={len(surv_mirna)} | "
                  f"TF={len(surv_tf)} (survived) | split seed={s} ===")
            summary = runner.run_experiment(cfg, experiment_name="run")
            rows.append({
                "model": "mognn-tf", "genes": g, "mirna": len(surv_mirna), "tf": len(surv_tf),
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

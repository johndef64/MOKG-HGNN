"""Recompute per-class metrics for an EXISTING run, from its saved checkpoint —
no retraining. Reads <run_dir>/config.json + model_best.pt, rebuilds the test
set, predicts, and writes per_class_metrics.{csv,json} + confusion_matrix.csv
into the run dir (same files a fresh run now produces).

Usage:
    conda run -n gnn python scripts/kg_hgnn/eval_per_class.py --run results/kg_hgnn_hgt/2026...
    conda run -n gnn python scripts/kg_hgnn/eval_per_class.py --all        # every run under results/
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from multiomics_kg_hgnn.pancancer_prediction.preprocessing.make_datasets import make_datasets, build_loaders
from multiomics_kg_hgnn.pancancer_prediction.training.trainer import HeteroTrainer
from multiomics_kg_hgnn.pancancer_prediction.training.per_class_metrics import save_per_class
from multiomics_kg_hgnn.models.hetero_gnn import HeteroMultiScaleGNN


def eval_run(run_dir):
    cfg_path = os.path.join(run_dir, "config.json")
    ckpt_path = os.path.join(run_dir, "model_best.pt")
    if not (os.path.exists(cfg_path) and os.path.exists(ckpt_path)):
        print(f"[skip] {run_dir}: missing config.json or model_best.pt")
        return False

    cfg = json.load(open(cfg_path))
    d, m = cfg["data"], cfg["model"]
    device = cfg.get("runtime", {}).get("device") or ("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds, test_ds, num_classes = make_datasets(
        split_dir=d["split_dir"], template_path=d["template_path"], hetero_dir=d["hetero_dir"],
        use_cnv=d.get("use_cnv", True), use_mirna=d.get("use_mirna", True),
        scaler=d.get("scaler", "standard"))
    _, _, test_loader = build_loaders(train_ds, val_ds, test_ds,
                                      batch_size=int(d.get("batch_size", 16)))

    feature_dims = {nt: t.shape[-1] for nt, t in train_ds.features.items()}
    model = HeteroMultiScaleGNN(
        metadata=train_ds.template.metadata(), num_classes=num_classes, feature_dims=feature_dims,
        backbone=m.get("backbone", "hgt"), hidden=int(m.get("hidden", 64)),
        num_layers=int(m.get("num_layers", 2)), heads=int(m.get("heads", 2)),
        dropout=float(m.get("dropout", 0.2)),
        readout_types=tuple(m.get("readout_types", ["gene", "pathway", "GO_term"]))).to(device)

    state = torch.load(ckpt_path, map_location=device)
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        # The template at data.template_path was overwritten since this run
        # (e.g. rebuilt with --metapath): the current graph has more edge types
        # than the checkpoint. Recomputing on a DIFFERENT graph would give wrong
        # numbers, so we refuse instead of silently loading with strict=False.
        print(f"[ERROR] {run_dir}: model/checkpoint mismatch — the template at "
              f"'{d['template_path']}' no longer matches the one this run was trained on.")
        print("        Rebuild that exact template (same --metapath / --top-genes / seed) "
              "before recomputing, or re-run training to regenerate the checkpoint.")
        print(f"        (details: {str(e).splitlines()[0]})")
        return False

    trainer = HeteroTrainer(model, optimizer=None, device=device)
    y_true, y_pred = trainer.predict(test_loader)
    res = save_per_class(run_dir, y_true, y_pred, num_classes=num_classes)

    print(f"[done] {run_dir}")
    print(f"       macro-F1 {res['macro_f1']:.4f} | weighted-F1 {res['weighted_f1']:.4f} | "
          f"accuracy {res['accuracy']:.4f}")
    print(f"       -> per_class_metrics.csv / .json / confusion_matrix.csv")
    return True


def main():
    ap = argparse.ArgumentParser(description="Recompute per-class metrics from a saved checkpoint.")
    ap.add_argument("--run", default=None, help="A single run dir.")
    ap.add_argument("--all", action="store_true", help="Every run with a checkpoint under results/.")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    if args.run:
        eval_run(args.run)
    elif args.all:
        runs = sorted(os.path.dirname(p) for p in
                      glob.glob(os.path.join(args.results, "**", "model_best.pt"), recursive=True))
        print(f"Found {len(runs)} run(s) with a checkpoint.")
        for r in runs:
            eval_run(r)
    else:
        ap.error("pass --run <dir> or --all")


if __name__ == "__main__":
    main()

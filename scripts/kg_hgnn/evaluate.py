"""Evaluation entrypoint for the heterogeneous model: rebuild datasets from a
config, load a trained checkpoint, and report test macro-F1 / accuracy.

Usage:
    conda run -n gnn python scripts/kg_hgnn/evaluate.py \
        --config configs/config_kg_hgnn.yml --checkpoint results/kg_hgnn_hgt_best.pt
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
from multiomics_gnn.config.loader import load_config
from multiomics_kg_hgnn.pancancer_prediction.preprocessing.make_datasets import make_datasets, build_loaders
from multiomics_kg_hgnn.pancancer_prediction.training.trainer import HeteroTrainer
from multiomics_kg_hgnn.models.hetero_gnn import HeteroMultiScaleGNN


def main():
    ap = argparse.ArgumentParser(description="Evaluate a trained hetero model on the test split.")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "config_kg_hgnn.yml"))
    ap.add_argument("--checkpoint", required=True, help="Path to a saved model state_dict (.pt).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    d, m = cfg["data"], cfg["model"]
    device = cfg.get("runtime", {}).get("device") or ("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds, test_ds, num_classes = make_datasets(
        split_dir=d["split_dir"], template_path=d["template_path"], hetero_dir=d["hetero_dir"],
        use_cnv=d.get("use_cnv", True), use_mirna=d.get("use_mirna", True),
        scaler=d.get("scaler", "standard"))
    _, _, test_loader = build_loaders(train_ds, val_ds, test_ds, batch_size=int(d.get("batch_size", 16)))

    model = HeteroMultiScaleGNN(
        metadata=train_ds.template.metadata(), num_classes=num_classes,
        backbone=m.get("backbone", "hgt"), hidden=int(m.get("hidden", 64)),
        num_layers=int(m.get("num_layers", 2)), heads=int(m.get("heads", 2)),
        dropout=float(m.get("dropout", 0.2)),
        readout_types=tuple(m.get("readout_types", ["gene", "pathway", "GO_term"])))
    # a dummy forward materializes the lazy layers before loading the checkpoint
    model(next(iter(test_loader)))
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    metrics = HeteroTrainer(model, optimizer=None, device=device).evaluate(test_loader)
    print(f"[TEST] macro-F1 {metrics['macro_f1']:.4f} | accuracy {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()

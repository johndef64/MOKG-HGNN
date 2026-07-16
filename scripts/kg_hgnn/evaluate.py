"""Evaluation entrypoint for the heterogeneous model: rebuild datasets from a
config, load a trained checkpoint, and report test macro-F1 / accuracy.

Usage:
    conda run -n gnn python scripts/kg_hgnn/evaluate.py \
        --config configs/config_kg_hgnn.yml --checkpoint results/kg_hgnn_hgt_best.pt
"""

import argparse
import glob
import os
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


def _latest_checkpoint(cfg):
    """Most recent model_best.pt under results/<experiment_name>/*/."""
    results_dir = cfg.get("paths", {}).get("results_dir", "results")
    exp = cfg["project"].get("experiment_name", "kg_hgnn")
    cands = glob.glob(os.path.join(results_dir, exp, "*", "model_best.pt"))
    if not cands:
        raise FileNotFoundError(
            f"No checkpoint under {results_dir}/{exp}/*/model_best.pt. Run training first, "
            f"or pass --checkpoint explicitly.")
    return max(cands, key=os.path.getmtime)


def main():
    ap = argparse.ArgumentParser(description="Evaluate a trained hetero model on the test split.")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "config_kg_hgnn.yml"))
    ap.add_argument("--checkpoint", default=None,
                    help="Path to a saved state_dict (.pt). Default: latest run for this config.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    d, m = cfg["data"], cfg["model"]
    device = cfg.get("runtime", {}).get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = args.checkpoint or _latest_checkpoint(cfg)
    print(f"checkpoint: {checkpoint}")

    train_ds, val_ds, test_ds, num_classes, _ = make_datasets(
        split_dir=d["split_dir"], template_path=d["template_path"], hetero_dir=d["hetero_dir"],
        use_cnv=d.get("use_cnv", True), use_mirna=d.get("use_mirna", True),
        scaler=d.get("scaler", "standard"))
    _, _, test_loader = build_loaders(train_ds, val_ds, test_ds, batch_size=int(d.get("batch_size", 16)))

    feature_dims = {nt: t.shape[-1] for nt, t in train_ds.features.items()}
    model = HeteroMultiScaleGNN(
        metadata=train_ds.template.metadata(), num_classes=num_classes, feature_dims=feature_dims,
        backbone=m.get("backbone", "hgt"), hidden=int(m.get("hidden", 64)),
        num_layers=int(m.get("num_layers", 2)), heads=int(m.get("heads", 2)),
        dropout=float(m.get("dropout", 0.2)),
        readout_types=tuple(m.get("readout_types", ["gene", "pathway", "GO_term"]))).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))

    metrics = HeteroTrainer(model, optimizer=None, device=device).evaluate(test_loader)
    print(f"[TEST] macro-F1 {metrics['macro_f1']:.4f} | accuracy {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()

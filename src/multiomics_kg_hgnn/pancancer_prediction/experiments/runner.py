"""Experiment runner for the heterogeneous model: config -> data -> model ->
train -> test. Analog of MOGNN-TF's ExperimentRunner, driven by a YAML config
(configs/config_kg_hgnn.yml) on the same schema.
"""

import os

import numpy as np
import torch

from multiomics_kg_hgnn.pancancer_prediction.preprocessing.make_datasets import (
    make_datasets, build_loaders)
from multiomics_kg_hgnn.pancancer_prediction.training.trainer import HeteroTrainer
from multiomics_kg_hgnn.models.hetero_gnn import HeteroMultiScaleGNN


def _seed_everything(seed):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_experiment(cfg, logger=print):
    proj, data_cfg, model_cfg, train_cfg = cfg["project"], cfg["data"], cfg["model"], cfg["train"]
    _seed_everything(int(proj.get("seed", 42)))
    device = cfg.get("runtime", {}).get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    logger(f"device: {device}")

    # --- data -------------------------------------------------------------
    train_ds, val_ds, test_ds, num_classes = make_datasets(
        split_dir=data_cfg["split_dir"],
        template_path=data_cfg.get("template_path", "data/prior_knowledge/hetero/hetero_graph_template.pt"),
        hetero_dir=data_cfg.get("hetero_dir", "data/prior_knowledge/hetero"),
        use_cnv=data_cfg.get("use_cnv", True),
        use_mirna=data_cfg.get("use_mirna", True),
        scaler=data_cfg.get("scaler", "standard"),
    )
    weighted = str(cfg.get("sampler_strategy", "none")).lower() == "weighted"
    train_loader, val_loader, test_loader = build_loaders(
        train_ds, val_ds, test_ds,
        batch_size=int(data_cfg.get("batch_size", 16)),
        num_workers=int(data_cfg.get("num_workers", 0)),
        weighted_sampling=weighted, gamma=float(cfg.get("sampler_gamma", 1.0)))

    # --- model ------------------------------------------------------------
    model = HeteroMultiScaleGNN(
        metadata=train_ds.template.metadata(), num_classes=num_classes,
        backbone=model_cfg.get("backbone", "hgt"), hidden=int(model_cfg.get("hidden", 64)),
        num_layers=int(model_cfg.get("num_layers", 2)), heads=int(model_cfg.get("heads", 2)),
        dropout=float(model_cfg.get("dropout", 0.2)),
        readout_types=tuple(model_cfg.get("readout_types", ["gene", "pathway", "GO_term"])),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 5e-4)))

    class_weights = None
    if train_cfg.get("class_weighted_loss", False):
        y = train_ds.y[train_ds.patient_idx].numpy()
        counts = np.bincount(y, minlength=num_classes).astype(float)
        class_weights = counts.sum() / (num_classes * np.clip(counts, 1, None))

    trainer = HeteroTrainer(
        model, optimizer, device=device, class_weights=class_weights,
        patience=int(cfg.get("early_stopping", {}).get("patience", 20)), logger=logger)

    # --- train + test -----------------------------------------------------
    best_val = trainer.fit(train_loader, val_loader, num_epochs=int(train_cfg.get("num_epochs", 100)))
    test_metrics = trainer.evaluate(test_loader)
    logger(f"[TEST] macro-F1 {test_metrics['macro_f1']:.4f} | "
           f"accuracy {test_metrics['accuracy']:.4f} | best val macro-F1 {best_val:.4f}")

    results_dir = cfg.get("paths", {}).get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)
    out = os.path.join(results_dir, f"{proj.get('experiment_name', 'kg_hgnn')}_test.txt")
    with open(out, "w") as fh:
        for k, v in test_metrics.items():
            fh.write(f"{k}\t{v}\n")
    logger(f"[saved] test metrics -> {out}")
    return test_metrics

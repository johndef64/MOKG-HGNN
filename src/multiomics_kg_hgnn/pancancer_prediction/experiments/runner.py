"""Experiment runner for the heterogeneous model: config -> data -> model ->
train -> test. Analog of MOGNN-TF's ExperimentRunner, driven by a YAML config
(configs/config_kg_hgnn.yml) on the same schema.
"""

import os
import csv
import json
import datetime

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


def _make_run_dir(results_dir, experiment_name):
    """results/<experiment_name>/<timestamp>/ — one folder per run."""
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_dir, experiment_name, stamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _file_logger(run_dir):
    """Return a logger that both prints and appends to run.log."""
    log_path = os.path.join(run_dir, "run.log")
    log_file = open(log_path, "a", encoding="utf-8")

    def log(msg):
        print(msg)
        log_file.write(str(msg) + "\n")
        log_file.flush()
    return log


def run_experiment(cfg, logger=None, save_artifacts=True):
    """save_artifacts=False (used during Optuna tuning) skips the per-run folder,
    checkpoint, logs and history — the tuner only needs the returned score, and
    writing a folder+checkpoint per trial just litters results/ and wastes I/O."""
    proj, data_cfg, model_cfg, train_cfg = cfg["project"], cfg["data"], cfg["model"], cfg["train"]
    _seed_everything(int(proj.get("seed", 42)))
    device = cfg.get("runtime", {}).get("device") or ("cuda" if torch.cuda.is_available() else "cpu")

    results_dir = cfg.get("paths", {}).get("results_dir", "results")
    run_dir = _make_run_dir(results_dir, proj.get("experiment_name", "kg_hgnn")) if save_artifacts else None
    if logger is None:
        logger = _file_logger(run_dir) if save_artifacts else print
    if run_dir:
        logger(f"run dir: {run_dir}")
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
    feature_dims = {nt: t.shape[-1] for nt, t in train_ds.features.items()}
    model = HeteroMultiScaleGNN(
        metadata=train_ds.template.metadata(), num_classes=num_classes, feature_dims=feature_dims,
        backbone=model_cfg.get("backbone", "hgt"), hidden=int(model_cfg.get("hidden", 64)),
        num_layers=int(model_cfg.get("num_layers", 2)), heads=int(model_cfg.get("heads", 2)),
        dropout=float(model_cfg.get("dropout", 0.2)),
        readout_types=tuple(model_cfg.get("readout_types", ["gene", "pathway", "GO_term"])),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 5e-4)))

    # LR scheduler (reuses MOGNN-TF's OmicScheduler: constant/step/exponential/
    # cosine/warmup_cosine/reduce_on_plateau). None -> constant LR.
    scheduler = None
    sched_name = cfg.get("scheduler", {}).get("name")
    if sched_name and str(sched_name).lower() not in {"none", "null", ""}:
        from multiomics_gnn.base_ml.scheduler import OmicScheduler
        scheduler = OmicScheduler(optimizer, str(sched_name),
                                  total_epochs=int(train_cfg.get("num_epochs", 100)))
        logger(f"scheduler: {sched_name}")

    class_weights = None
    if train_cfg.get("class_weighted_loss", False):
        y = train_ds.y[train_ds.patient_idx].numpy()
        counts = np.bincount(y, minlength=num_classes).astype(float)
        class_weights = counts.sum() / (num_classes * np.clip(counts, 1, None))

    trainer = HeteroTrainer(
        model, optimizer, device=device, class_weights=class_weights, scheduler=scheduler,
        patience=int(cfg.get("early_stopping", {}).get("patience", 20)), logger=logger)

    # --- train + test -----------------------------------------------------
    best_val = trainer.fit(train_loader, val_loader, num_epochs=int(train_cfg.get("num_epochs", 100)))
    test_metrics = trainer.evaluate(test_loader)
    logger(f"[TEST] macro-F1 {test_metrics['macro_f1']:.4f} | "
           f"accuracy {test_metrics['accuracy']:.4f} | best val macro-F1 {best_val:.4f}")

    # --- persist run artifacts (checkpoint + logs + metrics) --------------
    # skipped during tuning (save_artifacts=False): no per-trial folder/checkpoint
    ckpt_path = None
    if save_artifacts:
        ckpt_path = os.path.join(run_dir, "model_best.pt")
        torch.save(trainer.model.state_dict(), ckpt_path)

        with open(os.path.join(run_dir, "config.json"), "w") as fh:
            json.dump(cfg, fh, indent=2, default=str)

        if trainer.history:
            with open(os.path.join(run_dir, "history.csv"), "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(trainer.history[0].keys()))
                w.writeheader(); w.writerows(trainer.history)

        summary = {"best_val_macro_f1": best_val, **{f"test_{k}": v for k, v in test_metrics.items()}}
        with open(os.path.join(run_dir, "metrics.json"), "w") as fh:
            json.dump(summary, fh, indent=2)

        # graph provenance: record the ACTUAL node/edge types of the template this
        # run was trained on, so with/without --metapath is unambiguous later
        # (independent of the template file, which may be overwritten).
        node_types, edge_types = train_ds.template.metadata()
        rels = [rel for (_, rel, _) in edge_types]
        graph_info = {
            "template_path": data_cfg.get("template_path"),
            "node_types": list(node_types),
            "edge_types": [list(et) for et in edge_types],
            "has_metapath": any("shares_target" in rel for rel in rels),
            "num_relations": len(edge_types),
            "num_nodes": {nt: int(train_ds.template[nt].num_nodes) for nt in node_types},
        }
        with open(os.path.join(run_dir, "graph_info.json"), "w") as fh:
            json.dump(graph_info, fh, indent=2)
        logger(f"[graph] metapath={'yes' if graph_info['has_metapath'] else 'no'} "
               f"| {graph_info['num_relations']} relations")

        # per-class precision/recall/F1 + confusion matrix on the test set
        from multiomics_kg_hgnn.pancancer_prediction.training.per_class_metrics import save_per_class
        y_true, y_pred = trainer.predict(test_loader)
        save_per_class(run_dir, y_true, y_pred, num_classes=num_classes)

        logger(f"[saved] checkpoint -> {ckpt_path}")
        logger(f"[saved] history/metrics/config/log -> {run_dir}")
    # best_val_macro_f1 is the objective for hyperparameter tuning (tune on the
    # validation split, never on test); test_* are reported for the final table.
    return {**test_metrics, "best_val_macro_f1": best_val,
            "run_dir": run_dir, "checkpoint": ckpt_path}

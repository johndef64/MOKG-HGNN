# multiomics_kg_hgnn — the MOKG-HGNN framework package

Multi-scale heterogeneous GNN for pan-cancer molecular subtype classification. It
reuses the prior-knowledge backbone in `data/prior_knowledge/hetero/` (built by
`scripts/preprocessing/priors/build_hetero_graph.py`) and injects per-patient
omic features, producing `HeteroData` graphs ready for the RGCN / HeteroSAGE /
HGT backbones.

This package is self-contained: it has **no dependency** on the MOGNN-TF benchmark
package. Shared infrastructure (config loader, LR scheduler, split generator) is
vendored under `utils/`.

## Layout

```
utils/
  config.py               # minimal YAML config loader
  scheduler.py            # OmicScheduler (LR schedules)
  split.py                # stratified train/val/test split generator
models/
  hetero_gnn.py           # HeteroMultiScaleGNN: type-specific encoders + backbone + multi-scale readout
pancancer_prediction/
  preprocessing/
    make_splits.py        # stratified split indices (one per seed)
    feature_selection.py  # unified variance-based FS (gene/TF/miRNA), train-only
    patient_features.py   # feature injection: omics -> per-node-type tensors
    make_datasets.py      # template + omics + splits -> 3 datasets + loaders
  datasets/
    HeteroOmicDataset.py  # PyG dataset: one HeteroData per patient (shared topology)
    check_dataset.py      # end-to-end smoke test: real data -> model -> 27 classes
  training/
    trainer.py            # train/val/test loop, macro-F1, early stopping
    per_class_metrics.py  # per-class precision/recall/F1
  experiments/
    runner.py             # config -> data -> model -> train -> test
```

Backbones (registry in `models/hetero_gnn.py`): `rgcn`, `hetero_sage`, `hgt`.

## Usage

Entrypoints live in `scripts/kg_hgnn/` (`train.py`, `evaluate.py`,
`run_optuna.py`, ...). See the repository root `README.md` for the full pipeline;
in short:

```bash
conda run -n gnn python scripts/kg_hgnn/train.py --config configs/config_kg_hgnn.yml
conda run -n gnn python scripts/kg_hgnn/evaluate.py \
    --config configs/config_kg_hgnn.yml --checkpoint results/<...>/model_best.pt
```

The config drives everything: `data.split_dir`, `data.template_path`, ablation
switches (`use_cnv`, `use_mirna`), `model.backbone/hidden/num_layers/heads`,
`model.readout_types` (multi-scale pooling), and class-imbalance handling
(`train.class_weighted_loss` or `sampler_strategy: weighted`). Primary metric:
**macro-F1**.

Each run writes `results/<experiment>/<timestamp>/` with `model_best.pt`,
`run.log`, `history.csv`, `metrics.json`, `per_class_metrics.csv`,
`confusion_matrix.csv` and `graph_info.json`.

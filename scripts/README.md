# `scripts/` — CLI entrypoints and data pipeline

Guide to the scripts directory. All commands assume the **repository root** as the
working directory. For the end-to-end MOKG-HGNN pipeline see the root
[`README.md`](../README.md); MOGNN-TF (benchmark) entrypoints live in
[`benchmark/MOGNN-TF/scripts/`](../benchmark/MOGNN-TF/scripts).

## MOKG-HGNN entrypoints — `kg_hgnn/`

```bash
conda run -n gnn python scripts/kg_hgnn/train.py    --config configs/config_kg_hgnn.yml
conda run -n gnn python scripts/kg_hgnn/evaluate.py --config configs/config_kg_hgnn.yml --checkpoint results/<...>/model_best.pt
conda run -n gnn python scripts/kg_hgnn/run_optuna.py --config configs/config_kg_hgnn.yml
conda run -n gnn python scripts/kg_hgnn/explain.py  --run results/<...>/<ts> --per-class 15
```
Also here: `eval_per_class.py`, `plot_training.py`, `which_graph.py`,
`collapse_mokghgnn.sh`, `survived_features.py`, `extract_pkt_names.py`,
`build_hetero_graph.py` (under `preprocessing/priors/`).

## Shared data pipeline

Downloads and preprocessing that produce `data/training/` (used to build the
graph). Normally run through `bash prepare_data.sh`; the individual steps:

```bash
# 1. download TCGA Pan-Cancer omics (Xena) and the PKT knowledge graph
python scripts/download_pancan.py --out data/raw/tcga_pancan
python scripts/download_pkt.py

# 2-3. omics: raw -> parquet, per-omic cleaning, common-sample filtering
python scripts/preprocessing/omics/prepare_pancan.py --src data/raw/tcga_pancan --out data/raw/tcga_pancan/processed
python scripts/preprocessing/omics/process_data_pancan.py --normalization

# 4. final wrapper: alignments, top-variance miRNAs, training-set output
python scripts/preprocessing/omics/data_wrapper.py
```

## Feature collapse bridge

`collapse_mognntf.py` runs the MOGNN-TF benchmark on the same surviving feature
panel as MOKG-HGNN for a fair robustness comparison (see the root README,
"Feature collapse"). It imports the vendored MOGNN-TF package from
`benchmark/MOGNN-TF/`.

## Results aggregation and statistics

```bash
python scripts/collect_results.py --results-root results/pancan
python scripts/analysis/analize_metrics.py   # Friedman + Wilcoxon-Holm
```

# MOKG-HGNN — Multi-scale Heterogeneous GNN for pan-cancer molecular subtypes

**MOKG-HGNN** classifies pan-cancer molecular subtypes (27 iCluster classes) with
a **multi-scale heterogeneous graph** that links the molecular layer
(gene / miRNA / TF) to higher biological scales — **pathway → GO → disease** — by
integrating a biomedical Knowledge Graph (PKT / PheKnowLator, OWL-NETS). The task
is graph classification (one graph per patient); the message passing reasons
across several levels of biological organization rather than over molecular
features alone.

Molecular baselines (MOGNN-TF, MoGCN, Li & Nabavi) are provided under
[`benchmark/`](benchmark/) for a like-for-like comparison on the same split.

---

## What it contains

- **Heterogeneous graph** (`HeteroData`) with 6 node types (`gene`, `miRNA`,
  `TF`, `pathway`, `GO_term`, `disease`) and typed relations (gene↔gene,
  miRNA→gene, TF→gene, gene→pathway, gene→GO, GO→GO, gene→disease, plus optional
  metapaths).
- **Three heterogeneous backbones** (a complexity spectrum): `rgcn` (one weight
  per relation) · `hetero_sage` (SAGE per relation) · `hgt` (attention per
  meta-relation).
- **Unified variance-based feature selection** (gene/TF/miRNA), train-only.
- **Optuna tuning** (one study per backbone) and **multi-seed evaluation**.

Structure:
```
src/multiomics_kg_hgnn/   # the model package (framework)
scripts/kg_hgnn/          # entrypoints: train.py, evaluate.py, run_optuna.py, eval_per_class.py, explain.py, ...
scripts/preprocessing/    # omics + priors preprocessing + build_hetero_graph.py
configs/config_kg_hgnn*.yml
benchmark/                # molecular baselines (MOGNN-TF, MoGCN, Li & Nabavi)
*.sh                      # launchers (no make / no sudo — see below)
```

---

## Requirements

- Conda, CUDA GPU (tested on CUDA 12.8 / torch 2.7 / PyG 2.7).
- Raw/preprocessed data live in `data/` (gitignored): `data/prior_knowledge/PKT/`
  (the KG, fetched from Hugging Face) and `data/training/` (omics + splits). If
  missing, `prepare_data.sh` regenerates everything end-to-end: it downloads the
  **PKT** (`scripts/download_pkt.py`, nodes.zip + edges.zip, ~290 MB) and the
  **TCGA omics** (`scripts/download_pancan.py`), then preprocesses and builds the
  splits.

---

## Running the pipeline (no `make`, no `sudo`)

Every step runs through `conda run`, so neither `make` nor `sudo` is needed. From
the repo root:

```bash
bash setup_env.sh        # 1. create the `gnn` conda env + install the repo (pip install -e .)
bash prepare_data.sh     # 2. data preprocessing (ONLY if data/training/ is absent)
bash make_graph.sh       # 3. feature selection + graph (template) construction
bash train_and_eval.sh   # 4. training + evaluation
```

Each run writes `results/<experiment>/<timestamp>/` with `model_best.pt`
(checkpoint), `run.log`, `history.csv`, `metrics.json`, `per_class_metrics.csv`
(per-class precision/recall/F1), `confusion_matrix.csv` and `graph_info.json`.

### Choosing the backbone
```bash
BACKBONE=hgt          bash train_and_eval.sh   # attention
BACKBONE=hetero_sage  bash train_and_eval.sh   # SAGE per relation (best after tuning)
BACKBONE=rgcn         bash train_and_eval.sh   # one weight per relation
```
Results go to separate folders (`results/kg_hgnn_<backbone>/...`).

### Multi-seed evaluation (paper protocol)
```bash
BACKBONE=hetero_sage bash train_and_eval.sh --runs 5
```
Runs 5 independent splits (seeds 42..46), with feature selection + graph rebuilt
per seed (no leakage) and a fixed model seed; prints **mean ± s.d.** of every
metric.

---

## Metapaths (miRNA–miRNA / TF–TF)

Co-target metapaths (two miRNAs / two TFs sharing a target gene) are **off by
default**. They are decided **at graph-construction time**, so enable them by
passing `METAPATH=--metapath` to `train_and_eval.sh` (or `make_graph.sh`), which
rebuilds the template with those layers:

```bash
BACKBONE=hgt bash train_and_eval.sh                    # WITHOUT metapaths (default)
METAPATH=--metapath BACKBONE=hgt bash train_and_eval.sh # WITH (miRNA, shares_target, miRNA) + (TF, shares_target, TF)
```

> **Reproducibility caveat.** The template is always saved to
> `data/prior_knowledge/hetero/hetero_graph_template.pt`: building it with and
> without metapaths at the **same path** overwrites it, and old checkpoints no
> longer match the new template. For clean comparisons use `--runs` (which saves
> one template per seed) or change `OUT_DIR`.

Other knobs (env vars): `TOP_GENES` (default 700), `TOP_TF` (200), `TOP_MIRNA`
(100), `SEED`, `GO_MIN_SUPPORT` (3), `CONFIG`.

---

## Tuning (Optuna)

One study per backbone (each has different optimal hyperparameters). Objective =
**validation** macro-F1 (never test). Tuning uses the **already-built template**
(it does not rebuild it), so it inherits the metapath choice made with
`make_graph.sh`.

```bash
BACKBONE=hgt bash tune.sh    # one backbone (default 35 trials)
bash tune_all.sh             # all three in sequence
```
Output: `results/optuna/kg_hgnn_optuna_<backbone>/best.json` (+ CSV report). After
tuning, copy the winning params into `configs/config_kg_hgnn_<backbone>.yml` and
run `BACKBONE=<winner> bash train_and_eval.sh --runs 5`.

---

## Experiments

### Ablation (metapath + graph-component removal)
One-factor-at-a-time ablation from the full configuration, via `ablation.sh`
(rebuilds the template only for axes that change the graph, reuses it for
config-only axes):
```bash
bash ablation.sh                                   # all 9 variants, 5 seeds
VARIANTS="full no_pathway no_go" bash ablation.sh  # subset
```
Variants: `full` · `no_metapath` · `no_disease` · `no_pathway` · `no_go` ·
`readout_mol` · `readout_pathway` · `no_cnv` · `no_mirna`. Output in
`results/ablation/`: `ablation_table.csv` (mean ± s.d. macro-F1 + Δ vs `full`)
and `ablation_bars.png`.

Inspect which graph a run was trained on (metapath / scales):
```bash
conda run -n gnn python scripts/kg_hgnn/which_graph.py
```

### Per-class performance (27 classes)
Every run already saves `per_class_metrics.csv/.json` + `confusion_matrix.csv`;
the experiment is the **multi-seed aggregation** + comparison against MOGNN-TF.
```bash
bash experiments.sh per_class
```
Writes `per_class_aggregate.csv` (P/R/F1 mean ± s.d. per class),
`per_class_vs_mognntf.csv` (side-by-side + delta) and `per_class_vs_mognntf.png`.
Sanity check: the mean of the per-class F1 must equal the reported macro-F1.

Recompute per-class files for **old** runs from the checkpoint (no retraining):
```bash
conda run -n gnn python scripts/kg_hgnn/eval_per_class.py --all
```

### Feature collapse ("the graph saves the performance")
Both models over a decreasing gene grid (700→20), comparing the degradation
slope:
```bash
bash run_feature_collapse.sh   # both models, default grid, 5 seeds
```
The heterogeneous model rebuilds FS+template at each gene level; the MOGNN-TF
baseline runs via `scripts/collapse_mognntf.py` (a bridge that forces MOGNN-TF
onto the **same** surviving feature panel for a fair comparison, using
`benchmark/MOGNN-TF/configs/config_final.yml`). Output in
`results/feature_collapse/`: `feature_collapse_table.csv`,
`feature_collapse_robustness.csv` (the slope per model — the answer to the
question) and `feature_collapse_curve.png`. What matters is the **slope**, not the
absolute F1.

### Explainability (mechanistic interpretability)
Mechanism-level attribution: which `pathway`/`GO_term` drive the prediction for
each subtype (GNNExplainer on a trained checkpoint).
```bash
conda run -n gnn python scripts/kg_hgnn/explain.py --run results/<...>/<ts> --per-class 15 --epochs 50
```
Output in `results/explanations/<run>/`: `top_pathway_by_subtype.csv`,
`top_GO_term_by_subtype.csv` (real Reactome/GO ids, with `distinctive_score`) and
subtype×mechanism heatmaps. Biological validation (are the mechanisms sensible for
that tumour?) stays manual — the selling point a gene-only model cannot offer.

---

## Configuration

`configs/config_kg_hgnn.yml` drives a run: `data.split_dir`,
`data.template_path`, ablation switches (`use_cnv`, `use_mirna`),
`model.backbone` / `hidden` / `num_layers` / `heads` / `dropout` /
`readout_types` (multi-scale pooling), `scheduler`, early stopping and
class-imbalance handling (`class_weighted_loss`, `sampler_strategy`).

---

## Benchmarks

Molecular baselines live under [`benchmark/`](benchmark/) and are runnable for
reproduction on the same 27-class split; see [`benchmark/README.md`](benchmark/README.md).

**MOGNN-TF** is our previous single-scale molecular model, used here only as the
reference baseline. It is archived on Zenodo as software with a permanent DOI:

> De Filippis, G. M., Rinaldi, A. M., Tommasino, C. *MOGNN-TF: code and
> supplementary material*. Zenodo. https://doi.org/10.5281/zenodo.21475911

Its code is vendored (unmodified) under `benchmark/MOGNN-TF/` so its results can
be reproduced from this repository; the MOKG-HGNN framework does not depend on it.

# Benchmark baselines

This directory vendors the third-party implementations used to produce the
baseline columns of the model comparison reported in the manuscript and in
Supplementary Table S1.

Both are redistributed under their own licenses so that the comparison can be
re-run from a single checkout. **Copyright remains with the original authors.**

## Contents

### `Multimodal-GNN-for-Cancer-Subtype-Clasification/` — "Li24"

The multimodal GNN framework of Li & Nabavi, which this work extends. Builds a
heterogeneous multi-level supra-graph from prior knowledge, with intra-omic
(gene–gene) and inter-omic (miRNA–gene) connections, over GCN/GAT backbones.

| | |
| --- | --- |
| Reference | Li, B. & Nabavi, S., *A Multimodal Graph Neural Network Framework for Cancer Molecular Subtype Classification*, 2024 — [arXiv:2302.12838](https://arxiv.org/abs/2302.12838) |
| License | MIT — Copyright (c) 2023 CompGBI LAB (see `LICENSE` in the directory) |
| Provides | the `Li24 (GAT)` and `Li24 (GCN)` columns |

```bash
conda env create -f environment.yml
conda activate <env_name>

python cancer_test.py --model gat --num_gene 100 --cancer_subtype True \
  --omic_mode 4 --shuffle_index 0 --gene_gene True --mirna_gene True \
  --mirna_mirna True --parallel True --l2 True --decoder False \
  --poolsize 8 --edge_weight True --epochs 200 --train_ratio 0.7 --test_ratio 0.1
```

### `MoGCN/` — "MoGCN"

Multi-omics integration via a multimodal autoencoder, with a patient similarity
network built by Similarity Network Fusion (SNF) and classified with a GCN.

| | |
| --- | --- |
| Upstream | https://github.com/Lifoof/MoGCN |
| License | MIT — Copyright (c) 2021 Lifoof (see `LICENSE` in the directory) |
| Provides | the `MoGCN` column |

```bash
pip install -r requirements.txt

# 1. autoencoder for dimensionality reduction
python AE_run.py -p1 data/fpkm_data.csv -p2 data/gistic_data.csv \
  -p3 data/rppa_data.csv -m 0 -s 0 -d cpu

# 2. Similarity Network Fusion
python SNF.py -p data/fpkm_data.csv data/gistic_data.csv data/rppa_data.csv \
  -m sqeuclidean

# 3. GCN (or GCN_pyg_run.py for the PyTorch Geometric variant)
python GCN_run.py -fd result/latent_data.csv -ad result/SNF_fused_matrix.csv \
  -ld data/sample_classes.csv -ts data/test_sample.csv -m 1 -d gpu -p 20
```

### `run_tests.py`

Driver that sweeps the Li24 baseline over the same gene counts and backbones
used for MOGNN-TF, so that baseline and proposed model are evaluated under
matched settings.

## Provenance

Both baselines were vendored as source snapshots rather than as git submodules,
so the upstream commit hashes are not recorded in this repository. The snapshots
correspond to the versions available at the time the benchmark was run.

## Methods surveyed but not included

The following methods were reviewed during method selection and are cited in the
Related Work section of the manuscript, but contribute to no reported result.
Their implementations are not vendored here; refer to the original repositories.

| Method | Approach |
| --- | --- |
| MOGONET | Per-omic patient–patient graphs from cosine similarity, fused downstream by VCDN |
| MOGAT | As MOGONET, with GAT attention weighting; no explicit biological edges |
| SUPREME | One patient similarity network per omic, GCN on each, combined with raw features |
| AMOGEL | Graph derived by association rule mining, enriched with biological priors as auxiliary edges |
| MPK-GNN | Per-graph GNN encoders plus an MLP on raw omics, trained with a contrastive objective |

The reported comparison is restricted to methods that, like MOGNN-TF, build the
graph from biological prior knowledge over molecular entities rather than from
patient similarity — which makes Li24 and MoGCN the meaningful points of
reference.

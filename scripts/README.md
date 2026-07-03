# MOGNN-TF — pipeline scripts

Ordine di esecuzione end-to-end. Tutti i comandi assumono di essere lanciati
dalla **root** del repository.

## 1. Download dati TCGA Pan-Cancer

```bash
python scripts/download_pancan.py --out data/raw/tcga_pancan
```
Scarica da Xena: gene expression EB++, miRNA EB++, CNV GISTIC2,
iCluster molecular subtype.

## 2. Conversione raw → parquet (transpose, dtype)

```bash
python scripts/preprocessing/omics/prepare_pancan.py \
    --src data/raw/tcga_pancan \
    --out data/raw/tcga_pancan/processed
```

## 3. Pulizia per-omica, filtro per sample comuni

```bash
python scripts/preprocessing/omics/process_data_pancan.py \
    --normalization
```
Output: `data/omics/{clinical,fpkm,cnv,mirna}.zip`.

## 4. Costruzione *prior knowledge* (GGI, miRNA-target, TF-gene)

```bash
# 4a. Download BioGRID + miRDB + TFLink
python scripts/preprocessing/priors/get_raw_data.py

# 4b. Conversione RefSeq -> gene symbol (per miRDB)
python scripts/preprocessing/priors/refseq2gene.py

# 4c. Build adjacency matrices: GGI, miRNA-gene, miRNA-miRNA, TF-gene
python scripts/preprocessing/priors/load_interaction.py
```
Output: `data/prior_knowledge/{GGI,miRNA,TF}/*.npz` + node lists.

## 5. Wrapper finale: top-100 miRNA per varianza, allineamenti, output training

```bash
python scripts/preprocessing/omics/data_wrapper.py
```
Output: `data/training/{expression_data_pancan.tsv, cnv_data_pancan.tsv,
mirna_data_top_100.tsv, molecular_subtype.csv, adj_matrix_biogrid.npz,
standardized_mirna_mrna_edge_filtered_at_eight_with_top_100_mirna.npz,
tf_gene_adj_global.npz, tf_nodes_all_in_vocab.csv}`.

## 6. Training MOGNN-TF (configurazione finale tesi: GCN, 700 geni, 200 TF, γ=0.98)

```bash
python scripts/train.py --config configs/config.yml
```

## 7. Ablation studies (riproduzione tabelle tesi)

```bash
# Tab. confronto baseline, Tab. ablation geni, sampler, TF, split ratio
python scripts/run_sweep.py --sweep configs/sweep_config.yml
```

## 8. Hyperparameter tuning Optuna

```bash
python scripts/run_optuna.py
```

## 9. Aggregazione risultati + test statistici (Friedman, Wilcoxon-Holm)

```bash
python scripts/collect_results.py --results-root results/pancan
python scripts/analysis/analize_metrics.py
```

# TCGA-PANCAN
## Pipeline per i dati provenienti dal dataset PANCAN di Xena
1. **get_pancan_tcga.py**: Scarica i dati raw all'interno di data/pancan_tcga e li allinea in formato "sample x features" all'interno di data/pancan_tcga/processed
2. **process_data_pancan.py**: processa i dati dei file secondo quanto descritto nel readme in "Repo-tesi/datasets", vengono selezionati i campioni comuni. (eseguire prima Repo-tesi/datasets/load_interaction.py). Output in "data/omics" 
3. **data_wrapper.py**: Processa i file in "dataset/omics" ed i file di inerazione GGI e miRNA-gene in "Repo-tesi/datasets/data/prior_knowledge" per adattarli alla pipeline di Multimodal (Li 24):
    1. Pulizia dati clinici per ottenere le classi
    2. Filtraggio dei geni comuni tra expression, cnv e nodi di Biogrid
    3. Filtraggio dei miRNA comuni tra dati miRNA e mirna da MIRDB
    4. Allineamento CNV (Sample, colonne e lables)
    5. Allineamento Expression (Sample, colonne e lables) e clcolo della varianza per gene
    6. Allineamento miRNA (Sample, colonne) e calcolo dei top 100 miRNA (Li 24) per varianza
    7. Generazione della matrice di adiacenza GGI filtrata sui geni comuni con giusti indici
    8. Generazione della matrice di adiacenza mirna-genes filtrata per top 100 mirna
    9. Generazione della metrice di adiacenza TF-gene
main_preprocessing.py

## Struttura del repository (package layout)

```text
src/multiomics_gnn/
  base_ml/
    trainer.py            # loop training/val/test, checkpoint, metriche
    early_stopping.py     # logica early stopping
    scheduler.py          # scheduler LR (constant/step/exponential/cosine)
    sampler.py            # strategie di sampling (random/weighted/none)
  config/
    config.yml            # configurazione di default (paths, data, model, train, ecc.)
    loader.py             # load_config()
  models/
    gat.py                # GAT/GATv2 (conv) + pooling + head di classificazione (+ decoder opz.)
    gcn.py                # GCN + pooling + head di classificazione (+ decoder opz.)
    baseline.py           # baseline MLP (senza message passing)
    gat_tf.py             # variante GAT con supporto esplicito a TF (num_tf)
  pancancer_prediction/
    preprocessing/
      preprocess_pancan_traning.py  # preprocessing omici + costruzione reti + TF utilities
    datasets/
      OmicDataset.py      # OmicsGraphDataset (PyG Dataset)
    training/
      builder.py          # factory: build_model/optimizer/scheduler/dataloaders/trainer
    experiments/
      experiment_runner.py # entrypoint: orchestration end-to-end dell’esperimento
    utils/
      split.py            # split stratificato/indici shuffle
      save_metrics.py     # salvataggio curve/metriche/plot/report
  utils/
    logger.py             # logging configurabile
    paths.py              # run directories
    seed.py               # determinismo/seed
```
### File di configurazione

`config/config.yml` è strutturata per sezioni:

- **project**: seed, nome esperimento, device
- **paths**: directory dati e output (results, logs, figures, models)
- **data**:
  - `num_gene`, `num_mirna`
  - `omic_mode` (selezione omiche)
  - batch/worker
  - scelte rete: `gene_gene`, `mirna_gene`, `mirna_mirna`
- **model**: `name` (gat/gcn/baseline), `parallel`, pooling (`poolsize`, `poolrate`), `dropout`
- **graph**: `edge_weight`, `edge_attribute`
- **train**: epoche, lr, weight_decay, split validation, `l2`, `decoder`
- **early_stopping**: enabled/patience/metric/strategy
- **optimizer**: `adamw` (default) o `adam`
- **scheduler**: `constant|step|exponential|cosine`
- **sampler_strategy**: `random|weighted|none` (con `sampler_gamma`)
- **wandb**: flag di abilitazione + nome progetto (presente in config)

## Flusso di esecuzione end-to-end

### Entrypoint: `experiment_runner.py`

È l’orchestratore dell’esperimento.

1. Load config (`load_config`)
2. Seed (`set_seed`)
3. Load e preprocessing dati:
   - `load_exp_cnv_and_mirna_data(...)`
   - `down_unified_data(...)` (o la variante con TF)
   - `process_adj(...)` per ottenere `adj`, `edge_index`, `edge_weight`
4. Costruzione dataset (`OmicsGraphDataset`) e split train/val/test
5. Build components model/optimizer/scheduler/dataloaders/trainer (`build_all(...)` in `training/builder.py`)
6. Training (`Trainer.fit()`), poi test e salvataggio metriche/report

### Dataset: `OmicsGraphDataset` (PyG)

`pancancer_prediction/datasets/OmicDataset.py` implementa un `torch_geometric.data.Dataset` che, per ogni indice, crea un oggetto `Data` con:

- `x`: feature tensor del campione, shape tipica `(N_nodes, F)` o `(N_nodes, F, C)` a seconda dell’omic mode e della pipeline
- `edge_index`: topologia del grafo (condivisa per tutti i campioni)
- `y`: label del campione
- opzionale: `edge_weight` oppure `edge_attr`

Include anche una utility per **class imbalance**:

- `get_weight_pancan(gamma)` calcola pesi per classe e restituisce un `WeightedRandomSampler`

---
## Preprocessing e costruzione del grafo

La preparazione dei dati avviene in:
`pancancer_prediction/preprocessing/preprocess_pancan_traning.py`.

### 5.1 Omic mode (selezione feature)

La funzione `omic_mode_translation(omic_mode)` mappa il flag a una modalità operativa:

- `0`: solo **expression**
- `1`: solo **miRNA**
- `2`: **expression + miRNA**
- `3`: **expression + CNV**
- `4`: **expression + CNV + miRNA**

Inoltre è possibile includere nel grafo i Transcription Factors abilitando il flag `enable_tf` ottenendo:
- `0`: mRNA + TF network
- `2`: mRNA + miRNA + TF network
- `3`: mRNA + CNV + TF network
- `4`: mRNA + CNV + miRNA + TF network

Questa scelta impatta:

- numero canali/features per nodo
- quali reti (adjacency) siano ammissibili (es. miRNA–gene non disponibile se non hai miRNA)

### Adiacenza → `edge_index` / `edge_weight`

La funzione `process_adj(cfg, adj, logger=None)`:

- normalizza `adj` dividendo per `max(adj)`
- azzera diagonale e aggiunge self-loop (`adj + I`)
- converte in COO
- costruisce:
  - `edge_index` (shape `[2, E]`) da `(row, col)`
  - `edge_weight` (shape `[E]`) da `adj.data`

Scelta progettuale: **static topology**, cioè la rete biologica è comune a tutti i campioni.

### Estensione TF

Nel preprocessing sono presenti funzioni per TF:

- `filter_tf_adjacency_matrix(...)` (selezione TF, filtri su degree/top-N/percentili oppure filtrando i tf per varianza)
- `get_tf_inner_connection(...)` Costruzione della matrice di adiacenza TF-TF
- `get_tf_mirna_connection(...)` Costruzione della matrice di adiacenza TF-mirna
- `down_unified_data_with_TF(...)` Preprcessing dei dati per omic_mode con tf

In parallelo, esiste `models/gat_tf.py` che estende il modello includendo `num_tf` e la logica di slicing/packing dei nodi `[GENI | miRNA | TF]`.

---

## Training Engine (`base_ml/`)

### Trainer (`base_ml/trainer.py`)

Responsabilità principali:

- **Batch processing**: Costuzione dei batch
  - `_process_batch(batch)`
  - `batch = batch.to(device)`
  - ricostruisce:
    - `B = batch.num_graphs`
    - `N = batch.num_nodes // B`
    - `batch_x = batch.x.view(B, N, -1)`
    - `batch_y = batch.y.view(-1).long()`
    - `batch_edge_weight` se presente

- **Train epoch** Batch training della singola epoca
  - forward
  - loss
  - backward + optimizer step
  - tracking metriche batch/epoch: `accuracy_score`, `f1_score`

- **Eval epoch** (validation)
  - `no_grad`
  - calcolo metriche analoghe

- **Fit** Gestione dell'intera pipeline di addestramento
  - gestisce scheduler LR
  - early stopping (se abilitato)
  - checkpointing (es. `latest_checkpoint.pt` e/o “best”)

### Early stopping (`base_ml/early_stopping.py`)

Gestisce e monitora le best metriche di riferimento per lo stop del processo di addestramento:

- `patience`
- metrica di riferimento (es. `f1`)
- strategia `maximize`/`minimize`

### Scheduler (`base_ml/scheduler.py`)

Wrapper per LR scheduling:

- constant
- step
- exponential
- cosine

La scelta avviene via config (`scheduler.name`).

### Sampler (`base_ml/sampler.py`)

Modulo per il sampling dei campioni, per mitigare l'effetto dello sbilanciamento delle classi:

- `strategy`: `random`, `weighted`, `none`
- `gamma`: smoothing per class imbalance

Integra `WeightedRandomSampler` con pesi calcolati dal dataset.

---

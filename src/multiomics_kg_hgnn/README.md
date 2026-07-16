# multiomics_kg_hgnn — modello eterogeneo multi-scala (Proposta B)

Package del nuovo modello. Riusa il backbone di prior knowledge in
`data/prior_knowledge/hetero/` (costruito da
`scripts/preprocessing/priors/build_hetero_graph.py`) e vi inietta le feature
omiche per paziente, producendo le `HeteroData` pronte per HGT/RGCN/HeteroConv.

È l'analogo eterogeneo della pipeline dati di MOGNN-TF
(`down_unified_data_with_TF` + `OmicsGraphDataset`), scritto da zero per il nuovo
grafo tipizzato invece che per la supra-adiacenza omogenea.

## Struttura

```
models/
  hetero_gnn.py           # HeteroMultiScaleGNN: encoder type-specific + backbone + readout
pancancer_prediction/
  preprocessing/
    feature_selection.py  # FS unificata per varianza (gene/TF/miRNA), train-only
    patient_features.py   # feature injection: omiche -> tensori per tipo di nodo
    make_datasets.py      # orchestratore: template + omiche + split -> 3 dataset + loader
  datasets/
    HeteroOmicDataset.py  # Dataset PyG: 1 HeteroData per paziente (topologia condivisa)
    check_dataset.py      # smoke test end-to-end: dati reali -> HGT -> 27 classi
  training/
    trainer.py            # train/val/test loop, macro-F1, early stopping
  experiments/
    runner.py             # config -> data -> model -> train -> test
```

Entrypoint (sottili, in `scripts/kg_hgnn/`, separati da quelli di MOGNN-TF):
`train.py`, `evaluate.py`. Config: `configs/config_kg_hgnn.yml`.

## Training / test

```bash
# training (config YAML dedicato)
conda run -n gnn python scripts/kg_hgnn/train.py --config configs/config_kg_hgnn.yml
# evaluate su test da checkpoint
conda run -n gnn python scripts/kg_hgnn/evaluate.py \
    --config configs/config_kg_hgnn.yml --checkpoint results/<...>.pt
```

Il config guida tutto: `data.split_dir`, `data.template_path`, ablation
(`use_cnv`, `use_mirna`), `model.backbone` (**hgt** implementato; hetero_sage /
rgcn sono TODO nel registry `models/hetero_gnn.py:BACKBONES`), `model.hidden/
num_layers/heads`, `model.readout_types` (pooling multi-scala), sbilanciamento
(`train.class_weighted_loss` o `sampler_strategy: weighted`).

Metrica primaria: **macro-F1** (proposta sez. 5). Verificato end-to-end su GPU:
il modello impara (val macro-F1 0.16→0.40 in 3 epoch di smoke).

Ogni run scrive una cartella `results/<experiment_name>/<timestamp>/`:
- `model_best.pt` — checkpoint del best model (per validazione o riuso)
- `run.log` — log completo (oltre allo stdout)
- `history.csv` — metriche per-epoca (per i grafici di training)
- `metrics.json` — risultati finali (best val + test)
- `config.json` — config usato (riproducibilità)

`make evaluate` (o `scripts/kg_hgnn/evaluate.py`) ricarica automaticamente
l'ultimo `model_best.pt` per quell'experiment e riproduce le metriche di test
senza ri-addestrare (passa `CKPT=path` per un run specifico).

**Perché non si sovrappone a MOGNN-TF:** la logica sta in package separati
(`multiomics_gnn` vs `multiomics_kg_hgnn`), gli entrypoint in cartelle separate
(`scripts/` vs `scripts/kg_hgnn/`) e i config sono file distinti. Nessun file
condiviso viene modificato (si riusa solo `config.loader.load_config`).

## Come funziona

1. **`patient_features.build_features`** allinea per identificatore (non per
   posizione) le omiche ai vocabolari del template (`node_gene/TF/miRNA.csv`):
   - `gene.x`  = [N, n_gene, C]  (expression [, CNV])
   - `TF.x`    = [N, n_tf,   C]  (i TF sono geni: prendono expr/CNV)
   - `miRNA.x` = [N, n_mirna, 1] (miRNA expression)

   Lo scaler è fittato **solo sul train** e applicato a tutti (no leakage, come nel
   vecchio runner). Le scale superiori (pathway/GO/disease) restano **featureless**:
   il modello dà loro embedding appresi.

2. **`HeteroOmicsDataset`** condivide la topologia (edge_index statici) tra tutti i
   pazienti e attacca solo le feature per-paziente + la label graph-level.

3. **`make_datasets` / `build_loaders`** caricano template+omiche+split, creano i 3
   dataset e i `DataLoader` PyG (con opzione WeightedRandomSampler per lo
   sbilanciamento a 27 classi).

## Uso

```bash
# sanity build (stampa il primo batch)
conda run -n gnn python -m multiomics_kg_hgnn.pancancer_prediction.preprocessing.make_datasets \
    --split-dir data/training/splits/splits_seed_42 --batch-size 16

# end-to-end: dati reali -> HeteroData -> HGT -> logit 27 classi (+ backward)
conda run -n gnn python -m multiomics_kg_hgnn.pancancer_prediction.datasets.check_dataset
```

Da codice:

```python
from multiomics_kg_hgnn.pancancer_prediction.preprocessing.make_datasets import (
    make_datasets, build_loaders)
tr, va, te, ncls, classes = make_datasets("data/training/splits/splits_seed_42")
train_loader, val_loader, test_loader = build_loaders(tr, va, te, batch_size=16)
```

Ablation sulle modalità (Proposta B, sez. 6): `use_cnv=False`, `use_mirna=False`.

## Feature selection unificata (coerente, tutti i livelli insieme)

Per limiti di compute (full-batch RTX 2060) e per la comparabilità con MOGNN-TF
la feature selection va fatta, ma con **un solo criterio (varianza) e in un solo
momento, su tutti i livelli insieme**, partendo dai set COMPLETI — non miRNA
prima e geni dopo. MOGNN-TF seleziona per varianza gene/TF/miRNA sul solo train
(config `feature_selection_method: variance`), e questo lo replica.

`feature_selection.py`:
- materializza il miRNA completo (743) da `data/omics/mirna.zip`
  (`mirna_data_full.tsv`), perché `data/training` shippa solo il top-100 pre-tagliato;
- calcola la **varianza sul solo split di train** (no leakage) per gene / TF / miRNA;
- scrive i pannelli selezionati (default **700 geni / 200 TF / 100 miRNA**).

Flusso completo:

```bash
FS=data/training/feature_selection/splits_seed_42
# 1) FS unificata (varianza, train-only, tutti i livelli)
conda run -n gnn python -m multiomics_kg_hgnn.pancancer_prediction.preprocessing.feature_selection \
    --split-dir data/training/splits/splits_seed_42 --top-genes 700 --top-tf 200 --top-mirna 100
# 2) template ridotto sui pannelli selezionati
conda run -n gnn python scripts/preprocessing/priors/build_hetero_graph.py \
    --gene-list $FS/selected_genes.csv --tf-list $FS/selected_tf.csv \
    --mirna-list $FS/selected_mirna.txt --go-min-support 3 --force
# 3) dataset (copertura ~100% su tutti i tipi)
conda run -n gnn python -m multiomics_kg_hgnn.pancancer_prediction.datasets.check_dataset
```

Verificato: con 700/200/100 il grafo scende a ~3k nodi e la copertura è 100%
per gene/TF/miRNA.

### Numero di feature (700 / 800 / 900 …)

Si cambia con un flag, senza toccare il codice:

```bash
conda run -n gnn python -m multiomics_kg_hgnn.pancancer_prediction.preprocessing.feature_selection \
    --split-dir data/training/splits/splits_seed_42 \
    --top-genes 900 --top-tf 200 --top-mirna 100 \
    --out-dir data/training/feature_selection/g900_seed42
```

### Metapath miRNA-miRNA / TF-TF (opzionale, `--metapath`)

Aggiunge al template il layer di co-target di MOGNN-TF:
`(miRNA, shares_target, miRNA)` (dal file `metapath_mirna_mirna.csv`, due miRNA che
condividono un gene target) e `(TF, shares_target, TF)` (due TF che regolano un
gene selezionato comune, calcolato come A·Aᵀ). Con `--metapath` un nodo del
pannello **sopravvive se ha QUALSIASI arco (gene o metapath)**, recuperando i
miRNA/TF che non toccano direttamente un gene selezionato.

```bash
FS=data/training/feature_selection/g900_seed42
conda run -n gnn python scripts/preprocessing/priors/build_hetero_graph.py \
    --gene-list $FS/selected_genes.csv --tf-list $FS/selected_tf.csv \
    --mirna-list $FS/selected_mirna.txt --go-min-support 3 --metapath --force
```

Effetto verificato (900 geni): senza metapath miRNA=63 / TF=98; con `--metapath`
miRNA=91 / TF=107 (più nodi del pannello sopravvivono). È un asse di ablation:
costruisci due template (con/senza) per confrontarli.

**Note metodologiche:**
- Anche con `--metapath` un nodo del pannello del tutto isolato (nessun target
  selezionato e nessun co-target) resta escluso: è corretto, non influenzerebbe la
  predizione. Il TF-TF è calcolato sui soli geni selezionati (come MOGNN-TF), quindi
  un TF che non regola alcun gene selezionato non viene recuperato.
- Per il rigore multi-seed, rigenera FS + template **per ogni split** (la varianza
  è calcolata sul train di quel seed).

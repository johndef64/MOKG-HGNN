# MOKG-HGNN — Multi-scale Heterogeneous GNN per sottotipi tumorali pan-cancer

Evoluzione di **MOGNN-TF**: da un grafo molecolare *a singola scala*
(gene / miRNA / TF) a un **grafo eterogeneo gerarchico** che collega lo strato
molecolare a scale biologiche superiori — **pathway → GO → malattia** — integrando
un Knowledge Graph biomedico (PKT / PheKnowLator). Il task resta la classificazione
del sottotipo tumorale (27 classi iCluster), per confrontabilità diretta col
baseline; il *reasoning* però attraversa più livelli di organizzazione biologica.

Vedi `docs/proposta_B_tesi_multiscale_heterognn.md` per il posizionamento e
`docs/modelli_e_tuning_spiegazione.md` per la spiegazione dei modelli.

---

## Cosa contiene

- **Grafo eterogeneo** `HeteroData` con 6 tipi di nodo (`gene`, `miRNA`, `TF`,
  `pathway`, `GO_term`, `disease`) e relazioni tipizzate (gene↔gene, miRNA→gene,
  TF→gene, gene→pathway, gene→GO, GO→GO, gene→disease, + metapath opzionali).
- **Tre backbone eterogenei** selezionabili (spettro di complessità):
  `rgcn` (un peso per relazione) · `hetero_sage` (SAGE per relazione) ·
  `hgt` (attention per meta-relazione).
- **Feature selection unificata** per varianza (gene/TF/miRNA), train-only.
- **Tuning** con Optuna (uno studio per backbone) e **valutazione multi-seed**.

Struttura:
```
src/multiomics_kg_hgnn/   # package del modello nuovo (separato da multiomics_gnn = MOGNN-TF)
scripts/kg_hgnn/          # entrypoint: train.py, evaluate.py, run_optuna.py, eval_per_class.py, plot_training.py
scripts/preprocessing/    # preprocessing omiche + priors + build_hetero_graph.py
configs/config_kg_hgnn.yml
*.sh                      # launcher senza make/sudo (vedi sotto)
docs/                     # proposta, piani, TODO tesi, spiegazioni
```

---

## Requisiti

- Conda, GPU CUDA (testato su CUDA 12.8 / torch 2.7 / PyG 2.7).
- I dati grezzi/preprocessati: `data/prior_knowledge/PKT/` (il KG, fornito) e
  `data/training/` (omiche + split). Se `data/training/` manca, va rigenerato con
  `prepare_data.sh` (richiede i dati grezzi TCGA / connettività).

---

## Esecuzione (senza `make`, senza `sudo`)

Tutti i passi sono lanciabili con `conda run`, quindi non serve `make` né `sudo`
(conda installa nella home). Dalla root della repo:

```bash
bash setup_env.sh        # 1. crea l'env conda `gnn` + installa la repo
bash prepare_data.sh     # 2. preprocessing dati (SOLO se data/training/ non c'è)
bash make_graph.sh       # 3. feature selection + costruzione del grafo (template)
bash train_and_eval.sh   # 4. training + valutazione
```

Ogni run scrive `results/<experiment>/<timestamp>/` con: `model_best.pt`
(checkpoint), `run.log`, `history.csv`, `metrics.json`, `per_class_metrics.csv`
(precision/recall/F1 per classe) e `confusion_matrix.csv`.

### Scegliere il backbone
```bash
BACKBONE=hgt          bash train_and_eval.sh   # attention
BACKBONE=hetero_sage  bash train_and_eval.sh   # SAGE per relazione
BACKBONE=rgcn         bash train_and_eval.sh   # un peso per relazione
```
I risultati finiscono in cartelle separate (`results/kg_hgnn_<backbone>/...`).

### Valutazione multi-seed (protocollo del paper)
```bash
BACKBONE=rgcn bash train_and_eval.sh --runs 5
```
Fa 5 run su split indipendenti (seed 42..46), con feature selection + grafo
ricostruiti per ogni seed (nessun leakage) e seed del modello fisso; alla fine
stampa **media ± s.d.** di tutte le metriche.

---

## Provare i METAPATH (miRNA-miRNA / TF-TF)

I metapath di co-target (due miRNA / due TF che condividono un gene target) sono
**disattivati di default**. Si decidono **al momento della costruzione del grafo**,
quindi si attivano passando `METAPATH=--metapath` a `train_and_eval.sh` (o a
`make_graph.sh`), che ricostruisce il template con quei layer:

```bash
# SENZA metapath (default)
BACKBONE=hgt bash train_and_eval.sh

# CON metapath — aggiunge (miRNA, shares_target, miRNA) e (TF, shares_target, TF)
METAPATH=--metapath BACKBONE=hgt bash train_and_eval.sh
```

Effetto: con `--metapath` più nodi del pannello sopravvivono (un nodo resta se ha
**qualsiasi** arco, gene o metapath), non solo se tocca direttamente un gene
selezionato. È un asse di ablation: costruisci le due versioni e confrontale.

> **Attenzione (riproducibilità).** Il template si salva sempre in
> `data/prior_knowledge/hetero/hetero_graph_template.pt`: costruirlo con e senza
> metapath allo **stesso percorso** lo sovrascrive, e i checkpoint vecchi non
> combaciano più col template nuovo. Per confronti puliti usa `--runs` (che salva
> un template per seed) o cambia `OUT_DIR`.

Altri knob utili (env var): `TOP_GENES` (default 700), `TOP_TF` (200),
`TOP_MIRNA` (100), `SEED`, `GO_MIN_SUPPORT` (3), `CONFIG`.

---

## Tuning (Optuna)

Un studio per backbone (ognuno ha iperparametri ottimali diversi). Obiettivo =
macro-F1 di **validazione** (mai il test). Il tuning usa il **template già
costruito** (non lo ricostruisce), quindi eredita la scelta metapath fatta con
`make_graph.sh`.

```bash
BACKBONE=hgt bash tune.sh            # un backbone (default 35 trial, 10h)
bash tune_all.sh                     # tutti e tre in sequenza, ~20h totali (6.5h ciascuno)
```
Output: `results/optuna/kg_hgnn_optuna_<backbone>/best.json` (+ report CSV).
Dopo il tuning: copia i params vincenti in `configs/config_kg_hgnn.yml` e lancia
`BACKBONE=<vincente> bash train_and_eval.sh --runs 5`.

---

## Esperimenti (studi della tesi)

Gli esperimenti che giustificano il lavoro oltre l'accuratezza aggregata (vedi
`docs/TODO_esperimenti_tesi.md` per la motivazione di ciascuno).

### 0. Ablation: con / senza METAPATH e rimozione di componenti del grafo

I metapath si attivano alla **costruzione del grafo** (vedi sezione METAPATH sopra):
```bash
BACKBONE=hgt bash train_and_eval.sh                    # senza metapath (default)
METAPATH=--metapath BACKBONE=hgt bash train_and_eval.sh  # con metapath
```
Rimozione di componenti (asse di ablation, un fattore per volta):
```bash
# togli la scala 'disease' dal grafo (ricostruisce il template)
METAPATH= bash make_graph.sh   # poi rilancia build con --no-disease (vedi build_hetero_graph.py)
# togli scale dal readout multi-scala (nel config):
#   model.readout_types: [gene]                    -> solo molecolare
#   model.readout_types: [gene, pathway]           -> + pathway
#   model.readout_types: [gene, pathway, GO_term]  -> + GO
# togli una modalità omica (nel config):
#   data.use_cnv: false     |   data.use_mirna: false
```
Per sapere con che grafo è stata addestrata una run (metapath sì/no):
```bash
conda run -n gnn python scripts/kg_hgnn/which_graph.py
```
> Nota: l'ablation "rimozione componenti" non ha ancora un unico orchestratore
> (è nel TODO). I flag esistono già tutti; si combinano a mano come sopra.

### 1. Performance per-classe (27 classi)

Ogni run salva già `per_class_metrics.csv/.json` + `confusion_matrix.csv`. Per una
run esistente (dal checkpoint, senza riaddestrare):
```bash
conda run -n gnn python scripts/kg_hgnn/eval_per_class.py --run results/kg_hgnn_hgt/<...>
conda run -n gnn python scripts/kg_hgnn/eval_per_class.py --all
```
Confronta la tabella per-classe col baseline: l'eterogeneo può vincere su classi
rare pur perdendo sull'aggregato.

### 2. Crollo delle feature ("il grafo salva le performance")

Entrambi i modelli su una griglia decrescente di geni (700→20), con confronto
della pendenza di degrado:
```bash
bash run_feature_collapse.sh                                   # entrambi, griglia default, 5 seed
MODELS="mokghgnn" GENE_GRID="700 300 100 50" SEEDS="42 43 44" bash run_feature_collapse.sh
```
Output in `results/feature_collapse/`: `feature_collapse_table.csv` (macro-F1
media ± s.d. per modello×n-geni) e `feature_collapse_curve.png` (le due curve
sovrapposte). **Cosa guardare**: la curva MOKG-HGNN più alta a 50-100-20 geni.
Rigenerare solo tabella/curva:
```bash
conda run -n gnn python scripts/kg_hgnn/collapse_aggregate.py --results results/feature_collapse
```

### 3. Explainability (interpretabilità meccanicistica)

**Da implementare** (TODO P1.2): attribuzioni su nodi `pathway`/`GO_term`
(GNNExplainer / Captum / attention di HGT) → tabella "per sottotipo → top
pathway/GO", validata contro l'oncologia nota. È il *selling point* della tesi:
ciò che un modello gene-only non può dare. Nessuno script ancora disponibile.

---

## Analisi dei risultati

```bash
# curve di training (una figura per run; --aggregate per media±banda su piu run)
conda run -n gnn python scripts/kg_hgnn/plot_training.py --aggregate

# metriche per-classe da un checkpoint gia salvato (senza riaddestrare)
conda run -n gnn python scripts/kg_hgnn/eval_per_class.py --run results/kg_hgnn_hgt/<...>
conda run -n gnn python scripts/kg_hgnn/eval_per_class.py --all
```

---

## Configurazione

`configs/config_kg_hgnn.yml` guida un run: `data.split_dir`, `data.template_path`,
ablation (`use_cnv`, `use_mirna`), `model.backbone` / `hidden` / `num_layers` /
`heads` / `dropout` / `readout_types` (pooling multi-scala), `scheduler`,
early stopping e gestione dello sbilanciamento (`class_weighted_loss`,
`sampler_strategy`).

---

## Documentazione (`docs/`)

- `proposta_B_tesi_multiscale_heterognn.md` — proposta e posizionamento della tesi.
- `modelli_e_tuning_spiegazione.md` — i tre modelli, da dove vengono, come si usano.
- `piano-mapping-PKT-heterodata.md` — schema del KG e mapping verso `HeteroData`.
- `identificatori_MOGNN-TF.md` — identificatori (Entrez/HGNC/…) usati dai dati.
- `TODO_esperimenti_tesi.md` — esperimenti che giustificano il lavoro (per-classe,
  crollo sotto scarsità di feature, interpretabilità, connettività, …).
- `runtime_on_server.md` — stima dei tempi di training sul server.

---

## Rapporto con MOGNN-TF

Questo repo contiene **entrambi** i progetti, in package separati:
`src/multiomics_gnn/` (MOGNN-TF, baseline omogeneo) e `src/multiomics_kg_hgnn/`
(questo modello eterogeneo). Non condividono codice se non il caricamento del
config; gli entrypoint (`scripts/` vs `scripts/kg_hgnn/`) e i config sono distinti.

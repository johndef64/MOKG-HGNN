# Modelli di Benchmarking

Questo documento descrive i modelli selezionati per il benchmarking, approfondendo le loro caratteristiche principali. Di seguito sono riportate le modalità di fusione, la costruzione del grafo, i dettagli sui modelli e le omiche utilizzate.

## Modalità di Fusione

![Immagine del processo di benchmarking](images/image.png)
Le modalità di fusione rappresentano le strategie utilizzate per integrare i dati multi-omici:

- **Early Fusion**: L'integrazione avviene prima dell'elaborazione, combinando i dati grezzi provenienti da diverse omiche in un unico input
- **Intermediate Fusion**: Ogni omica viene elaborata singolarmente utilizzando sottoreti per generare embedding, che vengono poi combinate per l'addestramento del modello
- **Late Fusion**: Le omiche vengono elaborate separatamente e i risultati finali vengono combinati in una fase successiva

## Costruzione del Grafo
La costruzione del grafo è un passaggio cruciale per rappresentare le relazioni tra i dati. Esistono due tipi di approcci:

- **Patient Similarity Network**: Questo approccio si basa sulla costruzione di un grafo fondato sulla somiglianza tra i pazienti, utilizzando metriche di distanza o correlazione calcolate sui dati multi-omici.
- **Prior Knowledge Graphs**:Viene utilizzata una conoscenze a priori, come reti biologiche o pathway, per costruire un grafo che rappresenti le relazioni note tra geni, proteine o altre entità biologiche

## Modelli di Benchmarking

I modelli selezionati per il benchmarking che rappresentano lo stato dell'arte scelto:

- **MOGONET**: Costruisce, per ogni omica, un grafo paziente–paziente basato su similarità coseno. Integra le viste a valle con VCDN per identificare correlazioni nel dominio delle lables.
- **MOGAT**: Come MOGONET usa grafi paziente–paziente, ma applica GAT per pesare i vicini. Non inserisce archi biologici espliciti (PPI/KEGG). PK assente.
- **MoGCN**: Integra un autoencoder multimodale per estrarre rappresentazioni dai dati (RNA-seq, CNV, proteomica) e costruisce una rete di similarità tra pazienti tramite Similarity Network Fusion (SNF).
- **SUPREME**:Integra più reti di similarità tra pazienti (una per omica) e applica GCN su ciascuna rete; le embedding dei pazienti ottenute vengono combinate con le raw features
- **AMOGEL**: Genera un grafo dai dati via Association Rule Mining (intra/inter-omico) e lo arricchisce con prior biologici come archi ausiliari. Usa attenzione e produce ranking di geni.
- **Multimodal GNN (Li & Nabavi 2024)**: Implementa una GNN multimodale basata su un supra-grafo eterogeneo multilivello, costruito utilizzando PK, con connessioni intra-omiche (gene–gene) e inter-omiche (es. miRNA–gene). Fa uso di GCN/GAT.
- **MPK-GNN**: Combina rappresentazioni locali, apprese tramite encoder GNN per ciascun grafo biologico, e rappresentazioni globali, derivate da una rete MLP sui dati multi-omici originali. Utilizza un meccanismo di apprendimento contrastivo per massimizzare la coerenza tra rappresentazioni di campioni ottenuti da grafi differenti.

MOGONET → GCN + VCDN

MoGCN → AE + GCN

MOGAT → GAT

SUPREME → GCN (+ raw fusion)

AMOGEL → ARM + GNN(attn)

Multimodal GNN (Li & Nabavi 2024) → GCN/GAT (hetero)

MPK-GNN → GNN enc + MLP + Contrastive Learning


## Run

### AMOGEL
```bash
# Installazione dipendenze
pip install -r requirements.txt

# Esecuzione del modello con dataset BRCA
python src/amogel/model.py --dataset BRCA
```

### MOGAT-main
```bash
# Installazione dipendenze
pip install -r Requirements.txt

# Esecuzione del modello
python mogat1.py
python mogat2.py
```

### MoGCN
```bash
# Installazione dipendenze
pip install -r requirements.txt

# Esecuzione sequenziale del workflow
# 1. Autoencoder per riduzione dimensionalità
python AE_run.py -p1 data/fpkm_data.csv -p2 data/gistic_data.csv -p3 data/rppa_data.csv -m 0 -s 0 -d cpu

# 2. Similarity Network Fusion
python SNF.py -p data/fpkm_data.csv data/gistic_data.csv data/rppa_data.csv -m sqeuclidean

# 3. GCN standard
python GCN_run.py -fd result/latent_data.csv -ad result/SNF_fused_matrix.csv -ld data/sample_classes.csv -ts data/test_sample.csv -m 1 -d gpu -p 20

# oppure GCN con PyTorch Geometric
python GCN_pyg_run.py -fd result/latent_data.csv -ad result/SNF_fused_matrix.csv -ld data/sample_classes.csv -ts data/test_sample.csv -m 1 -d gpu -p 20
```

### MOGONET
```bash
# Installazione dipendenze
pip install -r requirements.txt

# Esecuzione per classificazione
python main_mogonet.py

# Esecuzione per identificazione biomarker
python main_biomarker.py
```

### MPK-GNN
```bash
# Installazione dipendenze
pip install -r requirements.txt

# Prima di eseguire bisogna scompattare data/processed_data/1000_0.1/test_data.zip

# Esecuzione del modello
python run_results.py
```

### Multimodal-GNN-for-Cancer-Subtype-Clasification
```bash
# Installazione environment conda
conda env create -f environment.yml
conda activate [nome_environment]

# Esecuzione con parametri test per dataset BRCA
python cancer_test.py --model gat --num_gene 100 --cancer_subtype True --omic_mode 4 --shuffle_index 0 --gene_gene True --mirna_gene True --mirna_mirna True --parallel True --l2 True --decoder False --poolsize 8 --edge_weight True --epochs 200 --train_ratio 0.7 --test_ratio 0.1
```

### SUPREME
```bash
# Installazione environment conda
conda env create -f environment.yml
conda activate [nome_environment]

# Esecuzione con dati di esempio
python SUPREME.py

# Esecuzione con dati personalizzati
python SUPREME.py -data user_defined_data
```




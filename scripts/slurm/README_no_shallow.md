# Esperimento no-shallow sul cluster iknos (Slurm)

Script per lanciare `experiments/no_shallow_matched/` (ticket del 2026-10-10) sul cluster
iknos, seguendo la guida del cluster (`server-guide.md`, tenuta solo in locale e non versionata:
contiene indirizzi e dettagli interni). Tutto è nuovo: nessuno script esistente viene modificato.

| File | Cosa fa |
|---|---|
| `no_shallow_preflight.py` | controlla ambiente Python, versioni (torch 2.7.0, PyG 2.7.0), file di input, GPU |
| `no_shallow_env.sh` | incluso dagli sbatch: attiva l'ambiente, va nel repo, stampa nodo, GPU e risorse |
| `no_shallow_sanity.sbatch` | passo 1, GPU: preflight + sanity check (statico + 2 riproduzioni con shallow) |
| `no_shallow_runs.sbatch` | passo 2, GPU: i 35 run come job array (7 livelli, max 3 task insieme) |
| `no_shallow_analysis.sbatch` | passo 3, CPU: tabelle, figure, report |
| `no_shallow_submit.sh` | sottomette la catena 1 → 2 → 3 con `--dependency=afterok` |

## Perché il sanity check va rifatto sul cluster

Le serie esistenti (multi-scala e molecolare con shallow) sono state calcolate sul server
precedente; il cluster ha GPU diverse. Il sanity check verifica che, su questa macchina, il
modello originale con shallow riproduca il valore esistente (tolleranza fissata a priori: 0.02
di macro-F1). `run_no_shallow.py` parte solo se `sanity_check.json` dice `passed: true` ed è
stato prodotto sullo **stesso modello di GPU**. Per questo tutti i passi GPU chiedono lo stesso
tipo, `gpu:rtx3090` (iknos-gpu1 e iknos-gpu3 hanno entrambi una RTX 3090). Se si preferisce la
RTX 5000 Ada (iknos-gpu2), cambiare `--gres` in **entrambi** gli sbatch GPU.

## 1. Portare codice e dati in `~/private/MOKG-HGNN`

Struttura attesa (percorsi relativi alla radice del repo; serve circa 1.8 GB):

- il codice, via git (`git clone https://github.com/johndef64/MOKG-HGNN.git`, poi `git pull`):
  porta `experiments/no_shallow_matched/`, `scripts/slurm/` e
  `scripts/preprocessing/priors/hgcn_mart_conversion_table.zip`;
- **fuori da git** (dati, gitignored), da copiare a parte:
  - `data/training/`: `expression_data_pancan.tsv`, `cnv_data_pancan.tsv`, `mirna_data_top_100.tsv`,
    `mirna_data_full.tsv` (oppure `data/omics/mirna.zip`), `molecular_subtype.csv`,
    `tf_nodes_all_in_vocab.csv`, `adj_matrix_biogrid.npz`, `tf_gene_adj_global.npz`,
    `standardized_mirna_mrna_edge_filtered_at_eight_with_top_100_mirna.npz`, `splits/splits_seed_4{2..6}/`;
  - `data/prior_knowledge/`: `GGI/GGI_edges_undirected.csv`, `miRNA/miRNA_gene_interactions.csv`,
    `TF/TF_interactions.tsv`;
  - `results/feature_collapse/` (serie esistenti, sola lettura): bastano i file piccoli, non i
    checkpoint: `collapse_mognntf_summary.csv`, `mokghgnn_g*/<ts>/{metrics,config,graph_info}.json`,
    `mognntf_g*/run_*/{used_config.yaml,experiment.log}`, `_aligned_fs/`.

Usare la copia **più completa** di `results/feature_collapse/`. Quella del portatile è parziale
(34/35 config dei run con shallow, 30/35 cartelle `_aligned_fs`): funziona, ma 5 pannelli a 700
geni risultano "non verificabili" invece che verificati. Se i dati del server precedente sono
stati migrati (guida §3, per esempio in `~/private/old_home`), prendere quella.

Da un computer Linux o macOS, con rsync, il sottoinsieme dei risultati senza checkpoint:

```bash
rsync -av --prune-empty-dirs \
  --include='*/' --include='*.json' --include='*.csv' --include='*.yaml' \
  --include='experiment.log' --include='*.txt' --exclude='*' \
  MOKG-HGNN/results/feature_collapse/ tuo-utente@<host-del-cluster>:private/MOKG-HGNN/results/feature_collapse/
```

Da Windows: `scp -r` dal PowerShell, oppure Files nel portale (upload). Gli output
dell'esperimento (`experiments/no_shallow_matched/results/`, tranne `preliminary_check.md`)
sono gitignored: sanity e run si producono sul cluster.

**Terminazioni di riga.** Gli script sono in LF. Se il trasferimento li converte in CRLF, bash
e sbatch falliscono (`$'\r': command not found`). Controllo sul cluster:
`grep -c $'\r' scripts/slurm/*.sh scripts/slurm/*.sbatch` deve dare 0 per ogni file
(altrimenti `sed -i 's/\r$//' scripts/slurm/*`).

## 2. L'ambiente: `gnn` più il venv `gnn-ns`

L'ambiente di base è quello ufficiale `gnn` (Python 3.10, torch 2.7.0 con CUDA 12.8, PyTorch
Geometric 2.7.0: le stesse versioni con cui sono state prodotte le serie esistenti sul server
precedente; il preflight lo verifica). Il runner di MOGNN-TF importa all'avvio anche `optuna` e
`statsmodels`, che `gnn` non ha. Negli ambienti ufficiali non si installa (guida §4), quindi si
crea una volta un venv sopra `gnn` che aggiunge solo quei due pacchetti, con le versioni di
`gnn` bloccate (numpy, scipy, pandas, torch non devono cambiare):

```bash
source /home/apps/iknos-env.sh
iknos_activate gnn
pip list --format=freeze > ~/gnn-pins.txt
iknos_venv_create gnn-ns gnn
iknos_activate $USER/gnn-ns
python -m pip install -c ~/gnn-pins.txt optuna statsmodels
```

Gli script scelgono l'ambiente da soli: `$USER/gnn-ns` se esiste, altrimenti `gnn`. Per forzarne
un altro: `export NS_ENV=<ambiente>` prima di sottomettere.

Prima di lanciare, verificare ambiente e input dal nodo di login (terminale del portale,
Clusters → Iknos Cluster Shell Access, o SSH), dalla radice del repo:

```bash
source /home/apps/iknos-env.sh && iknos_activate $USER/gnn-ns
python scripts/slurm/no_shallow_preflight.py
```

`no_shallow_submit.sh` ripete comunque il preflight e non accoda nulla se fallisce.

## 3. Lanciare

```bash
cd <radice del repo>
bash scripts/slurm/no_shallow_submit.sh
```

Sottomette i tre passi con le dipendenze: i run partono solo se il sanity finisce bene,
l'analisi solo se tutti i 7 task dei run finiscono bene. I log vanno in
`experiments/no_shallow_matched/slurm_logs/`. Gli sbatch vanno sottomessi **dalla radice del
repo**: i percorsi `#SBATCH --output` sono relativi alla cartella da cui si lancia, e i job
usano quella cartella come repo (per forzarne un'altra: `export NS_REPO=<percorso>`).
Lanciare un passo da solo: `mkdir -p experiments/no_shallow_matched/slurm_logs` e poi
`sbatch scripts/slurm/no_shallow_<passo>.sbatch`.

Seguire:

```bash
squeue -u $USER -o '%.10i %.12j %.8T %.10M %.10L %R'
tail -f experiments/no_shallow_matched/slurm_logs/ns-sanity-<jobid>.out
grep -E "\[repro\]|\[sanity\]" experiments/no_shallow_matched/slurm_logs/ns-sanity-*.out
ls experiments/no_shallow_matched/results/runs/ | wc -l        # 35 a fine passo 2
sacct -j <jobid> -o JobID,JobName%15,State,Elapsed,MaxRSS,ExitCode
```

## 4. Se qualcosa va storto

- **Sanity fallito** (job `FAILED`, riproduzione fuori tolleranza o controlli statici falsi):
  i passi successivi restano in coda con `DependencyNeverSatisfied`. Cancellarli
  (`scancel <id>`), leggere `sanity_check.json` e il `.err`, e **non** procedere con i run: per
  il ticket il disegno va rivisto prima.
- **Task dei run interrotto** (`PREEMPTED`/`REQUEUED`, coda `low`): riparte da solo; i run già
  conclusi hanno il loro JSON e vengono saltati, si rifà solo quello in corso.
- **Task fallito per altro motivo**: correggere e rilanciare solo i task che servono, per
  esempio `sbatch --array=2,5 scripts/slurm/no_shallow_runs.sbatch`, poi l'analisi.
- **Sanity già superato su questo cluster** e si vogliono rilanciare solo run e analisi:
  `bash scripts/slurm/no_shallow_submit.sh --skip-sanity`.

## 5. Tempi attesi

Sul server precedente un run di MOGNN-TF a 700 geni durava circa 3 minuti (3.5 s/epoca). Una RTX
3090 dovrebbe essere comparabile: sanity circa 30-40 minuti (rigenerazione dei 35 pannelli più
2 run), i 35 run circa 1 ora con 3 task in parallelo, analisi pochi minuti. I tetti `--time`
(2:30, 3:00, 0:30) lasciano ampio margine.

## 6. Riportare i risultati

A fine catena, `experiments/no_shallow_matched/results/` contiene `sanity_check.json`, `runs/`
(35 JSON), `tables/`, `figures/` (PDF e PNG) e `REPORT_no_shallow.md`. Copiarli sul proprio
computer (Files nel portale, oppure `scp -r`). `runs_raw/` contiene i checkpoint e le cartelle
complete di MOGNN-TF: servono solo per eventuali analisi successive.

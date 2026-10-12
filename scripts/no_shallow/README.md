# Confronto appaiato senza ramo shallow nel modello molecolare

Esperimento aggiuntivo (ticket del 2026-10-10). Rimuove il ramo shallow MLP (theta2) dal solo
modello molecolare MOGNN-TF nell'esperimento appaiato di scarsità di feature (tesi Sez. 4.5.3)
e lo confronta con le serie già ottenute, che si leggono soltanto:

- **MS**: multi-scala MOKG-HGNN (HeteroSAGE), 7 livelli x 5 split;
- **WS**: molecolare MOGNN-TF con shallow, stessi pannelli e split;
- **NS**: molecolare MOGNN-TF senza shallow (nuovo, 35 run).

Regola: nessun file esistente della pipeline viene modificato; l'esperimento riusa per import
il codice esistente. Come gli altri esperimenti della repo, il codice sta in `scripts/` e gli
output in `results/`.

## Cosa riusa (per import, senza copiarlo)

| Elemento | Dove sta | Come è usato |
|---|---|---|
| Script dell'esperimento appaiato | `scripts/collapse_mognntf.py` | `_panel` (pannello per livello e split) e `_set` (override di config) |
| Feature selection unificata e sopravvivenza miRNA/TF | `src/.../preprocessing/feature_selection.py`, `scripts/kg_hgnn/survived_features.py` | chiamati da `_panel` |
| Modello molecolare con ramo shallow | `benchmark/MOGNN-TF/multiomics_gnn/models/gcn_tf.py` (classe `GCN`, flag `parallel`) | sottoclasse `GCNNoShallow` |
| Config molecolare dell'esperimento appaiato | `benchmark/MOGNN-TF/configs/config_final.yml` | copiata in `configs/mognntf_no_shallow.yaml`, cambia solo `model.parallel` |
| Caricamento dati, training, valutazione | `ExperimentDataLoader`, `ExperimentRunner` | invariati |
| Split congelati | `data/training/splits/splits_seed_{42..46}/` | letti dal runner |
| Risultati multi-scala | `results/feature_collapse/mokghgnn_g*/<ts>/` | sola lettura |
| Risultati molecolare con shallow | `results/feature_collapse/collapse_mognntf_summary.csv`, `mognntf_g*/run_*/used_config.yaml` | sola lettura |
| Tabella pubblicata (controllo) | `supplementary/tables/S2_feature_collapse.csv` | sola lettura |

## File

```
scripts/no_shallow/
├── README.md
├── preliminary_check.md    verifica §0bis (scritta prima del codice)
├── common.py               helper condivisi (percorsi, pannelli, hash, un run, lettura serie)
├── model_no_shallow.py     variante senza shallow + verifica strutturale e funzionale
├── sanity_check.py         split, pannelli, config, variante, serie esistenti, riproduzione
├── run_no_shallow.py       i 35 run (ripartibile, protetto dal sanity check)
├── compare.py              tabelle CSV + Markdown, lettura secondo il §8, sensibilità post hoc
├── plot_comparison.py      4 figure in PDF + PNG 300 dpi
├── graph_topology.py       connettività dei due grafi per (livello, split)
├── plot_paper_figure.py    figura composita a tre pannelli
└── report.py               assembla REPORT_no_shallow.md

configs/mognntf_no_shallow.yaml   config della variante (solo model.parallel: false)

results/no_shallow/               output (fuori da git)
├── sanity_check.json
├── panels/                 pannello rigenerato per (livello, split) + panel.json
├── runs/                   un JSON per run
├── runs_raw/               cartelle di run di MOGNN-TF (checkpoint, log, confusion matrix)
├── sanity_runs/            run di riproduzione del modello con shallow
├── slurm_logs/             log dei job sul cluster
├── tables/  figures/
└── REPORT_no_shallow.md
```

## Come si lancia

Dalla radice del repository, nell'ambiente `gnn`:

```bash
conda run --no-capture-output -n gnn python -u scripts/no_shallow/model_no_shallow.py
conda run --no-capture-output -n gnn python -u scripts/no_shallow/sanity_check.py --repeats 2 --num-workers 0
conda run --no-capture-output -n gnn python -u scripts/no_shallow/run_no_shallow.py --num-workers 0
conda run -n gnn python scripts/no_shallow/compare.py
conda run -n gnn python scripts/no_shallow/plot_comparison.py
conda run -n gnn python scripts/no_shallow/graph_topology.py
conda run -n gnn python scripts/no_shallow/plot_paper_figure.py
conda run -n gnn python scripts/no_shallow/report.py
```

Sul cluster iknos (Slurm) si usa la catena in `scripts/slurm/` (`no_shallow_submit.sh`): vedi
`scripts/slurm/README_no_shallow.md`.

`run_no_shallow.py` parte solo se `results/no_shallow/sanity_check.json` riporta `passed: true` ed è stato
prodotto con la stessa versione di torch: su una macchina nuova va rieseguito
`sanity_check.py`. Un modello di GPU diverso da quello del sanity è ammesso ma segnalato: ogni
JSON registra la sua GPU e il report le elenca. Il comando è ripartibile: salta i run il cui JSON esiste già.
`--smoke` esegue 3 epoche in `results/no_shallow/smoke/` solo per collaudare la pipeline; quei run non
entrano mai nel confronto.

## Esecuzione su Windows e tempi (collaudo del 2026-10-11, Quadro M2200 4 GB)

- Con `num_workers: 1` (valore di config) su Windows ogni passata su un DataLoader rilancia un
  processo Python (spawn): epoche da 130 a 700 s, run da ore. Usare `--num-workers 0` sia in
  `sanity_check.py` sia in `run_no_shallow.py` (il runner esige lo stesso valore della
  riproduzione). È un parametro di sola esecuzione: gli indici dei batch li estrae il sampler
  pesato nel processo principale con il proprio generatore, e il seed base del DataLoader è
  estratto allo stesso modo con 0 o N worker. Verificato anche empiricamente: l'epoca 1 coincide
  alla quarta cifra con il run originale del server.
- Il runner originale cancella `checkpoints/` dopo il test; su Windows un lock transitorio sui
  `.pth` appena scritti può dare `PermissionError` e perdere il run. `common.py` ripete la
  cancellazione (solo nel proprio processo, nessun effetto sui numeri).
- Tempi misurati con `num_workers 0`: circa 20 s/epoca con shallow e 14 s/epoca senza shallow a
  700 geni (meno ai livelli bassi). Riproduzione g700/seed 42: 15-16 min per run. Il server su
  cui girarono le serie esistenti fa circa 3.5 s/epoca.
- Esito del sanity check su questa macchina: superato (0.7877 e 0.7937 contro 0.7810).

## Protocollo (invariato rispetto all'originale)

- Livelli: 700, 500, 300, 150, 100, 50, 20 geni richiesti; split 42..46 (60/20/20, indici
  congelati); seed del modello fisso 2025; la varianza riportata è tra partizioni dei dati.
- Pannello per livello e split: FS unificata per varianza sul solo train (top 200 TF e 100
  miRNA richiesti), poi solo miRNA e TF sopravvissuti nel grafo multi-scala; MOGNN-TF è forzato
  su quei geni, TF e miRNA per nome e nello stesso ordine.
- Iperparametri: quelli di `config_final.yml`, nessun re-tuning. Decoder di ricostruzione,
  penalità L2, sampler pesato e normalizzazione restano invariati.

## Sanity check

- **Statici**: split con la stessa sequenza di indici di una rigenerazione deterministica (i
  byte possono differire solo per le terminazioni di riga del sistema operativo); pannelli
  identici a quelli registrati nei run con shallow (`used_config.yaml`), ai conteggi del summary
  e ai nodi del grafo multi-scala; config ricostruita identica a quella dei run originali;
  serie esistenti complete e coerenti con la Tab. S2 pubblicata.
- **Riproduzione**: il modello con shallow, rieseguito con questa pipeline su un livello e uno
  split, deve riprodurre il valore esistente entro **0.02** di macro-F1. La tolleranza è fissata
  a priori, circa due volte la deviazione standard run-to-run (~0.01) osservata su repliche dello
  stesso setup, dovuta alle operazioni scatter non deterministiche della GPU.
- Ciò che non è verificabile sulla macchina corrente (cartelle non trasferite dal server) è
  riportato come tale, mai come superato.

## Lettura

Con 5 split il Wilcoxon appaiato non scende sotto p = 0.0625, quindi le differenze sono
riportate come effect size (differenza media appaiata, deviazione standard, dz, concordanza
dei segni). I criteri di lettura del §8 del ticket sono fissati in `compare.py` prima dei run.

Dopo i run (2026-10-12) è stata aggiunta un'analisi di sensibilità **post hoc**, dichiarata
come tale (`tables/sensitivity_*`, `sensitivity.json`, sezione dedicata nel report): stesse
soglie del §8, applicate alla serie senza shallow privata dei run degeneri (predittori
costanti: 2 run a 20 geni) e alle mediane per livello. Non sostituisce l'analisi primaria.

## Nota su git

Gli output stanno in `results/no_shallow/`, escluso da git come il resto di `results/`. I run
scritti prima dello spostamento in `scripts/` + `results/` (2026-10-12) registrano la propria
cartella grezza con il prefisso `experiments/no_shallow_matched/results`: `common.resolve_run_dir`
lo traduce, senza riscrivere i dati registrati.

# Verifica preliminare (ticket §0bis): il multi-scala ha una via che aggira il grafo?

Data: 2026-10-11. Commit ispezionato: `8bf291e` (main). Ispezione in sola lettura.
Modello verificato: MOKG-HGNN, backbone `hetero_sage`, configurazione delle 35 run del
feature collapse (`results/feature_collapse/mokghgnn_g*/<ts>/config.json`).

## Esito

**Premessa confermata.** Nel multi-scala non esiste un ramo, una skip connection, un
jumping knowledge, un decoder o una loss ausiliaria che porti gli omici del campione al
classificatore senza passare dal backbone relazionale. Al classificatore arrivano solo i
vettori ottenuti dal pooling per scala dopo il backbone. Si procede con il ticket.

Una precisazione, da dichiarare ma che non cambia l'esito: ogni `SAGEConv` contiene il
termine "root" (`lin_r` applicato allo stato del nodo stesso). È parte standard
dell'aggiornamento GraphSAGE, non porta nulla direttamente al classificatore e attraversa
comunque tutti i layer del backbone; però non richiede archi. Il modello molecolare ha
l'analogo esatto nel termine di ordine zero della `ChebConv` (`Tx_0 = x`). La componente
"indipendente dagli archi" è quindi presente in entrambi i modelli, dentro la convoluzione.

## Punti verificati

| # | Domanda del ticket | File e righe | Trovato |
|---|---|---|---|
| 1 | Esiste un ramo (MLP o altro) che legge gli omici e arriva al classificatore senza message passing? | `src/multiomics_kg_hgnn/models/hetero_gnn.py:181-185` (`forward`) | No. Il forward è `encoder -> backbone -> global_mean_pool per scala -> classifier`. Nessun altro tensore raggiunge `self.classifier`. |
| 2 | Al classificatore arrivano solo le rappresentazioni del pooling per scala dopo il backbone? | `hetero_gnn.py:176-179`, `:184-185` | Sì. Input del classificatore = `torch.cat(pooled)`, con `pooled` = `global_mean_pool(x_dict[t], ...)` su `x_dict` restituito dal backbone (riga 183). Dimensione `hidden * len(readout_types)`. |
| 3 | Skip, residual o jumping knowledge che portano feature di input (o la sola proiezione per tipo) al classificatore? | `hetero_gnn.py:84-100` (`_HeteroSAGEStack`), `:92` | No residual, no JK. Ogni layer sostituisce `x_dict` con l'uscita di `HeteroConv(aggr="sum")` seguita da ReLU e dropout. La proiezione per tipo dell'encoder (`:39-48`) entra solo nel primo layer. Unico termine self: `SAGEConv(root_weight=True)` (default PyG 2.7.0, verificato nel sorgente installato: `out = lin_l(mean_neigh) + lin_r(x_dst)`), interno alla convoluzione. |
| 4 | Decoder di ricostruzione o altra loss ausiliaria sugli omici? | `src/multiomics_kg_hgnn/pancancer_prediction/training/trainer.py:39`, `:50-51` | No. Unica loss: `nn.CrossEntropyLoss(weight=class_weights)` sui logit. Nessun termine di ricostruzione, nessuna penalità esplicita (solo `weight_decay` di AdamW, `runner.py:91-93`). |
| 5 | La configurazione dell'esperimento appaiato attiva componenti di questo tipo tramite flag? | `results/feature_collapse/mokghgnn_g*/<ts>/config.json` (35/35 lette), `runner.py:84-90` | No. Tutte e 35: `backbone=hetero_sage`, `readout_types=[gene, pathway, GO_term, disease]`, `class_weighted_loss=true`, `sampler_strategy=weighted`. Il modello non espone flag per skip, JK, decoder o rami paralleli. |

## Note di contorno (non modificano l'esito)

- I nodi pathway, GO e disease non hanno feature del paziente: ricevono un unico embedding
  appreso per tipo (`hetero_gnn.py:32`, dataset `HeteroOmicDataset.py:46-48`). Tutta
  l'informazione del paziente entra dai nodi gene, miRNA e TF.
- Il backbone HGT (non usato qui) ha una skip gated interna a `HGTConv`; non riguarda
  questo esperimento perché tutte le run usano `hetero_sage`.

## Riferimento: dove sta il bypass nel molecolare

`benchmark/MOGNN-TF/multiomics_gnn/models/gcn_tf.py`:
- ramo shallow: `:105-111` (definizione), `:260-262` (input `x_parallel` = feature dopo il
  proiettore per tipo, prima di ogni convoluzione), `:335-343` (MLP a due layer e concat con
  la rappresentazione del grafo);
- classificatore: `:111` (`Linear(128, C)` con shallow) oppure `:113` (`Linear(64, C)` senza);
- decoder: `:94-102`, `:327-333`, legge l'uscita di `linear2`, cioè la rappresentazione del
  grafo (theta1): non è un bypass.

Il ramo è già governato dal flag di configurazione `model.parallel`
(`benchmark/MOGNN-TF/multiomics_gnn/pancancer_prediction/training/builder.py:37`).

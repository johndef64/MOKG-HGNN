Ottimo, entriamo nel merito architetturale. La risposta breve: **il task resta graph classification** — non cambiarlo. Quello che cambia con il grafo eterogeneo *non è il task, è l'inductive bias e soprattutto il pooling*. Ti spiego perché, e poi quali reti PyG usare in progressione.

## Perché il task rimane graph classification

Il tuo setup è "population of graphs": ogni paziente = un grafo, la topologia di prior knowledge è condivisa tra pazienti, le feature multi-omiche (expression, CNV, miRNA) vivono sui nodi, e l'etichetta è a livello di grafo (il sottotipo tumorale, le tue ~27 classi). Questo è esattamente graph classification, ed è la formulazione giusta per tre motivi:

1. **Comparabilità 1:1 con MOGNN-TF.** La tua sezione 5 punta su un confronto controllato single→multi scala. Se cambi task, il confronto salta. Tenere fisso il task e variare solo la topologia (mono-scala vs multi-scala) è proprio il disegno sperimentale che ti serve.
2. **Il segnale paziente-specifico sta solo su gene/miRNA.** I nodi pathway/GO/disease del KG sono *scaffold condiviso*: non portano feature per-paziente, servono come impalcatura per il message passing e per il pooling gerarchico. Non generano un task nuovo, arricchiscono la struttura del task esistente.
3. Node classification non è naturale (non hai label per-nodo per paziente) e link prediction è, al più, un pretext ausiliario.

Quindi: **non cambiare task, cambia l'architettura**. Se vuoi *aggiungere* qualcosa, le opzioni sensate sono task ausiliari sulla stessa graph-embedding, non un rimpiazzo:
- **Multi-task**: testa di classificazione (sottotipo) + testa di sopravvivenza (Cox/regressione, TCGA ha i dati) → sfrutti lo stesso grafo e aumenti il segnale. Cambia lo scope della tesi, valutalo solo se hai margine.
- **Self-supervised pretraining** (masked feature reconstruction sui geni, o contrastive con augmentation del grafo-paziente) → fine-tuning sulla classificazione. Utile dato che con ~27 classi i conteggi per classe sono modesti.
- **Link prediction** sugli archi di prior knowledge come pretext per pre-inizializzare gli embedding dei nodi featureless (pathway/GO). Ausiliario, non primario.

## Quali reti HeteroGNN da PyG (in progressione)

Ti conviene una scala di complessità crescente, così hai baseline e ablation pronti per la tesi:

1. **`to_hetero(modello_omogeneo)` — il cavallo di battaglia.** Scrivi una GNN omogenea (SAGEConv, GATConv o GINConv) e la avvolgi con `torch_geometric.nn.to_hetero(model, metadata, aggr='sum')`: PyG duplica automaticamente i messaggi per ogni tipo di arco, con pesi separati per relazione, e aggrega tra relazioni. È la via più flessibile e la baseline giusta. Parti da **SAGE 2–3 layer**: robusto, pochi parametri, gestisce bene grafi con dimensioni di feature diverse per tipo.

2. **`HeteroConv({edge_type: Conv})` — controllo fine per relazione.** Assegni conv diverse a relazioni diverse (es. GAT su gene↔gene per catturare interazioni pesate, SAGE su gene→pathway, un conv semplice sulla gerarchia GO→GO). Ha senso quando le relazioni hanno semantica molto diversa — che è il tuo caso.

3. **`RGCNConv` / `FastRGCNConv` — stile KG.** Una matrice di peso per relazione, con basis/block decomposition per contenere i parametri. Naturale se pensi al grafo come "multi-relational KG"; meno naturale di `to_hetero` quando i tipi di nodo hanno dimensioni di feature diverse (dovresti proiettarli prima).

4. **`HGTConv` (Heterogeneous Graph Transformer) e `HANConv` (metapath-based) — i "flagship" per il confronto.** HGT usa attention parametrizzata per meta-relazione; HAN richiede di definire metapath (es. gene–pathway–gene, gene–GO–gene) — cosa che sposa bene la tua idea gerarchica. **Caveat importante**: sono modelli con molti parametri; con qualche migliaio di pazienti e 27 classi rischiano overfitting. Usali come termine di paragone "un modello più sofisticato aiuta davvero?", con regolarizzazione forte, non come default.

## Il vero punto: il pooling è dove le scale KG ripagano

Questa è la parte che giustifica tutto il lavoro sul grafo eterogeneo. Per il readout hai due strade:

- **Pooling generico**: `global_mean_pool`/`global_add_pool` sui nodi gene (ed eventualmente su pathway/GO), concateni i readout per tipo di nodo → MLP → softmax. Semplice, robusto, ottima baseline.
- **Pooling gerarchico biologicamente informato** — la potenziale novità della tesi. Invece di cluster appresi (DiffPool/`SAGPooling`/`ASAPooling`/`TopKPooling`), usi **gli archi del KG come matrice di assegnamento fissa**: aggreghi i geni nei pathway/GO tramite `gene→pathway`/`gene→GO`, poi i pathway/GO in scale superiori tramite `GO→GO`/`pathway→pathway`. È l'idea alla base dei modelli pathway-aware (tipo P-NET/Elmarakeby): la gerarchia biologica *è* la gerarchia di pooling. Questo è esattamente il motivo per cui hai aggiunto pathway/GO/disease — e ti dà anche interpretabilità gratis (quali pathway pesano sul sottotipo).

Metti i due a confronto (pooling biologico vs DiffPool appreso vs pooling piatto): è un'ablation forte e pubblicabile.

## Dettagli pratici PyG che ti risparmiano tempo

- **Nodi senza feature** (pathway/GO/disease): usa `torch.nn.Embedding` per tipo di nodo, oppure sfrutta la **lazy init** di PyG (`in_channels=-1` nei conv/`Linear`) che deduce le dimensioni al primo forward — comodissimo con tipi eterogenei. In alternativa inizializza gli embedding dei nodi featureless con il pretraining self-supervised di prima.
- **Direzionalità**: applica `T.ToUndirected()` per generare gli archi inversi tipizzati (es. `pathway__rev_member_of__gene`) dove vuoi bidirezionalità nel message passing; lascia direzionale la gerarchia GO se vuoi far fluire il segnale solo verso l'alto.
- **Batching e memoria**: ogni paziente è un `HeteroData`; li batchizzi con il `DataLoader` su lista di `HeteroData`. Attenzione: replicare lo scaffold KG (migliaia di geni + pathway + GO) per ogni paziente nel batch è pesante in memoria. Mitiga tenendo il set di geni piccolo (post feature-selection, come già fai), usando batch piccoli, o un collate custom che condivide l'`edge_index` statico tra i grafi del batch. Con GPU di memoria modesta questo è il collo di bottiglia reale.
- **Sbilanciamento classi** (27 sottotipi): `weighted cross-entropy` o focal loss, e stratifica gli split.
- **Interpretabilità**: attention di GAT/HGT o `GNNExplainer`/`CaptumExplainer` per attribuire la predizione a geni/pathway — chiude il cerchio con la motivazione biologica della tesi.

## Raccomandazione concreta

Come default di partenza: **`to_hetero(GraphSAGE)` a 2–3 layer + pooling gerarchico biologicamente informato lungo gene→pathway→(GO/disease) + readout concatenato per tipo + MLP → softmax a 27 classi**. Poi due ablation: (a) stesso modello con pooling piatto e con DiffPool appreso; (b) `HGTConv` come "modello sofisticato" per vedere se la maggiore capacità aiuta o overfitta. Task invariato — graph classification — così il confronto con MOGNN-TF e tra le scale resta pulito.

Vuoi che ti abbozzi lo scheletro del modello `to_hetero` + il modulo di hierarchical pooling che usa gli archi `gene→pathway`/`gene→GO` come assegnamento?
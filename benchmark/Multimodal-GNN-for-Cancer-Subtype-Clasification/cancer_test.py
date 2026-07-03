## IMPORTAZIONE DELLE LIBRERIE NECESSARIE
# Librerie standard per operazioni di sistema e gestione del tempo
import os
import time

# Librerie per manipolazione dati e calcolo numerico
import numpy as np  # per operazioni numeriche su array
import pandas as pd  # per manipolazione di dataframe e dati tabulari
import scipy.sparse as sp  # per gestire matrici sparse (con molti zeri)
import argparse  # per gestire argomenti da linea di comando
import sklearn.metrics  # per metriche di valutazione del modello

# Librerie PyTorch per deep learning e reti neurali
import torch
torch.manual_seed(2022)  # fissa il seed risultati riproducibili
import torch.nn as nn  # moduli di rete neurale
import torch.nn.functional as F  # funzioni di attivazione e loss
from torch.utils.data import Dataset, TensorDataset  # per gestire dataset

# Librerie PyTorch Geometric per Graph Neural Networks
from torch_geometric.loader import DataLoader  # per caricare dati in batch

# Importa funzioni personalizzate dai file del progetto
from utils import *  # funzioni di utilità per caricamento e preprocessing dati
from layer_model import *  # definizioni dei modelli GNN personalizzati

# Gestione memoria: libera memoria inutilizzata
import gc
gc.collect()  # garbage collection per liberare memoria RAM
torch.cuda.empty_cache()  # libera cache della GPU se disponibile

## CONFIGURAZIONE DEI PARAMETRI DEL MODELLO
# Crea un parser per gestire gli argomenti da linea di comando
parser = argparse.ArgumentParser()

# PARAMETRI DI TRAINING
parser.add_argument('--lr', type=float, default = 0.01, help='learning rate - velocità di apprendimento del modello')
parser.add_argument('--big_lr', type=str2bool, nargs='?', default = True, help='usa un learning rate più grande con decadimento')
parser.add_argument('--batch_size', type=int, default = 16, help='numero di campioni per batch durante il training')
parser.add_argument('--epochs', type=int, default = 100, help='numero di epoche di training')
parser.add_argument('--dropout', type=float, default = 0.6, help='tasso di dropout per evitare overfitting (0.6 = 60% neuroni disattivati)')
parser.add_argument('--decay', type=float, default = 0.9, help='tasso di decadimento del learning rate nel tempo')

# PARAMETRI DEI DATI OMICI
parser.add_argument('--num_gene', type=int, default = 1000, help='numero di geni da utilizzare nel modello')
parser.add_argument('--omic_mode', type=int, default = 0, help='modalità dati omici: 0=mRNA, 1=miRNA, 2=mRNA+miRNA, 3=mRNA+CNV, 4=tutti')
parser.add_argument('--num_omic', type=int, default = 1, help='numero di tipi di dati omici utilizzati')

# PARAMETRI PER CLASSIFICAZIONE TUMORI
parser.add_argument('--cancer_subtype', type=str2bool, nargs='?', default = False, help='classifica sottotipi di cancro invece che cancro vs normale')
parser.add_argument('--specific_type',type=str, default='brca', choices=['brca','luad'], help='tipo di cancro specifico (brca=breast cancer, luad=lung cancer)')
parser.add_argument('--shuffle_index',type=int, default=0, help='indice per la randomizzazione dei dati')

# PARAMETRI DEL MODELLO
parser.add_argument('--model', type=str, default = 'gat', choices=['gat','gatv2','gcn','multi-gcn','baseline'], help='tipo di modello GNN da usare')
parser.add_argument('--poolsize', type=int, default = 8, help='dimensione massima del pooling layer')
parser.add_argument('--poolrate', type=float, default = 0.8, help='tasso di pooling nel layer di self-attention')

# PARAMETRI DELLA RETE BIOLOGICA
parser.add_argument('--gene_gene', type=str2bool, nargs='?', default = True, help='usa connessioni gene-gene dalla rete biologica')
parser.add_argument('--mirna_gene', type=str2bool, nargs='?', default = True, help='usa connessioni miRNA-mRNA')
parser.add_argument('--mirna_mirna', type=str2bool, nargs='?', default = True, help='usa connessioni miRNA-miRNA')

# PARAMETRI ARCHITETTURA MODELLO
parser.add_argument('--parallel', type=str2bool, nargs='?', default = True, help='usa struttura parallela nel modello')
parser.add_argument('--l2', type=str2bool, nargs='?', default = True, help='usa regolarizzazione L2 per evitare overfitting')
parser.add_argument('--decoder', type=str2bool, nargs='?', default = True, help='usa decoder per ricostruire il grafo (autoencoder)')

# PARAMETRI DEGLI EDGE (CONNESSIONI)
parser.add_argument('--edge_attribute', type=str2bool, nargs='?', default = False, help='usa attributi multidimensionali per le connessioni')
parser.add_argument('--edge_weight', type=str2bool, nargs='?', default = False, help='usa pesi continui invece di connessioni binarie')

# PARAMETRI DIVISIONE DATASET
parser.add_argument('--train_ratio', type=float, default = 0.8, help='percentuale dati per training (80%)')
parser.add_argument('--test_ratio', type=float, default = 0.1, help='percentuale dati per test (10%, rimanente 10% per validazione)')

# Parsa tutti gli argomenti forniti
args = parser.parse_args()

## VALIDAZIONE E AGGIUSTAMENTO DEI PARAMETRI
# Traduce la modalità omica nel numero corrispondente di tipi di dati
# mode 0: solo mRNA (espressione genica)
# mode 1: solo miRNA (micro RNA regolatori)  
# mode 2: mRNA + miRNA (dati multimodali)
# mode 3: mRNA + CNV (copy number variations - variazioni nel numero di copie)
# mode 4: mRNA + CNV + miRNA (tutti e tre i tipi di dati)
args.num_omic = omic_mode_translation(args.omic_mode)

# Valida le scelte di rete biologica in base alla modalità omica selezionata
# Restituisce anche il numero di miRNA da utilizzare
args.gene_gene, args.mirna_gene, args.mirna_mirna, num_mirna = validate_network_choice(
    args.omic_mode, args.gene_gene, args.mirna_gene, args.mirna_mirna)

# Se usiamo solo miRNA (mode 1), non abbiamo bisogno di geni
if args.omic_mode == 1:
    args.num_gene = 0

# Per il modello baseline (senza GNN), disabilita decoder e struttura parallela
if args.model == 'baseline':
    args.decoder = False
    args.parallel = False

# Stampa tutti i parametri validati per verifica
print('Parametri correnti del modello:')
print(args)

## DEFINIZIONE DEI PERCORSI DEI FILE DI DATI
# Cartella principale contenente tutti i dati sul cancro  
path = 'data/'
#path = 'data_new/'

# FILE COMUNI A TUTTE LE ANALISI
expression_variance_path = path + 'expression_variance.tsv'  # varianza dell'espressione genica
non_null_index_path = path + 'biogrid_non_null.csv'  # indici non nulli della rete biologica
adjacency_matrix_path = path + 'adj_matrix_biogrid.npz'  # matrice di adiacenza della rete gene-gene
mirna_to_gene_matrix_path = path + 'standardized_mirna_mrna_edge_filtered_at_eight_with_top_100_mirna.npz'  # connessioni miRNA-mRNA

# SCELTA DEI FILE IN BASE AL TIPO DI ANALISI
if args.cancer_subtype:
    # Se stiamo classificando SOTTOTIPI di cancro (es. HER2+, luminal A, etc.)
    if args.specific_type.lower() == 'brca':
        # Dati specifici per il cancro al seno (breast cancer)
        shuffle_index_path = path + 'brca_shuffle_index.tsv'  # indici randomizzati per BRCA
        cancer_subtype_label_path = path + 'brca_subtype.csv'  # etichette dei sottotipi di BRCA
        expression_data_path = path + 'expression_data_brca.tsv'  # dati di espressione genica BRCA
        cnv_data_path = path + 'cnv_data_brca.tsv'  # variazioni numero copie BRCA
        mirna_data_path = path +'mirna_data_brca.tsv'  # dati miRNA BRCA
else:
    # Se stiamo classificando CANCRO vs NORMALE (classificazione binaria)
    expression_data_path = path + 'standardized_expression_data_with_labels.tsv'  # dati espressione normalizzati
    cnv_data_path = path + 'standardized_cnv_data_with_labels.tsv'  # dati CNV normalizzati  
    mirna_data_path = path +'top_100_mirna_data.tsv'  # top 100 miRNA più informativi
    shuffle_index_path = path + 'common_trimmed_shuffle_index_'+ str(args.shuffle_index) + '.tsv'  # indici randomizzati

## CARICAMENTO E PREPROCESSING DEI DATI
# Sceglie la funzione di caricamento in base ai tipi di dati omici richiesti
if args.omic_mode < 3:
    # Per modalità 0,1,2: carica solo dati di espressione e miRNA
    expr_all_data, mirna_all_data = load_exp_and_real_mirna_data(expression_data_path, mirna_data_path)

    # Effettua il downsampling e preprocessing dei dati
    # Restituisce: matrice di adiacenza, dati processati, etichette, indici randomizzati
    adj, train_data_all, labels, shuffle_index = down_sampling_exp_and_real_mirna_data(
        expression_variance_path=expression_variance_path,  # file con varianza geni
        expression_data=expr_all_data,                      # dati espressione genica
        mirna_data=mirna_all_data,                         # dati miRNA
        omic_mode=args.omic_mode,                          # modalità dati omici
        non_null_index_path=non_null_index_path,           # indici validi rete biologica  
        shuffle_index_path=shuffle_index_path,             # indici per randomizzazione
        adjacency_matrix_path=adjacency_matrix_path,       # matrice connessioni gene-gene
        mirna_to_gene_matrix_path=mirna_to_gene_matrix_path,  # connessioni miRNA-mRNA
        gene_gene=args.gene_gene,                          # usa connessioni gene-gene
        mirna_gene=args.mirna_gene,                        # usa connessioni miRNA-gene
        mirna_mirna=args.mirna_mirna,                      # usa connessioni miRNA-miRNA
        number_gene=args.num_gene,                         # numero di geni da selezionare
        singleton=False)                                   # non includere nodi isolati
else:
    # Per modalità 3,4: carica anche i dati CNV (Copy Number Variations)
    expr_all_data, cnv_all_data, mirna_all_data = load_exp_cnv_and_real_mirna_data(
        expression_data_path, cnv_data_path, mirna_data_path)
    print('Dati di espressione, CNV e miRNA caricati con successo.')
    # Preprocessing includendo anche i dati CNV
    adj, train_data_all, labels, shuffle_index = down_sampling_exp_cnv_and_real_mirna_data(
        expression_variance_path=expression_variance_path,
        expression_data=expr_all_data,                     # dati espressione genica
        cnv_data=cnv_all_data,                            # dati Copy Number Variations
        mirna_data=mirna_all_data,                        # dati miRNA
        omic_mode=args.omic_mode,
        non_null_index_path=non_null_index_path,
        shuffle_index_path=shuffle_index_path,
        adjacency_matrix_path=adjacency_matrix_path,
        mirna_to_gene_matrix_path=mirna_to_gene_matrix_path,
        gene_gene=args.gene_gene,
        mirna_gene=args.mirna_gene,
        mirna_mirna=args.mirna_mirna,
        number_gene=args.num_gene,
        singleton=False)

# Stampa la forma dei dati caricati per verifica
print('Dati caricati con successo. Forma dei dati di training:', np.asarray(train_data_all).shape)
print(labels)

## FILTRAGGIO PER SOTTOTIPI DI CANCRO (se richiesto)
if args.cancer_subtype:
    # Filtra i dati per mantenere solo i campioni del tipo di cancro specificato
    # e associa le etichette dei sottotipi corrispondenti
    train_data_all, labels = filter_data_by_cancer_type(
        cancer_subtype_label_path,  # file con etichette sottotipi (es. HER2+, luminal A, etc.)
        train_data_all,             # tutti i dati dei campioni
        expr_all_data)              # dati di espressione originali

print(labels)
## PREPROCESSING DELLE ETICHETTE
# Importa strumenti per preprocessing
from sklearn import preprocessing

# Trasforma le etichette testuali in numeri interi sequenziali (0, 1, 2, ...)
# Es: ['luminal_A', 'HER2+', 'basal'] -> [0, 1, 2]
le = preprocessing.LabelEncoder()
labels = le.fit_transform(labels)
print("Etichette preprocessate (numeriche):", labels)
# Codice commentato per rimuovere connessioni a zero (attualmente non utilizzato)
# if not args.singleton:      
#     adj, train_data_all = removeZeroAdj(adj, train_data_all)

## PREPROCESSING DELLA MATRICE DI ADIACENZA (RETE BIOLOGICA)
# Salva copia densa della matrice per la funzione di loss
adj_for_loss = adj.todense()

# Normalizza i pesi delle connessioni dividendo per il valore massimo
# Questo porta tutti i valori nell'intervallo [0,1]
adj = adj/np.max(adj)
adj = adj.astype('float32')  # conversione per efficienza computazionale

# Rimuove le self-connections (connessioni di un nodo con se stesso)
adj.setdiag(0)
# Aggiunge self-loops con peso 1 (ogni nodo è connesso a se stesso)
# Questo è importante per i GNN per preservare le informazioni del nodo
adj = adj + sp.eye(adj.shape[0])

# Converte in formato COOrdinate (efficiente per operazioni sparse)
adj = sp.coo_matrix(adj)
# Crea edge_index: matrice 2×N con indici delle connessioni [source_nodes; target_nodes]
edge_index = torch.stack([torch.tensor(adj.row), torch.tensor(adj.col)], dim=0)
# Estrae i pesi delle connessioni
edge_weight = torch.Tensor(adj.data)
print("Matrice di adiacenza preprocessata con", edge_index.shape[1], "connessioni.")

## CONVERSIONE DEGLI EDGE WEIGHTS IN ATTRIBUTI MULTIDIMENSIONALI (opzionale)
if args.edge_attribute:
    # Trasforma i pesi singoli in vettori di attributi multidimensionali
    # Utile per catturare diversi tipi di relazioni biologiche
    edge_attribute = disassemble_edge_weights(edge_weight, edge_index, args.num_gene, args.num_omic)

## DIVISIONE DEL DATASET IN TRAINING, VALIDATION E TEST
# Converte gli indici di randomizzazione in formato intero
shuffle_index = shuffle_index.astype(np.int32).reshape(-1)
print("Indice di randomizzazione caricato con", len(shuffle_index), "campioni.")
# Calcola le dimensioni dei set basandosi sui ratio specificati
# train_ratio=0.8 -> 80% training
# test_ratio=0.1 -> 10% test, 10% validation  
train_size = int(len(shuffle_index) * args.train_ratio)  
print("Dimensione training set:", train_size)         # 80% per training
val_size = int(len(shuffle_index) * (1 - args.test_ratio))       # 90% per training+validation

# Divide i DATI usando gli indici randomizzati
train_data = np.asarray(train_data_all).astype(np.float32)[shuffle_index[0:train_size]]  
print("Dimensione training set dati:", train_data.shape)         # primi 80%
val_data = np.asarray(train_data_all).astype(np.float32)[shuffle_index[train_size:val_size]]      # 80%-90%  
test_data = np.asarray(train_data_all).astype(np.float32)[shuffle_index[val_size:]]               # ultimi 10%

# Divide le ETICHETTE usando gli stessi indici
train_labels = labels[np.array(shuffle_index[0:train_size])] 
print("Dimensione training set etichette:", train_labels.shape)         # etichette training
val_labels = labels[shuffle_index[train_size:val_size]]          # etichette validation
test_labels = labels[shuffle_index[val_size:]]                   # etichette test

# CODICE COMMENTATO: opzione per ridurre ulteriormente il training set
# train_data, train_labels = dropout_data(train_data, train_labels, 0.75)

# CODICE COMMENTATO: conta le classi nel training set
# ll, cnt = np.unique(train_labels,return_counts=True)

# Calcola il numero totale di classi per la classificazione
nclass = len(np.unique(labels))
print("Numero di classi da predire:", nclass)
## CONVERSIONE IN TENSORI PYTORCH
# Converte le etichette in formato intero a 64 bit (richiesto da PyTorch per classificazione)
train_labels = train_labels.astype(np.int64)
test_labels = test_labels.astype(np.int64)  
val_labels = val_labels.astype(np.int64)

# Converte i dati numpy in tensori PyTorch
train_data = torch.FloatTensor(train_data)
print("Dati training convertiti in tensori float32:", train_data.shape)  
test_data = torch.FloatTensor(test_data)        # dati test in formato float32
val_data = torch.FloatTensor(val_data)          # dati validation in formato float32

# Converte le etichette in LongTensor (richiesto per CrossEntropyLoss)
train_labels = torch.LongTensor(train_labels)
print("Etichette training convertite in LongTensor:", train_labels.shape)   # etichette training
test_labels = torch.LongTensor(test_labels)     # etichette test  
val_labels = torch.LongTensor(val_labels)       # etichette validation

# Crea dataset PyTorch che combina dati e etichette
dset_train = TensorDataset(train_data, train_labels)
print("Dataset di training creato con", len(dset_train), "campioni.")
## CREAZIONE DEI DATA LOADER
# Data loader per il training: batch casuali per migliorare l'apprendimento
train_loader = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True)
print("Data loader di training creato con batch size:", args.batch_size)

# Data loader per il test: nessun shuffle per risultati riproducibili
dset_test = TensorDataset(test_data, test_labels)
test_loader = DataLoader(dset_test, shuffle=False)

# Data loader per la validation: batch casuali per valutazione robusta
dset_val = TensorDataset(val_data, val_labels)
val_loader = DataLoader(dset_val, batch_size=args.batch_size, shuffle=True)

## SELEZIONE DEL DEVICE DI COMPUTAZIONE
# Priorità: CUDA (GPU NVIDIA) > MPS (GPU Apple) > CPU
# CUDA è preferito per le prestazioni, MPS per Mac con chip Apple Silicon
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

print('Device utilizzato per il training:', device)

## INIZIALIZZAZIONE DEL MODELLO
# Sceglie il tipo di modello in base al parametro --model
if args.model == 'gcn':
    # Graph Convolutional Network: usa convoluzione sui grafi per aggregare informazioni dai vicini
    model = GCN(args.model,           # tipo di modello
                args.parallel,        # architettura parallela per diversi tipi di dati
                args.l2,              # regolarizzazione L2
                args.decoder,         # decoder per ricostruzione (autoencoder)
                args.poolsize,        # dimensione del pooling
                args.poolrate,        # tasso di pooling
                args.edge_weight,     # usa pesi delle connessioni
                args.edge_attribute,  # usa attributi multidimensionali
                args.num_gene,        # numero di geni
                num_mirna,            # numero di miRNA
                args.omic_mode,       # modalità dati omici
                nclass,               # numero di classi da predire
                args.dropout).to(device)  # tasso di dropout
elif args.model == 'baseline':
    # Modello baseline: rete neurale tradizionale senza struttura a grafo
    model = Baseline(args.model, 
                args.parallel, 
                args.l2, args.decoder, 
                args.poolsize, 
                args.poolrate,
                args.edge_weight, 
                args.edge_attribute, 
                args.num_gene,
                num_mirna, 
                args.omic_mode, 
                nclass, 
                args.dropout).to(device)
else:
    # Graph Attention Network (default): usa meccanismo di attenzione per pesare i vicini
    model = GAT(args.model,           # supporta 'gat', 'gatv2', 'multi-gcn'
                args.parallel, 
                args.l2, args.decoder, 
                args.poolsize, 
                args.poolrate,
                args.edge_weight, 
                args.edge_attribute, 
                args.num_gene,
                num_mirna, 
                args.omic_mode, 
                nclass, 
                args.dropout).to(device)

## CONFIGURAZIONE DELL'OTTIMIZZATORE E LEARNING RATE
# Variabili globali per il controllo del learning rate
global_lr = args.lr          # learning rate iniziale
global_step = 0              # contatore passi di training globali  
decay = args.decay           # fattore di decadimento (0.9 = 10% riduzione)
decay_steps = train_size     # numero di campioni dopo cui applicare il decadimento

def adjust_learning_rate(optimizer, epoch):
    """
    Funzione per aggiustare dinamicamente il learning rate durante il training
    Due strategie disponibili:
    - big_lr=True: decadimento basato sui passi globali (più aggressivo)
    - big_lr=False: decadimento ogni 12 epoche (più conservativo)
    """
    if args.big_lr:
        # Strategia aggressiva: riduce LR in base ai campioni processati
        lr = args.lr * pow(decay, float(global_step // decay_steps))
    else:
        # Strategia conservativa: riduce LR del 20% ogni 12 epoche
        lr = args.lr * (0.8 ** (epoch // 12))
    
    # Applica il nuovo learning rate a tutti i gruppi di parametri
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

# Inizializza l'ottimizzatore Adam con regolarizzazione del peso
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
# Alternativa commentata: SGD con momentum
# optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=args.lr)

# Coefficiente per regolarizzazione L2 (previene overfitting)
l2_regularization = 5e-4

## INIZIO DEL CICLO DI TRAINING
t_total_train = time.time()  # cronometro per tempo totale di training

for epoch in range(args.epochs):
    ## AGGIORNAMENTO DEL LEARNING RATE
    cur_lr = adjust_learning_rate(optimizer, epoch)

    ## INIZIALIZZAZIONE DELL'EPOCA
    t_start = time.time()      # cronometro per questa epoca
    model.train()              # mette il modello in modalità training
    loss_all = 0.0            # accumulatore per la loss totale
    accuracy_all = 0.0        # accumulatore per l'accuratezza totale
    count = 0                 # contatore dei batch processati

    # ITERAZIONE SU TUTTI I BATCH DI TRAINING
    for i, (batch_x, batch_y) in enumerate(train_loader):
        # Sposta dati e etichette sul device (GPU/CPU)
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        ## PREPARAZIONE DEL GRAFO PER IL BATCH
        # I modelli GAT/GATv2 richiedono grafi separati per ogni campione nel batch
        count += 1
        batch_edge_index = edge_index.type(torch.int64)  # indici delle connessioni

        ## PREPARAZIONE DEI PESI/ATTRIBUTI DELLE CONNESSIONI
        if args.edge_attribute:
            # Usa attributi multidimensionali per le connessioni
            batch_edge_weight = edge_attribute
        else:
            # Usa pesi scalari per le connessioni
            batch_edge_weight = edge_weight

        ## REPLICAZIONE DEL GRAFO PER OGNI CAMPIONE NEL BATCH
        # Ogni campione nel batch ha bisogno della sua copia del grafo
        for j in range(batch_y.shape[0] - 1):  # per ogni campione aggiuntivo nel batch
            if args.edge_weight and args.edge_attribute == False:
                # Replica i pesi scalari
                batch_edge_weight = torch.cat([batch_edge_weight, edge_weight], axis=0)
            elif args.edge_attribute:
                # Replica gli attributi multidimensionali
                batch_edge_weight = torch.cat([batch_edge_weight, edge_attribute], axis=0)
            
            # Replica gli indici del grafo con offset per il nuovo campione
            # Offset = j * (numero_geni + numero_mirna)
            batch_edge_index = torch.cat([batch_edge_index, edge_index+j*(args.num_gene+num_mirna)], axis=1)

        # Sposta grafo e pesi sul device
        batch_edge_index = batch_edge_index.to(device)
        batch_edge_weight = batch_edge_weight.to(device)


        ## FORWARD PASS E CALCOLO DELLA LOSS
        optimizer.zero_grad()  # azzera i gradienti accumulati dal batch precedente
        
        if args.decoder:
            # Modello con decoder (autoencoder): restituisce dati ricostruiti + predizioni
            x_reconstruct, out = model(batch_x, batch_edge_index, batch_edge_weight)
        else:
            # Modello senza decoder: solo predizioni di classificazione
            out = model(batch_x, batch_edge_index, batch_edge_weight)

        ## CALCOLO DELLA LOSS FUNCTION
        if args.decoder:
            # Loss combinata: ricostruzione + classificazione + regolarizzazione L2
            loss_batch = model.loss(x_reconstruct, batch_x, out, batch_y, l2_regularization)
        else:
            # Solo loss di classificazione + regolarizzazione L2
            loss_batch = model.loss(batch_x.view(batch_x.size()[0], -1), batch_x, out, batch_y, l2_regularization)
        
        # Calcola l'accuratezza per questo batch
        accuracy_batch = accuracy(out, batch_y)
        
        ## BACKWARD PASS E AGGIORNAMENTO PARAMETRI
        loss_batch.backward()    # calcola i gradienti tramite backpropagation
        optimizer.step()         # aggiorna i pesi del modello
        
        # Accumula metriche per il monitoraggio
        loss_all += loss_batch.item()        # somma la loss di questo batch
        accuracy_all += accuracy_batch       # somma l'accuratezza di questo batch
        global_step += args.batch_size       # aggiorna il contatore globale

    ## CODICE COMMENTATO: VALUTAZIONE SU VALIDATION SET
    # Questo codice (attualmente disabilitato) valuterebbe il modello sui dati di validation
    # durante il training per monitorare l'overfitting
    # model.eval()
    # running_vloss = 0.0
    # for i, (batch_x, batch_y) in enumerate(val_loader):
    #     batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    #     batch_edge_index = edge_index
    #     for i in range(batch_y.shape[0] - 1):
    #         tmp = torch.cat([batch_edge_index, edge_index+i*(args.num_gene+100)], axis=1)
    #     batch_edge_index = batch_edge_index.to(device)
    #     voutputs = model(batch_x, batch_edge_index)
    #     vloss = nn.CrossEntropyLoss()(voutputs, batch_y)
    #     running_vloss += vloss
    
    ## CALCOLO E STAMPA DELLE METRICHE DELL'EPOCA
    t_stop = time.time() - t_start           # tempo impiegato per questa epoca
    accuracy_all = accuracy_all / count      # accuratezza media sui batch
    
    # Stampa le metriche di performance per monitorare il training
    print(f'Epoca: {epoch}, Loss totale: {loss_all:.4f}, Accuratezza: {accuracy_all:.4f}')
    print(f'Tempo di training epoca: {t_stop:.2f} secondi')

def test(loader, num_classes):
    """
    Valuta le performance del modello su un dataset (tipicamente il test set).

    Calcola:
    - accuratezza globale
    - accuratezza media per batch
    - confusion matrix (grezza e normalizzata)
    - F1 macro, weighted e per classe
    - ROC-AUC macro/weighted (OVR) e per classe
    """
    model.eval()  # modalità valutazione

    correct = 0                     # predizioni corrette totali
    sum_batch_accuracy = 0.0        # somma delle accuracy di batch

    y_true_list = []                # etichette vere aggregate
    y_pred_list = []                # etichette predette aggregate
    y_score_list = []               # probabilità (o logit) aggregate per ROC

    print(f'Numero di campioni da valutare: {len(loader.dataset)}')

    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        # PREPARAZIONE DEL GRAFO PER IL BATCH (stessa logica del training)
        batch_edge_index = edge_index.type(torch.int64)

        if args.edge_attribute:
            batch_edge_weight = edge_attribute
        else:
            batch_edge_weight = edge_weight

        # replica il grafo per ogni campione del batch
        for i in range(batch_y.shape[0] - 1):
            if args.edge_weight and args.edge_attribute == False:
                batch_edge_weight = torch.cat([batch_edge_weight, edge_weight], axis=0)
            elif args.edge_attribute:
                batch_edge_weight = torch.cat([batch_edge_weight, edge_attribute], axis=0)
            batch_edge_index = torch.cat(
                [batch_edge_index, edge_index + i * (args.num_gene + num_mirna)],
                axis=1
            )

        batch_edge_index = batch_edge_index.to(device)
        batch_edge_weight = batch_edge_weight.to(device)

        # FORWARD PASS (senza gradienti)
        with torch.no_grad():
            if args.decoder:
                x_reconstruct, out = model(batch_x, batch_edge_index, batch_edge_weight)
            else:
                out = model(batch_x, batch_edge_index, batch_edge_weight)

        # PREDIZIONI E PROBABILITÀ
        # out ha shape (batch_size, num_classes) – logit
        probs = F.softmax(out, dim=1)  # punteggi per ROC AUC

        pred_labels = out.argmax(dim=1)

        # METRICHE DI ACCURATEZZA
        correct += int((pred_labels == batch_y).sum())
        batch_acc = accuracy(out, batch_y)  # funzione accuracy in utils.py
        sum_batch_accuracy += batch_acc

        # ACCUMULA VETTORI PER METRICHE GLOBALI
        y_true_list.append(batch_y.detach().cpu().numpy())
        y_pred_list.append(pred_labels.detach().cpu().numpy())
        y_score_list.append(probs.detach().cpu().numpy())

    # CONCATENA TUTTI I BATCH
    y_true = np.concatenate(y_true_list)              # shape (n_samples,)
    y_pred = np.concatenate(y_pred_list)              # shape (n_samples,)
    y_scores = np.concatenate(y_score_list, axis=0)   # shape (n_samples, num_classes)

    # ACCURATEZZE
    global_accuracy = correct / len(loader.dataset)
    mean_batch_accuracy = sum_batch_accuracy / len(loader)

    # ===========================
    #   CONFUSION MATRIX
    # ===========================
    cm = sklearn.metrics.confusion_matrix(
        y_true,
        y_pred,
        labels=range(num_classes)
    )

    print("\n=== MATRICE DI CONFUSIONE (righe = vera, colonne = predetta) ===")
    print(cm)

    with np.errstate(all='ignore'):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    print("\n=== MATRICE DI CONFUSIONE NORMALIZZATA PER RIGA ===")
    print(cm_norm)

    # Salva confusion matrix su file CSV
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{i}" for i in range(num_classes)],
        columns=[f"pred_{i}" for i in range(num_classes)]
    )
    cm_df.to_csv("confusion_matrix_test.csv")

    cm_norm_df = pd.DataFrame(
        cm_norm,
        index=[f"true_{i}" for i in range(num_classes)],
        columns=[f"pred_{i}" for i in range(num_classes)]
    )
    cm_norm_df.to_csv("confusion_matrix_test_normalized.csv")

    # ===========================
    #   F1-SCORE
    # ===========================
    from sklearn.metrics import f1_score, classification_report, roc_auc_score, roc_curve, auc
    from sklearn.preprocessing import label_binarize

    # F1 macro / weighted / per classe
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    f1_per_class = f1_score(
        y_true, y_pred, average=None, labels=range(num_classes)
    )

    print("\n=== F1-SCORE ===")
    print(f"F1 macro:    {f1_macro:.4f}")
    print(f"F1 weighted: {f1_weighted:.4f}")
    print("F1 per classe (indice di classe -> F1):")
    for i, f1_c in enumerate(f1_per_class):
        print(f"Classe {i}: F1 = {f1_c:.4f}")

    # Salva F1 su CSV
    f1_df = pd.DataFrame({
        "class": list(range(num_classes)),
        "f1_score": f1_per_class
    })
    f1_df.loc[len(f1_df)] = ["macro_avg", f1_macro]
    f1_df.loc[len(f1_df)] = ["weighted_avg", f1_weighted]
    f1_df.to_csv("f1_scores_test.csv", index=False)

    # ===========================
    #   REPORT DI CLASSIFICAZIONE
    # ===========================
    classif_report = classification_report(
        y_true,
        y_pred,
        labels=range(num_classes),
        digits=4
    )

    print("\n=== REPORT DI CLASSIFICAZIONE ===")
    print(classif_report)

    # ===========================
    #   ROC-AUC (multi-classe)
    # ===========================
    # Binarizza le etichette: shape (n_samples, num_classes)
    y_true_bin = label_binarize(y_true, classes=range(num_classes))

    # ROC-AUC globale (macro e weighted) con strategia OVR
    try:
        roc_auc_macro_ovr = roc_auc_score(
            y_true_bin, y_scores, average="macro", multi_class="ovr"
        )
        roc_auc_weighted_ovr = roc_auc_score(
            y_true_bin, y_scores, average="weighted", multi_class="ovr"
        )

        print("\n=== ROC-AUC GLOBALI (OVR) ===")
        print(f"ROC-AUC macro (OVR):    {roc_auc_macro_ovr:.4f}")
        print(f"ROC-AUC weighted (OVR): {roc_auc_weighted_ovr:.4f}")
    except ValueError as e:
        print("\n[ATTENZIONE] Impossibile calcolare ROC-AUC globale (OVR):", e)
        roc_auc_macro_ovr = None
        roc_auc_weighted_ovr = None

    # ROC-AUC per classe + salvataggio punti ROC per plottare
    per_class_auc = {}
    for c in range(num_classes):
        try:
            fpr, tpr, _ = roc_curve(y_true_bin[:, c], y_scores[:, c])
            auc_c = auc(fpr, tpr)
            per_class_auc[c] = auc_c

            # Salva i punti ROC della classe c su CSV
            roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
            roc_df.to_csv(f"roc_curve_class_{c}.csv", index=False)
        except ValueError as e:
            print(f"[ATTENZIONE] ROC-AUC non calcolabile per classe {c}: {e}")
            per_class_auc[c] = None

    print("\n=== ROC-AUC PER CLASSE ===")
    for c, auc_c in per_class_auc.items():
        if auc_c is not None:
            print(f"Classe {c}: AUC = {auc_c:.4f}")
        else:
            print(f"Classe {c}: AUC non disponibile")

    # ritorna un dizionario con tutte le metriche principali
    results = {
        "global_accuracy": global_accuracy,
        "mean_batch_accuracy": mean_batch_accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "f1_per_class": f1_per_class,
        "roc_auc_macro_ovr": roc_auc_macro_ovr,
        "roc_auc_weighted_ovr": roc_auc_weighted_ovr,
        "per_class_auc": per_class_auc
    }

    return results


print('\n=== VALUTAZIONE FINALE SUL TEST SET ===')
results = test(test_loader, nclass)

print("\n=== RISULTATI FINALI SINTETICI ===")
print(f"Accuratezza totale:        {results['global_accuracy']:.4f} ({results['global_accuracy']*100:.2f}%)")
print(f"Accuratezza media per batch: {results['mean_batch_accuracy']:.4f} ({results['mean_batch_accuracy']*100:.2f}%)")
print(f"F1 macro:                  {results['f1_macro']:.4f}")
print(f"F1 weighted:               {results['f1_weighted']:.4f}")
print(f"ROC-AUC macro (OVR):       {results['roc_auc_macro_ovr']}")
print(f"ROC-AUC weighted (OVR):    {results['roc_auc_weighted_ovr']}")


# Calcola e stampa il tempo totale di training
total_training_time = time.time() - t_total_train
print(f'\nTempo totale di training: {total_training_time:.2f} secondi ({total_training_time / 60:.1f} minuti)')

# ESEMPIO DI COMANDO:
# python cancer_test.py --model gat --num_gene 100 --cancer_subtype True --specific_type brca --batch_size 16 --dropout 0.6 --omic_mode 4 --shuffle_index 0 --gene_gene True --mirna_gene True --mirna_mirna True --parallel True --l2 True --decoder True --poolsize 8 --edge_weight True --epochs 100 --train_ratio 0.8 --test_ratio 0.1
# ~84-85% accuracy
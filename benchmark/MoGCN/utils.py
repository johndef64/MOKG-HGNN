#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/8/8 16:21
# @Author  : Li Xiao
# @File    : utils.py

"""
Modulo di utilità per il caricamento e preprocessing dei dati per MoGCN
Questo modulo contiene funzioni di supporto per il caricamento dei dati,
il calcolo della matrice di adiacenza laplaciana e la valutazione delle metriche.
"""

import pandas as pd     # Manipolazione e analisi dei dati tabulari
import numpy as np
from sklearn.metrics import f1_score      # Operazioni numeriche e calcoli matriciali
import torch           # Framework di deep learning
from torch_geometric.utils import dense_to_sparse  # Conversione da matrice densa a edge_index

# load data può restituire edge_index per torch geometric

def load_data(adj, fea, lab, mode=0, threshold=0.005):
    """
    Carica e preprocessa i dati per l'addestramento della GCN
    
    Questa funzione carica i file di input (matrice di similarità, caratteristiche e etichette),
    li allinea per campione e calcola la matrice di adiacenza laplaciana normalizzata
    utilizzata dalla Graph Convolutional Network.
    
    Args:
        adj (str): Percorso del file contenente la matrice di similarità (SNF fused matrix)
        fea (str): Percorso del file contenente le caratteristiche omiche vettoriali
        lab (str): Percorso del file contenente le etichette dei campioni
        threshold (float): Soglia per filtrare gli archi deboli (default: 0.005)
    
    Returns:
        tuple: Una tupla contenente:
            - adj_hat (np.ndarray): Matrice di adiacenza laplaciana normalizzata
            - fea_df (pd.DataFrame): DataFrame delle caratteristiche allineate
            - label_df (pd.DataFrame): DataFrame delle etichette allineate
    
    Note:
        La matrice SNF è completamente connessa, quindi è consigliabile filtrare
        gli archi con una soglia per migliorare le prestazioni della GCN.
    """
    print('loading data...')
    
    # Carica i file CSV in DataFrame pandas
    adj_df = pd.read_csv(adj, header=0, index_col=None)    # Matrice di similarità/adiacenza
    fea_df = pd.read_csv(fea, header=0, index_col=None)    # Caratteristiche omiche
    label_df = pd.read_csv(lab, header=0, index_col=None)  # Etichette dei campioni

    # Verifica che tutti i file abbiano lo stesso numero di campioni
    if adj_df.shape[0] != fea_df.shape[0] or adj_df.shape[0] != label_df.shape[0]:
        print('Input files must have same samples.')
        exit(1)

    # Standardizza i nomi delle colonne dei campioni per tutti i DataFrame
    adj_df.rename(columns={adj_df.columns.tolist()[0]: 'Sample'}, inplace=True)
    fea_df.rename(columns={fea_df.columns.tolist()[0]: 'Sample'}, inplace=True)
    label_df.rename(columns={label_df.columns.tolist()[0]: 'Sample'}, inplace=True)

    # Allinea i campioni di tutti i dataset ordinandoli alfabeticamente
    # Questo garantisce che l'ordine dei campioni sia coerente tra tutti i file
    adj_df.sort_values(by='Sample', ascending=True, inplace=True)
    fea_df.sort_values(by='Sample', ascending=True, inplace=True)
    label_df.sort_values(by='Sample', ascending=True, inplace=True)

    print('Calculating the laplace adjacency matrix...')
    
    # Estrae la matrice di similarità numerica (escludendo la colonna dei campioni)
    adj_m = adj_df.iloc[:, 1:].values
    
    # Filtra gli archi deboli utilizzando la soglia specificata
    # La matrice SNF è completamente connessa, quindi è meglio filtrare gli archi con una soglia
    # per migliorare le prestazioni e ridurre il rumore
    adj_m[adj_m < threshold] = 0

    if mode == 0:
        # Crea la matrice di adiacenza binaria dopo il filtraggio
        # exist[i,j] = 1 se esiste un arco tra i nodi i e j, 0 altrimenti
        exist = (adj_m != 0) * 1.0
        #np.savetxt('result/adjacency_matrix.csv', exist, delimiter=',', fmt='%d')

        # Calcola la matrice dei gradi per la normalizzazione laplaciana
        # Il grado di un nodo è il numero di archi connessi ad esso
        factor = np.ones(adj_m.shape[1])     # Vettore di uni per il calcolo della somma
        res = np.dot(exist, factor)          # Calcola il grado di ciascun nodo (somma di riga)
        diag_matrix = np.diag(res)           # Crea la matrice diagonale dei gradi
        #np.savetxt('result/diag.csv', diag_matrix, delimiter=',', fmt='%d')

        # Calcola la matrice laplaciana normalizzata: D^(-1) * A
        # Questa normalizzazione è standard nelle Graph Convolutional Networks
        # e aiuta a stabilizzare l'addestramento prevenendo l'esplosione dei gradienti
        d_inv = np.linalg.inv(diag_matrix)   # Inverso della matrice dei gradi
        adj_hat = d_inv.dot(exist)           # Matrice di adiacenza laplaciana normalizzata
        
        return adj_hat, fea_df, label_df
    elif mode == 1:
        # === PyTorch Geometric: uso edge_index ===
        adj_tensor = torch.tensor(adj_m, dtype=torch.float32)
        edge_index, _ = dense_to_sparse(adj_tensor)

        features = torch.tensor(fea_df.iloc[:, 1:].values, dtype=torch.float32)
        labels = torch.tensor(label_df.iloc[:, 1].values, dtype=torch.long)
        sample_names = fea_df['Sample'].tolist()
        return edge_index, features, labels, sample_names

def accuracy(output, labels):
    """
    Calcola l'accuratezza delle predizioni del modello
    
    Questa funzione confronta le predizioni del modello con le etichette vere
    per calcolare la percentuale di classificazioni corrette.
    
    Args:
        output (torch.Tensor): Output del modello (logits o probabilità)
                              Forma: (n_samples, n_classes)
        labels (torch.Tensor): Etichette vere dei campioni
                              Forma: (n_samples,)
    
    Returns:
        torch.Tensor: Accuratezza come frazione di predizioni corrette
                     Valore compreso tra 0.0 e 1.0
    
    Note:
        L'output viene convertito in predizioni prendendo l'indice della classe
        con il valore massimo (argmax). Questo è appropriato per problemi di
        classificazione multi-classe.
    """
    # Converte l'output in predizioni prendendo l'indice della classe con probabilità massima
    # output.max(1)[1] restituisce gli indici delle classi predette
    pred = output.max(1)[1].type_as(labels)
    
    # Confronta le predizioni con le etichette vere
    # eq() restituisce un tensor booleano, double() lo converte in float
    correct = pred.eq(labels).double()
    
    # Somma il numero di predizioni corrette
    correct = correct.sum()
    
    # Calcola l'accuratezza come frazione: corrette / totali
    return correct / len(labels)

def metrics_calc(y_true_file, y_pred_file):
    """
    Calcola accuratezza e F1-score pesato
    
    Questa funzione calcola due metriche di valutazione comuni per problemi di classificazione:
    l'accuratezza e il F1-score pesato. L'accuratezza misura la frazione di predizioni corrette,
    mentre il F1-score bilancia precisione e richiamo, ed è utile in presenza di classi sbilanciate.
    
    Args:
        y_true (list or np.ndarray): Etichette vere dei campioni
        y_pred (list or np.ndarray): Etichette predette dai campioni

    Returns:
        dict: Un dizionario contenente l'accuratezza e il F1-score pesato
    """
    from sklearn.metrics import accuracy_score

    y_true, y_pred = get_labels(y_true_file, y_pred_file)
    
    accuracy_value = accuracy_score(y_true, y_pred)
    f1_value = f1_score(y_true, y_pred, average='weighted')
    print(f'Accuracy: {accuracy_value:.4f}, F1-score (weighted): {f1_value:.4f}')
    return accuracy_value, f1_value

def get_labels(sample_labels_file, prediction_file):
    """
    Carica le etichette vere e predette da file CSV per i campioni comuni
    
    Questa funzione legge due file CSV: uno contenente le etichette vere dei campioni
    e l'altro contenente le etichette predette. Seleziona solo i campioni comuni
    tra i due file ed estrae le colonne 'predicted_label' e 'pam50'.
    
    Args:
        prediction_file (str): Percorso del file CSV con le etichette predette
        sample_lables_files (str): Percorso del file CSV con le etichette vere
    
    Returns:
        tuple: Una tupla contenente:
            - y_true (list): Lista delle etichette vere (pam50) per i campioni comuni
            - y_pred (list): Lista delle etichette predette per i campioni comuni
    """
    # Carica il file delle etichette vere in un DataFrame pandas
    label_df = pd.read_csv(sample_labels_file)
    
    # Carica il file delle etichette predette in un DataFrame pandas
    pred_df = pd.read_csv(prediction_file)

    # Trova i campioni comuni tra i due file
    common_samples = set(label_df['Sample_ID']).intersection(set(pred_df['Sample']))
    print(f"Campioni comuni trovati: {len(common_samples)}")
    
    # Filtra i DataFrame per includere solo i campioni comuni
    label_common = label_df[label_df['Sample_ID'].isin(common_samples)].copy()
    pred_common = pred_df[pred_df['Sample'].isin(common_samples)].copy()
    
    # Ordina entrambi i DataFrame per campione per garantire l'allineamento
    label_common = label_common.sort_values('Sample_ID')
    pred_common = pred_common.sort_values('Sample')
    
    # Verifica che l'ordine dei campioni sia lo stesso
    if not label_common['Sample_ID'].equals(pred_common['Sample']):
        print("ERRORE: L'ordine dei campioni non corrisponde dopo l'allineamento!")
        return None, None
    
    # Estrae le etichette vere dalla colonna 'pam50' se esiste, altrimenti dalla seconda colonna
    if 'class' in label_common.columns:
        y_true = label_common['class'].tolist()
        print("Usando colonna 'class' per le etichette vere")
    else:
        y_true = label_common.iloc[:, 1].tolist()
        print(f"Colonna 'pam50' non trovata, usando colonna: {label_common.columns[1]}")
    
    # Estrae le etichette predette dalla colonna 'predicted_label' se esiste, altrimenti dalla seconda colonna
    if 'predict_label' in pred_common.columns:
        y_pred = pred_common['predict_label'].tolist()
        print("Usando colonna 'predict_label' per le predizioni")
    else:
        y_pred = pred_common.iloc[:, 1].tolist()
        print(f"Colonna 'predicted_label' non trovata, usando colonna: {pred_common.columns[1]}")

    print(f"Etichette estratte per {len(y_true)} campioni comuni")
    
    return y_true, y_pred

metrics_calc('data/sample_classes.csv', 'result/GCN_predicted_data.csv')

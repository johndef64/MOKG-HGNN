#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/8/8 14:01
# @Author  : Li Xiao
# @File    : SNF.py

"""
Script per la Similarity Network Fusion (SNF) di dati multi-omici
Questo script implementa l'algoritmo SNF per integrare tre tipi diversi di dati omici
creando una rete di similarità fusa che cattura le relazioni complementari tra i campioni.
"""

import snf                      # Libreria per Similarity Network Fusion
import pandas as pd             # Manipolazione e analisi dei dati tabulari
import numpy as np              # Operazioni numeriche e array
import argparse                 # Parsing degli argomenti da riga di comando
import seaborn as sns           # Visualizzazione statistica avanzata

if __name__ == '__main__':
    """
    Blocco principale per l'esecuzione della Similarity Network Fusion
    
    Questo script carica tre tipi di dati omici, calcola le reti di affinità per ciascuno,
    le fonde utilizzando l'algoritmo SNF e salva la matrice risultante insieme a una
    visualizzazione clustermap.
    """
    
    # Configura il parser per gli argomenti da riga di comando
    parser = argparse.ArgumentParser()
    
    # Definisce i parametri configurabili per l'algoritmo SNF
    parser.add_argument('--path', '-p', type=str, nargs=3, required=True,
                        help='Percorsi dei file di input, devono essere esattamente 3 file')
    
    # Metrica di distanza per il calcolo della similarità
    parser.add_argument('--metric', '-m', type=str, choices=['braycurtis', 'canberra', 'chebyshev', 'cityblock',
                        'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski',
                        'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
                        'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'], default='sqeuclidean',
                        help='Metrica di distanza da utilizzare. Deve essere una delle metriche disponibili in scipy.spatial.distance.pdist.')
    
    # Numero di vicini più prossimi per la costruzione della matrice di affinità
    parser.add_argument('--K', '-k', type=int, default=20,
                        help='(0, N) int, numero di vicini da considerare nella creazione della matrice di affinità. Vedi Note di snf.compute.affinity_matrix per dettagli. Default: 20.')
    
    # Fattore di normalizzazione per il kernel di similarità
    parser.add_argument('--mu', '-mu', type=int, default=0.5,
                        help='(0, 1) float, fattore di normalizzazione per scalare il kernel di similarità nella costruzione della matrice di affinità. Vedi Note di snf.compute.affinity_matrix per dettagli. Default: 0.5.')
    
    # Analizza gli argomenti forniti
    args = parser.parse_args()

    print('Load data files...')
    
    # Carica i tre tipi di dati omici dai file CSV specificati
    omics_data_1 = pd.read_csv(args.path[0], header=0, index_col=None)  # Primo tipo di dati omici
    omics_data_2 = pd.read_csv(args.path[1], header=0, index_col=None)  # Secondo tipo di dati omici
    omics_data_3 = pd.read_csv(args.path[2], header=0, index_col=None)  # Terzo tipo di dati omici
    
    # Mostra le dimensioni dei dataset caricati per verifica
    print(omics_data_1.shape, omics_data_2.shape, omics_data_3.shape)

    # Verifica che tutti i dataset abbiano lo stesso numero di campioni
    # Questa è una condizione necessaria per l'algoritmo SNF
    if omics_data_1.shape[0] != omics_data_2.shape[0] or omics_data_1.shape[0] != omics_data_3.shape[0]:
        print('Input files must have same samples.')
        exit(1)

    # Standardizza i nomi delle colonne dei campioni per tutti i dataset
    # Questo assicura coerenza nella nomenclatura
    omics_data_1.rename(columns={omics_data_1.columns.tolist()[0]: 'Sample'}, inplace=True)
    omics_data_2.rename(columns={omics_data_2.columns.tolist()[0]: 'Sample'}, inplace=True)
    omics_data_3.rename(columns={omics_data_3.columns.tolist()[0]: 'Sample'}, inplace=True)

    # Allinea i campioni di tutti i dataset ordinandoli alfabeticamente
    # Questo garantisce che l'ordine dei campioni sia coerente tra tutti i tipi di dati
    omics_data_1.sort_values(by='Sample', ascending=True, inplace=True)
    omics_data_2.sort_values(by='Sample', ascending=True, inplace=True)
    omics_data_3.sort_values(by='Sample', ascending=True, inplace=True)

    print('Start similarity network fusion...')
    
    # Crea le reti di affinità per ciascun tipo di dato omico
    # L'algoritmo SNF richiede prima di calcolare una matrice di affinità per ogni tipo di dato
    # che cattura le similarità tra campioni basate su quel specifico tipo di informazione
    affinity_nets = snf.make_affinity([omics_data_1.iloc[:, 1:].values.astype(np.float64), 
                                       omics_data_2.iloc[:, 1:].values.astype(np.float64), 
                                       omics_data_3.iloc[:, 1:].values.astype(np.float64)],
                                      metric=args.metric,    # Metrica di distanza specificata
                                      K=args.K,             # Numero di vicini più prossimi
                                      mu=args.mu)           # Fattore di normalizzazione

    # Esegue la fusione delle reti di similarità utilizzando l'algoritmo SNF
    # SNF integra le diverse reti di affinità in una singola rete consenso
    # che cattura le informazioni complementari di tutti i tipi di dati omici
    fused_net = snf.snf(affinity_nets, K=args.K)

    print('Save fused adjacency matrix...')
    
    # Converte la rete fusa in un DataFrame pandas per facilità di manipolazione
    fused_df = pd.DataFrame(fused_net)
    
    # Imposta i nomi dei campioni come etichette per righe e colonne
    # Questo rende la matrice più interpretabile e tracciabile
    fused_df.columns = omics_data_1['Sample'].tolist()  # Etichette delle colonne
    fused_df.index = omics_data_1['Sample'].tolist()    # Etichette delle righe
    
    # Salva la matrice di adiacenza fusa in formato CSV
    # Questa matrice sarà utilizzata successivamente dalla GCN
    fused_df.to_csv('result/SNF_fused_matrix.csv', header=True, index=True)

    # Prepara la matrice per la visualizzazione rimuovendo la diagonale
    # I valori sulla diagonale (auto-similarità) sono impostati a 0 per una migliore visualizzazione
    np.fill_diagonal(fused_df.values, 0)
    
    # Crea una clustermap per visualizzare la struttura della rete fusa
    # La clustermap raggruppa automaticamente campioni simili e mostra la struttura dei cluster
    fig = sns.clustermap(fused_df.iloc[:, :],     # Utilizza tutta la matrice
                        cmap='vlag',              # Colormap simmetrica centrata su zero
                        figsize=(8, 8))          # Dimensioni del grafico
    
    # Salva la visualizzazione come immagine ad alta risoluzione
    fig.savefig('result/SNF_fused_clustermap.png', dpi=300)
    
    print('Success! Results can be seen in result file')
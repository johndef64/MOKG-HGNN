#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/8/7 14:43
# @Author  : Li Xiao
# @File    : AE_run.py

"""
Script per l'esecuzione di autoencoder multi-omici (MMAE - Multi-Modal AutoEncoder)
Questo script implementa l'addestramento e l'utilizzo di un autoencoder per l'integrazione 
di dati multi-omici, estrazione di caratteristiche e riduzione della dimensionalità.
"""

# Fix per errore OpenMP: imposta la variabile d'ambiente prima di importare le librerie
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import pandas as pd          # Manipolazione e analisi dei dati tabulari
import numpy as np           # Operazioni numeriche e array
import argparse             # Parsing degli argomenti da riga di comando
from tqdm import tqdm       # Barre di progresso per i loop
import autoencoder_model    # Modulo personalizzato per il modello autoencoder
import torch                # Framework deep learning PyTorch
import torch.utils.data as Data  # Utilità per la gestione dei dataset

def setup_seed(seed):
    """
    Imposta il seed per la riproducibilità dei risultati
    
    Args:
        seed (int): Valore del seed per i generatori di numeri casuali
    
    Note:
        Imposta il seed sia per PyTorch che per NumPy per garantire
        risultati deterministici e riproducibili
    """
    torch.manual_seed(seed)      # Imposta seed per PyTorch (CPU)
    np.random.seed(seed)         # Imposta seed per NumPy

def work(data, in_feas, lr=0.001, bs=32, epochs=100, device=torch.device('cpu'), a=0.4, b=0.3, c=0.3, mode=0, topn=100):
    """
    Funzione principale per l'addestramento dell'autoencoder multi-omico e l'estrazione delle caratteristiche
    
    Args:
        data (pd.DataFrame): DataFrame contenente i dati multi-omici concatenati
        in_feas (list): Lista con le dimensioni di ciascun tipo di dato omico [omics1_dim, omics2_dim, omics3_dim]
        lr (float): Tasso di apprendimento per l'ottimizzatore (default: 0.001)
        bs (int): Dimensione del batch per l'addestramento (default: 32)
        epochs (int): Numero di epoche di addestramento (default: 100)
        device (torch.device): Dispositivo di calcolo (CPU o GPU) (default: CPU)
        a (float): Peso per il primo tipo di dati omici (default: 0.4)
        b (float): Peso per il secondo tipo di dati omici (default: 0.3)
        c (float): Peso per il terzo tipo di dati omici (default: 0.3)
        mode (int): Modalità di esecuzione (0: addestra e integra, 1: solo addestramento, 2: solo integrazione)
        topn (int): Numero di caratteristiche top da estrarre (default: 100)
    """
    # Estrae i nomi dei campioni dalla prima colonna del dataset
    sample_name = data['Sample'].tolist()

    # Converte i dati in tensori PyTorch per l'elaborazione
    # X contiene le caratteristiche, Y è un array di zeri (non utilizzato per l'autoencoder)
    X, Y = data.iloc[:,1:].values, np.zeros(data.shape[0])
    TX, TY = torch.tensor(X, dtype=torch.float, device=device), torch.tensor(Y, dtype=torch.float, device=device)
    
    # Fase di addestramento del modello autoencoder (mode 0 o 1)
    if mode == 0 or mode == 1:
        print('Training model...')
        # Crea un dataset PyTorch dai tensori
        Tensor_data = Data.TensorDataset(TX, TY)
        # Crea un DataLoader per il batch processing durante l'addestramento
        train_loader = Data.DataLoader(Tensor_data, batch_size=bs, shuffle=True)

        # Inizializza il modello Multi-Modal AutoEncoder (MMAE)
        mmae = autoencoder_model.MMAE(in_feas, latent_dim=100, a=a, b=b, c=c)
        mmae.to(device)          # Sposta il modello sul dispositivo specificato
        mmae.train()             # Imposta il modello in modalità addestramento
        # Addestra il modello MMAE
        mmae.train_MMAE(train_loader, learning_rate=lr, device=device, epochs=epochs)
        mmae.eval()              # Imposta il modello in modalità valutazione prima del salvataggio
        # Salva il modello addestrato
        if not os.path.exists('model/AE'):
            os.makedirs('model/AE')
        torch.save(mmae, 'model/AE/MMAE_model.pkl')

    # Fase di integrazione dei dati e riduzione della dimensionalità (mode 0 o 2)
    if mode == 0 or mode == 2:
        print('Get the latent layer output...')
        # Carica il modello salvato per l'integrazione dei dati
        torch.serialization.add_safe_globals([autoencoder_model.MMAE])
        mmae = torch.load('model/AE/MMAE_model.pkl', weights_only=False)
        
        # Separa i dati multi-omici in base alle dimensioni specificate
        omics_1 = TX[:, :in_feas[0]]                                                    # Primo tipo di dati omici
        omics_2 = TX[:, in_feas[0]:in_feas[0]+in_feas[1]]                             # Secondo tipo di dati omici
        omics_3 = TX[:, in_feas[0]+in_feas[1]:in_feas[0]+in_feas[1]+in_feas[2]]       # Terzo tipo di dati omici
        
        # Esegue il forward pass attraverso l'autoencoder per ottenere:
        # - latent_data: rappresentazione compressa nello spazio latente
        # - decoded_omics_*: ricostruzioni dei dati omici originali
        latent_data, decoded_omics_1, decoded_omics_2, decoded_omics_3 = mmae.forward(omics_1, omics_2, omics_3)
        
        # Converte i dati latenti in DataFrame per il salvataggio
        latent_df = pd.DataFrame(latent_data.detach().cpu().numpy())
        latent_df.insert(0, 'Sample', sample_name)  # Aggiunge i nomi dei campioni come prima colonna
        
        # Salva i dati integrati (dimensionalità ridotta a 100) in formato CSV
        latent_df.to_csv('result/latent_data.csv', header=True, index=False)

    print('Extract features...')
    # Esegue l'estrazione delle caratteristiche più importanti
    extract_features(data, in_feas, epochs, topn)
    return

def extract_features(data, in_feas, epochs, topn=100):
    """
    Estrae le caratteristiche più importanti da ciascun tipo di dati omici
    basandosi sui pesi dell'autoencoder e sulla deviazione standard
    
    Args:
        data (pd.DataFrame): DataFrame contenente i dati multi-omici
        in_feas (list): Lista delle dimensioni di ciascun tipo di dato omico
        epochs (int): Numero totale di epoche di addestramento
        topn (int): Numero di caratteristiche top da estrarre (default: 100)
    
    Note:
        L'importanza delle caratteristiche è calcolata come: Peso * Deviazione_Standard
        I modelli salvati ogni 10 epoche vengono utilizzati per l'analisi temporale
    """
    # Estrae i dati di ciascun tipo omico separatamente
    data_omics_1 = data.iloc[:, 1: 1+in_feas[0]]                                      # Primo tipo di dati omici
    data_omics_2 = data.iloc[:, 1+in_feas[0]: 1+in_feas[0]+in_feas[1]]              # Secondo tipo di dati omici
    data_omics_3 = data.iloc[:, 1+in_feas[0]+in_feas[1]: 1+in_feas[0]+in_feas[1]+in_feas[2]]  # Terzo tipo di dati omici

    # Ottiene i nomi delle caratteristiche per ciascun tipo di dato omico
    feas_omics_1 = data_omics_1.columns.tolist()
    feas_omics_2 = data_omics_2.columns.tolist()
    feas_omics_3 = data_omics_3.columns.tolist()

    # Calcola la deviazione standard di ciascuna caratteristica
    # La deviazione standard indica la variabilità di una caratteristica tra i campioni
    std_omics_1 = data_omics_1.std(axis=0)
    std_omics_2 = data_omics_2.std(axis=0)
    std_omics_3 = data_omics_3.std(axis=0)

    # Inizializza DataFrame per memorizzare le top N caratteristiche per ogni epoca
    topn_omics_1 = pd.DataFrame()
    topn_omics_2 = pd.DataFrame()
    topn_omics_3 = pd.DataFrame()

    # Crea una lista di epoche per l'estrazione delle caratteristiche (ogni 10 epoche)
    # Se il numero totale di epoche non è divisibile per 10, aggiunge l'ultima epoca
    epoch_ls = list(range(10, epochs+10, 10))
    if epochs % 10 != 0:
        epoch_ls.append(epochs)
    
    # Itera attraverso le epoche selezionate per l'analisi delle caratteristiche
    for epoch in tqdm(epoch_ls):
        # Carica il modello salvato all'epoca specifica
        mmae = torch.load('model/AE/model_{}.pkl'.format(epoch), weights_only=False)
        
        # Ottiene i parametri del modello (pesi e bias)
        model_dict = mmae.state_dict()

        # Estrae i pesi assoluti dai layer encoder per ciascun tipo di dato omico
        # I pesi indicano l'importanza di ciascuna caratteristica nella codifica
        # Forma della matrice: (n_features, latent_layer_dim) -> trasposta per avere (n_features, latent_dim)
        weight_omics1 = np.abs(model_dict['encoder_omics_1.0.weight'].detach().cpu().numpy().T)
        weight_omics2 = np.abs(model_dict['encoder_omics_2.0.weight'].detach().cpu().numpy().T)
        weight_omics3 = np.abs(model_dict['encoder_omics_3.0.weight'].detach().cpu().numpy().T)

        # Converte i pesi in DataFrame con i nomi delle caratteristiche come indici
        weight_omics1_df = pd.DataFrame(weight_omics1, index=feas_omics_1)
        weight_omics2_df = pd.DataFrame(weight_omics2, index=feas_omics_2)
        weight_omics3_df = pd.DataFrame(weight_omics3, index=feas_omics_3)

        # Calcola la somma dei pesi per ciascuna caratteristica (somma di ogni riga)
        # Questo rappresenta l'importanza totale della caratteristica nella rete
        weight_omics1_df['Weight_sum'] = weight_omics1_df.apply(lambda x: x.sum(), axis=1)
        weight_omics2_df['Weight_sum'] = weight_omics2_df.apply(lambda x: x.sum(), axis=1)
        weight_omics3_df['Weight_sum'] = weight_omics3_df.apply(lambda x: x.sum(), axis=1)
        # Aggiunge le deviazioni standard calcolate precedentemente
        weight_omics1_df['Std'] = std_omics_1
        weight_omics2_df['Std'] = std_omics_2
        weight_omics3_df['Std'] = std_omics_3

        # Calcola l'importanza finale di ciascuna caratteristica
        # Importanza = Somma_Pesi * Deviazione_Standard
        # Questo combina l'importanza nella rete neurale con la variabilità nei dati
        weight_omics1_df['Importance'] = weight_omics1_df['Weight_sum'] * weight_omics1_df['Std']
        weight_omics2_df['Importance'] = weight_omics2_df['Weight_sum'] * weight_omics2_df['Std']
        weight_omics3_df['Importance'] = weight_omics3_df['Weight_sum'] * weight_omics3_df['Std']

        # Seleziona le top N caratteristiche per ciascun tipo di dato omico
        # basandosi sul punteggio di importanza calcolato
        fea_omics_1_top = weight_omics1_df.nlargest(topn, 'Importance').index.tolist()
        fea_omics_2_top = weight_omics2_df.nlargest(topn, 'Importance').index.tolist()
        fea_omics_3_top = weight_omics3_df.nlargest(topn, 'Importance').index.tolist()

        # Salva le top N caratteristiche nel DataFrame corrispondente
        # con il nome della colonna che indica l'epoca
        col_name = 'epoch_' + str(epoch)
        topn_omics_1[col_name] = fea_omics_1_top
        topn_omics_2[col_name] = fea_omics_2_top
        topn_omics_3[col_name] = fea_omics_3_top

    # Salva tutti i risultati delle top N caratteristiche in file CSV separati
    topn_omics_1.to_csv('result/topn_omics_1.csv', header=True, index=False)
    topn_omics_2.to_csv('result/topn_omics_2.csv', header=True, index=False)
    topn_omics_3.to_csv('result/topn_omics_3.csv', header=True, index=False)

if __name__ == '__main__':
    """
    Blocco principale di esecuzione dello script
    Gestisce il parsing degli argomenti, la preparazione dei dati e l'esecuzione del pipeline
    """
    
    # Configura il parser per gli argomenti da riga di comando
    parser = argparse.ArgumentParser()
    
    # Definisce tutti i parametri configurabili dello script
    parser.add_argument('--mode', '-m', type=int, choices=[0,1,2], default=0,
                        help='Modalità di esecuzione: 0=addestra&integra, 1=solo addestramento, 2=solo integrazione, default: 0.')
    parser.add_argument('--seed', '-s', type=int, default=0, 
                        help='Seed per la riproducibilità, default=0.')
    parser.add_argument('--path1', '-p1', type=str, required=True, 
                        help='Nome del file del primo tipo di dati omici.')
    parser.add_argument('--path2', '-p2', type=str, required=True, 
                        help='Nome del file del secondo tipo di dati omici.')
    parser.add_argument('--path3', '-p3', type=str, required=True, 
                        help='Nome del file del terzo tipo di dati omici.')
    parser.add_argument('--batchsize', '-bs', type=int, default=32, 
                        help='Dimensione del batch per l\'addestramento, default: 32.')
    parser.add_argument('--learningrate', '-lr', type=float, default=0.001, 
                        help='Tasso di apprendimento, default: 0.001.')
    parser.add_argument('--epoch', '-e', type=int, default=100, 
                        help='Numero di epoche di addestramento, default: 100.')
    parser.add_argument('--latent', '-l', type=int, default=100, 
                        help='Dimensione del layer latente, default: 100.')
    parser.add_argument('--device', '-d', type=str, choices=['cpu', 'gpu'], default='cpu', 
                        help='Dispositivo per l\'addestramento (cpu o gpu), default: cpu.')
    parser.add_argument('--a', '-a', type=float, default=0.6, 
                        help='[0,1], peso per il primo tipo di dati omici')
    parser.add_argument('--b', '-b', type=float, default=0.1, 
                        help='[0,1], peso per il secondo tipo di dati omici.')
    parser.add_argument('--c', '-c', type=float, default=0.3, 
                        help='[0,1], peso per il terzo tipo di dati omici.')
    parser.add_argument('--topn', '-n', type=int, default=100, 
                        help='Estrae le top N caratteristiche ogni 10 epoche, default: 100.')
    
    # Analizza gli argomenti forniti
    args = parser.parse_args()

    # Carica i dati multi-omici dai file CSV specificati
    omics_data1 = pd.read_csv(args.path1, header=0, index_col=None)  # Primo tipo di dati omici
    omics_data2 = pd.read_csv(args.path2, header=0, index_col=None)  # Secondo tipo di dati omici  
    omics_data3 = pd.read_csv(args.path3, header=0, index_col=None)  # Terzo tipo di dati omici

    # Configura il dispositivo di calcolo (CPU o GPU)
    device = torch.device('cpu')  # Dispositivo predefinito
    if args.device == 'gpu':
        # Utilizza GPU se disponibile, altrimenti fallback su CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Imposta il seed per la riproducibilità dei risultati
    setup_seed(args.seed)

    # Verifica che la somma dei pesi sia uguale a 1.0
    if args.a + args.b + args.c != 1.0:
        print('La somma dei pesi deve essere uguale a 1.')
        exit(1)

    # Calcola le dimensioni di ciascun tipo di dato omico (escludendo la colonna dei campioni)
    in_feas = [omics_data1.shape[1] - 1, omics_data2.shape[1] - 1, omics_data3.shape[1] - 1]
    
    # Standardizza i nomi delle colonne dei campioni per tutti i dataset
    omics_data1.rename(columns={omics_data1.columns.tolist()[0]: 'Sample'}, inplace=True)
    omics_data2.rename(columns={omics_data2.columns.tolist()[0]: 'Sample'}, inplace=True)
    omics_data3.rename(columns={omics_data3.columns.tolist()[0]: 'Sample'}, inplace=True)

    # Ordina tutti i dataset per nome del campione per garantire coerenza
    omics_data1.sort_values(by='Sample', ascending=True, inplace=True)
    omics_data2.sort_values(by='Sample', ascending=True, inplace=True)
    omics_data3.sort_values(by='Sample', ascending=True, inplace=True)

    # Unisce i dati multi-omici basandosi sui campioni comuni
    # Utilizza inner join per mantenere solo i campioni presenti in tutti e tre i dataset
    Merge_data = pd.merge(omics_data1, omics_data2, on='Sample', how='inner')  # Unisce primo e secondo
    Merge_data = pd.merge(Merge_data, omics_data3, on='Sample', how='inner')   # Aggiunge il terzo dataset
    Merge_data.sort_values(by='Sample', ascending=True, inplace=True)          # Ordina il dataset finale

    # Esegue il pipeline completo: addestramento del modello, riduzione della dimensionalità ed estrazione delle caratteristiche
    work(Merge_data, in_feas, lr=args.learningrate, bs=args.batchsize, epochs=args.epoch, 
         device=device, a=args.a, b=args.b, c=args.c, mode=args.mode, topn=args.topn)
    
    print('Successo! I risultati possono essere visualizzati nella cartella result')

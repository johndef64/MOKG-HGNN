#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/8/7 14:01
# @Author  : Li Xiao
# @File    : autoencoder_model.py

"""
Modulo per l'implementazione di un Multi-Modal AutoEncoder (MMAE)
Questo modulo contiene la definizione della classe MMAE che implementa un autoencoder
per l'integrazione di dati multi-omici con tre canali di input separati.
"""

import torch                    # Framework deep learning PyTorch
from torch import nn           # Moduli delle reti neurali di PyTorch
from matplotlib import pyplot as plt  # Libreria per la visualizzazione dei grafici

class MMAE(nn.Module):
    """
    Multi-Modal AutoEncoder (MMAE) per l'integrazione di dati multi-omici
    
    Questa classe implementa un autoencoder che può elaborare tre tipi diversi
    di dati omici simultaneamente, creando una rappresentazione latente integrata
    pesata secondo i coefficienti specificati.
    
    Architettura:
    - Tre encoder separati per ciascun tipo di dato omico
    - Uno spazio latente condiviso per l'integrazione
    - Tre decoder separati per la ricostruzione di ciascun tipo di dato
    """
    
    def __init__(self, in_feas_dim, latent_dim, a=0.4, b=0.3, c=0.3):
        """
        Inizializza il Multi-Modal AutoEncoder
        
        Args:
            in_feas_dim (list): Lista contenente le dimensioni di input per ciascun tipo di dato omico
                               [dim_omics_1, dim_omics_2, dim_omics_3]
            latent_dim (int): Dimensione dello spazio latente condiviso
            a (float): Peso per il primo tipo di dati omici nella combinazione latente (default: 0.4)
            b (float): Peso per il secondo tipo di dati omici nella combinazione latente (default: 0.3)
            c (float): Peso per il terzo tipo di dati omici nella combinazione latente (default: 0.3)
        
        Note:
            La somma di a + b + c dovrebbe essere uguale a 1.0 per una combinazione pesata corretta
        """
        super(MMAE, self).__init__()
        
        # Salva i parametri di configurazione come attributi della classe
        self.a = a                    # Peso per il primo tipo di dati omici
        self.b = b                    # Peso per il secondo tipo di dati omici
        self.c = c                    # Peso per il terzo tipo di dati omici
        self.in_feas = in_feas_dim    # Dimensioni di input per ciascun tipo di dato
        self.latent = latent_dim      # Dimensione dello spazio latente

        # Definizione degli encoder - Tre encoder separati per i diversi tipi di dati omici
        # Ogni encoder comprime i dati di input nello spazio latente condiviso
        
        # Encoder per il primo tipo di dati omici
        self.encoder_omics_1 = nn.Sequential(
            nn.Linear(self.in_feas[0], self.latent),  # Layer lineare: input_dim -> latent_dim
            nn.BatchNorm1d(self.latent),              # Normalizzazione batch per stabilizzare l'addestramento
            nn.Sigmoid()                              # Funzione di attivazione sigmoide (output [0,1])
        )
        
        # Encoder per il secondo tipo di dati omici
        self.encoder_omics_2 = nn.Sequential(
            nn.Linear(self.in_feas[1], self.latent),  # Layer lineare: input_dim -> latent_dim
            nn.BatchNorm1d(self.latent),              # Normalizzazione batch
            nn.Sigmoid()                              # Funzione di attivazione sigmoide
        )
        
        # Encoder per il terzo tipo di dati omici
        self.encoder_omics_3 = nn.Sequential(
            nn.Linear(self.in_feas[2], self.latent),  # Layer lineare: input_dim -> latent_dim
            nn.BatchNorm1d(self.latent),              # Normalizzazione batch
            nn.Sigmoid()                              # Funzione di attivazione sigmoide
        )
        
        # Definizione dei decoder - Tre decoder separati per ricostruire i dati originali
        # Ogni decoder ricostruisce il tipo di dato corrispondente dallo spazio latente
        
        # Decoder per il primo tipo di dati omici (latent -> original_dim_1)
        self.decoder_omics_1 = nn.Sequential(nn.Linear(self.latent, self.in_feas[0]))
        
        # Decoder per il secondo tipo di dati omici (latent -> original_dim_2)
        self.decoder_omics_2 = nn.Sequential(nn.Linear(self.latent, self.in_feas[1]))
        
        # Decoder per il terzo tipo di dati omici (latent -> original_dim_3)
        self.decoder_omics_3 = nn.Sequential(nn.Linear(self.latent, self.in_feas[2]))

        # Inizializzazione dei parametri della rete neurale
        # Una buona inizializzazione è cruciale per la convergenza dell'addestramento
        for name, param in MMAE.named_parameters(self):
            if 'weight' in name:
                # Inizializzazione normale per i pesi con media=0 e deviazione standard=0.1
                # Questo aiuta a prevenire il problema dei gradienti che esplodono o svaniscono
                torch.nn.init.normal_(param, mean=0, std=0.1)
            if 'bias' in name:
                # Inizializzazione a zero per tutti i bias
                # I bias possono essere inizializzati a zero senza problemi
                torch.nn.init.constant_(param, val=0)

    def forward(self, omics_1, omics_2, omics_3):
        """
        Metodo forward per il passaggio in avanti attraverso la rete
        
        Questo metodo implementa il flusso completo di dati attraverso l'autoencoder:
        1. Codifica ciascun tipo di dato omico nello spazio latente
        2. Combina le rappresentazioni latenti con pesi specifici
        3. Decodifica la rappresentazione integrata per ricostruire i dati originali
        
        Args:
            omics_1 (torch.Tensor): Tensor contenente il primo tipo di dati omici
            omics_2 (torch.Tensor): Tensor contenente il secondo tipo di dati omici  
            omics_3 (torch.Tensor): Tensor contenente il terzo tipo di dati omici
        
        Returns:
            tuple: Una tupla contenente:
                - latent_data: Rappresentazione integrata nello spazio latente
                - decoded_omics_1: Ricostruzione del primo tipo di dati omici
                - decoded_omics_2: Ricostruzione del secondo tipo di dati omici
                - decoded_omics_3: Ricostruzione del terzo tipo di dati omici
        """
        # Fase di codifica: Trasforma ciascun tipo di dato omico nello spazio latente
        encoded_omics_1 = self.encoder_omics_1(omics_1)  # Codifica primo tipo di dati
        encoded_omics_2 = self.encoder_omics_2(omics_2)  # Codifica secondo tipo di dati
        encoded_omics_3 = self.encoder_omics_3(omics_3)  # Codifica terzo tipo di dati
        
        # Fase di integrazione: Combina le rappresentazioni codificate con pesi specifici
        # Formula: latent = a*encoded_1 + b*encoded_2 + c*encoded_3
        # Questo crea una rappresentazione integrata che cattura informazioni da tutti i tipi di dati
        latent_data = torch.mul(encoded_omics_1, self.a) + torch.mul(encoded_omics_2, self.b) + torch.mul(encoded_omics_3, self.c)
        
        # Fase di decodifica: Ricostruisce i dati originali dalla rappresentazione latente integrata
        decoded_omics_1 = self.decoder_omics_1(latent_data)  # Ricostruisce primo tipo di dati
        decoded_omics_2 = self.decoder_omics_2(latent_data)  # Ricostruisce secondo tipo di dati
        decoded_omics_3 = self.decoder_omics_3(latent_data)  # Ricostruisce terzo tipo di dati
        
        return latent_data, decoded_omics_1, decoded_omics_2, decoded_omics_3

    def train_MMAE(self, train_loader, learning_rate=0.001, device=torch.device('cpu'), epochs=100):
        """
        Metodo per l'addestramento del Multi-Modal AutoEncoder
        
        Questo metodo implementa il ciclo di addestramento completo dell'autoencoder,
        includendo l'ottimizzazione, il calcolo della loss pesata e il salvataggio periodico
        
        Args:
            train_loader (DataLoader): DataLoader contenente i dati di addestramento
            learning_rate (float): Tasso di apprendimento per l'ottimizzatore Adam (default: 0.001)
            device (torch.device): Dispositivo di calcolo (CPU o GPU) (default: CPU)
            epochs (int): Numero di epoche di addestramento (default: 100)
        
        Note:
            - Utilizza l'ottimizzatore Adam per l'aggiornamento dei parametri
            - La funzione di loss è MSE pesata per ciascun tipo di dato omico
            - Salva il modello ogni 10 epoche per l'analisi delle caratteristiche
            - Genera un grafico della loss di addestramento
        """
        # Configura l'ottimizzatore Adam con il tasso di apprendimento specificato
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        # Definisce la funzione di loss (Mean Squared Error)
        loss_fn = nn.MSELoss()
        
        # Lista per memorizzare la loss di ogni epoca (per la visualizzazione)
        loss_ls = []
        
        # Ciclo principale di addestramento
        for epoch in range(epochs):
            train_loss_sum = 0.0  # Accumula la loss totale per l'epoca corrente
            
            # Itera attraverso i batch di dati
            for (x, y) in train_loader:
                # Separa i dati multi-omici concatenati in base alle dimensioni specificate
                omics_1 = x[:, :self.in_feas[0]]                                                    # Primo tipo di dati omici
                omics_2 = x[:, self.in_feas[0]:self.in_feas[0]+self.in_feas[1]]                   # Secondo tipo di dati omici
                omics_3 = x[:, self.in_feas[0]+self.in_feas[1]:self.in_feas[0]+self.in_feas[1]+self.in_feas[2]]  # Terzo tipo di dati omici

                # Sposta i dati sul dispositivo specificato (CPU o GPU)
                omics_1 = omics_1.to(device)
                omics_2 = omics_2.to(device)
                omics_3 = omics_3.to(device)

                # Esegue il forward pass attraverso la rete
                latent_data, decoded_omics_1, decoded_omics_2, decoded_omics_3 = self.forward(omics_1, omics_2, omics_3)
                
                # Calcola la loss pesata per ciascun tipo di dato omico
                # Loss totale = a*Loss_1 + b*Loss_2 + c*Loss_3
                # Questo assicura che ogni tipo di dato contribuisca alla loss secondo il suo peso
                loss = self.a*loss_fn(decoded_omics_1, omics_1) + self.b*loss_fn(decoded_omics_2, omics_2) + self.c*loss_fn(decoded_omics_3, omics_3)
                
                # Azzeramento dei gradienti dell'ottimizzatore
                optimizer.zero_grad()
                
                # Calcolo dei gradienti tramite backpropagation
                loss.backward()
                
                # Aggiornamento dei parametri del modello
                optimizer.step()

                # Accumula la loss per il monitoraggio
                train_loss_sum += loss.sum().item()

            # Memorizza la loss dell'epoca per la visualizzazione
            loss_ls.append(train_loss_sum)
            print('epoch: %d | loss: %.4f' % (epoch + 1, train_loss_sum))

            # Salva il modello ogni 10 epoche per l'estrazione delle caratteristiche
            # Questi modelli intermedi sono utilizzati per analizzare l'evoluzione delle caratteristiche
            if (epoch+1) % 10 == 0:
                torch.save(self, 'model/AE/model_{}.pkl'.format(epoch+1))

        # Genera e salva il grafico della curva di loss durante l'addestramento
        # Questo grafico è utile per monitorare la convergenza del modello
        plt.plot([i + 1 for i in range(epochs)], loss_ls)  # Asse x: epoche, Asse y: loss
        plt.xlabel('epochs')          # Etichetta per l'asse x
        plt.ylabel('loss')           # Etichetta per l'asse y
        plt.savefig('result/AE_train_loss.png')  # Salva il grafico come immagine PNG
# Importazione delle librerie necessarie
import numpy as np
import pandas as pd
import math

# Librerie per PyTorch (framework di deep learning)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset
# from torch.utils.loader import DataLoader

# Librerie per Graph Neural Networks (GNN) con PyTorch Geometric
# from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv, GATv2Conv, ChebConv  # Diversi tipi di layer convolutivi per grafi
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import MNISTSuperpixels

import torch_geometric.transforms as T

# Configurazione del dispositivo (GPU se disponibile, altrimenti CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Definizione della classe GAT (Graph Attention Network)
# Questa rete neurale utilizza meccanismi di attenzione per analizzare dati genomici su grafi
class GAT(torch.nn.Module):
    def __init__(self, 
                    method,          # Tipo di metodo GAT da utilizzare ('gat' o 'gatv2')
                    parallel,        # Se utilizzare elaborazione parallela
                    l2,              # Se applicare regolarizzazione L2
                    decoder,         # Se includere un decoder per ricostruzione
                    poolsize,        # Dimensione del pooling per ridurre i nodi
                    poolrate,        # Tasso di pooling
                    edge_weights,    # Se utilizzare pesi degli archi
                    edge_attributes, # Se utilizzare attributi degli archi
                    num_gene,        # Numero di geni nel dataset
                    num_mirna,       # Numero di microRNA nel dataset
                    omic_mode,       # Modalità di dati omici (1=expr, 2=mirna, 3=expr+cnv, etc.)
                    num_classes,     # Numero di classi per la classificazione
                    dropout_rate):   # Tasso di dropout per evitare overfitting

        super(GAT, self).__init__()
        
        # Salvataggio dei parametri di configurazione
        self.omic_mode = omic_mode
        self.method = method
        self.parallel = parallel
        self.decoder = decoder
        self.l2 = l2
        self.poolsize = poolsize
        self.poolrate = poolrate
        self.edge_weights = edge_weights
        self.edge_attributes = edge_attributes
        
        # Iperparametri della rete
        self.hid = 6                    # Dimensione hidden layer
        self.head = 8                   # Numero di teste di attenzione
        self.num_gene = num_gene
        self.num_mirna = num_mirna
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.raised_dimension = 8       # Dimensione dopo la trasformazione lineare iniziale
        self.concate_layer = 64         # Dimensione del layer finale prima della classificazione

        # Determina il numero di features in base alla modalità omics
        if self.omic_mode < 3:
            self.num_features = 1       # Solo espressione genica o solo miRNA
        else:
            self.num_features = 2       # Espressione genica + CNV (Copy Number Variation)

        # Layer lineari pre-convolutivi per trasformare i dati iniziali
        self.pre_conv_linear_gene = nn.Linear(self.num_features, self.raised_dimension)  # Per i geni
        self.pre_conv_linear_mirna = nn.Linear(1, self.raised_dimension)                 # Per i miRNA

        # Configurazione dei layer convolutivi GAT in base al metodo scelto
        if method == 'gatv2':  # GAT versione 2 (più recente)
            if self.edge_attributes:
                # Con attributi degli archi (2 dimensioni)
                self.conv1 = GATv2Conv(self.raised_dimension, self.hid, heads=self.head, edge_dim=2)
                self.conv2 = GATv2Conv(self.hid * self.head, self.hid, heads=self.head, edge_dim=2)
            elif self.edge_weights:
                # Con pesi degli archi (1 dimensione)
                self.conv1 = GATv2Conv(self.raised_dimension, self.hid, heads=self.head, edge_dim=1)
                self.conv2 = GATv2Conv(self.hid * self.head, self.hid, heads=self.head, edge_dim=1)
            else:
                # Senza informazioni aggiuntive sugli archi
                self.conv1 = GATv2Conv(self.raised_dimension, self.hid, heads=self.head)
                self.conv2 = GATv2Conv(self.hid * self.head, self.hid, heads=self.head)

        elif method == 'gat':  # GAT versione originale
            if self.edge_attributes:
                self.conv1 = GATConv(self.raised_dimension, self.hid, heads=self.head, edge_dim=2)
                self.conv2 = GATConv(self.hid * self.head, self.hid, heads=self.head, edge_dim=2)
            elif self.edge_weights:
                self.conv1 = GATConv(self.raised_dimension, self.hid, heads=self.head, edge_dim=1)
                self.conv2 = GATConv(self.hid * self.head, self.hid, heads=self.head, edge_dim=1)
            else:
                self.conv1 = GATConv(self.raised_dimension, self.hid, heads=self.head)
                self.conv2 = GATConv(self.hid * self.head, self.hid, heads=self.head)

        # Calcolo della dimensione input per i layer lineari finali
        # Tiene conto del pooling applicato ai nodi del grafo
        self.linear_input = math.floor((self.num_gene + self.num_mirna) / self.poolsize) * self.hid * self.head
        print(self.linear_input)

        # Layer lineari per la classificazione finale
        self.linear1 = nn.Linear(self.linear_input, self.linear_input//4)    # Primo layer di riduzione
        self.linear2 = nn.Linear(self.linear_input//4, self.concate_layer)   # Secondo layer di riduzione

        # Configurazione del decoder per la ricostruzione dei dati (se richiesto)
        if self.decoder:
            if self.num_features == 1:
                # Modalità omics: solo espressione, solo miRNA, o espressione+miRNA
                self.decoder_1 = nn.Linear(self.concate_layer, self.concate_layer*2)
                self.decoder_2 = nn.Linear(self.concate_layer*2, self.num_gene+self.num_mirna)
            elif self.num_features == 2:
                # Modalità omics: espressione+CNV, o espressione+CNV+miRNA
                self.decoder_1 = nn.Linear(self.concate_layer, self.concate_layer*2)
                self.decoder_2 = nn.Linear(self.concate_layer*2, self.num_gene*self.num_features + self.num_mirna)


        if self.parallel:

            parallel_input = self.raised_dimension*(self.num_gene+self.num_mirna)

            self.parallel_linear1 = nn.Linear(parallel_input, parallel_input//4)
            self.parallel_linear2 = nn.Linear(parallel_input//4, self.concate_layer)
            self.classifier = nn.Linear(self.concate_layer*2, num_classes)
        else:
            self.classifier = nn.Linear(self.concate_layer, num_classes)

    # Max pooling of size p. Must be a power of 2.
    def graph_max_pool(self, x, p):
        if p > 1:
            x = x.permute(0,2,1).contiguous()  # x = B x F x V
            x = nn.MaxPool1d(p)(x)             # B x F x V/p
            x = x.permute(0,2,1).contiguous()  # x = B x V/p x F
            return x
        else:
            return x
    
    ## create the batch index for each nodes in the batch
    def create_batch_index(self, batches):
        batch_index = []
        for i in range(batches):
            batch_index += [i]*(self.num_gene+self.num_mirna)
        return(torch.Tensor(batch_index).type(torch.int64))
        
    def forward(self, x, edge_index, edge_weight):
        """
        Forward pass della rete GAT
        x: dati di input [batch_size, num_nodes, num_features]
        edge_index: indici degli archi del grafo [2, num_edges]
        edge_weight: pesi degli archi [num_edges]
        """
        batches = x.shape[0]
        num_node = x.shape[1]
        
        # Processamento dei dati in base alla modalità omics
        if self.num_mirna == 0 or self.num_features == 1:
            # Caso semplice: solo espressione genica o solo miRNA
            x = self.pre_conv_linear_gene(x)
            x = F.relu(x)
        else:
            # Caso multimodale: espressione + CNV + miRNA
            x_exp_mirna = x[:,:,0]  # Prima dimensione: espressione + miRNA
            x_cnv = x[:,:,1]        # Seconda dimensione: CNV (Copy Number Variation)

            # Separazione dei miRNA dal resto (ultimi 100 elementi)
            x_cnv = x_cnv[:,:-100]      # CNV dei geni (esclusi miRNA)
            x_exp = x_exp_mirna[:,:-100] # Espressione dei geni (esclusi miRNA)

            # Riorganizzazione dei dati per la concatenazione
            x_cnv = x_cnv.view(batches,-1,1)
            x_exp = x_exp.view(batches,-1,1)
            x_gene = torch.cat([x_exp,x_cnv],dim=1)  # Combina espressione e CNV
            x_gene = x_gene.view(-1,self.num_features)
            x_mirna = x_exp_mirna[:,-100:]  # Estrae i dati dei miRNA

            x_mirna = torch.flatten(x_mirna)
            x_mirna = x_mirna.view(-1, 1)

            # Applicazione delle trasformazioni lineari pre-convolutive
            x_gene = self.pre_conv_linear_gene(x_gene)
            x_gene = F.relu(x_gene)

            x_mirna = self.pre_conv_linear_mirna(x_mirna)
            x_mirna = F.relu(x_mirna)

            # Riorganizzazione per la concatenazione finale
            x_gene = x_gene.view(batches, -1, self.raised_dimension)
            x_mirna = x_mirna.view(batches, -1, self.raised_dimension)

            x = torch.cat([x_gene,x_mirna],dim=1)  # Combina geni e miRNA



        # Preparazione per elaborazione parallela e convolutiva
        x_parallel = x  # Copia per il branch parallelo
        x = x.view(-1, self.raised_dimension)  # Riorganizza per le convoluzioni
        x_parallel = x_parallel.view(batches,-1)  # Appiattisce per il branch parallelo

        # Primo layer convolutivo GAT
        if self.edge_weights:
            x = self.conv1(x, edge_index, edge_weight)  # Con pesi degli archi
            x = F.leaky_relu(x)  # Funzione di attivazione LeakyReLU
        else:
            x = self.conv1(x, edge_index)  # Senza pesi degli archi
            x = F.leaky_relu(x)

        # Secondo layer convolutivo GAT
        if self.edge_weights:
            x = self.conv2(x, edge_index, edge_weight)
            x = F.leaky_relu(x)
        else:
            x = self.conv2(x, edge_index)  # Output: [batches * num_node, hid * head]
            x = F.leaky_relu(x)

        # Pooling per ridurre il numero di nodi nel grafo
        x = x.view(batches, num_node, -1)  # Riorganizza: [batches, num_node, hid * head]
        x = self.graph_max_pool(x, self.poolsize)  # Applica max pooling
        # Output: [batches, floor(num_node / poolsize), hid * head]

        x = x.view(-1, self.hid * self.head)  # Riorganizza per i layer lineari
        # Output: [batches * floor(num_node / poolsize), hid * head]

        # Layer lineari finali per la classificazione
        x = x.view(batches, -1)  # Output: [batches, floor(num_node / poolsize) * hid * head]
        x = self.linear1(x)      # Primo layer di riduzione dimensionale
        x = F.relu(x)
        x = self.linear2(x)      # Secondo layer di riduzione
        x = F.relu(x)

        # Decoder per la ricostruzione dei dati (se abilitato)
        if self.decoder:
            x_reconstruct = x
            x_reconstruct = self.decoder_1(x_reconstruct)  # Primo layer del decoder
            x_reconstruct = F.relu(x_reconstruct)
            
            x_reconstruct = nn.Dropout(self.dropout_rate)(x_reconstruct)  # Dropout per regolarizzazione
            x_reconstruct = self.decoder_2(x_reconstruct)  # Secondo layer del decoder

        # Branch parallelo (se abilitato)
        if self.parallel:
            # Rete neurale fully-connected a due layer che bypassa le convoluzioni
            x_parallel = self.parallel_linear1(x_parallel)
            x_parallel = F.relu(x_parallel)
            
            x_parallel = self.parallel_linear2(x_parallel)
            x_parallel = F.relu(x_parallel)
            
            # Concatenazione dell'output GAT con l'output parallelo
            x = torch.cat((x,x_parallel),1)
        
        # Classificazione finale
        x = F.dropout(x, p=self.dropout_rate, training=self.training)  # Dropout per prevenire overfitting
        x = self.classifier(x)  # Layer di classificazione

        # Ritorna i risultati
        if self.decoder:
            return x_reconstruct, F.log_softmax(x, dim=1)  # Ricostruzione + classificazione
        else:
            return F.log_softmax(x, dim=1)  # Solo classificazione
    
    def loss(self, x_reconstruct, x_target, y, y_target, l2_regularization):
        """
        Calcola la funzione di loss combinata
        x_reconstruct: output del decoder
        x_target: dati target per la ricostruzione
        y: predizioni di classificazione
        y_target: labels vere per la classificazione
        l2_regularization: fattore di regolarizzazione L2
        """
        # Loss di ricostruzione (se il decoder è abilitato)
        if self.decoder:
            if self.num_mirna == 0 or self.num_features == 1:
                # Caso semplice: ricostruzione diretta
                x_target = x_target.view(x_target.size()[0], -1)
                loss1 = nn.MSELoss()(x_reconstruct, x_target)  # Mean Squared Error
            else:
                # Caso multimodale: riorganizza i target per la ricostruzione
                x_target_exp_mirna = x_target[:,:,0]
                x_target_cnv = x_target[:,:,1]

                # Separa miRNA dal resto
                x_target_cnv = x_target_cnv[:,:-100]      # CNV dei geni
                x_target_exp = x_target_exp_mirna[:,:-100] # Espressione dei geni
                x_target_mirna = x_target_exp_mirna[:,-100:] # miRNA
                
                # Concatena tutto in un vettore flat per la ricostruzione
                x_target_flatten = torch.cat([x_target_exp, x_target_cnv, x_target_mirna], dim=1)
                loss1 = nn.MSELoss()(x_reconstruct, x_target_flatten)
        else:
            loss1 = 0  # Nessuna loss di ricostruzione
        
        # Loss di classificazione
        loss2 = nn.CrossEntropyLoss()(y, y_target)
        
        # Combina le due loss
        loss = 1*loss1 + 1*loss2
        
        # Aggiunge regolarizzazione L2 (se abilitata)
        if self.l2:
            l2_loss = 0.0
            for param in self.parameters():
                data = param * param  # Quadrato dei parametri
                l2_loss += data.sum()
            
            loss += 0.2 * l2_regularization * l2_loss  # Penalità L2
        
        return loss


# Definizione della classe GCN (Graph Convolutional Network)
# Simile a GAT ma usa convoluzioni Chebyshev invece di meccanismi di attenzione
class GCN(torch.nn.Module):
    def __init__(self, 
                    method,          # Tipo di metodo ('gcn' per Graph Convolutional Network)
                    parallel,        # Se utilizzare elaborazione parallela
                    l2,              # Se applicare regolarizzazione L2
                    decoder,         # Se includere un decoder per ricostruzione
                    poolsize,        # Dimensione del pooling per ridurre i nodi
                    poolrate,        # Tasso di pooling
                    edge_weights,    # Se utilizzare pesi degli archi
                    edge_attributes, # Se utilizzare attributi degli archi
                    num_gene,        # Numero di geni nel dataset
                    num_mirna,       # Numero di microRNA nel dataset
                    omic_mode,       # Modalità di dati omici
                    num_classes,     # Numero di classi per la classificazione
                    dropout_rate):   # Tasso di dropout

        super(GCN, self).__init__()
        self.omic_mode = omic_mode
        self.method = method
        self.parallel = parallel
        self.decoder = decoder
        self.l2 = l2
        self.poolsize = poolsize
        self.poolrate = poolrate
        self.edge_weights = edge_weights
        self.edge_attributes = edge_attributes
        self.hid = 6
        self.num_gene = num_gene
        self.num_mirna = num_mirna
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.raised_dimension = 8
        self.concate_layer = 64

        if self.omic_mode < 3:
            self.num_features = 1
        else:
            self.num_features = 2

        self.pre_conv_linear_gene = nn.Linear(self.num_features, self.raised_dimension)
        self.pre_conv_linear_mirna = nn.Linear(1, self.raised_dimension)
    
        if method == 'gcn':
            self.conv1 = ChebConv(self.raised_dimension, self.hid, K=5)
            self.conv2 = ChebConv(self.hid, self.hid, K=5)

        if self.poolsize <= 1:
            if method == 'gcn':
                self.linear_input = (self.num_gene + self.num_mirna) * self.hid
        else:
            if method == 'gcn':
                self.linear_input = math.floor((self.num_gene + self.num_mirna) / self.poolsize) * self.hid

        self.linear1 = nn.Linear(self.linear_input, self.linear_input//4)
        self.linear2 = nn.Linear(self.linear_input//4, self.concate_layer)

        if self.decoder:
            if self.num_features == 1:
                ## Omic mode: Exp, mi, Exp+mi
                self.decoder_1 = nn.Linear(self.concate_layer, self.concate_layer*2)
                self.decoder_2 = nn.Linear(self.concate_layer*2, self.num_gene+self.num_mirna)
            elif self.num_features == 2:
                ## omic_mode: Exp+CNV, Exp+CNV+mi
                self.decoder_1 = nn.Linear(self.concate_layer, self.concate_layer*2)
                self.decoder_2 = nn.Linear(self.concate_layer*2, self.num_gene*self.num_features + self.num_mirna)


        # Configurazione del branch parallelo (se richiesto)
        if self.parallel:
            # Il branch parallelo processa i dati senza convoluzioni sul grafo
            parallel_input = self.raised_dimension*(self.num_gene + self.num_mirna)

            self.parallel_linear1 = nn.Linear(parallel_input, parallel_input//4)
            self.parallel_linear2 = nn.Linear(parallel_input//4, self.concate_layer)
            # Classifier che combina output del GAT e del branch parallelo
            self.classifier = nn.Linear(self.concate_layer*2, num_classes)
        else:
            # Classifier che usa solo l'output del GAT
            self.classifier = nn.Linear(self.concate_layer, num_classes)

    # Funzione di Max pooling per ridurre il numero di nodi nel grafo
    # La dimensione p deve essere una potenza di 2
    def graph_max_pool(self, x, p):
        if p > 1:
            x = x.permute(0,2,1).contiguous()  # Riordina: Batch x Features x Vertices
            x = nn.MaxPool1d(p)(x)             # Applica pooling: Batch x Features x (Vertices/p)
            x = x.permute(0,2,1).contiguous()  # Riordina: Batch x (Vertices/p) x Features
            return x
        else:
            return x
    
    # Crea un indice di batch per ogni nodo nel batch
    # Serve per identificare a quale campione appartiene ogni nodo
    def create_batch_index(self, batches):
        batch_index = []
        for i in range(batches):
            # Ogni batch ha (num_gene + num_mirna) nodi
            batch_index += [i]*(self.num_gene + self.num_mirna)
        return(torch.Tensor(batch_index).type(torch.int64))
        
    def forward(self, x, edge_index, edge_weight):
        batches = x.shape[0]
        num_node = x.shape[1]
        
        if self.num_mirna == 0 or self.num_features == 1:
            x = self.pre_conv_linear_gene(x)
            x = F.relu(x)
        else:
            ## the second matrix cnv_data has padding
            x_exp_mirna = x[:,:,0]
            x_cnv = x[:,:,1]

            ## separate mirna from the rest
            x_cnv = x_cnv[:,:-100]
            x_exp = x_exp_mirna[:,:-100]

            x_cnv = x_cnv.view(batches,-1,1)
            x_exp = x_exp.view(batches,-1,1)
            x_gene = torch.cat([x_exp,x_cnv],dim=1)
            x_gene = x_gene.view(-1,self.num_features)
            x_mirna = x_exp_mirna[:,-100:]
            x_mirna = torch.flatten(x_mirna)
            x_mirna = x_mirna.view(-1, 1)

            x_gene = self.pre_conv_linear_gene(x_gene)
            x_gene = F.relu(x_gene)

            x_mirna = self.pre_conv_linear_mirna(x_mirna)
            x_mirna = F.relu(x_mirna)

            x_gene = x_gene.view(batches, -1, self.raised_dimension)
            x_mirna = x_mirna.view(batches, -1, self.raised_dimension)

            x = torch.cat([x_gene,x_mirna],dim=1)



        x_parallel = x
        x = x.view(-1, self.raised_dimension)
        x_parallel = x_parallel.view(batches,-1)

        if self.edge_weights:
            x = self.conv1(x, edge_index, edge_weight)

            x = F.relu(x)
        else:
            x = self.conv1(x, edge_index)

            x = F.relu(x)

        if self.edge_weights:
            x = self.conv2(x, edge_index, edge_weight)

            x = F.relu(x)
        else:
            x = self.conv2(x, edge_index) ## output shape: [batches * num_node, hid * head]

            x = F.relu(x)

        ## pooling on the graph to reduce nodes
        x = x.view(batches, num_node, -1) ## output shape: [batches, num_node, hid * head]
        x = self.graph_max_pool(x, self.poolsize)   ## if "gat", then output shape: [batches, floor(num_node / poolsize), hid * head]
                                                    ## if "gcn", then output shape: [batches, floor(num_node / poolsize), hid]

        if self.method == 'gcn':
            x = x.view(-1, self.hid) ## output shape:[batches * floor(num_node / poolsize), hid]

        x = x.view(batches, -1) ## output size: [batches, floor(num_node / poolsize) * hid * head]
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)

        if self.decoder:
            x_reconstruct = x
            x_reconstruct = self.decoder_1(x_reconstruct)
            x_reconstruct = F.relu(x_reconstruct)

            x_reconstruct  = nn.Dropout(0.2)(x_reconstruct)
            x_reconstruct = self.decoder_2(x_reconstruct)

        if self.parallel:
            ## the two layer shallow FC network
            x_parallel = self.parallel_linear1(x_parallel)
            x_parallel = F.relu(x_parallel)
            x_parallel = self.parallel_linear2(x_parallel)
            x_parallel = F.relu(x_parallel)

            x = torch.cat((x,x_parallel),1)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.classifier(x)

        if self.decoder:
            return x_reconstruct, F.log_softmax(x, dim=1)
        else:
            return F.log_softmax(x, dim=1)
    
    def loss(self, x_reconstruct, x_target, y, y_target, l2_regularization):
        if self.decoder:
            if self.num_mirna == 0 or self.num_features == 1:
                x_target = x_target.view(x_target.size()[0], -1)
                loss1 = nn.MSELoss()(x_reconstruct, x_target)
            else:
                x_target_exp_mirna = x_target[:,:,0]
                x_target_cnv = x_target[:,:,1]

                ## separate mirna from the rest
                x_target_cnv = x_target_cnv[:,:-100]
                x_target_exp = x_target_exp_mirna[:,:-100]
                x_target_mirna = x_target_exp_mirna[:,-100:]
                x_target_flatten = torch.cat([x_target_exp, x_target_cnv, x_target_mirna], dim=1)
                loss1 = nn.MSELoss()(x_reconstruct, x_target_flatten)
        else:
            loss1 = 0
        
        loss2 = nn.CrossEntropyLoss()(y, y_target)
        loss = 1*loss1 + 1*loss2
        
        if self.l2:
            l2_loss = 0.0
            for param in self.parameters():
                data = param* param
                l2_loss += data.sum()

            loss += 0.2* l2_regularization* l2_loss
        return loss


# Definizione della classe Baseline
# Modello di riferimento che usa solo layer fully-connected senza convoluzioni sul grafo
class Baseline(torch.nn.Module):
    def __init__(self, 
                    method,          # Tipo di metodo (non utilizzato nel baseline)
                    parallel,        # Se utilizzare elaborazione parallela (non utilizzato)
                    l2,              # Se applicare regolarizzazione L2
                    decoder,         # Se includere un decoder (non utilizzato nel baseline)
                    poolsize,        # Dimensione del pooling (non utilizzato)
                    poolrate,        # Tasso di pooling (non utilizzato)
                    edge_weights,    # Se utilizzare pesi degli archi (non utilizzato)
                    edge_attributes, # Se utilizzare attributi degli archi (non utilizzato)
                    num_gene,        # Numero di geni nel dataset
                    num_mirna,       # Numero di microRNA nel dataset
                    omic_mode,       # Modalità di dati omici
                    num_classes,     # Numero di classi per la classificazione
                    dropout_rate):   # Tasso di dropout

        super(Baseline, self).__init__()
        
        # Salvataggio dei parametri (molti non sono utilizzati nel baseline)
        self.omic_mode = omic_mode
        self.method = method
        self.parallel = parallel
        self.decoder = decoder
        self.l2 = l2
        self.poolsize = poolsize
        self.poolrate = poolrate
        self.edge_weights = edge_weights
        self.edge_attributes = edge_attributes
        
        # Iperparametri per il modello baseline
        self.hid = 6
        self.num_gene = num_gene
        self.num_mirna = num_mirna
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.raised_dimension = 8       # Dimensione dopo trasformazione iniziale
        self.concate_layer = 64         # Dimensione finale prima della classificazione

        # Determina il numero di features come negli altri modelli
        if self.omic_mode < 3:
            self.num_features = 1       # Solo espressione o solo miRNA
        else:
            self.num_features = 2       # Espressione + CNV

        # Layer di preprocessing (come negli altri modelli)
        self.pre_conv_linear_gene = nn.Linear(self.num_features, self.raised_dimension)
        self.pre_conv_linear_mirna = nn.Linear(1, self.raised_dimension)

        # Rete fully-connected a tre layer (senza convoluzioni sul grafo)
        parallel_input = self.raised_dimension*(self.num_gene + self.num_mirna)

        self.parallel_linear1 = nn.Linear(parallel_input, parallel_input//2)     # Primo layer di riduzione
        self.parallel_linear2 = nn.Linear(parallel_input//2, parallel_input//4) # Secondo layer di riduzione
        self.parallel_linear3 = nn.Linear(parallel_input//4, self.concate_layer) # Terzo layer di riduzione
        self.classifier = nn.Linear(self.concate_layer, num_classes)             # Classificatore finale
    
    ## create the batch index for each nodes in the batch
    def create_batch_index(self, batches):
        batch_index = []
        for i in range(batches):
            batch_index += [i]*(self.num_gene + self.num_mirna)
        return(torch.Tensor(batch_index).type(torch.int64))
        
    def forward(self, x, edge_index, edge_weight):
        """
        Forward pass del modello Baseline
        Nota: edge_index e edge_weight non sono utilizzati perché non ci sono convoluzioni sul grafo
        """
        batches = x.shape[0]
        num_node = x.shape[1]
        
        # Preprocessing dei dati (identico agli altri modelli)
        if self.num_mirna == 0 or self.num_features == 1:
            # Caso semplice: solo un tipo di dato omico
            x = self.pre_conv_linear_gene(x)
            x = F.relu(x)
        else:
            # Caso multimodale: gestisce espressione + CNV + miRNA
            x_exp_mirna = x[:,:,0]  # Prima dimensione
            x_cnv = x[:,:,1]        # Seconda dimensione

            # Separazione dei miRNA (ultimi 100 elementi)
            x_cnv = x_cnv[:,:-100]      # CNV dei geni
            x_exp = x_exp_mirna[:,:-100] # Espressione dei geni

            # Riorganizzazione e concatenazione
            x_cnv = x_cnv.view(batches,-1,1)
            x_exp = x_exp.view(batches,-1,1)
            x_gene = torch.cat([x_exp,x_cnv],dim=1)
            x_gene = x_gene.view(-1,self.num_features)
            x_mirna = x_exp_mirna[:,-100:]
            x_mirna = torch.flatten(x_mirna)
            x_mirna = x_mirna.view(-1, 1)

            # Trasformazioni lineari
            x_gene = self.pre_conv_linear_gene(x_gene)
            x_gene = F.relu(x_gene)

            x_mirna = self.pre_conv_linear_mirna(x_mirna)
            x_mirna = F.relu(x_mirna)

            # Riorganizzazione finale
            x_gene = x_gene.view(batches, -1, self.raised_dimension)
            x_mirna = x_mirna.view(batches, -1, self.raised_dimension)

            x = torch.cat([x_gene,x_mirna],dim=1)

        # Elaborazione con rete fully-connected (senza convoluzioni sul grafo)
        x_parallel = x
        x_parallel = x_parallel.view(batches,-1)  # Appiattisce i dati
        
        # Passaggio attraverso i tre layer fully-connected
        x_parallel = self.parallel_linear1(x_parallel)
        x_parallel = F.relu(x_parallel)
        x_parallel = self.parallel_linear2(x_parallel)
        x_parallel = F.relu(x_parallel)
        x_parallel = self.parallel_linear3(x_parallel)
        x_parallel = F.relu(x_parallel)

        # Classificazione finale
        x_parallel = F.dropout(x_parallel, p=self.dropout_rate, training=self.training)
        x_parallel = self.classifier(x_parallel)
        return F.log_softmax(x_parallel, dim=1)  # Output con log-softmax
    
    def loss(self, x_reconstruct, x_target, y, y_target, l2_regularization):
        """
        Calcola la funzione di loss per il modello Baseline
        Usa solo la loss di classificazione (senza ricostruzione)
        """
        # Solo loss di classificazione (il baseline non ha decoder)
        loss2 = nn.CrossEntropyLoss()(y, y_target)
        loss = 1*loss2
        
        return loss
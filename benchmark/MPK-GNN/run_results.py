'''
    @Project: MPK-GNN
    @File   : run_results.py
    @Author : Shunxin Xiao
    @Email  : xiaoshunxin.tj@gmail
    @Desc   : Script principale per l'esecuzione di MPK-GNN
    
    Questo file è il punto di ingresso principale del progetto MPK-GNN.
    Si occupa di:
    - Configurare i parametri del modello tramite argparse
    - Caricare e preprocessare i dati multi-omici (expression + CNV)
    - Istanziare il modello MPK-GNN
    - Eseguire il training con supervised contrastive learning
    - Valutare le performance sui dati di test
    - Salvare i risultati in file di testo
    
    Il sistema supporta esperimenti ripetuti per validazione statistica
    e utilizza tre diverse matrici di adiacenza per catturare diversi
    tipi di conoscenza biologica a priori.
'''
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

# Importa il modello principale MPK-GNN
from models.mpk import MPK
# Utilities per inizializzazione pesi, loss function e operazioni di training/testing
from utils.ops_al import weight_init
from utils.ops_loss import SupConLoss
from utils.ops_tt import adjust_learning_rate, test_model
# Utilities per caricamento e processamento dati
from utils.ops_io import process_data, load_processed_data
# Utilities per valutazione delle performance
from utils.ops_ev import accuracy, get_classification_results


if __name__ == '__main__':
    # Configurazione dei parametri tramite argparse
    parser = argparse.ArgumentParser()
    
    # === PARAMETRI DI BASE ===
    parser.add_argument('--seed', type=int, default=1009, help='Seed per la riproducibilità degli esperimenti')
    parser.add_argument('--n_repeated', type=int, default=10, help='Numero di esperimenti ripetuti per validazione statistica')
    parser.add_argument('--cuda', action='store_true', default=True, help='Abilita training su GPU')
    parser.add_argument('--cuda_device', type=str, default='3', help='Numero del dispositivo CUDA da utilizzare')
    
    # === PARAMETRI PER IL CARICAMENTO DATI ===
    parser.add_argument('--load_saved', action='store_true', default=True, help='Se caricare i dati già preprocessati')
    parser.add_argument('--dataset_name', type=str, default='1000', help='Nome del dataset (numero di geni)')
    parser.add_argument('--ratio', type=float, default=0.1, help='Percentuale di campioni per il training')
    parser.add_argument('--batch_size', type=int, default=128, help='Dimensione del batch per il training')
    
    # === PARAMETRI ARCHITETTURA DI RETE ===
    parser.add_argument('--num_omics', type=int, default=2, help='Numero di tipi di dati omici (expression + CNV)')
    # Parametri per Sample Learning Module (SLM)
    parser.add_argument('--slm_dim_1', type=int, default=1024, help='Dimensione output primo layer SLM')
    parser.add_argument('--slm_dim_2', type=int, default=256, help='Dimensione output secondo layer SLM')
    # Parametri per Feature Learning Module (FLM)
    parser.add_argument('--flm_gcn_dim_1', type=int, default=64, help='Dimensione output primo layer GCN in FLM')
    parser.add_argument('--flm_gcn_dim_2', type=int, default=8, help='Dimensione output secondo layer GCN in FLM')
    parser.add_argument('--pool_size', type=int, default=8, help='Dimensione pooling layer (deve essere potenza di 2)')
    parser.add_argument('--flm_fl_dim', type=int, default=1024, help='Dimensione flatten layer in FLM')
    # Parametri per Projection Module (PM)
    parser.add_argument('--pm_dim_1', type=int, default=32, help='Prima dimensione output Projection Module')
    parser.add_argument('--pm_dim_2', type=int, default=1024, help='Seconda dimensione output Projection Module')
    
    # === PARAMETRI DI TRAINING ===
    parser.add_argument('--lr', type=float, default=0.05, help='Learning rate iniziale')
    parser.add_argument('--decay', type=float, default=0.95, help='Fattore di decay per learning rate')
    parser.add_argument('--num_epochs', type=int, default=30, help='Numero di epoche di training')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Valore di regolarizzazione L2')
    parser.add_argument('--lambda_1', type=float, default=0.5, help='Peso della contrastive loss')
    parser.add_argument('--temperature', type=float, default=0.2, help='Parametro temperatura per contrastive learning')
    args = parser.parse_args()

    # Liste per raccogliere i risultati di tutti gli esperimenti ripetuti
    all_ACC = []      # Accuracy
    all_MaP = []      # Macro Precision
    all_MaR = []      # Macro Recall
    all_MaF = []      # Macro F1-score
    all_Time = []     # Tempi di training

    # === LOOP PRINCIPALE PER ESPERIMENTI RIPETUTI ===
    for i in range(args.n_repeated):
        # Prima iterazione: può processare dati raw o caricare preprocessati
        if i == 0:
            # args.load_saved = False  # Decommentare per processare dati raw
            args.load_saved = True
        else:
            args.load_saved = True  # Successive iterazioni usano sempre dati preprocessati

        # === CARICAMENTO DATI ===
        args.num_genes = int(args.dataset_name)
        if args.load_saved:
            # Carica dati già preprocessati
            train_data, train_labels, val_data, val_labels, test_data, test_labels, L_1, L_2, L_3, num_classes = \
                load_processed_data(num_genes=args.num_genes, ratio=args.ratio)
        else:
            # Processa dati raw (più lento)
            train_data, train_labels, val_data, val_labels, test_data, test_labels, L_1, L_2, L_3, num_classes = \
                process_data(num_genes=args.num_genes, ratio=args.ratio)

        # === CREAZIONE DATA LOADERS ===
        # Dataset e DataLoader per training (con shuffle)
        dset_train = Data.TensorDataset(train_data, train_labels)
        train_loader = Data.DataLoader(dset_train, batch_size=args.batch_size, shuffle=True)
        
        # Dataset e DataLoader per validation (senza shuffle)
        dset_val = Data.TensorDataset(val_data, val_labels)
        val_loader = Data.DataLoader(dset_val, shuffle=False)
        
        # Dataset e DataLoader per testing (senza shuffle)
        dset_test = Data.TensorDataset(test_data, test_labels)
        test_loader = Data.DataLoader(dset_test, shuffle=False)

        # === ISTANZIAZIONE MODELLO E OTTIMIZZATORE ===
        # Crea il modello MPK-GNN con tutti i parametri configurati
        net = MPK(num_omics=args.num_omics, num_genes=args.num_genes, num_classes=num_classes, 
                  slm_dim_1=args.slm_dim_1, slm_dim_2=args.slm_dim_2, 
                  flm_gcn_dim_1=args.flm_gcn_dim_1, flm_gcn_dim_2=args.flm_gcn_dim_2,
                  pool_size=args.pool_size, flm_fl_dim=args.flm_fl_dim, 
                  pm_dim_1=args.pm_dim_1, pm_dim_2=args.pm_dim_2)
        
        # Inizializza i pesi del modello con Xavier initialization
        net.apply(weight_init)
        
        # Configura l'ottimizzatore SGD con momentum
        optimizer = optim.SGD(net.parameters(), momentum=0.9, lr=args.lr, weight_decay=args.weight_decay)
        
        # Configura il dispositivo di calcolo (GPU/CPU)
        device = torch.device("cuda:" + args.cuda_device if args.cuda else "cpu")
        
        # Crea la funzione di loss per contrastive learning
        criterion = SupConLoss(temperature=args.temperature, cuda_device=args.cuda_device)

        # Sposta modello e matrici di adiacenza su GPU/CPU
        net = net.to(device)
        L_1 = L_1.to(device)  # Prima matrice di adiacenza (es. BioGRID)
        L_2 = L_2.to(device)  # Seconda matrice di adiacenza (es. co-expression)
        L_3 = L_3.to(device)  # Terza matrice di adiacenza (es. STRING)

        # === INIZIO TRAINING ===
        t_total_train = time.time()  # Cronometro per tempo totale di training
        cur_lr = args.lr             # Learning rate corrente
        global_step = 0              # Contatore globale dei passi di training
        train_size = train_data.shape[0]  # Numero di campioni di training

        # Loop principale di training per le epoche
        for epoch in range(args.num_epochs):
            net.train()  # Imposta il modello in modalità training
            
            # Aggiorna il learning rate con decay
            cur_lr = adjust_learning_rate(optimizer, cur_lr, args.decay, global_step, train_size)
            
            t_start = time.time()  # Cronometro per l'epoca corrente
            
            # Variabili per statistiche dell'epoca
            epoch_loss = 0.0
            epoch_acc = 0.0
            count = 0
            
            # Loop sui batch di training
            for i, (batch_x, batch_y) in enumerate(train_loader):
                # Sposta batch su dispositivo appropriato
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                # Azzera i gradienti
                optimizer.zero_grad()
                
                # Forward pass: ottieni output del modello e rappresentazioni proiettate
                output, r_x_0, r_x_1, r_x_2 = net(batch_x, L_1, L_2, L_3)

                # Calcola loss di classificazione (CrossEntropy)
                loss = nn.CrossEntropyLoss()(output, batch_y)
                
                # Aggiunge contrastive loss pesata con lambda_1
                # Combina le tre rappresentazioni proiettate per contrastive learning
                loss += args.lambda_1 * criterion(
                    torch.cat([r_x_0.unsqueeze(1), r_x_1.unsqueeze(1), r_x_2.unsqueeze(1)], dim=1), batch_y)

                # Calcola accuracy del batch
                acc_batch = accuracy(output, batch_y).item()

                # Backward pass e ottimizzazione
                loss.backward()
                optimizer.step()

                # Aggiorna statistiche
                count += 1
                epoch_loss += loss.item()
                epoch_acc += acc_batch
                global_step += args.batch_size

            # Calcola statistiche medie dell'epoca
            epoch_loss /= count
            epoch_acc /= count
            t_stop = time.time() - t_start
            
            # Stampa progresso dell'epoca
            print('epoch= %d, loss(train)= %.3f, accuracy(train)= %.3f, time= %.3f, lr= %.5f' %
                  (epoch + 1, epoch_loss, epoch_acc, t_stop, cur_lr))
        # Calcola tempo totale di training
        t_total_train = time.time() - t_total_train
        print("Total training time: ", t_total_train)

        # === FASE DI TESTING ===
        t_start_test = time.time()
        
        # Valuta il modello sui dati di test
        # Restituisce: accuracy, matrice di confusione, predizioni e etichette predette
        test_acc, confusionGCN, predictions, preds_labels = test_model(net, test_loader, device, L_1, L_2, L_3,
                                                                       num_classes)

        # Calcola metriche di classificazione dettagliate
        ACC, MACRO_P, MACRO_R, MACRO_F1, MICRO_F1 = get_classification_results(test_labels, preds_labels)
        
        # Aggiungi risultati alle liste per calcolo statistiche finali
        all_ACC.append(ACC)
        all_MaP.append(MACRO_P)
        all_MaR.append(MACRO_R)
        all_MaF.append(MACRO_F1)
        all_Time.append(t_total_train)

    # === SALVATAGGIO RISULTATI FINALI ===
    # Salva statistiche medie e deviazioni standard di tutti gli esperimenti
    # fp = open("results.txt", "a+", encoding="utf-8")  # File generico
    fp = open(str(args.num_genes) + ".txt", "a+", encoding="utf-8")  # File specifico per numero geni
    fp.write("Ratio: {}\n".format(args.ratio))
    fp.write("ACC: {:.2f}\t{:.2f}\n".format(np.mean(all_ACC) * 100, np.std(all_ACC) * 100))
    fp.write("MaP: {:.2f}\t{:.2f}\n".format(np.mean(all_MaP) * 100, np.std(all_MaP) * 100))
    fp.write("MaR: {:.2f}\t{:.2f}\n".format(np.mean(all_MaR) * 100, np.std(all_MaR) * 100))
    fp.write("MaF: {:.2f}\t{:.2f}\n\n".format(np.mean(all_MaF) * 100, np.std(all_MaF) * 100))
    fp.write("Train Time: {:.2f}\t{:.2f}\n\n".format(np.mean(all_Time), np.std(all_Time)))
    fp.close()

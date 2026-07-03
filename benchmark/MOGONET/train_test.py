"""
Training and testing of the MOGONET model.

Questo modulo implementa le funzioni principali per:
1. Preparazione dei dati multi-omici (mRNA, miRNA, metilazione)
2. Generazione delle matrici di adiacenza per i grafi
3. Training dei modelli (pre-training + training principale)
4. Testing e valutazione delle performance

MOGONET utilizza Graph Convolutional Networks (GCN) per ogni vista omica
e un Cross-View Decoder Network (VCDN) per l'integrazione multi-vista.
"""

# Importazione delle librerie standard e di machine learning
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Importazione di PyTorch per deep learning
import torch
import torch.nn.functional as F

# Importazione dei moduli personalizzati
from models import init_model_dict, init_optim
from utils import one_hot_tensor, cal_sample_weight, gen_adj_mat_tensor, gen_test_adj_mat_tensor, cal_adj_mat_parameter, save_model_dict

# Controllo automatico della disponibilità di GPU CUDA
cuda = True if torch.cuda.is_available() else False


def prepare_trte_data(data_folder, view_list):
    """
    Prepara e carica i dati multi-omici per training e testing.
    
    Questa funzione è fondamentale per MOGONET in quanto:
    1. Carica i dati delle diverse viste omiche (mRNA, miRNA, metilazione)
    2. Combina i set di training e test per creare grafi completi
    3. Converte i dati in tensori PyTorch compatibili con GPU
    
    Args:
        data_folder (str): Cartella contenente i dati (es. 'BRCA', 'ROSMAP')
        view_list (list): Lista delle viste da caricare (es. [1,2,3])
        
    Returns:
        data_train_list (list): Dati di training per ogni vista
        data_all_list (list): Dati completi (training + test) per ogni vista
        idx_dict (dict): Indici per distinguere campioni di training e test
        labels (array): Labels complete (training + test)
    """
    # Numero di viste omiche da analizzare
    num_view = len(view_list)
    
    # =============================================================================
    # CARICAMENTO DELLE LABELS
    # =============================================================================
    
    # Caricamento delle etichette di training e test
    labels_tr = np.loadtxt(os.path.join(data_folder, "labels_tr.csv"), delimiter=',')
    labels_te = np.loadtxt(os.path.join(data_folder, "labels_te.csv"), delimiter=',')
    
    # Conversione a interi (classi discrete)
    labels_tr = labels_tr.astype(int)
    labels_te = labels_te.astype(int)
    
    # =============================================================================
    # CARICAMENTO DEI DATI MULTI-OMICI
    # =============================================================================
    
    # Liste per contenere i dati di ogni vista omica
    data_tr_list = []  # Dati di training
    data_te_list = []  # Dati di test
    
    # Caricamento dei dati per ogni vista (1=mRNA, 2=miRNA, 3=metilazione)
    for i in view_list:
        data_tr_list.append(np.loadtxt(os.path.join(data_folder, str(i)+"_tr.csv"), delimiter=','))
        data_te_list.append(np.loadtxt(os.path.join(data_folder, str(i)+"_te.csv"), delimiter=','))
    
    # Calcolo delle dimensioni dei dataset
    num_tr = data_tr_list[0].shape[0]  # Numero di campioni di training
    num_te = data_te_list[0].shape[0]  # Numero di campioni di test
    
    # =============================================================================
    # COMBINAZIONE DEI DATI PER LA COSTRUZIONE DEI GRAFI
    # =============================================================================
    
    # Concatenazione verticale di training e test per ogni vista
    # Questo è necessario per costruire grafi che includano tutti i campioni
    data_mat_list = []
    for i in range(num_view):
        data_mat_list.append(np.concatenate((data_tr_list[i], data_te_list[i]), axis=0))
    
    # =============================================================================
    # CONVERSIONE A TENSORI PYTORCH E TRASFERIMENTO SU GPU
    # =============================================================================
    
    # Conversione a tensori PyTorch Float (necessario per i calcoli di deep learning)
    data_tensor_list = []
    for i in range(len(data_mat_list)):
        data_tensor_list.append(torch.FloatTensor(data_mat_list[i]))
        # Trasferimento su GPU se disponibile (accelerazione computazionale)
        if cuda:
            data_tensor_list[i] = data_tensor_list[i].cuda()
    
    # =============================================================================
    # CREAZIONE DEGLI INDICI PER TRAINING/TEST SPLIT
    # =============================================================================
    
    # Dizionario con gli indici per identificare campioni di training e test
    idx_dict = {}
    idx_dict["tr"] = list(range(num_tr))                    # Indici 0 to num_tr-1
    idx_dict["te"] = list(range(num_tr, (num_tr+num_te)))   # Indici num_tr to num_tr+num_te-1
    
    # =============================================================================
    # PREPARAZIONE DEI DATASET FINALI
    # =============================================================================
    
    # Estrazione dei dati separati per training e completi
    data_train_list = []  # Solo dati di training
    data_all_list = []    # Dati completi (training + test)
    
    for i in range(len(data_tensor_list)):
        # Estrazione dei soli dati di training
        data_train_list.append(data_tensor_list[i][idx_dict["tr"]].clone())
        
        # Creazione del dataset completo (necessario per i grafi)
        data_all_list.append(torch.cat((data_tensor_list[i][idx_dict["tr"]].clone(),
                                       data_tensor_list[i][idx_dict["te"]].clone()),0))
    
    # Combinazione delle labels complete
    labels = np.concatenate((labels_tr, labels_te))
    
    return data_train_list, data_all_list, idx_dict, labels


def gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter):
    """
    Genera le matrici di adiacenza per i grafi di training e test.
    
    Le matrici di adiacenza definiscono le connessioni tra i campioni nei grafi.
    MOGONET utilizza la similarità coseno per determinare quali campioni
    sono "vicini" nello spazio delle features.
    
    Args:
        data_tr_list (list): Dati di training per ogni vista
        data_trte_list (list): Dati completi (training + test) per ogni vista
        trte_idx (dict): Indici per distinguere training e test
        adj_parameter (float): Parametro per il controllo della sparsità del grafo
        
    Returns:
        adj_train_list (list): Matrici di adiacenza per il training
        adj_test_list (list): Matrici di adiacenza per il test
    """
    # Metrica utilizzata per calcolare la similarità tra campioni
    adj_metric = "cosine"  # Distanza coseno (misura angolare tra vettori)
    
    # Liste per le matrici di adiacenza
    adj_train_list = []
    adj_test_list = []
    
    # Generazione delle matrici per ogni vista omica
    for i in range(len(data_tr_list)):
        # Calcolo del parametro adattivo basato sui dati di training
        # Questo parametro controlla quanto "denso" sarà il grafo
        adj_parameter_adaptive = cal_adj_mat_parameter(adj_parameter, data_tr_list[i], adj_metric)
        
        # Generazione della matrice di adiacenza per il training
        # Connette solo i campioni di training tra loro
        adj_train_list.append(gen_adj_mat_tensor(data_tr_list[i], adj_parameter_adaptive, adj_metric))
        
        # Generazione della matrice di adiacenza per il test
        # Include connessioni tra tutti i campioni (training + test)
        adj_test_list.append(gen_test_adj_mat_tensor(data_trte_list[i], trte_idx, adj_parameter_adaptive, adj_metric))
    
    return adj_train_list, adj_test_list


def train_epoch(data_list, adj_list, label, one_hot_label, sample_weight, model_dict, optim_dict, train_VCDN=True):
    """
    Esegue una singola epoca di training per MOGONET.
    
    Questa funzione implementa la strategia di training in due fasi:
    1. Training individuale dei classificatori specifici per ogni vista (C1, C2, C3)
    2. Training del Cross-View Decoder Network (VCDN) per l'integrazione multi-vista
    
    Args:
        data_list (list): Dati di training per ogni vista omica
        adj_list (list): Matrici di adiacenza per ogni vista
        label (tensor): Labels vere per il training
        one_hot_label (tensor): Labels in formato one-hot encoding
        sample_weight (tensor): Pesi per bilanciare le classi sbilanciate
        model_dict (dict): Dizionario contenente tutti i modelli (E1,E2,E3,C1,C2,C3,C)
        optim_dict (dict): Dizionario contenente gli ottimizzatori
        train_VCDN (bool): Se True, addestra anche il VCDN (default: True)
        
    Returns:
        loss_dict (dict): Dizionario con le loss per ogni componente
    """
    # Dizionario per raccogliere le loss di ogni componente
    loss_dict = {}
    
    # Funzione di loss: Cross-Entropy con riduzione None per applicare i pesi
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    
    # Impostazione di tutti i modelli in modalità training
    for m in model_dict:
        model_dict[m].train()    
    
    # Numero di viste omiche
    num_view = len(data_list)
    
    # =============================================================================
    # FASE 1: TRAINING DEI CLASSIFICATORI SPECIFICI PER VISTA
    # =============================================================================
    
    # Training separato per ogni vista omica (mRNA, miRNA, metilazione)
    for i in range(num_view):
        # Reset dei gradienti per il classificatore della vista i-esima
        optim_dict["C{:}".format(i+1)].zero_grad()
        
        # Forward pass:
        # 1. Encoder E_i processa i dati attraverso la GCN
        # 2. Classificatore C_i produce le predizioni per la vista i-esima
        ci = model_dict["C{:}".format(i+1)](
            model_dict["E{:}".format(i+1)](data_list[i], adj_list[i])
        )
        
        # Calcolo della loss pesata per bilanciare le classi
        ci_loss = torch.mean(torch.mul(criterion(ci, label), sample_weight))
        
        # Backward pass e aggiornamento dei parametri
        ci_loss.backward()
        optim_dict["C{:}".format(i+1)].step()
        
        # Salvataggio della loss per monitoraggio
        loss_dict["C{:}".format(i+1)] = ci_loss.detach().cpu().numpy().item()
    
    # =============================================================================
    # FASE 2: TRAINING DEL CROSS-VIEW DECODER NETWORK (VCDN)
    # =============================================================================
    
    # Il VCDN viene addestrato solo se abbiamo almeno 2 viste e train_VCDN=True
    if train_VCDN and num_view >= 2:
        # Reset dei gradienti per il classificatore globale
        optim_dict["C"].zero_grad()
        
        # Raccolta delle rappresentazioni da tutti gli encoder
        ci_list = []
        for i in range(num_view):
            # Estrazione delle rappresentazioni latenti da ogni vista
            ci_list.append(
                model_dict["C{:}".format(i+1)](
                    model_dict["E{:}".format(i+1)](data_list[i], adj_list[i])
                )
            )
        
        # Il VCDN combina le rappresentazioni di tutte le viste
        c = model_dict["C"](ci_list)    
        
        # Calcolo della loss per il classificatore globale
        c_loss = torch.mean(torch.mul(criterion(c, label), sample_weight))
        
        # Backward pass e aggiornamento
        c_loss.backward()
        optim_dict["C"].step()
        
        # Salvataggio della loss
        loss_dict["C"] = c_loss.detach().cpu().numpy().item()
    
    return loss_dict
    

def test_epoch(data_list, adj_list, te_idx, model_dict):
    """
    Esegue la valutazione del modello sui dati di test.
    
    Durante il test, il modello utilizza tutte le informazioni disponibili
    (training + test) per fare predizioni sui campioni di test, sfruttando
    la struttura del grafo per propagare l'informazione.
    
    Args:
        data_list (list): Dati completi (training + test) per ogni vista
        adj_list (list): Matrici di adiacenza complete per ogni vista
        te_idx (list): Indici dei campioni di test
        model_dict (dict): Dizionario contenente tutti i modelli addestrati
        
    Returns:
        prob (array): Probabilità predette per ogni classe sui campioni di test
    """
    # Impostazione di tutti i modelli in modalità evaluation (disabilita dropout, etc.)
    for m in model_dict:
        model_dict[m].eval()
    
    # Numero di viste omiche
    num_view = len(data_list)
    
    # =============================================================================
    # FORWARD PASS PER OGNI VISTA
    # =============================================================================
    
    # Lista per raccogliere le rappresentazioni da ogni vista
    ci_list = []
    
    # Estrazione delle rappresentazioni da ogni encoder-classificatore
    for i in range(num_view):
        # Forward pass attraverso encoder + classificatore per la vista i-esima
        ci_list.append(
            model_dict["C{:}".format(i+1)](
                model_dict["E{:}".format(i+1)](data_list[i], adj_list[i])
            )
        )
    
    # =============================================================================
    # INTEGRAZIONE MULTI-VISTA E PREDIZIONE FINALE
    # =============================================================================
    
    # Se abbiamo multiple viste, usa il VCDN per l'integrazione
    if num_view >= 2:
        c = model_dict["C"](ci_list)    # Cross-View Decoder Network
    else:
        # Se abbiamo una sola vista, usa direttamente il suo output
        c = ci_list[0]
    
    # Estrazione delle predizioni solo per i campioni di test
    c = c[te_idx, :]
    
    # Conversione a probabilità usando softmax e trasferimento su CPU
    prob = F.softmax(c, dim=1).data.cpu().numpy()
    
    return prob


def train_test(data_folder, view_list, num_class,
               lr_e_pretrain, lr_e, lr_c, 
               num_epoch_pretrain, num_epoch):
    """
    Funzione principale per il training e testing completo di MOGONET.
    
    Implementa la strategia di training in due fasi:
    1. Pre-training: Training separato degli encoder per ogni vista
    2. Training principale: Training congiunto di tutti i componenti
    
    Args:
        data_folder (str): Cartella contenente i dati ('BRCA' o 'ROSMAP')
        view_list (list): Lista delle viste da utilizzare (es. [1,2,3])
        num_class (int): Numero di classi per la classificazione
        lr_e_pretrain (float): Learning rate per il pre-training degli encoder
        lr_e (float): Learning rate per gli encoder nel training principale
        lr_c (float): Learning rate per i classificatori
        num_epoch_pretrain (int): Numero di epoche per il pre-training
        num_epoch (int): Numero di epoche per il training principale
    """
    # Intervallo per la valutazione durante il training
    test_inverval = 50
    
    # Numero di viste e calcolo delle dimensioni
    num_view = len(view_list)
    dim_hvcdn = pow(num_class, num_view)  # Dimensione per il Cross-View Decoder Network
    
    # =============================================================================
    # CONFIGURAZIONE PARAMETRI SPECIFICI PER DATASET
    # =============================================================================
    
    if data_folder == 'ROSMAP':
        adj_parameter = 2  # Parametro per la sparsità del grafo
        dim_he_list = [200, 200, 100]  # Dimensioni nascoste degli encoder
    if data_folder == 'BRCA':
        adj_parameter = 10
        dim_he_list = [400, 400, 200]
    
    # =============================================================================
    # PREPARAZIONE DEI DATI E PREPROCESSING
    # =============================================================================
    
    # Caricamento e preparazione dei dati multi-omici
    data_tr_list, data_trte_list, trte_idx, labels_trte = prepare_trte_data(data_folder, view_list)
    
    # Conversione delle labels di training a tensori PyTorch
    labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])
    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class)
    
    # Calcolo dei pesi per bilanciare le classi sbilanciate
    sample_weight_tr = cal_sample_weight(labels_trte[trte_idx["tr"]], num_class)
    sample_weight_tr = torch.FloatTensor(sample_weight_tr)
    
    # Trasferimento su GPU se disponibile
    if cuda:
        labels_tr_tensor = labels_tr_tensor.cuda()
        onehot_labels_tr_tensor = onehot_labels_tr_tensor.cuda()
        sample_weight_tr = sample_weight_tr.cuda()
    
    # Generazione delle matrici di adiacenza per i grafi
    adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter)
    
    # =============================================================================
    # INIZIALIZZAZIONE DEI MODELLI
    # =============================================================================
    
    # Calcolo delle dimensioni di input per ogni vista
    dim_list = [x.shape[1] for x in data_tr_list]
    
    # Inizializzazione di tutti i modelli (encoder + classificatori + VCDN)
    model_dict = init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hvcdn)
    
    # Trasferimento dei modelli su GPU
    for m in model_dict:
        if cuda:
            model_dict[m].cuda()
    
    # =============================================================================
    # FASE 1: PRE-TRAINING DEGLI ENCODER
    # =============================================================================
    
    print("\nPretrain GCNs...")
    
    # Inizializzazione degli ottimizzatori per il pre-training
    optim_dict = init_optim(num_view, model_dict, lr_e_pretrain, lr_c)
    
    # Training separato degli encoder per ogni vista (senza VCDN)
    for epoch in range(num_epoch_pretrain):
        train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor, 
                    onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict, 
                    train_VCDN=False)  # Disabilita il training del VCDN
    
    # =============================================================================
    # FASE 2: TRAINING PRINCIPALE CON INTEGRAZIONE MULTI-VISTA
    # =============================================================================
    
    print("\nTraining...")
    
    # Nuovi ottimizzatori con learning rate diverse per il training principale
    optim_dict = init_optim(num_view, model_dict, lr_e, lr_c)
    
    # Training completo con VCDN attivo
    for epoch in range(num_epoch+1):
        # Training di una singola epoca
        train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor, 
                    onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict)
        
        # =============================================================================
        # VALUTAZIONE PERIODICA SUL TEST SET
        # =============================================================================
        
        if epoch % test_inverval == 0:
            # Predizione sui dati di test
            te_prob = test_epoch(data_trte_list, adj_te_list, trte_idx["te"], model_dict)
            
            print("\nTest: Epoch {:d}".format(epoch))
            
            # Metriche per classificazione binaria
            if num_class == 2:
                print("Test ACC: {:.3f}".format(
                    accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
                print("Test F1: {:.3f}".format(
                    f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
                print("Test AUC: {:.3f}".format(
                    roc_auc_score(labels_trte[trte_idx["te"]], te_prob[:,1])))
            
            # Metriche per classificazione multi-classe
            else:
                print("Test ACC: {:.3f}".format(
                    accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
                print("Test F1 weighted: {:.3f}".format(
                    f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='weighted')))
                print("Test F1 macro: {:.3f}".format(
                    f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')))
            print()
    
    # =============================================================================
    # SALVATAGGIO DEI MODELLI ADDESTRATI
    # =============================================================================
    
    # Creazione della cartella per salvare i modelli
    model_folder = os.path.join(data_folder, 'models')
    
    print(f"\nSalvataggio modelli in: {model_folder}")
    
    # Salvataggio di tutti i modelli addestrati
    save_model_dict(model_folder, model_dict)
    
    print("Modelli salvati con successo!")
    
    # Calcolo delle predizioni finali per il ritorno
    final_predictions = test_epoch(data_trte_list, adj_te_list, trte_idx["te"], model_dict)
    
    # Ritorna le predizioni finali e informazioni utili
    return {
        'predictions': final_predictions,
        'true_labels': labels_trte[trte_idx["te"]],
        'model_dict': model_dict,
        'test_indices': trte_idx["te"]
    }
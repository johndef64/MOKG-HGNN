# ===== CONFIGURAZIONE PARAMETRI =====
addRawFeat = True  # Se aggiungere le features originali agli embeddings
base_path = ''
feature_networks_integration = ['clinical', 'cna', 'exp'] # Tipi di dati da concatenare come features
node_networks = ['clinical', 'cna', 'exp'] # Tipi di reti da usare per gli embeddings
int_method = 'MLP' # Algoritmo ML per integrare gli embeddings finali


# ===== PARAMETRI HYPERPARAMETER TUNING =====
learning_rates = [0.01, 0.001, 0.0001] # Learning rates da testare per le GCN
hid_sizes = [32, 64, 128, 256] # Dimensioni hidden layer da testare per le GCN
xtimes = 50 # Numero di iterazioni per il tuning degli algoritmi ML
xtimes2 = 10 # Numero di ripetizioni per calcolare deviazione standard delle metriche

# ===== PARAMETRI SELEZIONE FEATURES (OPZIONALE) =====
feature_selection_per_network = [False, False, False]  # Attiva selezione features per ogni rete
top_features_per_network = [50, 50, 50]  # Numero di top features da selezionare per rete
optional_feat_selection = False  # Selezione features sulle features concatenate finali
boruta_runs = 100  # Numero di run dell'algoritmo Boruta per feature selection
boruta_top_features = 50  # Numero di top features da selezionare con Boruta

# ===== PARAMETRI TRAINING GCN =====
max_epochs = 500  # Massimo numero di epoche per training GCN
min_epochs = 200  # Minimo numero di epoche prima di applicare early stopping
patience = 30     # Patience per early stopping

# ===== SEED PER RIPRODUCIBILITÀ =====
random_state = 404

# ===== AVVIO SUPREME =====
print('SUPREME Framework - Multi-omics Cancer Subtype Classification')
print('=' * 60)
print('Setting up SUPREME...')

# Importazione librerie necessarie
print('Loading required libraries...')
from lib import module  # Modulo custom per le GCN
import time
import os, itertools
import pickle5 as pickle
#import pickle
from sklearn.metrics import f1_score, accuracy_score
import statistics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split, RandomizedSearchCV, GridSearchCV
import pandas as pd
import numpy as np
from torch_geometric.data import Data
import os
import torch
import argparse
import errno
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Importazione librerie R per feature selection (solo se necessario)
if ((True in feature_selection_per_network) or (optional_feat_selection == True)):
    import rpy2
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
    utils = importr('utils')
    rFerns = importr('rFerns')    # Per algoritmo Boruta
    Boruta = importr('Boruta')    # Algoritmo per feature selection
    pracma = importr('pracma')
    dplyr = importr('dplyr')
    import re

# ===== PARSING ARGOMENTI E SETUP =====
parser = argparse.ArgumentParser(description='''Framework SUPREME per classificazione sottotipi tumorali 
usando Graph Convolutional Networks su dati multi-omici''')
parser.add_argument('-data', "--data_location", nargs = 1, default = ['sample_data'])

args = parser.parse_args()
dataset_name = args.data_location[0]  # Nome del dataset da usare
print(f'Dataset selected: {dataset_name}')

# Verifica esistenza cartella dati
path = base_path + "data/" + dataset_name
print(f'Checking data directory: {path}')
if not os.path.exists(path):
    print(f'ERROR: Data directory not found: {path}')
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
else:
    print(f'Data directory found successfully')
        
device = torch.device('cpu')  # Dispositivo per PyTorch (CPU)
print(f'PyTorch device: {device}')


# ===== FUNZIONI TRAINING E VALIDAZIONE GCN =====
def train():
    """Funzione per un step di training della GCN"""
    model.train()
    optimizer.zero_grad()
    out, emb1 = model(data)  # Forward pass: ottieni output e embeddings
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Calcola loss sui dati di training
    loss.backward()  # Backpropagation
    optimizer.step()  # Aggiorna pesi
    return emb1


def validate():
    """Funzione per validazione della GCN"""
    model.eval()
    with torch.no_grad():
        out, emb2 = model(data)  # Forward pass senza gradienti
        pred = out.argmax(dim=1)  # Predizioni
        loss = criterion(out[data.valid_mask], data.y[data.valid_mask])  # Loss su validation set
    return loss, emb2

criterion = torch.nn.CrossEntropyLoss()  # Funzione di loss per classificazione

# ===== SETUP PATHS E CARICAMENTO DATI =====
data_path_node =  base_path + 'data/' + dataset_name +'/'
run_name = 'SUPREME_'+  dataset_name + '_results'
save_path = base_path + run_name + '/'

# Crea cartella per salvare i risultati
print(f'Setting up output directory: {save_path}')
if not os.path.exists(base_path + run_name):
    os.makedirs(base_path + run_name + '/')
    print('Output directory created')
else:
    print('Output directory already exists')

# Carica le label (classi/sottotipi)
file = base_path + 'data/' + dataset_name +'/labels.pkl'
print(f'Loading labels from: {file}')
try:
    with open(file, 'rb') as f:
        labels = pickle.load(f)
    print(f'Labels loaded successfully. Shape: {len(labels)} samples')
    print(f'Unique classes: {set(labels)}')
except Exception as e:
    print(f'ERROR loading labels: {e}')
    raise

# Carica o crea split train/test
file = base_path + 'data/' + dataset_name + '/mask_values.pkl'
print(f'Checking for existing train/test split: {file}')
if os.path.exists(file):
    print('Loading existing train/test split...')
    with open(file, 'rb') as f:
        train_valid_idx, test_idx = pickle.load(f)
    print(f'Train/validation samples: {len(train_valid_idx)}')
    print(f'Test samples: {len(test_idx)}')
else:
    print('Creating new train/test split (80/20)...')
    # Crea split 80/20 stratificato
    train_valid_idx, test_idx= train_test_split(np.arange(len(labels)), test_size=0.20, shuffle=True, stratify=labels)
    print(f'Train/validation samples: {len(train_valid_idx)}')
    print(f'Test samples: {len(test_idx)}')

start = time.time()

is_first = 0

print('\n' + '=' * 60)
print('SUPREME EXECUTION STARTED')
print('=' * 60)

start = time.time()
print(f'Start time: {time.ctime()}')

# ===== FASE 1: GENERAZIONE NODE FEATURES =====
print('\nPHASE 1: GENERATING NODE FEATURES')
print('-' * 40)
print(f'Dataset name: {dataset_name}')
print(f'Node networks to process: {node_networks}')
print(f'Feature networks for integration: {feature_networks_integration}')
print(f'Add raw features: {addRawFeat}')

# Concatena le features da tutti i tipi di dati multi-omici            
for netw in node_networks:
    print(f'\nProcessing network: {netw}')
    print('-' * 20)
    # Carica features per questo tipo di dato (es: clinical, cna, exp)
    file = base_path + 'data/' + dataset_name +'/'+ netw +'.pkl'
    print(f'Loading file: {file}')
    
    # Check if file exists
    if not os.path.exists(file):
        print(f'ERROR: File not found: {file}')
        print(f'Available files in directory:')
        data_dir = base_path + 'data/' + dataset_name
        if os.path.exists(data_dir):
            for f in os.listdir(data_dir):
                print(f'  - {f}')
        else:
            print(f'  Directory does not exist: {data_dir}')
        print(f'Skipping network: {netw}')
        continue
    
    try:
        with open(file, 'rb') as f:
            feat = pickle.load(f)
        print(f'Successfully loaded {netw} features')
        print(f'Feature shape: {feat.shape if hasattr(feat, "shape") else "Unknown shape"}')
        if hasattr(feat, 'columns'):
            print(f'Feature columns (first 5): {list(feat.columns)[:5]}{"..." if len(feat.columns) > 5 else ""}')
        
        # Se abilitata, applica feature selection con Boruta
        print(f'Checking feature selection settings for {netw}...')
        if feature_selection_per_network[node_networks.index(netw)] and top_features_per_network[node_networks.index(netw)] < feat.values.shape[1]:     
            feat_flat = [item for sublist in feat.values.tolist() for item in sublist]
            feat_temp = robjects.FloatVector(feat_flat)
            robjects.globalenv['feat_matrix'] = robjects.r('matrix')(feat_temp)
            robjects.globalenv['feat_x'] = robjects.IntVector(feat.shape)
            robjects.globalenv['labels_vector'] = robjects.IntVector(labels.tolist())
            robjects.globalenv['top'] = top_features_per_network[node_networks.index(netw)]
            robjects.globalenv['maxBorutaRuns'] = boruta_runs
            robjects.r('''
                require(rFerns)
                require(Boruta)
                labels_vector = as.factor(labels_vector)
                feat_matrix <- Reshape(feat_matrix, feat_x[1])
                feat_data = data.frame(feat_matrix)
                colnames(feat_data) <- 1:feat_x[2]
                feat_data <- feat_data %>%
                    mutate('Labels' = labels_vector)
                boruta.train <- Boruta(feat_data$Labels ~ ., data= feat_data, doTrace = 0, getImp=getImpFerns, holdHistory = T, maxRuns = maxBorutaRuns)
                thr = sort(attStats(boruta.train)$medianImp, decreasing = T)[top]
                boruta_signif = rownames(attStats(boruta.train)[attStats(boruta.train)$medianImp >= thr,])
                    ''')
            boruta_signif = robjects.globalenv['boruta_signif']
            robjects.r.rm("feat_matrix")
            robjects.r.rm("labels_vector")
            robjects.r.rm("feat_data")
            robjects.r.rm("boruta_signif")
            robjects.r.rm("thr")
            topx = []
            for index in boruta_signif:
                t_index=re.sub("`","",index)
                topx.append((np.array(feat.values).T)[int(t_index)-1])
            topx = np.array(topx)
            values = torch.tensor(topx.T, device=device)
            print(f'Applied Boruta feature selection: {len(topx)} features selected from {feat.values.shape[1]} original features')
        elif feature_selection_per_network[node_networks.index(netw)] and top_features_per_network[node_networks.index(netw)] >= feat.values.shape[1]:
            values = feat.values
            print(f'Feature selection requested but limit ({top_features_per_network[node_networks.index(netw)]}) >= available features ({feat.values.shape[1]})')
            print(f'Using all available features')
        else:
            values = feat.values
            print(f'Using all features: {feat.values.shape}')
    
        # Concatena le features da tutti i tipi di dati
        print(f'Concatenating {netw} features to feature matrix...')
        if is_first == 0:
            new_x = torch.tensor(values, device=device).float()
            is_first = 1
            print(f'Initial feature matrix created - Shape: {new_x.shape}')
        else:
            prev_shape = new_x.shape
            new_x = torch.cat((new_x, torch.tensor(values, device=device).float()), dim=1)
            print(f'Feature matrix updated - Previous: {prev_shape}, New: {new_x.shape}')
        print(f'Successfully processed {netw} network\n')
            
    except Exception as e:
        print(f'ERROR processing {netw}: {e}')
        continue
    
# Check if we have successfully loaded any features
if is_first == 0:
    print('\n================= FEATURE LOADING FAILED =================')
    print('ERROR: No features were successfully loaded!')
    print('Please check if the data files exist and are properly formatted.')
    exit(1)

print(f'\n================= FEATURE LOADING COMPLETE =================')
print(f'Final feature matrix shape: {new_x.shape}')
print(f'Total features concatenated from {len(node_networks)} networks')

print(f'\nFeature loading completed!')
print(f'Final concatenated feature matrix shape: {new_x.shape}')
print(f'Total features per sample: {new_x.shape[1]}')

# ===== FASE 2: GENERAZIONE NODE EMBEDDINGS CON GCN =====
print(f'\n================= PHASE 2: GCN EMBEDDING GENERATION =================')
print(f'Generating node embeddings using Graph Convolutional Networks...')
print(f'Will train {len(node_networks)} separate GCNs for different network types')

# Addestra una GCN separata per ogni rete con hyperparameter tuning   
for n in range(len(node_networks)):
    netw_base = node_networks[n]  # Tipo di rete corrente (es: clinical, cna, exp)
    print(f'\n--- Processing GCN {n+1}/{len(node_networks)}: {netw_base} network ---')
    
    # Carica la struttura della rete (edges)
    edges_file = data_path_node + 'edges_' + netw_base + '.pkl'
    print(f'Loading edges file: {edges_file}')
    
    if not os.path.exists(edges_file):
        print(f'ERROR: Edge file not found: {edges_file}')
        print(f'Skipping network: {netw_base}')
        continue
    
    try:
        with open(edges_file, 'rb') as f:
            edge_index = pickle.load(f)
        print(f'Successfully loaded {netw_base} edges')
        print(f'Edge matrix shape: {edge_index.shape}')
        if hasattr(edge_index, 'columns'):
            print(f'Edge columns: {list(edge_index.columns)}')
    except Exception as e:
        print(f'ERROR loading edges for {netw_base}: {e}')
        continue
        
    best_ValidLoss = np.Inf
    print(f'Starting hyperparameter tuning for {netw_base} network')
    print(f'Parameter grid: {len(learning_rates)} learning rates × {len(hid_sizes)} hidden sizes = {len(learning_rates)*len(hid_sizes)} combinations')
    print(f'Learning rates: {learning_rates}')
    print(f'Hidden sizes: {hid_sizes}')
    
    # Hyperparameter tuning: prova tutte le combinazioni di lr e hidden size
    param_combo = 0
    for learning_rate in learning_rates:
        for hid_size in hid_sizes:
            param_combo += 1
            print(f'\n   Testing combination {param_combo}/{len(learning_rates)*len(hid_sizes)}: lr={learning_rate}, hidden={hid_size}')
            av_valid_losses = list()

            for ii in range(xtimes2):
                print(f'     Run {ii+1}/{xtimes2}')
                # Crea oggetto grafo con features, edges e label
                print(f'     Creating graph data object...')
                data = Data(x=new_x, edge_index=torch.tensor(edge_index[edge_index.columns[0:2]].transpose().values, device=device).long(),
                            edge_attr=torch.tensor(edge_index[edge_index.columns[2]].transpose().values, device=device).float(), y=labels) 
                X = data.x[train_valid_idx]
                y = data.y[train_valid_idx]
                print(f'     Train/valid data shape: {X.shape}, labels: {y.shape}')
                
                # Cross-validation per split train/validation
                rskf = RepeatedStratifiedKFold(n_splits=4, n_repeats=1)

                for train_part, valid_part in rskf.split(X, y):
                    train_idx = train_valid_idx[train_part]
                    valid_idx = train_valid_idx[valid_part]
                    break

                train_mask = np.array([i in set(train_idx) for i in range(data.x.shape[0])])
                valid_mask = np.array([i in set(valid_idx) for i in range(data.x.shape[0])])
                data.valid_mask = torch.tensor(valid_mask, device=device)
                data.train_mask = torch.tensor(train_mask, device=device)
                test_mask = np.array([i in set(test_idx) for i in range(data.x.shape[0])])
                data.test_mask = torch.tensor(test_mask, device=device)
                print(f'     Masks created - Train: {train_mask.sum()}, Valid: {valid_mask.sum()}, Test: {test_mask.sum()}')

                # Crea e configura il modello GCN
                in_size = data.x.shape[1]  # Dimensione input features
                out_size = torch.unique(data.y).shape[0]  # Numero di classi
                print(f'     Creating GCN model - Input: {in_size}, Hidden: {hid_size}, Output: {out_size}')
                model = module.Net(in_size=in_size, hid_size=hid_size, out_size=out_size)
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                print(f'     Model parameters: {sum(p.numel() for p in model.parameters())}')

                min_valid_loss = np.Inf
                patience_count = 0

                # Training loop con early stopping
                print(f'     Training GCN (max {max_epochs} epochs, patience {patience})...')
                for epoch in range(max_epochs):
                    emb = train()
                    this_valid_loss, emb = validate()

                    if this_valid_loss < min_valid_loss:
                        min_valid_loss = this_valid_loss
                        patience_count = 0
                        this_emb = emb
                    else:
                        patience_count += 1

                    if epoch >= min_epochs and patience_count >= patience:
                        print(f'     Early stopping at epoch {epoch+1}')
                        break
                    elif epoch % 50 == 0:
                        print(f'     Epoch {epoch+1}/{max_epochs}, Valid Loss: {this_valid_loss:.4f}')

                av_valid_losses.append(min_valid_loss.item())
                print(f'     Final validation loss: {min_valid_loss.item():.4f}')

            av_valid_loss = round(statistics.median(av_valid_losses), 3)
            print(f'   Average validation loss for lr={learning_rate}, hidden={hid_size}: {av_valid_loss}')
            
            if av_valid_loss < best_ValidLoss:
                best_ValidLoss = av_valid_loss
                best_emb_lr = learning_rate
                best_emb_hs = hid_size
                selected_emb = this_emb
                print(f'   New best parameters found!')

    print(f'Best hyperparameters for {netw_base}: lr={best_emb_lr}, hidden={best_emb_hs}, loss={best_ValidLoss}')
    emb_file = save_path + 'Emb_' +  netw_base + '.pkl'
    with open(emb_file, 'wb') as f:
        pickle.dump(selected_emb, f)
        pd.DataFrame(selected_emb).to_csv(emb_file[:-4] + '.csv')
    print(f'Embeddings saved to: {emb_file}')
    
start2 = time.time()    
print(f'\n================= GCN TRAINING COMPLETE =================')
print(f'Node embedding generation completed in {round(start2 - start, 1)} seconds')
print(f'Total hyperparameter trials: {len(learning_rates)*len(hid_sizes)} per network × {len(node_networks)} networks')
print(f'Embeddings saved for all {len(node_networks)} networks')

print(f'\n================= PHASE 3: EMBEDDING INTEGRATION =================')
print('Integrating embeddings for machine learning classification...')
    
# Running Machine Learning for each possible combination of input network
# Input for Machine Learning algorithm is the concatanation of node embeddings (specific to each combination) and node features (if node feature integration is True)    
addFeatures = []
t = range(len(node_networks))
trial_combs = []
for r in range(1, len(t) + 1):
    trial_combs.extend([list(x) for x in itertools.combinations(t, r)])

print(f'Testing {len(trial_combs)} different network combinations:')
for i, combo in enumerate(trial_combs):
    combo_names = [node_networks[j] for j in combo]
    print(f'  {i+1}. {" + ".join(combo_names)}')

for trials in range(len(trial_combs)):
    node_networks2 = [node_networks[i] for i in trial_combs[trials]]
    print(f'\n--- Combination {trials+1}/{len(trial_combs)}: {" + ".join(node_networks2)} ---')
    
    # Load first network embeddings
    netw_base = node_networks2[0]
    emb_file = save_path + 'Emb_' +  netw_base + '.pkl'
    print(f'Loading embeddings: {emb_file}')
    with open(emb_file, 'rb') as f:
        emb = pickle.load(f)
    print(f'Initial embedding shape: {emb.shape}')

    # Concatenate additional network embeddings if multiple networks
    if len(node_networks2) > 1:
        for netw_base in node_networks2[1:]:
            emb_file = save_path + 'Emb_' +  netw_base + '.pkl'
            print(f'Loading and concatenating: {emb_file}')
            with open(emb_file, 'rb') as f:
                cur_emb = pickle.load(f)
            prev_shape = emb.shape
            emb = torch.cat((emb, cur_emb), dim=1)
            print(f'Embedding shape after concatenation: {prev_shape} -> {emb.shape}')
            
    # Add raw features if requested
    if addRawFeat == True:
        print('Adding raw features to embeddings...')
        is_first = 0
        addFeatures = feature_networks_integration
        print(f'Feature networks to integrate: {addFeatures}')
        
        for netw in addFeatures:
            file = base_path + 'data/' + dataset_name +'/'+ netw +'.pkl'
            print(f'Loading raw features: {file}')
            with open(file, 'rb') as f:
                feat = pickle.load(f)
            print(f'Raw feature shape for {netw}: {feat.values.shape}')
            
            if is_first == 0:
                allx = torch.tensor(feat.values, device=device).float()
                is_first = 1
                print(f'Initial raw feature matrix: {allx.shape}')
            else:
                prev_shape = allx.shape
                allx = torch.cat((allx, torch.tensor(feat.values, device=device).float()), dim=1)   
                print(f'Raw feature matrix after {netw}: {prev_shape} -> {allx.shape}')   
        
        # Optional feature selection on raw features
        if optional_feat_selection == True:
            print(f'Applying Boruta feature selection on raw features (top {boruta_top_features})...')
            allx_flat = [item for sublist in allx.tolist() for item in sublist]
            allx_temp = robjects.FloatVector(allx_flat)
            robjects.globalenv['allx_matrix'] = robjects.r('matrix')(allx_temp)
            robjects.globalenv['allx_x'] = robjects.IntVector(allx.shape)
            robjects.globalenv['labels_vector'] = robjects.IntVector(labels.tolist())
            robjects.globalenv['top'] = boruta_top_features
            robjects.globalenv['maxBorutaRuns'] = boruta_runs
            robjects.r('''
                require(rFerns)
                require(Boruta)
                labels_vector = as.factor(labels_vector)
                allx_matrix <- Reshape(allx_matrix, allx_x[1])
                allx_data = data.frame(allx_matrix)
                colnames(allx_data) <- 1:allx_x[2]
                allx_data <- allx_data %>%
                    mutate('Labels' = labels_vector)
                boruta.train <- Boruta(allx_data$Labels ~ ., data= allx_data, doTrace = 0, getImp=getImpFerns, holdHistory = T, maxRuns = maxBorutaRuns)
                thr = sort(attStats(boruta.train)$medianImp, decreasing = T)[top]
                boruta_signif = rownames(attStats(boruta.train)[attStats(boruta.train)$medianImp >= thr,])
                    ''')
            boruta_signif = robjects.globalenv['boruta_signif']
            robjects.r.rm("allx_matrix")
            robjects.r.rm("labels_vector")
            robjects.r.rm("allx_data")
            robjects.r.rm("boruta_signif")
            robjects.r.rm("thr")
            topx = []
            for index in boruta_signif:
                t_index=re.sub("`","",index)
                topx.append((np.array(allx).T)[int(t_index)-1])
            topx = np.array(topx)
            prev_emb_shape = emb.shape
            emb = torch.cat((emb, torch.tensor(topx.T, device=device)), dim=1)
            print(f'Boruta feature selection completed: {len(topx)} features selected')
            print(f'Embedding shape after Boruta: {prev_emb_shape} -> {emb.shape}')
        else:
            prev_emb_shape = emb.shape
            emb = torch.cat((emb, allx), dim=1)
            print(f'Added all raw features: {prev_emb_shape} -> {emb.shape}')
    
    print(f'Final embedding shape for ML: {emb.shape}')
    data = Data(x=emb, y=labels)
    train_mask = np.array([i in set(train_valid_idx) for i in range(data.x.shape[0])])
    data.train_mask = torch.tensor(train_mask, device=device)
    test_mask = np.array([i in set(test_idx) for i in range(data.x.shape[0])])
    data.test_mask = torch.tensor(test_mask, device=device)
    
    X_train = pd.DataFrame(data.x[data.train_mask].numpy())
    X_test = pd.DataFrame(data.x[data.test_mask].numpy())
    y_train = pd.DataFrame(data.y[data.train_mask].numpy()).values.ravel()
    y_test = pd.DataFrame(data.y[data.test_mask].numpy()).values.ravel()
    
    print(f'Training data shape: {X_train.shape}, Test data shape: {X_test.shape}')
    print(f'Training labels: {len(y_train)}, Test labels: {len(y_test)}')
    
    print(f'Training {int_method} classifier...')
    if int_method == 'MLP':
        print('Hyperparameter tuning for MLP...')
        params = {'hidden_layer_sizes': [(16,), (32,),(64,),(128,),(256,),(512,), (32, 32), (64, 32), (128, 32), (256, 32), (512, 32)]}
        search = RandomizedSearchCV(estimator = MLPClassifier(solver = 'adam', activation = 'relu', early_stopping = True), 
                                    return_train_score = True, scoring = 'f1_macro', 
                                    param_distributions = params, cv = 4, n_iter = xtimes, verbose = 0)
        search.fit(X_train, y_train)
        print(f'Best MLP parameters: {search.best_params_}')
        model = MLPClassifier(solver = 'adam', activation = 'relu', early_stopping = True,
                              hidden_layer_sizes = search.best_params_['hidden_layer_sizes'])
        
    elif int_method == 'XGBoost':
        print('Hyperparameter tuning for XGBoost...')
        params = {'reg_alpha':range(0,6,1), 'reg_lambda':range(1,5,1),
                  'learning_rate':[0, 0.001, 0.01, 1]}
        fit_params = {'early_stopping_rounds': 10,
                     'eval_metric': 'mlogloss',
                     'eval_set': [(X_train, y_train)]}
        
              
        search = RandomizedSearchCV(estimator = XGBClassifier(use_label_encoder=False, n_estimators = 1000, 
                                                                  fit_params = fit_params, objective="multi:softprob", eval_metric = "mlogloss", 
                                                                  verbosity = 0), return_train_score = True, scoring = 'f1_macro',
                                        param_distributions = params, cv = 4, n_iter = xtimes, verbose = 0)
        
        search.fit(X_train, y_train)
        print(f'Best XGBoost parameters: {search.best_params_}')
        
        model = XGBClassifier(use_label_encoder=False, objective="multi:softprob", eval_metric = "mlogloss", verbosity = 0,
                              n_estimators = 1000, fit_params = fit_params,
                              reg_alpha = search.best_params_['reg_alpha'],
                              reg_lambda = search.best_params_['reg_lambda'],
                              learning_rate = search.best_params_['learning_rate'])
                            
    elif int_method == 'RF':
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        params = {'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 100)]}
        search = RandomizedSearchCV(estimator = RandomForestClassifier(), return_train_score = True,
                                    scoring = 'f1_macro', param_distributions = params, cv=4,  n_iter = xtimes, verbose = 0)
        search.fit(X_train, y_train)
        model=RandomForestClassifier(n_estimators = search.best_params_['n_estimators'])

    elif int_method == 'SVM':
        params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001]}
        search = RandomizedSearchCV(SVC(), return_train_score = True,
                                    scoring = 'f1_macro', param_distributions = params, cv=4, n_iter = xtimes, verbose = 0)
        search.fit(X_train, y_train)
        model=SVC(C = search.best_params_['C'],
                  gamma = search.best_params_['gamma'])

 
    av_result_acc = list()
    av_result_wf1 = list()
    av_result_mf1 = list()
    av_tr_result_acc = list()
    av_tr_result_wf1 = list()
    av_tr_result_mf1 = list()
 
        
    for ii in range(xtimes2):
        model.fit(X_train,y_train)
        predictions = model.predict(X_test)
        y_pred = [round(value) for value in predictions]
        preds = model.predict(pd.DataFrame(data.x.numpy()))
        av_result_acc.append(round(accuracy_score(y_test, y_pred), 3))
        av_result_wf1.append(round(f1_score(y_test, y_pred, average='weighted'), 3))
        av_result_mf1.append(round(f1_score(y_test, y_pred, average='macro'), 3))
        tr_predictions = model.predict(X_train)
        tr_pred = [round(value) for value in tr_predictions]
        av_tr_result_acc.append(round(accuracy_score(y_train, tr_pred), 3))
        av_tr_result_wf1.append(round(f1_score(y_train, tr_pred, average='weighted'), 3))
        av_tr_result_mf1.append(round(f1_score(y_train, tr_pred, average='macro'), 3))
        
    if xtimes2 == 1:
        av_result_acc.append(round(accuracy_score(y_test, y_pred), 3))
        av_result_wf1.append(round(f1_score(y_test, y_pred, average='weighted'), 3))
        av_result_mf1.append(round(f1_score(y_test, y_pred, average='macro'), 3))
        av_tr_result_acc.append(round(accuracy_score(y_train, tr_pred), 3))
        av_tr_result_wf1.append(round(f1_score(y_train, tr_pred, average='weighted'), 3))
        av_tr_result_mf1.append(round(f1_score(y_train, tr_pred, average='macro'), 3))
        

    result_acc = str(round(statistics.median(av_result_acc), 3)) + '+-' + str(round(statistics.stdev(av_result_acc), 3))
    result_wf1 = str(round(statistics.median(av_result_wf1), 3)) + '+-' + str(round(statistics.stdev(av_result_wf1), 3))
    result_mf1 = str(round(statistics.median(av_result_mf1), 3)) + '+-' + str(round(statistics.stdev(av_result_mf1), 3))
    tr_result_acc = str(round(statistics.median(av_tr_result_acc), 3)) + '+-' + str(round(statistics.stdev(av_tr_result_acc), 3))
    tr_result_wf1 = str(round(statistics.median(av_tr_result_wf1), 3)) + '+-' + str(round(statistics.stdev(av_tr_result_wf1), 3))
    tr_result_mf1 = str(round(statistics.median(av_tr_result_mf1), 3)) + '+-' + str(round(statistics.stdev(av_tr_result_mf1), 3))
    
    print('Combination ' + str(trials) + ' ' + str(node_networks2) + ' >  selected parameters = ' + str(search.best_params_) + 
      ', train accuracy = ' + str(tr_result_acc) + ', train weighted-f1 = ' + str(tr_result_wf1) +
      ', train macro-f1 = ' +str(tr_result_mf1) + ', test accuracy = ' + str(result_acc) + 
      ', test weighted-f1 = ' + str(result_wf1) +', test macro-f1 = ' +str(result_mf1))


end = time.time()
print('It took ' + str(round(end - start, 1)) + ' seconds in total.')
print('SUPREME is done.')

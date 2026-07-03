"""
MOGAT (Multi-Omics Graph Attention Network) Implementation
=========================================================

This script implements MOGAT, a graph attention network framework for integrating 
multi-omics data using prior knowledge networks (e.g., PPI networks) for cancer 
subtype classification.

The framework consists of two main stages:
1. GAT-based feature extraction: Uses Graph Attention Networks to learn embeddings 
   from multi-omics data on biological networks
2. Integration classifier: Combines learned embeddings for final classification

Multi-omics data types:
- exp: Gene expression data
- coe: Copy number alterations  
- cli: Clinical features
- met: DNA methylation data
- mut: Mutation data
- cna: Copy number variations
- lnc: Long non-coding RNA expression
- mir: microRNA expression
"""

# ===========================
# CONFIGURATION PARAMETERS
# ===========================

# Feature integration options
addRawFeat = True  # Whether to add raw features to GAT embeddings in final classification
base_path = ''     # Base directory path for data and results

# Multi-omics data types to use for feature integration and node networks
# Each represents a different biological data modality
feature_networks_integration = ['exp','coe','cli','met','mut','cna', 'lnc', 'mir']
node_networks = ['exp','coe','cli','met','mut','cna', 'lnc', 'mir']
# Alternative: single modality testing
# feature_networks_integration = ['exp']
# node_networks = ['exp']

# Integration method for final classification stage
int_method = 'MLP'  # Options: 'MLP', 'XGBoost', 'RF', 'SVM'

# Hyperparameter search and training repetitions
xtimes = 50   # Number of iterations for hyperparameter search
xtimes2 = 10  # Number of repetitions for final model training

# Feature selection configuration using Boruta algorithm
feature_selection_per_network = [False] * len(feature_networks_integration)
top_features_per_network = [50, 50, 50]  # Top features to select per network
optional_feat_selection = False           # Additional feature selection flag
boruta_runs = 100                        # Number of Boruta algorithm runs
boruta_top_features = 50                 # Top features from Boruta selection

# GAT training parameters
max_epochs = 500          # Maximum training epochs
min_epochs = 200          # Minimum epochs before early stopping
patience = 30             # Early stopping patience
learning_rates = [0.01, 0.001, 0.0001]  # Learning rates to search
hid_sizes = [512]         # Hidden layer sizes for GAT
# Alternative: full search space
# hid_sizes = [16, 32, 64, 128, 256, 512]
# learning_rates = [0.0001]

random_state = 404        # Random seed for reproducibility

# ===========================
# IMPORTS AND SETUP
# ===========================

print('MOGAT is setting up!')

# Custom modules for GAT implementation and utility functions
from lib import module2, function

# Standard libraries
import time
import os
import pyreadr
import itertools
import pickle
import statistics
import pandas as pd
import numpy as np
import torch
import argparse
import errno
import warnings
import re

# PyTorch Geometric for graph neural networks
from torch_geometric.data import Data

# Scikit-learn for machine learning models and evaluation
from sklearn.metrics import f1_score, accuracy_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (RepeatedStratifiedKFold, train_test_split, 
                                   RandomizedSearchCV, GridSearchCV)
from sklearn.ensemble import RandomForestClassifier

# XGBoost for gradient boosting
from xgboost import XGBClassifier

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# R interface for Boruta feature selection (only imported if feature selection is enabled)
if ((True in feature_selection_per_network) or (optional_feat_selection == True)):
    import rpy2
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
    # Import R packages for feature selection
    utils = importr('utils')
    rFerns = importr('rFerns')      # Random ferns for importance calculation
    Boruta = importr('Boruta')      # Boruta feature selection algorithm
    pracma = importr('pracma')      # Practical numerical math functions
    dplyr = importr('dplyr')        # Data manipulation functions

# ===========================
# COMMAND LINE ARGUMENT PARSING
# ===========================

# Set up argument parser for dataset selection
parser = argparse.ArgumentParser(description='''
MOGAT: Multi-Omics Graph Attention Network
==========================================

An integrative node classification framework for cancer subtype prediction that utilizes 
graph attention networks on multiple datatype-specific networks annotated with 
multi-omics datasets as node features.

This framework is model-agnostic and can be applied to any classification problem with 
properly processed datatypes and networks. In this implementation, MOGAT is applied 
to breast cancer subtype prediction using patient similarity networks constructed from 
multiple biological datasets.

The method works in two stages:
1. GAT stage: Learn embeddings for each omics type using graph attention on biological networks
2. Integration stage: Combine embeddings using traditional ML classifiers (MLP/XGBoost/RF/SVM)
''')

parser.add_argument('-data', "--data_location", nargs=1, default=['sample_data'],
                   help='Directory name containing the input data files')

# Parse arguments and setup paths
args = parser.parse_args()
dataset_name = args.data_location[0]

# Verify data directory exists
path = base_path + "data/" + dataset_name
if not os.path.exists(path):
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)

# ===========================
# CUDA SETUP
# ===========================

# Configure CUDA device for GPU acceleration
device = torch.device('cuda:0')
torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.cuda.set_device(0)


# ===========================
# GAT TRAINING FUNCTIONS
# ===========================

def train():
    """
    Training function for Graph Attention Network
    
    Performs one training step:
    1. Forward pass through GAT model
    2. Calculate loss on training nodes
    3. Backward propagation
    4. Update model parameters
    
    Returns:
        emb1 (torch.Tensor): Node embeddings learned by the GAT
    """
    model.train()                           # Set model to training mode
    optimizer.zero_grad()                   # Clear gradients
    out, emb1 = model(data)                # Forward pass: get predictions and embeddings
    
    # Extract training nodes and calculate loss
    train_idx = data.train_mask
    obj1 = out[train_idx]                  # Predictions for training nodes
    obj2 = data.y[train_idx]               # True labels for training nodes
    loss = criterion(obj1, obj2)           # Calculate cross-entropy loss
    
    # Backpropagation and parameter update
    loss.backward()                        # Compute gradients
    optimizer.step()                       # Update parameters
    return emb1


def validate():
    """
    Validation function for Graph Attention Network
    
    Evaluates model performance on validation set without updating parameters:
    1. Forward pass in evaluation mode
    2. Calculate validation loss
    3. Return loss and embeddings
    
    Returns:
        loss (torch.Tensor): Validation loss
        emb2 (torch.Tensor): Node embeddings from validation
    """
    model.eval()                           # Set model to evaluation mode
    with torch.no_grad():                  # Disable gradient computation
        out, emb2 = model(data)            # Forward pass
        pred = out.argmax(dim=1)           # Get predicted classes
        valid_idx = data.valid_mask
        loss = criterion(out[valid_idx], data.y[valid_idx])  # Calculate validation loss
    return loss, emb2

# Define loss function for node classification
criterion = torch.nn.CrossEntropyLoss()

# ===========================
# DATA LOADING AND PREPARATION
# ===========================

# Setup paths for data and results
data_path_node = base_path + 'data/' + dataset_name + '/'
run_name = 'MOGAT_' + dataset_name + '_results_1'
save_path = base_path + run_name + '/'
excel_file = save_path + "MOGAT_results.xlsx"

# Create results directory if it doesn't exist
if not os.path.exists(base_path + run_name):
    os.makedirs(base_path + run_name + '/')

# Load class labels for all samples
file = base_path + 'data/' + dataset_name + '/labels.pkl'
print("Reading:", file)
with open(file, 'rb') as f:
    labels = pickle.load(f)

# Load train/test split if available, otherwise create new split
file = base_path + 'data/' + dataset_name + '/mask_values.pkl'
if os.path.exists(file):
    print("Loading existing train/test split")
    with open(file, 'rb') as f:
        train_valid_idx, test_idx = pickle.load(f)
else:
    print("Creating new train/test split (80/20)")
    train_valid_idx, test_idx = train_test_split(
        np.arange(len(labels)), 
        test_size=0.20, 
        shuffle=True, 
        stratify=labels, 
        random_state=random_state
    )

# Start timing the execution
start = time.time()

# ===========================
# FEATURE LOADING AND SELECTION
# ===========================

is_first = 0  # Flag to track first network for concatenation

print('MOGAT is running..')

# Process each omics network to load and optionally select features
for netw in node_networks:
    file = base_path + 'data/' + dataset_name + '/' + netw + '.pkl'
    print("Reading:", file)
    
    with open(file, 'rb') as f:
        feat = pickle.load(f)
        
        # Apply Boruta feature selection if enabled for this network
        network_idx = node_networks.index(netw)
        should_select_features = (feature_selection_per_network[network_idx] and 
                                top_features_per_network[network_idx] < feat.values.shape[1])
        
        if should_select_features:
            print(f"Applying Boruta feature selection for {netw} network")
            
            # Prepare data for R Boruta algorithm
            feat_flat = [item for sublist in feat.values.tolist() for item in sublist]
            feat_temp = robjects.FloatVector(feat_flat)
            robjects.globalenv['feat_matrix'] = robjects.r('matrix')(feat_temp)
            robjects.globalenv['feat_x'] = robjects.IntVector(feat.shape)
            robjects.globalenv['labels_vector'] = robjects.IntVector(labels.tolist())
            robjects.globalenv['top'] = top_features_per_network[network_idx]
            robjects.globalenv['maxBorutaRuns'] = boruta_runs
            
            # Execute Boruta feature selection in R
            robjects.r('''
                require(rFerns)
                require(Boruta)
                labels_vector = as.factor(labels_vector)
                feat_matrix <- Reshape(feat_matrix, feat_x[1])
                feat_data = data.frame(feat_matrix)
                colnames(feat_data) <- 1:feat_x[2]
                feat_data <- feat_data %>%
                    mutate('Labels' = labels_vector)
                
                # Run Boruta algorithm for feature selection
                boruta.train <- Boruta(feat_data$Labels ~ ., data= feat_data, doTrace = 0, 
                                     getImp=getImpFerns, holdHistory = T, maxRuns = maxBorutaRuns)
                
                # Select top features based on median importance
                thr = sort(attStats(boruta.train)$medianImp, decreasing = T)[top]
                boruta_signif = rownames(attStats(boruta.train)[attStats(boruta.train)$medianImp >= thr,])
            ''')
            
            # Extract selected features
            boruta_signif = robjects.globalenv['boruta_signif']
            
            # Clean up R environment
            robjects.r.rm("feat_matrix")
            robjects.r.rm("labels_vector")
            robjects.r.rm("feat_data")
            robjects.r.rm("boruta_signif")
            robjects.r.rm("thr")
            
            # Convert selected feature indices and extract features
            topx = []
            for index in boruta_signif:
                t_index = re.sub("`", "", index)  # Remove backticks from R
                topx.append((np.array(feat.values).T)[int(t_index)-1])
            topx = np.array(topx)
            values = torch.tensor(topx.T, device=device)
            
        elif (feature_selection_per_network[network_idx] and 
              top_features_per_network[network_idx] >= feat.values.shape[1]):
            # If requested features >= available features, use all
            values = feat.values
        else:
            # No feature selection, use all features
            values = feat.values
    
    # Concatenate features from different networks
    if is_first == 0:
        new_x = torch.tensor(values, device=device).float()
        is_first = 1
    else:
        new_x = torch.cat((new_x, torch.tensor(values, device=device).float()), dim=1)
    
# ===========================
# GAT TRAINING AND HYPERPARAMETER OPTIMIZATION
# ===========================

# Train GAT for each network type to learn network-specific embeddings
for n in range(len(node_networks)):
    netw_base = node_networks[n]
    
    # Load the biological network (e.g., PPI, co-expression, etc.) for current omics type
    with open(data_path_node + 'edges_' + netw_base + '.pkl', 'rb') as f:
        print("Reading", data_path_node + 'edges_' + netw_base + '.pkl')
        edge_index = pickle.load(f)
    
    best_ValidLoss = np.Inf  # Track best validation loss for hyperparameter selection

    # Hyperparameter search over learning rates and hidden sizes
    print(f"Starting hyperparameter search for {netw_base} network")
    for learning_rate in learning_rates:
        for hid_size in hid_sizes:
            av_valid_losses = list()

            # Multiple runs for robust hyperparameter evaluation
            for ii in range(xtimes2):
                # Create PyTorch Geometric Data object
                # Node features: multi-omics data (new_x)
                # Edge indices: biological network connections (first 2 columns)
                # Edge attributes: edge weights/features (3rd column)
                data = Data(
                    x=new_x, 
                    edge_index=torch.tensor(edge_index[edge_index.columns[0:2]].transpose().values, device=device).long(),
                    edge_attr=torch.tensor(edge_index[edge_index.columns[2]].transpose().values, device=device).float(), 
                    y=labels
                )

                # Create train/validation split from the training set
                X = data.x[train_valid_idx]
                y = data.y[train_valid_idx]
                rskf = RepeatedStratifiedKFold(n_splits=4, n_repeats=1)

                # Use only the first fold for hyperparameter search
                for train_part, valid_part in rskf.split(X, y):
                    train_idx = train_valid_idx[train_part]
                    valid_idx = train_valid_idx[valid_part]
                    break

                # Create boolean masks for train/validation/test sets
                train_mask = np.array([i in set(train_idx) for i in range(data.x.shape[0])])
                valid_mask = np.array([i in set(valid_idx) for i in range(data.x.shape[0])])
                test_mask = np.array([i in set(test_idx) for i in range(data.x.shape[0])])
                
                data.valid_mask = torch.tensor(valid_mask, device=device)
                data.train_mask = torch.tensor(train_mask, device=device)
                data.test_mask = torch.tensor(test_mask, device=device)

                # Determine input and output dimensions
                in_size = data.x.shape[1]              # Number of input features
                out_size = torch.unique(data.y).shape[0]  # Number of classes

                print(f"GAT training for hyperparameters: lr={learning_rate}, hidden_size={hid_size}")
                
                # Initialize GAT model and optimizer
                model = module2.Net(in_size=in_size, hid_size=hid_size, out_size=out_size)
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

                # Training loop with early stopping
                min_valid_loss = np.Inf
                patience_count = 0
                data.cuda()  # Move data to GPU
                
                for epoch in range(max_epochs):
                    emb = train()                           # Training step
                    this_valid_loss, emb = validate()       # Validation step

                    # Early stopping logic
                    if this_valid_loss < min_valid_loss:
                        min_valid_loss = this_valid_loss
                        patience_count = 0
                    else:
                        patience_count += 1

                    # Stop if minimum epochs reached and no improvement
                    if epoch >= min_epochs and patience_count >= patience:
                        break

                av_valid_losses.append(min_valid_loss.item())

            # Calculate median validation loss for this hyperparameter combination
            av_valid_loss = round(statistics.median(av_valid_losses), 3)
            
            # Update best hyperparameters if this combination is better
            if av_valid_loss < best_ValidLoss:
                best_ValidLoss = av_valid_loss
                best_emb_lr = learning_rate     # Best learning rate
                best_emb_hs = hid_size         # Best hidden size
    # ===========================
    # FINAL GAT TRAINING WITH BEST HYPERPARAMETERS
    # ===========================
    
    print(f"Training final GAT model for {netw_base} with best hyperparameters: lr={best_emb_lr}, hidden_size={best_emb_hs}")
    
    # Recreate data object for final training
    data = Data(
        x=new_x, 
        edge_index=torch.tensor(edge_index[edge_index.columns[0:2]].transpose().values, device=device).long(),
        edge_attr=torch.tensor(edge_index[edge_index.columns[2]].transpose().values, device=device).float(), 
        y=labels
    )
    data.cuda()
    
    # For final training, use entire train_valid set for training and test set for validation
    X = data.x[train_valid_idx]
    y = data.y[train_valid_idx]
    
    # Create masks: train on train_valid set, validate on test set
    train_mask = np.array([i in set(train_valid_idx) for i in range(data.x.shape[0])])
    valid_mask = np.array([i in set(test_idx) for i in range(data.x.shape[0])])
    
    data.train_mask = torch.tensor(train_mask, device=device)
    data.valid_mask = torch.tensor(valid_mask, device=device)
    
    # Model dimensions
    in_size = data.x.shape[1]
    out_size = torch.unique(data.y).shape[0]
    
    # Initialize final GAT model with best hyperparameters
    print("GAT training Started 2")
    model = module2.Net(in_size=in_size, hid_size=best_emb_hs, out_size=out_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_emb_lr)

    # Training loop for final model
    min_valid_loss = np.Inf
    patience_count = 0
    selected_emb = None  # Will store the best embeddings
                
    for epoch in range(max_epochs):
        emb = train()                               # Training step
        this_valid_loss, emb = validate()           # Validation step

        # Save embeddings when validation improves
        if this_valid_loss < min_valid_loss:
            min_valid_loss = this_valid_loss
            patience_count = 0
            selected_emb = emb                      # Save best embeddings
        else:
            patience_count += 1

        # Early stopping
        if epoch >= min_epochs and patience_count >= patience:
            break

    # Save learned embeddings for this network
    emb_file = save_path + 'Emb_' + netw_base + '.pkl'
    with open(emb_file, 'wb') as f:
        pickle.dump(selected_emb, f)
        # Also save as CSV for inspection
        pd.DataFrame(selected_emb.cpu().numpy()).to_csv(emb_file[:-4] + '.csv')
    
    print(f"GAT training completed for {netw_base}. Embeddings saved to {emb_file}")
    
'''     
addFeatures = []
t = range(len(node_networks))
trial_combs = []
for r in range(1, len(t) + 1):
    trial_combs.extend([list(x) for x in itertools.combinations(t, r)])

device = torch.device('cpu')
for trials in range(len(trial_combs)):
    node_networks2 = [node_networks[i] for i in trial_combs[trials]] # list(set(a) & set(feature_networks))
    netw_base = node_networks2[0]
    emb_file = save_path + 'Emb_' +  netw_base + '.pkl'
    with open(emb_file, 'rb') as f:
        emb = pickle.load(f)
    
    if len(node_networks2) > 1:
        for netw_base in node_networks2[1:]:
            emb_file = save_path + 'Emb_' +  netw_base + '.pkl'
            with open(emb_file, 'rb') as f:
                cur_emb = pickle.load(f)
            emb = torch.cat((emb, cur_emb), dim=1)
    emb = emb.cpu()        
    if addRawFeat == True:
        is_first = 0
        addFeatures = feature_networks_integration
        for netw in addFeatures:
            file = base_path + 'data/' + dataset_name +'/'+ netw +'.pkl'
            with open(file, 'rb') as f:
                feat = pickle.load(f)
            if is_first == 0:
                allx = torch.tensor(feat.values, device=device).float()
                is_first = 1
            else:
                allx = torch.cat((allx, torch.tensor(feat.values, device=device).float()), dim=1)   
        
        else:
            # print(emb.get_device())
            # print(allx.get_device())
            emb = torch.cat((emb, allx), dim=1)
    
    data = Data(x=emb, y=labels)
    
    data.cpu()
    train_mask = np.array([i in set(train_valid_idx) for i in range(data.x.shape[0])])
    data.train_mask = torch.tensor(train_mask, device=device)
    test_mask = np.array([i in set(test_idx) for i in range(data.x.shape[0])])
    data.test_mask = torch.tensor(test_mask, device=device)
    X_train = pd.DataFrame(data.x[data.train_mask].numpy())
    X_test = pd.DataFrame(data.x[data.test_mask].numpy())
    y_train = pd.DataFrame(data.y[data.train_mask].numpy()).values.ravel()
    y_test = pd.DataFrame(data.y[data.test_mask].numpy()).values.ravel()
    print("Second Model Training Started")
    if int_method == 'MLP':
        params = {'hidden_layer_sizes': [(16,), (32,),(64,),(128,),(256,),(512,), (32, 32), (64, 32), (128, 32), (256, 32), (512, 32)],
                  'learning_rate_init': [0.1, 0.01, 0.001, 0.0001, 0.00001, 1, 2, 3],
                  'max_iter': [250, 500, 1000, 1500, 2000],
                  'n_iter_no_change': range(10,110,10)}
        search = RandomizedSearchCV(estimator = MLPClassifier(solver = 'adam', activation = 'relu', early_stopping = True), 
                                    return_train_score = True, scoring = 'f1_macro', 
                                    param_distributions = params, cv = 4, n_iter = xtimes, verbose = 0)
        search.fit(X_train, y_train)
        model = MLPClassifier(solver = 'adam', activation = 'relu', early_stopping = True,
                              max_iter = search.best_params_['max_iter'], 
                              n_iter_no_change = search.best_params_['n_iter_no_change'],
                              hidden_layer_sizes = search.best_params_['hidden_layer_sizes'],
                              learning_rate_init = search.best_params_['learning_rate_init'])
        
    elif int_method == 'XGBoost':
        params = {'reg_alpha':range(0,10,1), 'reg_lambda':range(1,10,1) ,'max_depth': range(1,6,1), 
                  'min_child_weight': range(1,10,1), 'gamma': range(0,6,1),
                  'learning_rate':[0, 1e-5, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1],
                  'max_delta_step': range(0,10,1), 'colsample_bytree': [0.5, 0.7, 1.0],
                  'colsample_bylevel': [0.5, 0.7, 1.0], 'colsample_bynode': [0.5, 0.7, 1.0]}
        fit_params = {'early_stopping_rounds': 10,
                     'eval_metric': 'mlogloss',
                     'eval_set': [(X_train, y_train)]}
        
              
        search = RandomizedSearchCV(estimator = XGBClassifier(use_label_encoder=False, n_estimators = 1000, 
                                                                  fit_params = fit_params, objective="multi:softprob", eval_metric = "mlogloss", 
                                                                  verbosity = 0), return_train_score = True, scoring = 'f1_macro',
                                        param_distributions = params, cv = 4, n_iter = xtimes, verbose = 0)
        
        search.fit(X_train, y_train)
        
        model = XGBClassifier(use_label_encoder=False, objective="multi:softprob", eval_metric = "mlogloss", verbosity = 0,
                              n_estimators = 1000, fit_params = fit_params,
                              reg_alpha = search.best_params_['reg_alpha'],
                              reg_lambda = search.best_params_['reg_lambda'],
                              max_depth = search.best_params_['max_depth'],
                              min_child_weight = search.best_params_['min_child_weight'],
                              gamma = search.best_params_['gamma'],
                              learning_rate = search.best_params_['learning_rate'],
                              max_delta_step = search.best_params_['max_delta_step'],
                              colsample_bytree = search.best_params_['colsample_bytree'],
                              colsample_bylevel = search.best_params_['colsample_bylevel'],
                              colsample_bynode = search.best_params_['colsample_bynode'])
                            
    elif int_method == 'RF':
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        params = {'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
                  'max_depth': max_depth,
                  'min_samples_split': [2, 5, 7, 10],
                  'min_samples_leaf': [1, 2, 5, 7, 10], 
                 'min_impurity_decrease':[0,0.5, 0.7, 1, 5, 10],
                 'max_leaf_nodes': [None, 5, 10, 20]}
        
        search = RandomizedSearchCV(estimator = RandomForestClassifier(), return_train_score = True,
                                    scoring = 'f1_macro', param_distributions = params, cv=4,  n_iter = xtimes, verbose = 0)
        
        search.fit(X_train, y_train)
        model=RandomForestClassifier(n_estimators = search.best_params_['n_estimators'],
                                     max_depth = search.best_params_['max_depth'],
                                     min_samples_split = search.best_params_['min_samples_split'],
                                     min_samples_leaf = search.best_params_['min_samples_leaf'],
                                     min_impurity_decrease = search.best_params_['min_impurity_decrease'],
                                     max_leaf_nodes = search.best_params_['max_leaf_nodes'])

    elif int_method == 'SVM':
        params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 'scale', 'auto'],
                  'kernel': ['linear', 'rbf']}
        
        search = RandomizedSearchCV(SVC(), return_train_score = True,
                                    scoring = 'f1_macro', param_distributions = params, cv=4, n_iter = xtimes, verbose = 0)
        
        search.fit(X_train, y_train)
        model = SVC(kernel=search.best_params_['kernel'],
                  C = search.best_params_['C'],
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
    
    
    df = pd.DataFrame(columns=['Comb No', 'Used Embeddings', 'Added Raw Features', 'Selected Params', 'Train Acc', 'Train wF1','Train mF1', 'Test Acc', 'Test wF1','Test mF1'])
    x = [trials, node_networks2, addFeatures, search.best_params_, 
         tr_result_acc, tr_result_wf1, tr_result_mf1, result_acc, result_wf1, result_mf1]
    df = df.append(pd.Series(x, index=df.columns), ignore_index=True)
    
    print('Combination ' + str(trials) + ' ' + str(node_networks2) + ' >  selected parameters = ' + str(search.best_params_) + 
      ', train accuracy = ' + str(tr_result_acc) + ', train weighted-f1 = ' + str(tr_result_wf1) +
      ', train macro-f1 = ' +str(tr_result_mf1) + ', test accuracy = ' + str(result_acc) + 
      ', test weighted-f1 = ' + str(result_wf1) +', test macro-f1 = ' +str(result_mf1))

    if trials == 0:
        if addRawFeat == True:
            function.append_df_to_excel(excel_file, df, sheet_name = int_method + '+Raw', index = False, header = True)
        else:
            function.append_df_to_excel(excel_file, df, sheet_name = int_method, index = False, header = True)
    else:
        if addRawFeat == True:
            function.append_df_to_excel(excel_file, df, sheet_name = int_method + '+Raw', index = False, header = False)
        else:
            function.append_df_to_excel(excel_file, df, sheet_name = int_method, index = False, header = False)
'''
end = time.time()
print('It took ' + str(round(end - start, 1)) + ' seconds for all runs.')
print('MOGAT is done.')
print('Results are available at ' + excel_file)

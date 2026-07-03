"""
Feature Importance Analysis for MOGONET
======================================

This module calculates feature importance for multi-omics data in MOGONET by using
permutation-based importance scoring. The importance of each feature is determined
by measuring the performance drop when that feature is set to zero.
"""

import os
import copy
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import f1_score
from utils import load_model_dict
from models import init_model_dict
from train_test import prepare_trte_data, gen_trte_adj_mat, test_epoch

# Check if CUDA is available for GPU acceleration
cuda = True if torch.cuda.is_available() else False

def cal_feat_imp(data_folder, model_folder, view_list, num_class):
    """
    Calculate feature importance using permutation-based method.
    
    This function computes the importance of each feature by measuring the performance
    drop when individual features are set to zero (permutation importance).
    
    Parameters:
    -----------
    data_folder : str
        Path to the folder containing the data files (e.g., 'ROSMAP', 'BRCA')
    model_folder : str
        Path to the folder containing pre-trained model weights
    view_list : list
        List of view/omics indices to analyze (e.g., [1, 2, 3])
    num_class : int
        Number of classes in the classification task
        
    Returns:
    --------
    feat_imp_list : list
        List of pandas DataFrames containing feature names and their importance scores
        for each view/omics type
    """
    num_view = len(view_list)
    # Dimension of hidden vector for cross-view decoder network
    dim_hvcdn = pow(num_class, num_view)
    
    # Set dataset-specific parameters
    if data_folder == 'ROSMAP':
        adj_parameter = 2  # Adjacency matrix parameter for ROSMAP dataset
        dim_he_list = [200, 200, 100]  # Hidden layer dimensions for each encoder
    if data_folder == 'BRCA':
        adj_parameter = 10  # Adjacency matrix parameter for BRCA dataset
        dim_he_list = [400, 400, 200]  # Hidden layer dimensions for each encoder
    
    # Load and prepare training and test data
    data_tr_list, data_trte_list, trte_idx, labels_trte = prepare_trte_data(data_folder, view_list)
    
    # Generate adjacency matrices for training and testing
    adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter)
    
    # Load feature names for each view/omics type
    featname_list = []
    for v in view_list:
        df = pd.read_csv(os.path.join(data_folder, str(v)+"_featname.csv"), header=None)
        featname_list.append(df.values.flatten())
    
    # Get dimensions of each view for model initialization
    dim_list = [x.shape[1] for x in data_tr_list]
    
    # Initialize and load pre-trained model
    model_dict = init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hvcdn)
    for m in model_dict:
        if cuda:
            model_dict[m].cuda()
    model_dict = load_model_dict(model_folder, model_dict)
    
    # Get baseline performance with original data
    te_prob = test_epoch(data_trte_list, adj_te_list, trte_idx["te"], model_dict)
    if num_class == 2:
        f1 = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
    else:
        f1 = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')
    
    # Calculate feature importance for each view/omics type
    feat_imp_list = []
    for i in range(len(featname_list)):
        # Initialize feature importance dictionary for current view
        feat_imp = {"feat_name": featname_list[i]}
        feat_imp['imp'] = np.zeros(dim_list[i])
        
        # Calculate importance for each feature in the current view
        for j in range(dim_list[i]):
            # Store original feature values before setting them to zero
            feat_tr = data_tr_list[i][:, j].clone()
            feat_trte = data_trte_list[i][:, j].clone()
            
            # Set current feature to zero (permutation step)
            data_tr_list[i][:, j] = 0
            data_trte_list[i][:, j] = 0
            
            # Regenerate adjacency matrices with modified data
            adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter)
            
            # Test model performance with current feature set to zero
            te_prob = test_epoch(data_trte_list, adj_te_list, trte_idx["te"], model_dict)
            if num_class == 2:
                f1_tmp = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
            else:
                f1_tmp = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')
            
            # Calculate feature importance as performance drop scaled by feature dimension
            feat_imp['imp'][j] = (f1 - f1_tmp) * dim_list[i]
            
            # Restore original feature values
            data_tr_list[i][:, j] = feat_tr.clone()
            data_trte_list[i][:, j] = feat_trte.clone()
        
        # Store feature importance results as DataFrame
        feat_imp_list.append(pd.DataFrame(data=feat_imp))
    
    return feat_imp_list


def summarize_imp_feat(featimp_list_list, topn=30):
    """
    Summarize and rank feature importance across multiple repetitions/folds.
    
    This function aggregates feature importance scores from multiple runs and
    identifies the top most important features across all omics types.
    
    Parameters:
    -----------
    featimp_list_list : list of lists
        Nested list where each inner list contains feature importance DataFrames
        for different views/omics from a single repetition/fold
    topn : int, default=30
        Number of top features to display in the final ranking
        
    Returns:
    --------
    None
        Prints the ranking of top features to console
    """
    num_rep = len(featimp_list_list)  # Number of repetitions/folds
    num_view = len(featimp_list_list[0])  # Number of views/omics types
    
    # Initialize list to store temporary DataFrames
    df_tmp_list = []
    
    # Process first repetition and add omics type identifier
    for v in range(num_view):
        df_tmp = copy.deepcopy(featimp_list_list[0][v])
        df_tmp['omics'] = np.ones(df_tmp.shape[0], dtype=int) * v  # Add omics type column
        df_tmp_list.append(df_tmp.copy(deep=True))
    
    # Concatenate all views from first repetition
    df_featimp = pd.concat(df_tmp_list).copy(deep=True)
    
    # Process remaining repetitions and aggregate results
    for r in range(1, num_rep):
        for v in range(num_view):
            df_tmp = copy.deepcopy(featimp_list_list[r][v])
            df_tmp['omics'] = np.ones(df_tmp.shape[0], dtype=int) * v  # Add omics type column
            # Append to main DataFrame (using concat instead of deprecated append)
            df_featimp = pd.concat([df_featimp, df_tmp.copy(deep=True)], ignore_index=True)
    
    # Aggregate importance scores by feature name and omics type
    df_featimp_top = df_featimp.groupby(['feat_name', 'omics'])['imp'].sum()
    df_featimp_top = df_featimp_top.reset_index()
    
    # Sort by importance score in descending order and select top features
    df_featimp_top = df_featimp_top.sort_values(by='imp', ascending=False)
    df_featimp_top = df_featimp_top.iloc[:topn]
    
    # Display results
    print('{:}\t{:}'.format('Rank', 'Feature name'))
    for i in range(len(df_featimp_top)):
        print('{:}\t{:}'.format(i+1, df_featimp_top.iloc[i]['feat_name']))
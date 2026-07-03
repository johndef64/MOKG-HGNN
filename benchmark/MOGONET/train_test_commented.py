""" Training and testing of the model
"""
# Import necessary libraries
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
import torch.nn.functional as F
from models import init_model_dict, init_optim
from utils import one_hot_tensor, cal_sample_weight, gen_adj_mat_tensor, gen_test_adj_mat_tensor, cal_adj_mat_parameter

# Check if CUDA is available for GPU acceleration
cuda = True if torch.cuda.is_available() else False


def prepare_trte_data(data_folder, view_list):
    """
    Prepare training and testing data from the specified data folder.

    Args:
        data_folder (str): Path to the folder containing the data files.
        view_list (list): List of view identifiers (e.g., ['view1', 'view2']).

    Returns:
        tuple: A tuple containing:
            - data_train_list (list): List of training data tensors for each view.
            - data_all_list (list): List of combined training and testing data tensors for each view.
            - idx_dict (dict): Dictionary containing indices for training and testing data.
            - labels (numpy.ndarray): Concatenated labels for training and testing data.
    """

    # Get the number of views from the view_list
    num_view = len(view_list)

    # Load training and testing labels from CSV files
    labels_tr = np.loadtxt(os.path.join(data_folder, "labels_tr.csv"), delimiter=',')
    labels_te = np.loadtxt(os.path.join(data_folder, "labels_te.csv"), delimiter=',')

    # Convert labels to integers
    labels_tr = labels_tr.astype(int)
    labels_te = labels_te.astype(int)

    # Initialize lists to store training and testing data for each view
    data_tr_list = []
    data_te_list = []

    # Load data for each view
    for i in view_list:
        # Load training data for the current view
        data_tr_list.append(np.loadtxt(os.path.join(data_folder, str(i) + "_tr.csv"), delimiter=','))
        # Load testing data for the current view
        data_te_list.append(np.loadtxt(os.path.join(data_folder, str(i) + "_te.csv"), delimiter=','))

    # Get the number of training and testing samples
    num_tr = data_tr_list[0].shape[0]
    num_te = data_te_list[0].shape[0]

    # Create a list of data matrices by concatenating training and testing data for each view
    data_mat_list = []
    for i in range(num_view):
        data_mat_list.append(np.concatenate((data_tr_list[i], data_te_list[i]), axis=0))

    # Convert data matrices to tensors and move to GPU if CUDA is available
    data_tensor_list = []
    for i in range(len(data_mat_list)):
        data_tensor = torch.FloatTensor(data_mat_list[i])
        if cuda:
            data_tensor = data_tensor.cuda()
        data_tensor_list.append(data_tensor)

    # Create dictionaries to store indices for training and testing data
    idx_dict = {}
    idx_dict["tr"] = list(range(num_tr))
    idx_dict["te"] = list(range(num_tr, num_tr + num_te))

    # Prepare training data list
    data_train_list = []
    # Prepare combined data list (training + testing)
    data_all_list = []
    for i in range(len(data_tensor_list)):
        # Add training data for the current view
        data_train_list.append(data_tensor_list[i][idx_dict["tr"]].clone())
        # Combine training and testing data for the current view
        data_all_list.append(torch.cat((data_tensor_list[i][idx_dict["tr"]].clone(),
                                      data_tensor_list[i][idx_dict["te"]].clone()), 0))

    # Concatenate training and testing labels
    labels = np.concatenate((labels_tr, labels_te))

    # Return the prepared data
    return data_train_list, data_all_list, idx_dict, labels


def gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter):
    # Define the metric for calculating adjacency (cosine similarity)
    adj_metric = "cosine" 

    # Initialize lists to store adjacency matrices for training and testing
    adj_train_list = []
    adj_test_list = []

    # Iterate over each view's data
    for i in range(len(data_tr_list)):
        # Calculate adaptive parameter for adjacency matrix
        adj_parameter_adaptive = cal_adj_mat_parameter(adj_parameter, data_tr_list[i], adj_metric)

        # Generate training adjacency matrix tensor
        adj_train_list.append(gen_adj_mat_tensor(data_tr_list[i], adj_parameter_adaptive, adj_metric))

        # Generate testing adjacency matrix tensor using test indices
        adj_test_list.append(gen_test_adj_mat_tensor(data_trte_list[i], trte_idx, adj_parameter_adaptive, adj_metric))

    # Return lists of adjacency matrices for training and testing
    return adj_train_list, adj_test_list


def train_epoch(data_list, adj_list, label, one_hot_label, sample_weight, model_dict, optim_dict, train_VCDN=True):
    # Initialize dictionary to store loss for each component
    loss_dict = {}

    # Define loss criterion (CrossEntropyLoss with reduction='none' for element-wise loss)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    # Set all models to training mode
    for m in model_dict:
        model_dict[m].train()

    # Get number of views from data_list
    num_view = len(data_list)

    # Iterate over each view to compute loss and update optimizer
    for i in range(num_view):
        # Zero the gradients for the current view's optimizer
        optim_dict["C{:}".format(i+1)].zero_grad()

        # Initialize loss for current view
        ci_loss = 0

        # Forward pass: Encoder -> Node features, then Classifier
        ci = model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](data_list[i], adj_list[i]))

        # Compute loss using criterion and apply sample weights
        ci_loss = torch.mean(torch.mul(criterion(ci, label), sample_weight))

        # Backward pass and optimize
        ci_loss.backward()
        optim_dict["C{:}".format(i+1)].step()

        # Store loss value in dictionary
        loss_dict["C{:}".format(i+1)] = ci_loss.detach().cpu().numpy().item()

    # If training VCDN and multiple views are present, compute combined loss
    if train_VCDN and num_view >= 2:
        # Zero gradients for VCDN optimizer
        optim_dict["C"].zero_grad()

        # Initialize VCDN loss
        c_loss = 0

        # List to store outputs from each view's classifier
        ci_list = []

        # Forward pass for each view
        for i in range(num_view):
            ci_list.append(model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](data_list[i], adj_list[i])))

        # Combine outputs from all views using VCDN model
        c = model_dict["C"](ci_list)

        # Compute VCDN loss with sample weights
        c_loss = torch.mean(torch.mul(criterion(c, label), sample_weight))

        # Backward pass and optimize
        c_loss.backward()
        optim_dict["C"].step()

        # Store VCDN loss in dictionary
        loss_dict["C"] = c_loss.detach().cpu().numpy().item()

    # Return dictionary of losses
    return loss_dict


def test_epoch(data_list, adj_list, te_idx, model_dict):
    # Set all models to evaluation mode
    for m in model_dict:
        model_dict[m].eval()

    # Get number of views from data_list
    num_view = len(data_list)

    # List to store outputs from each view's classifier
    ci_list = []

    # Forward pass for each view
    for i in range(num_view):
        ci_list.append(model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](data_list[i], adj_list[i])))

    # Combine outputs if multiple views are present, otherwise use single view's output
    if num_view >= 2:
        c = model_dict["C"](ci_list)
    else:
        c = ci_list[0]

    # Select probabilities for test indices
    c = c[te_idx, :]

    # Apply softmax to get probability distribution
    prob = F.softmax(c, dim=1).data.cpu().numpy()

    # Return predicted probabilities
    return prob


def train_test(data_folder, view_list, num_class,
               lr_e_pretrain, lr_e, lr_c, 
               num_epoch_pretrain, num_epoch):
    # Set interval for testing during training
    test_interval = 50

    # Get number of views from view_list
    num_view = len(view_list)

    # Calculate dimension for hidden variables based on number of classes and views
    dim_hvcdn = pow(num_class, num_view)

    # Set dataset-specific parameters
    if data_folder == 'ROSMAP':
        adj_parameter = 2  # Adjacency parameter for ROSMAP dataset
        dim_he_list = [200, 200, 100]  # Hidden embedding dimensions
    if data_folder == 'BRCA':
        adj_parameter = 10  # Adjacency parameter for BRCA dataset
        dim_he_list = [400, 400, 200]  # Hidden embedding dimensions

    # Prepare training and testing data
    data_tr_list, data_trte_list, trte_idx, labels_trte = prepare_trte_data(data_folder, view_list)

    # Convert labels to tensor and one-hot encoding
    labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])
    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class)

    # Calculate sample weights for class balancing
    sample_weight_tr = cal_sample_weight(labels_trte[trte_idx["tr"]], num_class)
    sample_weight_tr = torch.FloatTensor(sample_weight_tr)

    # Move tensors to GPU if CUDA is available
    if cuda:
        labels_tr_tensor = labels_tr_tensor.cuda()
        onehot_labels_tr_tensor = onehot_labels_tr_tensor.cuda()
        sample_weight_tr = sample_weight_tr.cuda()

    # Generate adjacency matrices for training and testing
    adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter)

    # Get dimension list from training data
    dim_list = [x.shape[1] for x in data_tr_list]

    # Initialize model dictionary with specified dimensions
    model_dict = init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hvcdn)

    # Move models to GPU if CUDA is available
    for m in model_dict:
        if cuda:
            model_dict[m].cuda()

    # Pretrain GCNs
    print("\nPretrain GCNs...")
    optim_dict = init_optim(num_view, model_dict, lr_e_pretrain, lr_c)
    for epoch in range(num_epoch_pretrain):
        train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor, 
                   onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict, train_VCDN=False)

    # Main training loop
    print("\nTraining...")
    optim_dict = init_optim(num_view, model_dict, lr_e, lr_c)
    for epoch in range(num_epoch + 1):
        train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor, 
                   onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict)

        # Test model periodically
        if epoch % test_interval == 0:
            te_prob = test_epoch(data_trte_list, adj_te_list, trte_idx["te"], model_dict)
            print("\nTest: Epoch {:d}".format(epoch))

            # Calculate and print performance metrics
            if num_class == 2:
                print("Test ACC: {:.3f}".format(accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
                print("Test F1: {:.3f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
                print("Test AUC: {:.3f}".format(roc_auc_score(labels_trte[trte_idx["te"]], te_prob[:, 1])))
            else:
                print("Test ACC: {:.3f}".format(accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
                print("Test F1 weighted: {:.3f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='weighted')))
                print("Test F1 macro: {:.3f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')))
            print()
"""
Utility Functions for MOGONET
============================

This module contains essential utility functions for the MOGONET (Multi-Omics Graph cOnvolutional NETworks) framework.
It provides functions for:
- Sample weight calculation for handling class imbalance
- Graph construction from distance matrices
- Adjacency matrix generation for graph neural networks
- Cosine distance computation
- Tensor format conversions
- Model saving and loading utilities

The functions handle multi-omics data integration and graph-based learning operations.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F

# Check if CUDA is available for GPU acceleration
cuda = True if torch.cuda.is_available() else False

def cal_sample_weight(labels, num_class, use_sample_weight=True):
    """
    Calculate sample weights for handling class imbalance in multi-class classification.
    
    This function computes weights inversely proportional to class frequencies to balance
    the contribution of each class during training. Classes with fewer samples get higher
    weights, while classes with more samples get lower weights.
    
    Parameters:
    -----------
    labels : numpy.ndarray
        Array of class labels (integers from 0 to num_class-1)
    num_class : int
        Total number of classes in the dataset
    use_sample_weight : bool, default=True
        Whether to apply sample weighting. If False, returns uniform weights
        
    Returns:
    --------
    sample_weight : numpy.ndarray
        Array of weights for each sample, same length as labels
        
    Mathematical Formula:
    For class i: weight_i = count_i / total_samples
    where count_i is the number of samples in class i
    """
    if not use_sample_weight:
        # Return uniform weights if sample weighting is disabled
        return np.ones(len(labels)) / len(labels)
    
    # Count the number of samples in each class
    count = np.zeros(num_class)
    for i in range(num_class):
        count[i] = np.sum(labels == i)  # Count samples belonging to class i
    
    # Initialize sample weight array with same shape as labels
    sample_weight = np.zeros(labels.shape)
    
    # Assign weights based on class frequency
    for i in range(num_class):
        # Find indices of samples belonging to class i
        class_indices = np.where(labels == i)[0]
        # Assign weight = class_frequency / total_samples to all samples of class i
        sample_weight[class_indices] = count[i] / np.sum(count)
    
    return sample_weight


def one_hot_tensor(y, num_dim):
    """
    Convert class labels to one-hot encoded tensor format.
    
    This function transforms integer class labels into one-hot encoded vectors,
    which are commonly used in multi-class classification tasks. Each class
    is represented as a binary vector with 1 at the class position and 0 elsewhere.
    
    Parameters:
    -----------
    y : torch.Tensor
        Tensor of class labels (integers) with shape (batch_size,)
    num_dim : int
        Number of classes (dimension of one-hot vectors)
        
    Returns:
    --------
    y_onehot : torch.Tensor
        One-hot encoded tensor with shape (batch_size, num_dim)
        
    Example:
    --------
    Input: y = [0, 2, 1], num_dim = 3
    Output: [[1, 0, 0],
             [0, 0, 1], 
             [0, 1, 0]]
    """
    # Initialize zero tensor with shape (batch_size, num_classes)
    y_onehot = torch.zeros(y.shape[0], num_dim)
    
    # Use scatter_ to place 1s at the correct positions
    # scatter_(dim, index, value) fills the tensor along dim using index positions
    # y.view(-1,1) reshapes y to column vector for indexing
    y_onehot.scatter_(1, y.view(-1, 1), 1)
    
    return y_onehot


def cosine_distance_torch(x1, x2=None, eps=1e-8):
    """
    Compute pairwise cosine distances between samples in PyTorch tensors.
    
    Cosine distance measures the angular difference between vectors, calculated as:
    cosine_distance = 1 - cosine_similarity
    cosine_similarity = (x1 · x2) / (||x1|| * ||x2||)
    
    This metric is particularly useful for high-dimensional data where the magnitude
    of vectors is less important than their direction (e.g., gene expression data).
    
    Parameters:
    -----------
    x1 : torch.Tensor
        First set of vectors with shape (n_samples_1, n_features)
    x2 : torch.Tensor, optional
        Second set of vectors with shape (n_samples_2, n_features)
        If None, computes pairwise distances within x1
    eps : float, default=1e-8
        Small epsilon value to prevent division by zero
        
    Returns:
    --------
    distances : torch.Tensor
        Pairwise cosine distances with shape (n_samples_1, n_samples_2)
        Values range from 0 (identical direction) to 2 (opposite direction)
        
    Mathematical Details:
    - cosine_similarity = dot_product / (norm1 * norm2)
    - cosine_distance = 1 - cosine_similarity
    - For identical vectors: distance = 0
    - For orthogonal vectors: distance = 1  
    - For opposite vectors: distance = 2
    """
    # If x2 is not provided, compute pairwise distances within x1
    x2 = x1 if x2 is None else x2
    
    # Compute L2 norms (Euclidean norms) for each sample
    # norm(p=2, dim=1, keepdim=True) computes ||x||_2 for each row
    w1 = x1.norm(p=2, dim=1, keepdim=True)  # Shape: (n_samples_1, 1)
    
    # Reuse w1 if x2 is the same as x1, otherwise compute norm for x2
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)  # Shape: (n_samples_2, 1)
    
    # Compute cosine distance: 1 - (x1 @ x2.T) / (||x1|| * ||x2||)
    # torch.mm(x1, x2.t()) computes dot products between all pairs
    # (w1 * w2.t()) computes outer product of norms
    # clamp(min=eps) prevents division by zero
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def to_sparse(x):
    """
    Convert a dense tensor to sparse tensor format for memory efficiency.
    
    Sparse tensors store only non-zero elements, which is memory-efficient for
    matrices with many zero values (common in adjacency matrices of graphs).
    This function converts dense tensors to PyTorch's COOrdinate (COO) sparse format.
    
    Parameters:
    -----------
    x : torch.Tensor
        Dense tensor to be converted to sparse format
        
    Returns:
    --------
    sparse_tensor : torch.sparse.Tensor
        Sparse representation of the input tensor in COO format
        
    Technical Details:
    - COO format stores (indices, values, size) where:
      * indices: coordinates of non-zero elements
      * values: actual non-zero values
      * size: original tensor dimensions
    - If tensor is all zeros, returns empty sparse tensor with correct shape
    """
    # Get the tensor type name (e.g., 'FloatTensor', 'DoubleTensor')
    x_typename = torch.typename(x).split('.')[-1]
    
    # Get corresponding sparse tensor constructor
    sparse_tensortype = getattr(torch.sparse, x_typename)
    
    # Find coordinates of all non-zero elements
    indices = torch.nonzero(x)
    
    # Handle edge case: if all elements are zeros
    if len(indices.shape) == 0:
        return sparse_tensortype(*x.shape)
    
    # Transpose indices to get proper format for sparse tensor
    # PyTorch sparse expects indices as (n_dims, n_nonzeros)
    indices = indices.t()
    
    # Extract values at non-zero positions
    # Use tuple indexing to get values at each coordinate
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    
    # Create and return sparse tensor
    return sparse_tensortype(indices, values, x.size())


def cal_adj_mat_parameter(edge_per_node, data, metric="cosine"):
    """
    Calculate the distance threshold parameter for graph construction.
    
    This function determines the optimal threshold for creating adjacency matrices
    by ensuring each node has approximately 'edge_per_node' connections. The threshold
    is computed by finding the distance value that would result in the desired
    connectivity level across the entire graph.
    
    Parameters:
    -----------
    edge_per_node : int
        Target number of edges (connections) per node in the graph
    data : torch.Tensor
        Input data matrix with shape (n_samples, n_features)
    metric : str, default="cosine"
        Distance metric to use (currently only "cosine" is implemented)
        
    Returns:
    --------
    parameter : float
        Distance threshold value for graph construction
        Distances <= parameter will create edges in the graph
        
    Algorithm:
    ----------
    1. Compute pairwise distances between all data points
    2. Flatten distance matrix and sort values
    3. Select the (edge_per_node * n_samples)-th smallest distance
    4. This ensures approximately edge_per_node connections per node
    
    Example:
    --------
    If edge_per_node=5 and n_samples=100, we select the 500th smallest
    distance from 10,000 total pairwise distances.
    """
    assert metric == "cosine", "Only cosine distance implemented"
    
    # Compute pairwise cosine distances between all samples
    dist = cosine_distance_torch(data, data)
    
    # Flatten distance matrix and sort to find threshold
    # We want the (edge_per_node * n_samples)-th smallest distance
    # This ensures each node has approximately edge_per_node connections
    sorted_distances = torch.sort(dist.reshape(-1,)).values
    threshold_index = edge_per_node * data.shape[0]
    parameter = sorted_distances[threshold_index]
    
    # Convert to CPU numpy and extract scalar value
    return parameter.data.cpu().numpy().item()

def graph_from_dist_tensor(dist, parameter, self_dist=True):
    """
    Create a binary adjacency graph from a distance matrix using a threshold.
    
    This function converts a continuous distance matrix into a binary graph
    by applying a distance threshold. Pairs of nodes with distance <= parameter
    are connected (edge = 1), while pairs with distance > parameter are not
    connected (edge = 0).
    
    Parameters:
    -----------
    dist : torch.Tensor
        Distance matrix with shape (n_samples, n_samples) if self_dist=True
        or (n_samples_1, n_samples_2) if self_dist=False
    parameter : float
        Distance threshold for edge creation
        Pairs with distance <= parameter will be connected
    self_dist : bool, default=True
        Whether the distance matrix represents self-distances (square matrix)
        If True, diagonal elements are set to 0 (no self-loops)
        
    Returns:
    --------
    g : torch.Tensor
        Binary adjacency matrix with same shape as dist
        Values are 0 (no edge) or 1 (edge exists)
        
    Graph Construction Logic:
    ------------------------
    - Edge exists if distance <= threshold
    - Self-loops are removed when self_dist=True
    - Results in sparse graphs for appropriate threshold selection
    """
    if self_dist:
        # Verify that input is a square matrix for self-distance computation
        assert dist.shape[0] == dist.shape[1], "Input is not pairwise dist matrix"
    
    # Create binary adjacency matrix: 1 if distance <= parameter, 0 otherwise
    # .float() converts boolean tensor to float tensor (0.0 and 1.0)
    g = (dist <= parameter).float()
    
    if self_dist:
        # Remove self-loops by setting diagonal elements to 0
        # np.diag_indices returns (row_indices, col_indices) for diagonal
        diag_idx = np.diag_indices(g.shape[0])
        g[diag_idx[0], diag_idx[1]] = 0

    return g


def gen_adj_mat_tensor(data, parameter, metric="cosine"):
    """
    Generate a complete adjacency matrix for graph neural network training.
    
    This function creates a weighted, symmetric, and normalized adjacency matrix
    suitable for graph convolutional networks. The process involves:
    1. Computing pairwise distances
    2. Creating binary connectivity graph
    3. Converting distances to similarities
    4. Making the matrix symmetric
    5. Adding self-connections and normalizing
    
    Parameters:
    -----------
    data : torch.Tensor
        Input data matrix with shape (n_samples, n_features)
    parameter : float
        Distance threshold for determining graph connectivity
    metric : str, default="cosine"
        Distance metric for computing similarities
        
    Returns:
    --------
    adj : torch.sparse.Tensor
        Normalized sparse adjacency matrix ready for GNN operations
        
    Mathematical Process:
    --------------------
    1. Distance computation: D = cosine_distance(data, data)
    2. Binary graph: G = (D <= parameter)
    3. Similarity matrix: S = (1 - D) * G
    4. Symmetrization: A = S + S^T - S ⊙ (S^T > S)
    5. Self-connection: A' = A + I
    6. Row normalization: A_norm = A' / rowsum(A')
    7. Sparsification for memory efficiency
    """
    assert metric == "cosine", "Only cosine distance implemented"
    
    # Step 1: Compute pairwise cosine distances
    dist = cosine_distance_torch(data, data)
    
    # Step 2: Create binary connectivity graph based on distance threshold
    g = graph_from_dist_tensor(dist, parameter, self_dist=True)
    
    # Step 3: Convert distances to similarities and apply graph mask
    if metric == "cosine":
        # Cosine similarity = 1 - cosine_distance
        adj = 1 - dist
    else:
        raise NotImplementedError
    
    # Apply binary graph mask to keep only selected edges
    adj = adj * g
    
    # Step 4: Make the adjacency matrix symmetric
    # This ensures undirected graph properties
    adj_T = adj.transpose(0, 1)  # Transpose of adjacency matrix
    
    # Create identity matrix for self-connections
    I = torch.eye(adj.shape[0])
    if cuda:
        I = I.cuda()
    
    # Symmetrization formula: A_sym = A + A^T * (A^T > A) - A * (A^T > A)
    # This takes the maximum value between A[i,j] and A[j,i] for each edge
    adj = adj + adj_T * (adj_T > adj).float() - adj * (adj_T > adj).float()
    
    # Step 5: Add self-connections and normalize
    # F.normalize with p=1 performs row-wise L1 normalization (sum=1)
    adj = F.normalize(adj + I, p=1)
    
    # Step 6: Convert to sparse format for memory efficiency
    adj = to_sparse(adj)

    return adj


def gen_test_adj_mat_tensor(data, trte_idx, parameter, metric="cosine"):
    """
    Generate adjacency matrix for test phase with train-test split constraints.
    
    This function creates a specialized adjacency matrix for the test phase where:
    - Test nodes can only connect to training nodes (no test-test connections)
    - Training nodes can connect to test nodes
    - This prevents information leakage during testing
    - Maintains graph structure for semi-supervised learning
    
    Parameters:
    -----------
    data : torch.Tensor
        Complete dataset with shape (n_total_samples, n_features)
        Contains both training and test samples
    trte_idx : dict
        Dictionary with keys "tr" and "te" containing indices for
        training and test samples respectively
    parameter : float
        Distance threshold for edge creation
    metric : str, default="cosine"
        Distance metric for similarity computation
        
    Returns:
    --------
    adj : torch.sparse.Tensor
        Normalized sparse adjacency matrix for test phase
        Shape: (n_total_samples, n_total_samples)
        
    Matrix Structure:
    ----------------
    The resulting adjacency matrix has the structure:
    
         [Train | Test ]
    Train[ A_tt | A_te ]  
    Test [ A_et | 0    ]
    
    Where:
    - A_tt: train-train connections (from training phase)
    - A_te: train-test connections (allowed)
    - A_et: test-train connections (allowed)  
    - 0: test-test connections (forbidden to prevent leakage)
    """
    assert metric == "cosine", "Only cosine distance implemented"
    
    # Initialize adjacency matrix with zeros
    adj = torch.zeros((data.shape[0], data.shape[0]))
    if cuda:
        adj = adj.cuda()
    
    num_tr = len(trte_idx["tr"])  # Number of training samples
    
    # Part 1: Create train-to-test connections
    # Compute distances from training samples to test samples
    dist_tr2te = cosine_distance_torch(data[trte_idx["tr"]], data[trte_idx["te"]])
    
    # Create binary connectivity graph for train-to-test edges
    g_tr2te = graph_from_dist_tensor(dist_tr2te, parameter, self_dist=False)
    
    # Convert distances to similarities and apply to adjacency matrix
    if metric == "cosine":
        adj[:num_tr, num_tr:] = 1 - dist_tr2te  # Fill train-test block
    else:
        raise NotImplementedError
    
    # Apply binary graph mask to retain only selected edges
    adj[:num_tr, num_tr:] = adj[:num_tr, num_tr:] * g_tr2te

    # Part 2: Create test-to-train connections  
    # Compute distances from test samples to training samples
    dist_te2tr = cosine_distance_torch(data[trte_idx["te"]], data[trte_idx["tr"]])
    
    # Create binary connectivity graph for test-to-train edges
    g_te2tr = graph_from_dist_tensor(dist_te2tr, parameter, self_dist=False)
    
    # Convert distances to similarities and apply to adjacency matrix
    if metric == "cosine":
        adj[num_tr:, :num_tr] = 1 - dist_te2tr  # Fill test-train block
    else:
        raise NotImplementedError
    
    # Apply binary graph mask to retain only selected edges
    adj[num_tr:, :num_tr] = adj[num_tr:, :num_tr] * g_te2tr
    
    # Part 3: Symmetrization and normalization
    # Make the adjacency matrix symmetric (same process as gen_adj_mat_tensor)
    adj_T = adj.transpose(0, 1)
    
    # Add identity matrix for self-connections
    I = torch.eye(adj.shape[0])
    if cuda:
        I = I.cuda()
    
    # Symmetrization: take maximum between A[i,j] and A[j,i]
    adj = adj + adj_T * (adj_T > adj).float() - adj * (adj_T > adj).float()
    
    # Row-wise L1 normalization and add self-connections
    adj = F.normalize(adj + I, p=1)
    
    # Convert to sparse format for memory efficiency
    adj = to_sparse(adj)

    return adj


def save_model_dict(folder, model_dict):
    """
    Save all models in the model dictionary to disk.
    
    This function saves the state dictionaries of all models in the provided
    dictionary to separate .pth files. Each model is saved with its key name
    as the filename, allowing for organized model persistence.
    
    Parameters:
    -----------
    folder : str
        Directory path where model files will be saved
        Will be created if it doesn't exist
    model_dict : dict
        Dictionary where keys are model names (strings) and values are
        PyTorch model objects with state_dict() method
        
    File Structure:
    ---------------
    folder/
    ├── model1.pth  # First model's state dict
    ├── model2.pth  # Second model's state dict
    └── ...
    
    Usage Example:
    --------------
    model_dict = {'encoder': encoder_model, 'decoder': decoder_model}
    save_model_dict('checkpoints/', model_dict)
    # Creates: checkpoints/encoder.pth, checkpoints/decoder.pth
    """
    # Create directory if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Save each model's state dictionary
    for module in model_dict:
        # Get model's state dictionary (parameters and buffers)
        state_dict = model_dict[module].state_dict()
        
        # Save to file with module name
        file_path = os.path.join(folder, module + ".pth")
        torch.save(state_dict, file_path)


def load_model_dict(folder, model_dict):
    """
    Load pre-trained model weights from disk into model dictionary.
    
    This function loads saved state dictionaries from .pth files and applies
    them to the corresponding models in the model dictionary. It handles
    missing files gracefully and ensures models are moved to GPU if available.
    
    Parameters:
    -----------
    folder : str
        Directory path containing the saved model files (.pth files)
    model_dict : dict
        Dictionary where keys are model names and values are PyTorch model
        objects that will receive the loaded weights
        
    Returns:
    --------
    model_dict : dict
        Updated model dictionary with loaded weights
        Models are moved to CUDA if available
        
    Loading Process:
    ----------------
    1. Check if corresponding .pth file exists for each model
    2. Load state dictionary with appropriate device mapping
    3. Apply loaded weights to model
    4. Move model to GPU if CUDA is available
    5. Print warnings for missing model files
    
    Error Handling:
    ---------------
    - Missing files: Print warning but continue with other models
    - Device mapping: Automatically maps to current CUDA device
    - GPU availability: Conditionally moves models to CUDA
    """
    for module in model_dict:
        # Construct file path for current model
        model_file = os.path.join(folder, module + ".pth")
        
        if os.path.exists(model_file):
            # Load state dictionary with device mapping
            # map_location ensures compatibility across different devices
            # torch.cuda.current_device() gets the current CUDA device ID
            state_dict = torch.load(
                model_file, 
                map_location="cuda:{:}".format(torch.cuda.current_device())
            )
            
            # Apply loaded weights to the model
            model_dict[module].load_state_dict(state_dict)
            
            # Uncomment the following line if you want loading confirmations:
            # print("Module {:} loaded!".format(module))
        else:
            # Warn about missing model files but continue execution
            print("WARNING: Module {:} from model_dict is not loaded!".format(module))
        
        # Move model to GPU if CUDA is available
        if cuda:
            model_dict[module].cuda()
    
    return model_dict
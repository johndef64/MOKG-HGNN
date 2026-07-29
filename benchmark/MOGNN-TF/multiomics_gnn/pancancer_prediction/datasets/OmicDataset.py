
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import torch
import pandas as pd
from torch_geometric.data import Data, Dataset

class OmicsGraphDataset(Dataset):
    def __init__(self, X, y, edge_index, edge_weight=None, edge_attr=None, transform=None, pre_transform=None):
        super().__init__(root=None, transform=transform, pre_transform=pre_transform) 
        self.X = X
        self.y = y
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.edge_attr = edge_attr
        self.num_nodes = X.shape[1]
        self.feature_dim = X.shape[-1]
        
    def len(self):
        return int(self.X.size(0))
    
    def get(self, idx):
        d = Data(x=self.X[idx], edge_index=self.edge_index, y=self.y[idx])
        if self.edge_attr is not None:
            d.edge_attr = self.edge_attr
        elif self.edge_weight is not None:
            d.edge_weight = self.edge_weight
        return d 
    
    def get_weight_pancan(self, gamma: float = 1, alpha: float = 2, sample_type: str = 'nulite') -> torch.DoubleTensor:
        """
        Get a WeightedRandomSampler for pancan dataset to handle class imbalance.

        Args:
            gamma (float): Exponent to adjust the weights. Default is 1.
            alpha (float): Parameter to adjust the weights. Default is 2.
            sample_type (str): Type of sample weighting. Default is 'nulite'.
        Returns:
            WeightedRandomSampler: Sampler for DataLoader.
        """
        if sample_type == 'nulite':
            assert 0 <= gamma <= 1, "Gamma must be between 0 and 1"
        else:
            assert alpha >= 0 , "Alpha must be between 0 and 1"
        # Build subtype counts and per-sample weights, then return a sampler
        print('Calculating weights for Sampler...')
        labels = self.y.numpy()
        sub_df = pd.DataFrame({'index': np.arange(len(labels)), 'subtype': labels})
        print(sub_df.head(5))
        # get counts per subtype
        subtype_counts = sub_df['subtype'].value_counts().to_dict()
        print('Subtype counts:', subtype_counts)
        k = sum(subtype_counts.values())
        print('Total samples (k):', k)
        weights_dict = {}
        for c, n_c in subtype_counts.items():
            if sample_type == 'nulite':
                print("-----------using nulite weighting-----------")
                print("-----------gamma value:", gamma)
                w = k / (gamma * n_c + (1 - gamma) * k)
            elif sample_type == 'alpha':
                print("-----------using alpha weighting-----------")
                print("-----------alpha value:", alpha)
                w = (k / n_c) ** (1/alpha)
            weights_dict[c] = float(w)
        print('Weights dict:', weights_dict)
        # assign weight to each sample based on its subtype
        weights = [weights_dict[subtype] for subtype in sub_df['subtype']]
        weights_tensor = torch.tensor(weights, dtype=torch.double)
        return weights_tensor

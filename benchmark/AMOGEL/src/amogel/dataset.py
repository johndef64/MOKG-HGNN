from torch.utils.data import Dataset
import torch
import pandas as pd

class OmicDataset(Dataset):
    
    def __init__(self , data: pd.DataFrame , label: pd.DataFrame) -> None:
        
        assert len(data) == len(label) , "Data and label must have the same length"
        assert isinstance(data , pd.DataFrame) , "Data must be a pandas DataFrame"
        assert isinstance(label , pd.DataFrame) , "Label must be a pandas DataFrame"
        
        self.data = data
        self.label = label
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        data = torch.tensor(self.data.iloc[idx].values , dtype=torch.float32)
        label = torch.tensor(self.label.iloc[idx,0] , dtype=torch.long)
        
        return data , label
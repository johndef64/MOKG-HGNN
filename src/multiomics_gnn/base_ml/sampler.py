import torch
from torch.utils.data import WeightedRandomSampler, RandomSampler, Sampler
from torch.utils.data import Subset
import numpy as np
from multiomics_gnn.pancancer_prediction.datasets.OmicDataset import OmicsGraphDataset
class OmicSampler:
    def __init__(self, dataset: OmicsGraphDataset, batch_size, strategy='random', gamma=None, alpha=None, seed=None, mode='nulite'):
        self.strategy = strategy
        self.weight_mode = mode
        self.dataset = dataset
        self.batch_size = batch_size
        self.gamma = gamma if gamma is not None else 0.5
        self.alpha = alpha if alpha is not None else 1.0
        self.seed = seed if seed is not None else 42
        self.num_samples = len(dataset)
        self.sampler = self._get_sampler()

    def _get_sampler(self) -> torch.utils.data.Sampler:
        """Return the sampler (either RandomSampler or WeightedRandomSampler)

        Args:
            train_dataset (CellDataset): Dataset for training
            strategy (str, optional): Sampling strategy. Defaults to "random" (random sampling).
                Implemented are "random", "cell", "tissue", "cell+tissue".
            gamma (float, optional): Gamma scaling factor, between 0 and 1.
                1 means total balancing, 0 means original weights. Defaults to 1.

        Raises:
            NotImplementedError: Not implemented sampler is selected

        Returns:
            Sampler: Sampler for training
        """
        if self.strategy == 'random':
            sampling_generator = torch.Generator().manual_seed(self.seed)
            return RandomSampler(self.dataset, generator=sampling_generator)
        
        elif self.strategy == 'weighted':
            print("Using class-based WeightedRandomSampler")
            if isinstance(self.dataset, Subset):
                ds = self.dataset.dataset
            else:
                ds = self.dataset
            # Sonnet paper
            
            weights = ds.get_weight_pancan(gamma=self.gamma, sample_type=self.weight_mode, alpha=self.alpha)
            sampling_generator = torch.Generator().manual_seed(self.seed)
            sampler = WeightedRandomSampler(weights, num_samples=self.num_samples, replacement=True, generator=sampling_generator)

            print("Weights tensor shape:", weights.shape)
            print("Weights tensor sample values:", weights)
            print("Number of samples:", self.num_samples)
            return sampler
        elif self.strategy.lower() == "none":
            print("No sampler used")
            sampler = None
            return sampler
        else:
            raise ValueError(f"Sampling strategy '{self.strategy}' is not implemented.")
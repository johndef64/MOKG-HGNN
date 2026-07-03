"""Per-patient HeteroData dataset for the multi-scale backbone.

HeteroData analog of MOGNN-TF's ``OmicsGraphDataset``. The topology template is
shared across all patients (built once by build_hetero_graph.py); only the node
features change per patient. ``get(i)`` returns a fresh HeteroData that reuses the
template's edge_index tensors (by reference, they are never mutated) and carries
this patient's per-type feature matrices + the graph-level label.

Featureless scales (pathway / GO_term / disease) keep only ``num_nodes``; the
model assigns them learned embeddings (e.g. torch.nn.Embedding or lazy in_channels=-1).
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Dataset, HeteroData


class HeteroOmicsDataset(Dataset):
    def __init__(self, template, features, y, indices, transform=None):
        """
        template : HeteroData topology (edge_index_dict + num_nodes), no features
        features : dict {node_type: ndarray [N_patients, n_nodes, C]}
        y        : ndarray [N_patients] encoded labels
        indices  : ndarray of patient rows in this split (train/val/test)
        """
        super().__init__(root=None, transform=transform)
        self.template = template
        self.features = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in features.items()}
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.patient_idx = np.asarray(indices, dtype=int)  # not 'indices': clashes with Dataset.indices()

        # cache the static parts once
        self._edge_index = {et: template[et].edge_index for et in template.edge_types}
        self._num_nodes = {nt: template[nt].num_nodes for nt in template.node_types}
        self._featured = set(self.features.keys())

    def len(self):
        return len(self.patient_idx)

    def get(self, i):
        p = int(self.patient_idx[i])
        data = HeteroData()
        for nt in self._num_nodes:
            if nt in self._featured:
                data[nt].x = self.features[nt][p]           # [n_nodes, C]
            else:
                data[nt].num_nodes = self._num_nodes[nt]     # featureless scale
        for et, ei in self._edge_index.items():
            data[et].edge_index = ei                         # shared, static
        data.y = self.y[p].view(1)
        return data

    # --- class-imbalance sampler weights (ported from OmicDataset) -----------
    def get_weight_pancan(self, gamma=1.0, alpha=2.0, sample_type="nulite"):
        """Per-sample weights for a WeightedRandomSampler over the 27 subtypes."""
        labels = self.y[self.patient_idx].numpy()
        sub = pd.DataFrame({"subtype": labels})
        counts = sub["subtype"].value_counts().to_dict()
        k = sum(counts.values())
        wd = {}
        for c, n_c in counts.items():
            if sample_type == "nulite":
                assert 0.0 <= gamma <= 1.0, "gamma in [0,1]"
                wd[c] = k / (gamma * n_c + (1 - gamma) * k)
            else:
                assert alpha >= 0, "alpha >= 0"
                wd[c] = (k / n_c) ** (1.0 / alpha)
        return torch.tensor([wd[s] for s in sub["subtype"]], dtype=torch.double)

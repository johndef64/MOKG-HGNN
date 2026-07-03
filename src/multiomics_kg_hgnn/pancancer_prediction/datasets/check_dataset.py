"""End-to-end sanity check: real omics -> per-patient HeteroData batch ->
HGTConv forward -> 27-class graph-level logits. Featureless scales get a learned
embedding; molecular types get a type-specific linear projection (in_channels=-1).

    conda run -n gnn python -m multiomics_kg_hgnn.pancancer_prediction.datasets.check_dataset
"""

import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv, global_mean_pool

from multiomics_kg_hgnn.pancancer_prediction.preprocessing.make_datasets import (
    make_datasets, build_loaders)


class MultiScaleHGT(nn.Module):
    """Minimal proof model: type-specific input projection (learned embedding for
    featureless scales) -> HGTConv -> multi-scale mean-pool readout -> MLP."""

    def __init__(self, metadata, hidden=32, out=27):
        super().__init__()
        self.node_types, _ = metadata
        # lazy Linear for featured types; Embedding for featureless scales
        self.proj = nn.ModuleDict()
        self.emb = nn.ModuleDict()
        self.hgt = HGTConv(hidden, hidden, metadata, heads=2)
        self.head = nn.Linear(hidden * 3, out)  # readout over gene+pathway+GO_term
        self.hidden = hidden

    def _lin(self, nt):
        if nt not in self.proj:
            self.proj[nt] = nn.LazyLinear(self.hidden)
        return self.proj[nt]

    def forward(self, batch):
        x_dict = {}
        for nt in self.node_types:
            store = batch[nt]
            if "x" in store:
                x_dict[nt] = self._lin(nt)(store.x)
            else:
                if nt not in self.emb:
                    self.emb[nt] = nn.Embedding(1, self.hidden)
                x_dict[nt] = self.emb[nt](torch.zeros(store.num_nodes, dtype=torch.long))
        h = self.hgt(x_dict, batch.edge_index_dict)
        pooled = torch.cat(
            [global_mean_pool(h[nt], batch[nt].batch) for nt in ("gene", "pathway", "GO_term")],
            dim=1)
        return self.head(pooled)


def main():
    tr, va, te, ncls = make_datasets("data/training/splits/splits_seed_42")
    train_loader, _, _ = build_loaders(tr, va, te, batch_size=8)
    batch = next(iter(train_loader))

    model = MultiScaleHGT(tr.template.metadata(), hidden=32, out=ncls)
    logits = model(batch)
    loss = nn.functional.cross_entropy(logits, batch.y)
    loss.backward()  # exercise the backward pass too

    print(f"\nlogits: {tuple(logits.shape)} (batch x {ncls} classi) | "
          f"loss={loss.item():.4f} | backward OK")
    print("END-TO-END (dati reali -> HeteroData -> HGT -> 27 classi): OK")


if __name__ == "__main__":
    main()

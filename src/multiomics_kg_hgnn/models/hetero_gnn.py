"""Heterogeneous multi-scale GNN for pan-cancer subtype classification.

Graph-level classification over per-patient HeteroData (see the datasets package).
Type-specific input encoding -> backbone message passing -> multi-scale readout
-> MLP -> 27-class logits.

Backbone is selected by name via BACKBONES. First iteration ships HGTConv; the
registry makes adding HeteroConv+SAGE / RGCN a one-function change (see the TODOs).
"""

import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv, global_mean_pool


class HeteroInputEncoder(nn.Module):
    """Per-node-type encoding into a shared hidden space.
    Featured types get a concrete Linear projection (input dim from feature_dims);
    every other node type gets a learned Embedding (a single vector broadcast to
    all its nodes). Using concrete Linears — not LazyLinear — means the state_dict
    is fully materialized and portable (no UninitializedParameter to serialize)."""

    def __init__(self, node_types, hidden, feature_dims):
        super().__init__()
        self.node_types = list(node_types)
        self.hidden = hidden
        self.featured = dict(feature_dims)  # node_type -> in_channels
        self.proj = nn.ModuleDict(
            {self._key(nt): nn.Linear(c, hidden) for nt, c in self.featured.items()})
        self.emb = nn.ModuleDict(
            {self._key(nt): nn.Embedding(1, hidden) for nt in self.node_types
             if nt not in self.featured})

    @staticmethod
    def _key(nt):
        return nt.replace(".", "_")

    def forward(self, data):
        out = {}
        for nt in self.node_types:
            key = self._key(nt)
            if nt in self.featured:
                out[nt] = self.proj[key](data[nt].x)
            else:
                idx = torch.zeros(data[nt].num_nodes, dtype=torch.long, device=self.emb[key].weight.device)
                out[nt] = self.emb[key](idx)
        return out


# --------------------------------------------------------------------------
# Backbones (name -> builder). Each builder returns an nn.Module whose forward
# is (x_dict, edge_index_dict) -> x_dict.
# --------------------------------------------------------------------------
def _build_hgt(metadata, hidden, num_layers, heads, dropout):
    return _HGTStack(metadata, hidden, num_layers, heads, dropout)


class _HGTStack(nn.Module):
    def __init__(self, metadata, hidden, num_layers, heads, dropout):
        super().__init__()
        self.convs = nn.ModuleList(
            [HGTConv(hidden, hidden, metadata, heads=heads) for _ in range(num_layers)])
        self.drop = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: self.drop(v.relu()) for k, v in x_dict.items()}
        return x_dict


BACKBONES = {
    "hgt": _build_hgt,
    # TODO(hetero-sage): to_hetero(GraphSAGE) / HeteroConv({rel: SAGEConv}) baseline.
    # TODO(rgcn): RGCNConv via data.to_homogeneous() (KG-style, one weight per relation).
}


class HeteroMultiScaleGNN(nn.Module):
    """Full model: encoder -> backbone -> multi-scale readout -> classifier."""

    def __init__(self, metadata, num_classes, feature_dims, backbone="hgt", hidden=64,
                 num_layers=2, heads=2, dropout=0.2, readout_types=("gene", "pathway", "GO_term")):
        super().__init__()
        self.node_types, _ = metadata
        if backbone not in BACKBONES:
            raise ValueError(f"backbone '{backbone}' not in {list(BACKBONES)}")
        # readout only over scales that actually exist in this template
        self.readout_types = [t for t in readout_types if t in self.node_types]

        self.encoder = HeteroInputEncoder(self.node_types, hidden, feature_dims)
        self.backbone = BACKBONES[backbone](metadata, hidden, num_layers, heads, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * len(self.readout_types), hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, data):
        x_dict = self.encoder(data)
        x_dict = self.backbone(x_dict, data.edge_index_dict)
        pooled = [global_mean_pool(x_dict[t], data[t].batch) for t in self.readout_types]
        return self.classifier(torch.cat(pooled, dim=1))
